"""GPU worker (Phase 3b): batched manual denoising loop with preemption.

The worker now receives BATCHES -- N same-shape jobs sharing one UNet
forward pass per step. Per your H100 calibration this is strongly
sublinear (512^2: 65 -> 31 ms/image at batch 4), i.e. batch-1 serving
leaves most of the GPU idle.

Batch protocol (one message in, N messages out):
  in : {"type": "generate_batch", "batch_id", "width", "height", "steps",
        "start_index", "preemptible",
        "jobs": [{"request_id", "prompt", "seed",
                  "resume_latents": bytes|None, "return_image": bool}]}
  out: one {"type": "result"|"preempted"|"error"} message PER JOB.

Preemption targets the batch_id. On preempt the whole batch checkpoints:
the (B,4,h,w) latents tensor is sliced per job so each job carries its own
resume state. Jobs preempted together share (shape, steps, step_index) and
therefore the same batch key, so they re-form as a batch on resume.

Postprocessing is conditional: if no job in the batch wants the image,
VAE decode and PNG encode are skipped entirely (this was ~120ms of the
~205ms per-request overhead measured in the Phase 3a runs).
"""

import base64
import io
import os
import queue as queue_mod
import time
import traceback


def worker_main(gpu_id: int, model_id: str, request_q, control_q, response_q,
                warmup_batch_sizes=(1,), warmup_resolutions=((512, 512),)) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # before torch import

    import numpy as np
    import torch
    from diffusers import AutoPipelineForText2Image

    t0 = time.perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    comp = {
        "tokenizer": pipe.tokenizer,
        "text_encoder": pipe.text_encoder,
        "unet": pipe.unet,
        "vae": pipe.vae,
        "scheduler": pipe.scheduler,
        "image_processor": pipe.image_processor,
    }

    # Warm EVERY (batch_size, resolution) shape the scheduler can dispatch.
    # Lesson learned via tracing: the first forward pass of a NEW shape pays
    # cuDNN algorithm selection + allocator growth -- a cold batch-2 job
    # measured gpu_ms=504 vs ~40ms warm, and cold first steps delayed the
    # preemption boundary (queue_ms 208 vs ~25 expected). Warmup makes every
    # shape's first real request fast and keeps preempt latency ~one step.
    for (ww, wh) in warmup_resolutions:
        for b in warmup_batch_sizes:
            _run_batch(comp, {
                "batch_id": f"warmup-{ww}x{wh}-b{b}", "width": ww, "height": wh,
                "steps": 1, "start_index": 0, "preemptible": False,
                "jobs": [{"request_id": f"warmup-{k}", "prompt": "warmup",
                          "seed": k, "resume_latents": None,
                          "return_image": True}  # warm the VAE decode shapes too
                         for k in range(b)],
            }, control_q, gpu_id, np, torch)

    response_q.put({"type": "ready", "gpu_id": gpu_id,
                    "load_seconds": round(time.perf_counter() - t0, 1)})

    while True:
        batch = request_q.get()
        if batch is None:
            break
        try:
            msgs = _run_batch(comp, batch, control_q, gpu_id, np, torch)
        except Exception:
            err = traceback.format_exc()
            msgs = [{"type": "error", "request_id": j["request_id"],
                     "gpu_id": gpu_id, "error": err} for j in batch["jobs"]]
        for m in msgs:
            response_q.put(m)


def _preempt_requested(control_q, batch_id: str) -> bool:
    """Drain control messages; True if any targets the CURRENT batch."""
    hit = False
    while True:
        try:
            msg = control_q.get_nowait()
        except queue_mod.Empty:
            return hit
        if msg.get("type") == "preempt" and msg.get("batch_id") == batch_id:
            hit = True  # keep draining stale messages


def _run_batch(comp, batch, control_q, gpu_id: int, np, torch) -> list[dict]:
    jobs = batch["jobs"]
    B = len(jobs)
    steps, w, h = batch["steps"], batch["width"], batch["height"]
    start = batch["start_index"]
    scheduler, unet, vae = comp["scheduler"], comp["unet"], comp["vae"]

    t_start = time.perf_counter()
    with torch.inference_mode():
        # --- batched text encoding ---
        tok = comp["tokenizer"](
            [j["prompt"] for j in jobs], padding="max_length",
            max_length=comp["tokenizer"].model_max_length,
            truncation=True, return_tensors="pt")
        embeds = comp["text_encoder"](tok.input_ids.to("cuda"))[0]

        scheduler.set_timesteps(steps, device="cuda")
        timesteps = scheduler.timesteps

        # --- per-job latents (fresh randn with own seed, or resume slice) ---
        lat_shape = (1, unet.config.in_channels, h // 8, w // 8)
        lat_list = []
        for j in jobs:
            if j["resume_latents"] is not None:
                lat = torch.from_numpy(
                    np.frombuffer(j["resume_latents"], dtype=np.float16)
                    .reshape(lat_shape).copy()).to("cuda")
            else:
                gen = (torch.Generator(device="cuda").manual_seed(j["seed"])
                       if j.get("seed") is not None else None)
                lat = torch.randn(lat_shape, generator=gen, device="cuda",
                                  dtype=torch.float16) * scheduler.init_noise_sigma
            lat_list.append(lat)
        latents = torch.cat(lat_list)

        # One generator drives the ancestral per-step noise for the whole
        # batch (documented caveat: per-image reproducibility depends on
        # batch composition; acceptable for a scheduling project).
        step_gen = (torch.Generator(device="cuda").manual_seed(jobs[0]["seed"])
                    if jobs[0].get("seed") is not None else None)

        # --- denoising loop, preemption boundary at every step ---
        for i in range(start, len(timesteps)):
            if batch["preemptible"] and _preempt_requested(control_q, batch["batch_id"]):
                t_now = time.perf_counter()
                lat_cpu = latents.to("cpu").numpy()
                return [{
                    "type": "preempted",
                    "request_id": j["request_id"],
                    "gpu_id": gpu_id,
                    "step_index": i,
                    "steps_total": steps,
                    "latents": lat_cpu[k:k + 1].tobytes(),
                    "shape": list(lat_shape),
                    "batch_size": B,
                    "t_start": t_start,
                    "t_end": t_now,
                } for k, j in enumerate(jobs)]
            t = timesteps[i]
            latent_in = scheduler.scale_model_input(latents, t)
            noise = unet(latent_in, t, encoder_hidden_states=embeds).sample
            latents = scheduler.step(noise, t, latents,
                                     generator=step_gen).prev_sample

        # --- conditional postprocessing ---
        need = [j["return_image"] for j in jobs]
        pils = [None] * B
        if any(need):
            img_t = vae.decode(latents / vae.config.scaling_factor).sample
            decoded = comp["image_processor"].postprocess(img_t, output_type="pil")
            for k in range(B):
                if need[k]:
                    pils[k] = decoded[k]

    t_end = time.perf_counter()
    gpu_ms = round((t_end - t_start) * 1000.0, 1)
    out = []
    for k, j in enumerate(jobs):
        msg = {"type": "result", "request_id": j["request_id"],
               "gpu_id": gpu_id, "gpu_time_ms": gpu_ms, "batch_size": B,
               "t_start": t_start, "t_end": t_end}
        if pils[k] is not None:
            buf = io.BytesIO()
            pils[k].save(buf, format="PNG")
            msg["image_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append(msg)
    return out