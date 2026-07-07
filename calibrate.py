"""Calibration sweep (Phase 2).

Measures diffusion cost empirically on ONE GPU across
(resolution x steps x batch), producing calibration.json that the
CostEstimator fits. Run standalone, NOT through the gateway:

    python calibrate.py --gpu 0

Takes a few minutes with SD-Turbo. Key idea for the README: diffusion cost
is deterministic given (steps, resolution, batch), so a small empirical
table makes the admission-time estimator genuinely predictive -- unlike
LLM serving where output length is unknown.
"""

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("MODEL_ID", "stabilityai/sd-turbo"))
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    # Pin before torch import (same rule as worker.py).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from diffusers import AutoPipelineForText2Image

    print(f"loading {args.model} ...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    resolutions = [(512, 512), (768, 768)]
    steps_list = [1, 2, 4]
    batches = [1, 2, 4]
    prompt = "a calibration test image of a lighthouse"

    entries = []
    for (w, h) in resolutions:
        for batch in batches:
            for steps in steps_list:
                prompts = [prompt] * batch
                # warmup this exact shape (first run compiles kernels / resizes buffers)
                _ = pipe(prompt=prompts, num_inference_steps=steps,
                         guidance_scale=0.0, width=w, height=h)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                times = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    _ = pipe(prompt=prompts, num_inference_steps=steps,
                             guidance_scale=0.0, width=w, height=h)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000.0)

                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                total_ms = sum(times) / len(times)
                entry = {
                    "width": w, "height": h, "steps": steps, "batch": batch,
                    "total_ms_mean": round(total_ms, 1),
                    "per_image_ms": round(total_ms / batch, 1),
                    "peak_vram_gb": round(peak_gb, 2),
                }
                entries.append(entry)
                print(entry)

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                   "entries": entries}, f, indent=2)
    print(f"\nwrote {len(entries)} entries to {args.out}")


if __name__ == "__main__":
    main()