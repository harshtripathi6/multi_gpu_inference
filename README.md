# Mini Multi-GPU Inference Scheduler

A miniature but production-shaped scheduler for **diffusion image inference**
across multiple GPUs. Built to demonstrate GPU-aware scheduling reasoning:
the control plane (queues, admission, cost model, policies, preemption,
telemetry, benchmarking) is real; the data plane is scaled to a 2×H100
allocation. Model: SD-Turbo (1–4 step distilled diffusion).

Diffusion is a deliberate choice — it exposes scheduling properties that are
hard to demo with LLMs: **deterministic per-request cost** (known at
admission), **cheap step-level preemption** (checkpoint is one small latent
tensor, not a GB-scale KV cache), and **shape-constrained batching**.

```
clients ─HTTP─> gateway ─admit─> [ production | research | benchmark ]  queues
                (quota/SLO/VRAM)        │  policy.select() picks head
                                        │  + pop_mates(): same (w,h,steps,start)
                                        ▼  hold ≤ BATCH_WAIT_MS if underfilled (non-prod)
                          dispatcher ── least-utilized free GPU (NVML)
                           │    ▲                │ batch payload (+resume latents)
                           │    └── preempted ───┤
                           ▼                     ▼
                    control_q: preempt      worker (1 per GPU, pinned):
                    (by batch_id)           per-step [ preempt-check → batched
                                            UNet → scheduler.step ] → VAE decode
                                        │
                                   tracer + NVML history → /metrics, /dashboard
                                        │
                                   bench.py → results.db → report.py (plots)
```

## Verified results (2×H100, SD-Turbo)

- **Preemption** (40-step 768² background vs production burst):
  production p50 **1097 → 443 ms**, max 1319 → 670 ms; evicted jobs finish
  ~0.6 s later with **zero recomputed denoising** (latent checkpoint = one
  fp16 tensor).
- **Dynamic batching** (24-job 512²×4-step flood): **20.5 → 64.6 img/s
  (3.15×)**; research e2e p50 633 → 185 ms (queue drain dominated);
  mid-flood production probes held at ~60 ms e2e.
- **Policy comparison, production p99 e2e @ 15 rps**:
  FIFO 330 → priority 178 → EDF 141 → **priority_preempt 77 ms** (p50 ≈ 50 ms
  for all — policies differentiate only in the tail, as intended).
- **Debugging arc**: batching first *regressed* production probes to 680 ms
  with <half explained. Span tracing (queue/ipc/gpu/respond) attributed it to
  cold-shape first-use cost (cuDNN autotune + allocator growth): gpu_ms 504 on
  the first-ever batch-2, and cold flood steps delayed the preemption
  boundary. Fix: warm every (batch, resolution) shape at startup. Post-fix
  trace closes exactly: 5.7 + 1.1 + 53.1 + 0.5 = 60.4 ms. Tracing also
  exposed an over-preemption bug (2 batches evicted for a 2-request burst);
  evictions are now proportional to need.

## Components

| File | Role |
|------|------|
| `gateway.py` | FastAPI control plane: admission, dispatcher, preemption, tracing, metrics/dashboard endpoints |
| `worker.py` | Per-GPU pinned process; batched manual denoising loop with step-level preemption + resume |
| `scheduling.py` | Priority queues, admission control (depth/quota/VRAM), batch-mate selection, requeue |
| `policies.py` | Pluggable policies: `fifo`, `priority`, `edf`, `priority_preempt` |
| `estimator.py` | Cost + memory model: fits `latency = intercept + slope·steps` per shape |
| `calibrate.py` | Sweeps resolution × steps × batch to produce `calibration.json` |
| `gpu_monitor.py` | NVML telemetry thread (util/mem/temp) for placement + dashboard |
| `tracer.py` | Per-request span decomposition, rolling percentiles, event log |
| `bench.py` | Sustained Poisson mixed-traffic generator → SQLite regression DB |
| `report.py` | Comparison tables + policy-vs-load plots |
| `dashboard.html` | Self-contained live dashboard (no external assets; works air-gapped) |
| `client_*.py` | Focused demos: preemption A/B, batching A/B, admission/quota |

## Quickstart

```bash
# once, on a node with internet — keep HF cache off $HOME quota
export HF_HOME=/path/to/scratch/hf_cache
pip install torch --index-url https://download.pytorch.org/whl/cu121  # match cluster CUDA
pip install -r requirements.txt
huggingface-cli download stabilityai/sd-turbo

# calibrate the cost model (once)
python calibrate.py --gpu 0

# run the scheduler (pick a policy)
SCHED_POLICY=priority_preempt MAX_BATCH=4 \
    uvicorn gateway:app --host 0.0.0.0 --port 8000

# live dashboard:  ssh -L 8000:<node>:8000 <login-host>  →  localhost:8000/dashboard
```

## Demos

```bash
python client_preempt.py --long-steps 40 --production 6   # preemption A/B (vs SCHED_POLICY=priority)
python client_batch.py   --n 24 --steps 4 --probe         # batching throughput A/B (vs MAX_BATCH=1)
python client_mixed.py   --production 12 --research 6 --benchmark 8   # admission + quota (429s)
```

## Benchmark sweep + report

```bash
# for each policy, restart the gateway then sweep load:
SCHED_POLICY=fifo uvicorn gateway:app --port 8000
for rps in 5 10 15 20 25 30 35; do python bench.py --rps $rps --duration 20; done
# repeat for priority, edf, priority_preempt

python report.py            # comparison table from results.db
python report.py --plots    # prod_p99_vs_load.png (money plot), slo_vs_load.png, throughput_vs_load.png
```

`bench.py` reads the gateway's real policy/config from `/health` so runs
can't be mislabeled; records git commit, mean NVML utilization, and
preemption delta per run; treats 429s as backpressure (recorded, not
retried).

## Endpoints

`/generate` · `/health` · `/stats` · `/metrics` (aggregates + 120 s span
percentiles + time-series) · `/traces/{rid}` (full per-request timeline) ·
`/dashboard` (live).

## Configuration

| Env var | Default | Meaning |
|---------|---------|---------|
| `SCHED_POLICY` | `priority_preempt` | `fifo` / `priority` / `edf` / `priority_preempt` |
| `MAX_BATCH` | `4` | max jobs per shared UNet pass |
| `BATCH_WAIT_MS` | `10` | max hold to fill a non-production batch (latency/throughput knob) |
| `NUM_GPUS` | `2` | worker processes to spawn |
| `MODEL_ID` | `stabilityai/sd-turbo` | any diffusers text2img model |

## Design notes & honest limitations

- **Where the toy diverges from production**: real fleets use Ray Serve /
  Triton / K8s + Run:ai for placement, autoscaling, and fault tolerance.
  Building the scheduler by hand here is the point — it makes the mechanisms
  legible.
- **Cross-process timing** is safe: `perf_counter` is CLOCK_MONOTONIC
  (system-wide) on Linux, so worker and gateway spans are directly comparable
  on one node.
- **Preemption fidelity**: the ancestral sampler redraws per-step noise on
  resume and batches share one noise generator, so images aren't bit-identical
  across batch composition or preemption — acceptable for scheduling, not for
  reproducibility-critical serving.
- **Known next step — preemption thrash**: under heavy load, long jobs can be
  re-evicted repeatedly; a minimum run quantum (no re-eviction within N ms of
  resume) would bound this.
- **Batching reorders within a class** (mates skip non-matching shapes);
  cross-class priority is preserved because mates come only from the head's
  own queue.
