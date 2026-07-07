"""Request gateway (Phase 4).

Changes from Phase 3b:
  - Full per-request tracing (tracer.py): every request decomposes into
    queue_ms / ipc_ms / gpu_ms / respond_ms spans, with one dispatch
    attempt recorded per (re)dispatch so preempted requests show their
    whole history. GET /traces/{rid} returns a single request's timeline.
  - PREEMPTION FIX: evictions are now proportional to need. The Phase 3b
    runs showed 8 preemptions for a 2-request production probe -- the
    dispatcher kept firing while the first victim was still checkpointing
    and evicted a second batch for nothing. Now it evicts at most
    ceil(waiting_production / MAX_BATCH) minus evictions already in flight.
  - Background sampler records (queue depths, GPU util/mem, completed
    counter) twice a second into a ring buffer for the dashboard.
  - GET /metrics (JSON aggregates + history) and GET /dashboard (live view,
    no external assets -- works on cluster nodes without internet).

Run:
    SCHED_POLICY=priority_preempt MAX_BATCH=4 \
        uvicorn gateway:app --host 0.0.0.0 --port 8000
    # then open http://<node>:8000/dashboard (port-forward if needed)
"""

import asyncio
import math
import multiprocessing as mp
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import QUEUE_CONFIG
from estimator import CostEstimator
from gpu_monitor import GpuMonitor
from policies import make_policy
from scheduling import AdmissionError, Job, PriorityQueues
from tracer import Tracer
from worker import worker_main

NUM_GPUS = int(os.environ.get("NUM_GPUS", "2"))
MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/sd-turbo")
WORKER_READY_TIMEOUT_S = int(os.environ.get("WORKER_READY_TIMEOUT_S", "600"))
SCHED_POLICY = os.environ.get("SCHED_POLICY", "priority_preempt")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "4"))
BATCH_WAIT_MS = float(os.environ.get("BATCH_WAIT_MS", "10"))

ctx = mp.get_context("spawn")

state = {
    "workers": [], "request_queues": [], "control_queues": [],
    "response_queue": None, "ready": {},
    "busy": {},               # gpu_id -> batch group dict | None
    "pending": {},            # request_id -> (Future, Job)
    "queues": PriorityQueues(),
    "estimator": CostEstimator(),
    "policy": make_policy(SCHED_POLICY),
    "monitor": GpuMonitor(),
    "tracer": Tracer(),
    "history": deque(maxlen=600),   # ~5 min at 0.5s sampling
    "dispatch_event": None, "loop": None,
    "stats": {
        "completed": 0, "rejected": 0, "preemptions": 0, "resumes": 0,
        "batches_dispatched": 0, "batch_size_hist": {},
        "per_gpu_completed": {}, "per_queue_completed": {},
        "slo_met": 0, "slo_missed": 0,
    },
}


def _batch_key(job: Job):
    start = job.resume_state["step_index"] if job.resume_state else 0
    p = job.payload
    return (p["width"], p["height"], p["steps"], start)


# --------------------------------------------------------------------------
# Worker response plumbing
# --------------------------------------------------------------------------
def _drain_responses():
    rq = state["response_queue"]
    while True:
        msg = rq.get()
        if msg is None:
            break
        if msg["type"] == "ready":
            state["ready"][msg["gpu_id"]] = msg["load_seconds"]
            continue
        if state["loop"] is not None:
            state["loop"].call_soon_threadsafe(_on_worker_response, msg)


def _on_worker_response(msg: dict):
    gpu_id, rid = msg["gpu_id"], msg["request_id"]

    group = state["busy"].get(gpu_id)
    if group is not None and rid in group["jobs"]:
        group["remaining"] -= 1
        if group["remaining"] == 0:
            state["busy"][gpu_id] = None

    if msg["type"] == "preempted":
        entry = state["pending"].get(rid)
        if entry is not None:
            _, job = entry
            job.resume_state = {"latents": msg["latents"],
                                "shape": msg["shape"],
                                "step_index": msg["step_index"]}
            job.preempt_requested = False
            job.preempt_count += 1
            state["queues"].requeue(job)
            state["stats"]["preemptions"] += 1
            state["tracer"].preempted(rid, msg["step_index"])
    else:  # result | error
        entry = state["pending"].pop(rid, None)
        if entry is not None:
            fut, job = entry
            state["queues"].release(job)
            state["tracer"].complete(rid, msg, ok=(msg["type"] == "result"))
            if not fut.done():
                fut.set_result(msg)

    state["dispatch_event"].set()


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
async def dispatcher():
    ev = state["dispatch_event"]
    policy = state["policy"]
    loop = state["loop"]
    while True:
        await ev.wait()
        ev.clear()

        # 1) fill free GPUs
        while True:
            free = [g for g, grp in state["busy"].items() if grp is None]
            if not free:
                break
            head = policy.select(state["queues"])
            if head is None:
                break
            mates = state["queues"].pop_mates(head, MAX_BATCH - 1, _batch_key)
            group_jobs = [head] + mates

            if (head.queue != "production" and len(group_jobs) < MAX_BATCH
                    and BATCH_WAIT_MS > 0):
                age_ms = (time.perf_counter() - head.arrival) * 1000.0
                if age_ms < BATCH_WAIT_MS:
                    for j in reversed(group_jobs):
                        state["queues"].requeue(j)
                    loop.call_later((BATCH_WAIT_MS - age_ms) / 1000.0 + 0.001,
                                    ev.set)
                    break

            gpu_id = min(free, key=lambda g: state["monitor"].util(g))
            batch_id = uuid.uuid4().hex
            state["busy"][gpu_id] = {
                "batch_id": batch_id, "head": head,
                "jobs": {j.request_id: j for j in group_jobs},
                "remaining": len(group_jobs),
            }

            st = state["stats"]
            st["batches_dispatched"] += 1
            bs = str(len(group_jobs))
            st["batch_size_hist"][bs] = st["batch_size_hist"].get(bs, 0) + 1
            st["resumes"] += sum(1 for j in group_jobs if j.resume_state)
            for j in group_jobs:
                state["tracer"].dispatch(j.request_id, gpu_id, batch_id,
                                         len(group_jobs),
                                         j.resume_state is not None)

            w, h, steps, start = _batch_key(head)
            state["request_queues"][gpu_id].put({
                "type": "generate_batch",
                "batch_id": batch_id,
                "width": w, "height": h, "steps": steps, "start_index": start,
                "preemptible": head.preemptible,
                "jobs": [{
                    "request_id": j.request_id,
                    "prompt": j.payload["prompt"],
                    "seed": j.payload.get("seed"),
                    "resume_latents": (j.resume_state["latents"]
                                       if j.resume_state else None),
                    "return_image": j.payload.get("return_image", True),
                } for j in group_jobs],
            })

        # 2) preemption, proportional to need (Phase 4 fix)
        waiting = state["queues"].waiting("production")
        if policy.preemptive and waiting > 0:
            if all(grp is not None for grp in state["busy"].values()):
                in_flight_evictions = sum(
                    1 for grp in state["busy"].values()
                    if grp is not None and grp["head"].preempt_requested)
                needed = math.ceil(waiting / MAX_BATCH) - in_flight_evictions
                if needed > 0:
                    victims = sorted(
                        ((g, grp) for g, grp in state["busy"].items()
                         if grp["head"].preemptible
                         and not grp["head"].preempt_requested
                         and grp["head"].priority
                             > QUEUE_CONFIG["production"]["priority"]),
                        key=lambda v: -v[1]["head"].priority)
                    for gpu_id, grp in victims[:needed]:
                        grp["head"].preempt_requested = True
                        state["control_queues"][gpu_id].put(
                            {"type": "preempt", "batch_id": grp["batch_id"]})
                        state["tracer"].event(
                            "preempt_signal",
                            f"gpu{gpu_id} batch={grp['batch_id'][:8]} "
                            f"size={len(grp['jobs'])} queue={grp['head'].queue} "
                            f"(prod waiting={waiting})")


# --------------------------------------------------------------------------
# History sampler (feeds the dashboard time-series)
# --------------------------------------------------------------------------
async def sampler():
    while True:
        snap = state["monitor"].snapshot()
        state["history"].append({
            "ts": time.time(),
            "depths": state["queues"].depths(),
            "busy": {g: (len(grp["jobs"]) if grp else 0)
                     for g, grp in state["busy"].items()},
            "util": {g: snap.get(g, {}).get("util_pct") for g in range(NUM_GPUS)},
            "mem": {g: snap.get(g, {}).get("mem_used_gb") for g in range(NUM_GPUS)},
            "completed": state["stats"]["completed"],
            "preemptions": state["stats"]["preemptions"],
        })
        await asyncio.sleep(0.5)


# --------------------------------------------------------------------------
# Lifespan
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["loop"] = asyncio.get_running_loop()
    state["dispatch_event"] = asyncio.Event()
    state["response_queue"] = ctx.Queue()
    state["monitor"].start()

    for gpu_id in range(NUM_GPUS):
        req_q, ctl_q = ctx.Queue(), ctx.Queue()
        p = ctx.Process(target=worker_main,
                        args=(gpu_id, MODEL_ID, req_q, ctl_q,
                              state["response_queue"],
                              tuple(range(1, MAX_BATCH + 1)),   # warm batch 1..MAX_BATCH
                              ((512, 512), (768, 768))),        # both demo resolutions
                        daemon=True, name=f"gpu-worker-{gpu_id}")
        p.start()
        state["request_queues"].append(req_q)
        state["control_queues"].append(ctl_q)
        state["workers"].append(p)
        state["busy"][gpu_id] = None
        state["stats"]["per_gpu_completed"][gpu_id] = 0

    threading.Thread(target=_drain_responses, daemon=True).start()

    deadline = time.time() + WORKER_READY_TIMEOUT_S
    while len(state["ready"]) < NUM_GPUS:
        if time.time() > deadline:
            raise RuntimeError(f"workers ready {sorted(state['ready'])} of {NUM_GPUS}")
        if not all(p.is_alive() for p in state["workers"]):
            raise RuntimeError("a worker died during startup")
        await asyncio.sleep(1.0)

    est = "calibrated" if state["estimator"].calibrated else "heuristic (run calibrate.py)"
    print(f"[gateway] {NUM_GPUS} workers ready | policy={state['policy'].name} "
          f"| max_batch={MAX_BATCH} wait={BATCH_WAIT_MS}ms | estimator={est} "
          f"| nvml={state['monitor'].available}")

    dispatch_task = asyncio.create_task(dispatcher())
    sampler_task = asyncio.create_task(sampler())
    yield
    dispatch_task.cancel()
    sampler_task.cancel()
    for q in state["request_queues"]:
        q.put(None)
    state["response_queue"].put(None)


app = FastAPI(title="Mini GPU Inference Scheduler — Phase 4", lifespan=lifespan)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = Field(default=2, ge=1, le=50)
    width: int = Field(default=512, multiple_of=8, ge=256, le=1024)
    height: int = Field(default=512, multiple_of=8, ge=256, le=1024)
    seed: int | None = None
    queue: Literal["production", "research", "benchmark"] = "production"
    tenant_id: str = "default"
    return_image: bool = True


@app.post("/generate")
async def generate(req: GenerateRequest):
    if len(state["ready"]) < NUM_GPUS:
        raise HTTPException(503, "workers still loading")

    request_id = uuid.uuid4().hex
    est = state["estimator"].estimate(req.steps, req.width, req.height, batch=1)

    job = Job(
        request_id=request_id, queue=req.queue, tenant=req.tenant_id, est=est,
        payload={"request_id": request_id, "prompt": req.prompt,
                 "steps": req.steps, "width": req.width, "height": req.height,
                 "seed": req.seed, "return_image": req.return_image},
    )

    try:
        state["queues"].admit(job)
    except AdmissionError as e:
        state["stats"]["rejected"] += 1
        raise HTTPException(e.status, e.detail)

    state["tracer"].admit(request_id, req.queue, req.tenant_id, est.gpu_ms)
    fut: asyncio.Future = state["loop"].create_future()
    state["pending"][request_id] = (fut, job)
    state["dispatch_event"].set()

    msg = await fut
    done = time.perf_counter()

    if msg["type"] == "error":
        raise HTTPException(500, detail=msg["error"])

    queue_ms = (job.dispatched_at - job.arrival) * 1000.0
    e2e_ms = (done - job.arrival) * 1000.0
    deadline_met = None
    if job.deadline is not None:
        deadline_met = done <= job.deadline
        state["stats"]["slo_met" if deadline_met else "slo_missed"] += 1

    st = state["stats"]
    st["completed"] += 1
    st["per_gpu_completed"][msg["gpu_id"]] += 1
    st["per_queue_completed"][req.queue] = st["per_queue_completed"].get(req.queue, 0) + 1

    resp = {
        "request_id": request_id,
        "queue": req.queue,
        "gpu_id": msg["gpu_id"],
        "batch_size": msg.get("batch_size", 1),
        "queue_ms": round(queue_ms, 1),
        "gpu_time_ms": msg["gpu_time_ms"],
        "e2e_ms": round(e2e_ms, 1),
        "est_gpu_ms": est.gpu_ms,
        "deadline_met": deadline_met,
        "preempt_count": job.preempt_count,
    }
    if req.return_image and "image_b64" in msg:
        resp["image_b64"] = msg["image_b64"]
    return resp


@app.get("/health")
async def health():
    return {"workers_ready": sorted(state["ready"].keys()),
            "load_seconds": state["ready"],
            "policy": state["policy"].name,
            "max_batch": MAX_BATCH,
            "batch_wait_ms": BATCH_WAIT_MS}


@app.get("/stats")
async def stats():
    s = state["stats"]
    slo_total = s["slo_met"] + s["slo_missed"]
    return {
        "policy": state["policy"].name,
        "max_batch": MAX_BATCH,
        "queue_depths": state["queues"].depths(),
        "busy": {g: ({"batch_id": grp["batch_id"][:8],
                      "size": len(grp["jobs"]),
                      "queue": grp["head"].queue} if grp else None)
                 for g, grp in state["busy"].items()},
        "completed": s["completed"],
        "rejected": s["rejected"],
        "preemptions": s["preemptions"],
        "resumes": s["resumes"],
        "batches_dispatched": s["batches_dispatched"],
        "batch_size_hist": s["batch_size_hist"],
        "per_gpu_completed": s["per_gpu_completed"],
        "per_queue_completed": s["per_queue_completed"],
        "production_slo_attainment":
            round(s["slo_met"] / slo_total, 3) if slo_total else None,
        "gpu_telemetry": state["monitor"].snapshot(),
    }


@app.get("/metrics")
async def metrics():
    s = state["stats"]
    slo_total = s["slo_met"] + s["slo_missed"]
    return {
        "policy": state["policy"].name,
        "max_batch": MAX_BATCH,
        "counters": {
            "completed": s["completed"], "rejected": s["rejected"],
            "preemptions": s["preemptions"], "resumes": s["resumes"],
            "batches": s["batches_dispatched"],
            "slo_attainment": (round(s["slo_met"] / slo_total, 3)
                               if slo_total else None),
        },
        "batch_size_hist": s["batch_size_hist"],
        "latency": state["tracer"].summary(window_s=120),
        "recent": state["tracer"].recent(25),
        "events": state["tracer"].recent_events(30),
        "history": list(state["history"]),
    }


@app.get("/traces/{rid}")
async def trace(rid: str):
    tr = state["tracer"].get(rid)
    if tr is None:
        raise HTTPException(404, "unknown request_id")
    return tr


@app.get("/dashboard")
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return HTMLResponse(html_path.read_text())