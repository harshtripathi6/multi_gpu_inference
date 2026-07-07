"""Benchmark harness (Phase 5).

Drives SUSTAINED Poisson mixed traffic at a running gateway (short bursts
were shown in Phase 3b/4 to distort throughput via ramp/tail effects),
then records every request plus run metadata into results.db (SQLite),
keyed by (policy, max_batch, rps, git commit). report.py turns the DB
into comparison tables and plots.

The gateway's actual config (policy, max_batch, batch_wait) is read from
/health so runs can't be mislabeled.

Traffic mix (default 70/20/10):
  production  512^2, 2 steps, 2s SLO           (interactive users)
  research    512^2, 4 steps                    (batchable best-effort)
  benchmark   50% 512^2 x8 steps, 50% 768^2 x20 (long, preemptible)

429 rejections are recorded, not retried -- backpressure is a result,
not an error.

Usage (one run per invocation; sweep by re-invoking):
    SCHED_POLICY=fifo uvicorn gateway:app --port 8000
    python bench.py --rps 10 --duration 20
    python bench.py --rps 20 --duration 20
    # restart gateway with the next policy, repeat, then: python report.py
"""

import argparse
import asyncio
import json
import random
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

import httpx

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, ts REAL, policy TEXT, max_batch INTEGER,
    batch_wait_ms REAL, rps REAL, duration_s REAL, mix TEXT, git TEXT,
    notes TEXT, completed INTEGER, rejected INTEGER, preemptions INTEGER,
    avg_util REAL
);
CREATE TABLE IF NOT EXISTS requests (
    run_id TEXT, class TEXT, e2e_ms REAL, queue_ms REAL, gpu_ms REAL,
    batch_size INTEGER, slo_met INTEGER, preempts INTEGER, rejected INTEGER
);
"""


def sample_request(i: int, mix) -> dict:
    cls = random.choices(("production", "research", "benchmark"), mix)[0]
    if cls == "production":
        spec = {"steps": 2, "width": 512, "height": 512,
                "tenant_id": f"prod-{i % 4}"}
    elif cls == "research":
        spec = {"steps": 4, "width": 512, "height": 512,
                "tenant_id": f"res-{i % 6}"}
    else:
        if random.random() < 0.5:
            spec = {"steps": 8, "width": 512, "height": 512}
        else:
            spec = {"steps": 20, "width": 768, "height": 768}
        spec["tenant_id"] = f"bench-{i % 6}"
    spec.update({"queue": cls, "prompt": f"benchmark workload {i}",
                 "seed": i, "return_image": False})
    return spec


async def fire(client, url, spec, out):
    try:
        r = await client.post(f"{url}/generate", json=spec)
        if r.status_code == 429:
            out.append({"class": spec["queue"], "rejected": 1})
        elif r.status_code == 200:
            d = r.json()
            out.append({
                "class": d["queue"], "e2e_ms": d["e2e_ms"],
                "queue_ms": d["queue_ms"], "gpu_ms": d["gpu_time_ms"],
                "batch_size": d["batch_size"],
                "slo_met": (None if d["deadline_met"] is None
                            else int(d["deadline_met"])),
                "preempts": d["preempt_count"], "rejected": 0,
            })
        else:
            out.append({"class": spec["queue"], "rejected": 1})
    except Exception:
        out.append({"class": spec["queue"], "rejected": 1})


async def run_load(url, rps, duration, mix):
    out = []
    tasks = []
    async with httpx.AsyncClient(timeout=300) as client:
        loop = asyncio.get_running_loop()
        t_end = loop.time() + duration
        i = 0
        while loop.time() < t_end:
            await asyncio.sleep(random.expovariate(rps))
            tasks.append(asyncio.create_task(
                fire(client, url, sample_request(i, mix), out)))
            i += 1
        await asyncio.gather(*tasks)
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True,
            timeout=5).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rps", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--mix", default="0.7,0.2,0.1",
                    help="production,research,benchmark weights")
    ap.add_argument("--db", default="results.db")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    random.seed(args.seed)
    mix = tuple(float(x) for x in args.mix.split(","))

    health = httpx.get(f"{args.url}/health", timeout=10).json()
    before = httpx.get(f"{args.url}/stats", timeout=10).json()

    print(f"run: policy={health['policy']} max_batch={health['max_batch']} "
          f"rps={args.rps} duration={args.duration}s mix={mix}")

    t0_wall = time.time()
    results = asyncio.run(run_load(args.url, args.rps, args.duration, mix))
    t1_wall = time.time()

    after = httpx.get(f"{args.url}/stats", timeout=10).json()
    metrics = httpx.get(f"{args.url}/metrics", timeout=10).json()

    # mean NVML utilization across GPUs within the run window
    utils = [u for h in metrics.get("history", [])
             if t0_wall <= h["ts"] <= t1_wall
             for u in h["util"].values() if u is not None]
    avg_util = round(sum(utils) / len(utils), 1) if utils else None

    completed = sum(1 for r in results if not r["rejected"])
    rejected = sum(1 for r in results if r["rejected"])
    preempts = after["preemptions"] - before["preemptions"]

    run_id = uuid.uuid4().hex[:12]
    db = sqlite3.connect(args.db)
    db.executescript(SCHEMA)
    db.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, t0_wall, health["policy"], health["max_batch"],
         health["batch_wait_ms"], args.rps, args.duration,
         json.dumps(mix), git_commit(), args.notes,
         completed, rejected, preempts, avg_util))
    db.executemany(
        "INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?)",
        [(run_id, r["class"], r.get("e2e_ms"), r.get("queue_ms"),
          r.get("gpu_ms"), r.get("batch_size"), r.get("slo_met"),
          r.get("preempts", 0), r["rejected"]) for r in results])
    db.commit()
    db.close()

    thr = completed / (t1_wall - t0_wall)
    prod = sorted(r["e2e_ms"] for r in results
                  if r["class"] == "production" and not r["rejected"])
    p99 = prod[min(len(prod) - 1, round(0.99 * (len(prod) - 1)))] if prod else None
    print(f"done: run_id={run_id} completed={completed} rejected={rejected} "
          f"throughput={thr:.1f} req/s preemptions={preempts} "
          f"avg_gpu_util={avg_util}% prod_p99={p99}ms")


if __name__ == "__main__":
    main()