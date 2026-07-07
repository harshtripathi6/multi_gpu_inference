"""Phase 3b batching demo.

Floods the gateway with same-shape research jobs and measures throughput
plus the realized batch-size distribution. Run against two gateway configs
and compare:

    MAX_BATCH=1 SCHED_POLICY=priority_preempt uvicorn gateway:app --port 8000
    python client_batch.py

    MAX_BATCH=4 SCHED_POLICY=priority_preempt uvicorn gateway:app --port 8000
    python client_batch.py

Optionally (--probe) fires 2 production requests mid-flood to show that
batching does not hurt production latency: production either preempts the
running research batch or jumps the queue.

Predicted from calibration (512^2, 4 steps): batch 1 ~ 128ms/image,
batch 4 ~ 47ms/image -> ~2.7x throughput on the same hardware.

Usage:
    python client_batch.py --n 24 --steps 4 --probe
"""

import argparse
import concurrent.futures as cf
import statistics
import time

import httpx


def send(base_url, body):
    r = httpx.post(f"{base_url}/generate", json=body, timeout=600.0)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--probe", action="store_true",
                    help="send 2 production requests mid-flood")
    args = ap.parse_args()

    health = httpx.get(f"{args.url}/health", timeout=10).json()
    print(f"policy={health['policy']} max_batch={health['max_batch']} "
          f"batch_wait_ms={health['batch_wait_ms']}\n")

    # research quota is 4 in-flight per tenant -> spread across tenants
    tenants = [f"r{i}" for i in range((args.n + 3) // 4)]

    probe_results = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.n + 2) as ex:
        futs = [
            ex.submit(send, args.url, {
                "prompt": f"study of light and shadow, variation {i}",
                "steps": args.steps, "seed": i,
                "queue": "research", "tenant_id": tenants[i // 4],
                "return_image": False})
            for i in range(args.n)
        ]
        if args.probe:
            time.sleep(0.25)
            probe_futs = [
                ex.submit(send, args.url, {
                    "prompt": "product photo of a ceramic mug",
                    "steps": 2, "seed": 1000 + i, "queue": "production",
                    "return_image": False})
                for i in range(2)
            ]
            probe_results = [f.result() for f in probe_futs]
        results = [f.result() for f in futs]
    wall = time.perf_counter() - t0

    sizes = {}
    for r in results:
        sizes[r["batch_size"]] = sizes.get(r["batch_size"], 0) + 1
    e2e = [r["e2e_ms"] for r in results]
    per_img = [r["gpu_time_ms"] / r["batch_size"] for r in results]

    print(f"research flood: {len(results)} images in {wall:.2f}s "
          f"= {len(results)/wall:.1f} img/s")
    print(f"batch sizes seen (per request): {dict(sorted(sizes.items()))}")
    print(f"gpu ms/image: p50={statistics.median(per_img):.0f}")
    print(f"e2e ms: p50={statistics.median(e2e):.0f} max={max(e2e):.0f}")

    if probe_results:
        print("\nmid-flood production probes:")
        for p in probe_results:
            print(f"  prod e2e={p['e2e_ms']:7.1f}ms queue_ms={p['queue_ms']:6.1f} "
                  f"batch_size={p['batch_size']} slo_met={p['deadline_met']}")

    stats = httpx.get(f"{args.url}/stats", timeout=10).json()
    print(f"\nbatches_dispatched={stats['batches_dispatched']} "
          f"hist={stats['batch_size_hist']} "
          f"preemptions={stats['preemptions']} resumes={stats['resumes']}")


if __name__ == "__main__":
    main()