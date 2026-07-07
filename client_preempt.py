"""Phase 3a preemption demo.

Fills BOTH GPUs with long benchmark jobs (many steps), then fires a
production burst. Run it twice against two gateway configs:

    SCHED_POLICY=priority          -> production waits behind the long jobs
    SCHED_POLICY=priority_preempt  -> long jobs are checkpointed mid-denoise,
                                      production runs immediately, then the
                                      long jobs RESUME from their saved step

Compare the 'production e2e' line between the two runs, and check
preemptions/resumes in /stats plus preempt_count on the benchmark jobs.

Usage:
    python client_preempt.py --long-steps 40 --production 6
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
    ap.add_argument("--long-jobs", type=int, default=2)
    ap.add_argument("--long-steps", type=int, default=40)
    ap.add_argument("--production", type=int, default=6)
    args = ap.parse_args()

    policy = httpx.get(f"{args.url}/health", timeout=10).json()["policy"]
    print(f"gateway policy: {policy}\n")

    with cf.ThreadPoolExecutor(max_workers=args.long_jobs + args.production) as ex:
        # 1) occupy every GPU with a long background job
        long_futs = [
            ex.submit(send, args.url, {
                "prompt": "an extremely detailed fantasy map",
                "steps": args.long_steps, "width": 768, "height": 768,
                "seed": i, "queue": "benchmark",
                "tenant_id": f"bench-{i}",           # one tenant each: quota is 2
                "return_image": False})
            for i in range(args.long_jobs)
        ]

        time.sleep(0.8)  # let them get well into their denoising loops

        # 2) production burst arrives
        t0 = time.perf_counter()
        prod_futs = [
            ex.submit(send, args.url, {
                "prompt": "a product photo of a ceramic mug",
                "steps": 2, "seed": i, "queue": "production",
                "return_image": False})
            for i in range(args.production)
        ]
        prod = [f.result() for f in prod_futs]
        burst_s = time.perf_counter() - t0
        longs = [f.result() for f in long_futs]

    e2e = [p["e2e_ms"] for p in prod]
    slo = sum(1 for p in prod if p["deadline_met"])
    print(f"production burst ({len(prod)} reqs) drained in {burst_s:.2f}s")
    print(f"production e2e ms: p50={statistics.median(e2e):.0f} "
          f"max={max(e2e):.0f}  slo_met={slo}/{len(prod)}")
    for p in sorted(prod, key=lambda x: x["e2e_ms"]):
        print(f"  prod gpu={p['gpu_id']} queue_ms={p['queue_ms']:7.1f} "
              f"e2e={p['e2e_ms']:7.1f}")

    print("\nlong benchmark jobs:")
    for l in longs:
        print(f"  bench gpu={l['gpu_id']} steps_e2e={l['e2e_ms']:8.1f}ms "
              f"preempt_count={l['preempt_count']}")

    stats = httpx.get(f"{args.url}/stats", timeout=10).json()
    print(f"\npreemptions={stats['preemptions']} resumes={stats['resumes']} "
          f"slo_attainment={stats['production_slo_attainment']}")


if __name__ == "__main__":
    main()