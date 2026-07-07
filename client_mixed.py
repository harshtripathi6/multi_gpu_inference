"""Phase 2 mixed-traffic test.

Floods the gateway with a burst of production + research + benchmark
requests simultaneously and reports the queue_ms / gpu_time / e2e split
per class. With strict-priority dispatch you should see production
queue_ms stay low while benchmark requests wait at the back.

Also demonstrates quota rejection: the 'greedy' benchmark tenant sends
more than its in-flight quota and collects 429s.

Usage:
    python client_mixed.py --production 12 --research 6 --benchmark 8
"""

import argparse
import concurrent.futures as cf
import statistics
import time

import httpx

PROMPT = "a lighthouse on a cliff at golden hour"


def one(base_url: str, queue: str, tenant: str, i: int) -> dict:
    try:
        r = httpx.post(
            f"{base_url}/generate",
            json={"prompt": PROMPT, "steps": 2, "seed": i, "queue": queue,
                  "tenant_id": tenant, "return_image": False},
            timeout=300.0,
        )
        if r.status_code == 429:
            return {"queue": queue, "rejected": True, "detail": r.json()["detail"]}
        r.raise_for_status()
        d = r.json()
        d["rejected"] = False
        return d
    except httpx.HTTPStatusError as e:
        return {"queue": queue, "rejected": True, "detail": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--production", type=int, default=12)
    ap.add_argument("--research", type=int, default=6)
    ap.add_argument("--benchmark", type=int, default=8)
    args = ap.parse_args()

    tasks = (
        [("production", "tenant-prod") for _ in range(args.production)]
        + [("research", "tenant-research") for _ in range(args.research)]
        + [("benchmark", "tenant-greedy") for _ in range(args.benchmark)]
    )

    results = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futs = [ex.submit(one, args.url, q, t, i) for i, (q, t) in enumerate(tasks)]
        for f in cf.as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - t0

    print(f"\n{len(results)} requests finished in {wall:.1f}s\n")
    for qname in ("production", "research", "benchmark"):
        rs = [r for r in results if r["queue"] == qname and not r["rejected"]]
        rejected = sum(1 for r in results if r["queue"] == qname and r["rejected"])
        if not rs:
            print(f"{qname:<11} completed=0 rejected={rejected}")
            continue
        qms = [r["queue_ms"] for r in rs]
        e2e = [r["e2e_ms"] for r in rs]
        slo = [r["deadline_met"] for r in rs if r["deadline_met"] is not None]
        slo_str = (f"  slo_met={sum(slo)}/{len(slo)}" if slo else "")
        print(f"{qname:<11} completed={len(rs):<3} rejected={rejected:<3} "
              f"queue_ms p50={statistics.median(qms):7.0f} max={max(qms):7.0f}  "
              f"e2e p50={statistics.median(e2e):7.0f}{slo_str}")

    stats = httpx.get(f"{args.url}/stats", timeout=10).json()
    print(f"\n/stats: {stats}")


if __name__ == "__main__":
    main()