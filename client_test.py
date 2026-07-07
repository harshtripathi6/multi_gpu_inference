"""Phase 1 smoke test.

Fires N concurrent requests at the gateway, saves images, and prints
which GPU served each request plus latency stats. Verifies (a) both GPUs
are doing work and (b) end-to-end latency is sane.

Usage:
    python client_test.py --n 8 --concurrency 4
"""

import argparse
import base64
import concurrent.futures as cf
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = [
    "a cinematic photo of a red fox in snowy forest",
    "isometric voxel art of a tiny coffee shop",
    "watercolor painting of sailboats at dusk",
    "macro photo of a dew drop on a leaf",
]


def one_request(base_url: str, i: int, outdir: Path) -> dict:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{base_url}/generate",
        json={"prompt": PROMPTS[i % len(PROMPTS)], "steps": 2, "seed": i},
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    client_ms = (time.perf_counter() - t0) * 1000.0

    img_path = outdir / f"img_{i:03d}_gpu{data['gpu_id']}.png"
    img_path.write_bytes(base64.b64decode(data["image_b64"]))

    return {
        "i": i,
        "gpu_id": data["gpu_id"],
        "gpu_time_ms": data["gpu_time_ms"],
        "e2e_ms": data["e2e_ms"],
        "client_ms": round(client_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    health = httpx.get(f"{args.url}/health", timeout=10).json()
    print(f"health: {health}\n")

    results = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one_request, args.url, i, outdir) for i in range(args.n)]
        for f in cf.as_completed(futs):
            r = f.result()
            print(f"req {r['i']:03d}  gpu={r['gpu_id']}  "
                  f"gpu_time={r['gpu_time_ms']:7.1f}ms  e2e={r['e2e_ms']:7.1f}ms")
            results.append(r)
    wall_s = time.perf_counter() - t0

    per_gpu = {}
    for r in results:
        per_gpu[r["gpu_id"]] = per_gpu.get(r["gpu_id"], 0) + 1
    e2e = [r["e2e_ms"] for r in results]

    print(f"\n--- summary ---")
    print(f"requests: {len(results)} in {wall_s:.1f}s "
          f"({len(results)/wall_s:.2f} req/s)")
    print(f"per-GPU distribution: {per_gpu}")
    print(f"e2e latency ms: p50={statistics.median(e2e):.0f} "
          f"max={max(e2e):.0f}")
    print(f"images saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
