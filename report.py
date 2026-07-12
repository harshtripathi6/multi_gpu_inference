"""Regression report (Phase 5).

Reads results.db and produces:
  1. A comparison table grouped by (policy, max_batch, rps): per-class
     p50/p99 e2e, production SLO attainment, throughput, rejection rate,
     preemptions, mean GPU utilization.
  2. (--plots) The comparison charts, one line per (policy, max_batch):
       plots/prod_p99_vs_load.png    <- the money plot
       plots/slo_vs_load.png
       plots/throughput_vs_load.png

Usage:
    python report.py
    python report.py --plots
    python report.py --db results.db --classes production research
"""

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, round(p / 100 * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def load(db_path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    runs = [dict(r) for r in db.execute(
        "SELECT * FROM runs ORDER BY policy, max_batch, rps, ts")]
    reqs = defaultdict(list)
    for r in db.execute("SELECT * FROM requests"):
        reqs[r["run_id"]].append(dict(r))
    db.close()
    return runs, reqs


def summarize(runs, reqs):
    """One summary row per run; repeated (policy,max_batch,rps) runs are
    kept separate so regressions across commits stay visible."""
    rows = []
    for run in runs:
        rr = reqs.get(run["run_id"], [])
        row = {"run_id": run["run_id"], "policy": run["policy"],
               "max_batch": run["max_batch"], "rps": run["rps"],
               "git": run["git"],
               "throughput": (round(run["completed"] / run["duration_s"], 1)
                              if run["duration_s"] else None),
               "rejected_pct": round(100 * run["rejected"] /
                                     max(1, run["completed"] + run["rejected"]), 1),
               "preemptions": run["preemptions"],
               "avg_util": run["avg_util"]}
        for cls in ("production", "research", "benchmark"):
            ok = sorted(r["e2e_ms"] for r in rr
                        if r["class"] == cls and not r["rejected"]
                        and r["e2e_ms"] is not None)
            row[f"{cls}_p50"] = pct(ok, 50)
            row[f"{cls}_p99"] = pct(ok, 99)
        slo = [r["slo_met"] for r in rr
               if r["class"] == "production" and r["slo_met"] is not None]
        row["slo_pct"] = round(100 * sum(slo) / len(slo), 1) if slo else None
        rows.append(row)
    return rows


def print_table(rows, classes):
    cols = ["policy", "max_batch", "rps", "throughput", "slo_pct",
            "rejected_pct", "preemptions", "avg_util"]
    for c in classes:
        cols += [f"{c}_p50", f"{c}_p99"]
    fmt = lambda v: "–" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))


def make_plots(rows, outdir="plots"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(exist_ok=True)
    series = defaultdict(list)   # (policy, max_batch) -> [(rps, row)]
    for r in rows:
        series[(r["policy"], r["max_batch"])].append((r["rps"], r))
    for k in series:
        series[k].sort(key=lambda x: x[0])

    specs = [
        ("prod_p99_vs_load", "production_p99", "Production p99 e2e (ms)",
         "Production p99 vs offered load"),
        ("slo_vs_load", "slo_pct", "Production SLO attainment (%)",
         "SLO attainment vs offered load"),
        ("throughput_vs_load", "throughput", "Completed req/s",
         "Throughput vs offered load"),
    ]
    for fname, field, ylabel, title in specs:
        plt.figure(figsize=(7, 4.5))
        for (policy, mb), pts in series.items():
            xs = [p[0] for p in pts]
            ys = [p[1].get(field) for p in pts]
            plt.plot(xs, ys, marker="o", label=f"{policy} (b{mb})")
        plt.xlabel("offered load (req/s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = Path(outdir) / f"{fname}.png"
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results.db")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--classes", nargs="+",
                    default=["production", "research", "benchmark"])
    args = ap.parse_args()

    runs, reqs = load(args.db)
    if not runs:
        print("no runs in DB yet — run bench.py first")
        return
    rows = summarize(runs, reqs)
    print_table(rows, args.classes)
    if args.plots:
        make_plots(rows)


if __name__ == "__main__":
    main()