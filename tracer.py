"""Per-request tracing (Phase 4).

Motivated by a real observation from the Phase 3b runs: production probes
showed e2e=680ms while their queue_ms (192) + gpu time (~50) explained less
than half of it. The missing spans were invisible because we only had two
timestamps. This tracer records the full lifecycle:

    arrival -> dispatch (per attempt; preemption means multiple attempts)
            -> worker_start -> worker_end -> resolved

and decomposes every request into spans:
    queue_ms   arrival to FIRST dispatch
    ipc_ms     last dispatch to worker actually starting the batch
               (mp.Queue transfer + worker still draining its previous batch)
    gpu_ms     worker start to worker end (the batch's compute)
    respond_ms worker end to the gateway resolving the future

Cross-process timestamps are safe here: time.perf_counter() is
CLOCK_MONOTONIC on Linux, which is system-wide, so worker and gateway
clocks are directly comparable on the same node.
"""

import threading
import time
from collections import deque


def _pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, round(p / 100 * (len(sorted_vals) - 1)))
    return round(sorted_vals[k], 1)


class Tracer:
    def __init__(self, keep_done=5000, keep_events=300):
        self._lock = threading.Lock()
        self.active = {}                      # rid -> trace dict
        self.done = deque(maxlen=keep_done)   # completed traces
        self.events = deque(maxlen=keep_events)

    # ---------------- lifecycle marks ----------------
    def admit(self, rid, queue, tenant, est_gpu_ms):
        with self._lock:
            self.active[rid] = {
                "rid": rid, "queue": queue, "tenant": tenant,
                "est_gpu_ms": est_gpu_ms,
                "arrival": time.perf_counter(),
                "attempts": [], "preempts": 0,
            }

    def dispatch(self, rid, gpu_id, batch_id, batch_size, resumed):
        with self._lock:
            tr = self.active.get(rid)
            if tr is not None:
                tr["attempts"].append({
                    "t_dispatch": time.perf_counter(), "gpu": gpu_id,
                    "batch_id": batch_id[:8], "batch_size": batch_size,
                    "resumed": resumed,
                })

    def preempted(self, rid, step_index):
        with self._lock:
            tr = self.active.get(rid)
            if tr is not None:
                tr["preempts"] += 1
                if tr["attempts"]:
                    tr["attempts"][-1]["preempted_at_step"] = step_index

    def event(self, kind, detail):
        with self._lock:
            self.events.append({"wall_ts": time.time(), "kind": kind,
                                "detail": detail})

    def complete(self, rid, msg, ok=True):
        """msg may carry t_start/t_end from the worker (perf_counter)."""
        now = time.perf_counter()
        with self._lock:
            tr = self.active.pop(rid, None)
            if tr is None:
                return None
            tr["ok"] = ok
            tr["wall_done"] = time.time()
            tr["e2e_ms"] = round((now - tr["arrival"]) * 1000, 1)
            if tr["attempts"]:
                first, last = tr["attempts"][0], tr["attempts"][-1]
                tr["queue_ms"] = round(
                    (first["t_dispatch"] - tr["arrival"]) * 1000, 1)
                tr["final_batch_size"] = last["batch_size"]
                if msg.get("t_start") is not None:
                    tr["ipc_ms"] = round(
                        (msg["t_start"] - last["t_dispatch"]) * 1000, 1)
                    tr["gpu_ms"] = round(
                        (msg["t_end"] - msg["t_start"]) * 1000, 1)
                    tr["respond_ms"] = round((now - msg["t_end"]) * 1000, 1)
            self.done.append(tr)
            return tr

    # ---------------- aggregation ----------------
    def summary(self, window_s=120):
        cutoff = time.time() - window_s
        out = {}
        with self._lock:
            recent = [t for t in self.done if t["wall_done"] >= cutoff and t["ok"]]
        for qname in ("production", "research", "benchmark"):
            rows = [t for t in recent if t["queue"] == qname]
            if not rows:
                continue
            spans = {}
            for k in ("e2e_ms", "queue_ms", "ipc_ms", "gpu_ms", "respond_ms"):
                vals = sorted(t[k] for t in rows if k in t)
                spans[k] = {"p50": _pct(vals, 50), "p99": _pct(vals, 99)}
            spans["n"] = len(rows)
            spans["preempts"] = sum(t["preempts"] for t in rows)
            out[qname] = spans
        return out

    def recent(self, n=25):
        with self._lock:
            rows = list(self.done)[-n:]
        keep = ("rid", "queue", "e2e_ms", "queue_ms", "ipc_ms", "gpu_ms",
                "respond_ms", "final_batch_size", "preempts", "ok")
        return [{k: t.get(k) for k in keep} for t in reversed(rows)]

    def get(self, rid):
        with self._lock:
            if rid in self.active:
                return dict(self.active[rid])
            for t in self.done:
                if t["rid"] == rid:
                    return dict(t)
        return None

    def recent_events(self, n=40):
        with self._lock:
            return list(self.events)[-n:][::-1]