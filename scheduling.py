"""Priority queues + admission control (Phase 3a).

New vs Phase 2:
  - Job carries preemption fields (preemptible, resume_state, preempt_count).
    Production jobs are never preempted; research/benchmark are.
  - pop_by(key): generic "pop the job minimizing key" across all queues --
    this is what lets FIFO and EDF policies share one queue structure.
  - requeue(job): put a preempted job back at the FRONT of its class queue
    WITHOUT re-admission (it still holds its tenant quota slot).
"""

import time
from collections import deque
from dataclasses import dataclass, field

from config import GPU_VRAM_BUDGET_GB, QUEUE_CONFIG
from estimator import Estimate


@dataclass
class Job:
    request_id: str
    queue: str                 # "production" | "research" | "benchmark"
    tenant: str
    payload: dict
    est: Estimate
    arrival: float = field(default_factory=time.perf_counter)
    deadline: float | None = None
    dispatched_at: float | None = None
    # --- preemption ---
    preemptible: bool = True
    resume_state: dict | None = None
    preempt_requested: bool = False
    preempt_count: int = 0

    def __post_init__(self):
        cfg = QUEUE_CONFIG[self.queue]
        if cfg["slo_ms"] is not None:
            self.deadline = self.arrival + cfg["slo_ms"] / 1000.0
        self.preemptible = cfg.get("preemptible", True)
        self.payload["preemptible"] = self.preemptible

    @property
    def priority(self) -> int:
        return QUEUE_CONFIG[self.queue]["priority"]


class AdmissionError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail


class PriorityQueues:
    def __init__(self):
        self.queues = {name: deque() for name in QUEUE_CONFIG}
        self.tenant_inflight = {name: {} for name in QUEUE_CONFIG}

    # ---------------- admission ----------------
    def admit(self, job: Job) -> None:
        cfg = QUEUE_CONFIG[job.queue]

        if job.est.vram_gb > GPU_VRAM_BUDGET_GB:
            raise AdmissionError(
                413, f"estimated VRAM {job.est.vram_gb} GB exceeds "
                     f"budget {GPU_VRAM_BUDGET_GB} GB")

        if len(self.queues[job.queue]) >= cfg["max_depth"]:
            raise AdmissionError(
                429, f"queue '{job.queue}' at max depth {cfg['max_depth']}")

        inflight = self.tenant_inflight[job.queue].get(job.tenant, 0)
        if inflight >= cfg["tenant_inflight"]:
            raise AdmissionError(
                429, f"tenant '{job.tenant}' at in-flight quota "
                     f"{cfg['tenant_inflight']} for '{job.queue}'")

        self.tenant_inflight[job.queue][job.tenant] = inflight + 1
        self.queues[job.queue].append(job)

    # ---------------- dispatch primitives (used by policies) ----------------
    def pop_next(self) -> Job | None:
        """Strict priority order, FIFO within a class."""
        for name in sorted(QUEUE_CONFIG, key=lambda n: QUEUE_CONFIG[n]["priority"]):
            if self.queues[name]:
                return self._take(self.queues[name].popleft())
        return None

    def pop_by(self, key) -> Job | None:
        """Pop the job minimizing key(job) across ALL queues."""
        best = None
        for q in self.queues.values():
            for job in q:
                if best is None or key(job) < key(best):
                    best = job
        if best is not None:
            self.queues[best.queue].remove(best)
            self._take(best)
        return best

    def pop_mates(self, head: Job, max_extra: int, key) -> list[Job]:
        """Pop up to max_extra jobs from head's OWN class queue whose batch
        key matches head's. May skip over non-matching jobs (shape-aware
        batching reorders within a class; cross-class priority is untouched
        because mates only come from the head's queue)."""
        if max_extra <= 0:
            return []
        hk = key(head)
        q = self.queues[head.queue]
        mates = [j for j in q if key(j) == hk][:max_extra]
        for m in mates:
            q.remove(m)
            self._take(m)
        return mates

    def _take(self, job: Job) -> Job:
        job.dispatched_at = time.perf_counter()
        return job

    def requeue(self, job: Job) -> None:
        """Preempted job returns to the FRONT of its queue (no re-admission)."""
        job.dispatched_at = None
        self.queues[job.queue].appendleft(job)

    def release(self, job: Job) -> None:
        t = self.tenant_inflight[job.queue]
        t[job.tenant] = max(0, t.get(job.tenant, 1) - 1)

    # ---------------- introspection ----------------
    def depths(self) -> dict:
        return {name: len(q) for name, q in self.queues.items()}

    def waiting(self, queue_name: str) -> int:
        return len(self.queues[queue_name])