"""Pluggable scheduling policies (Phase 3a).

A policy answers one question -- "which queued job runs next?" -- and
optionally declares itself preemptive, in which case the dispatcher will
evict a running lower-priority job when production work is waiting and no
GPU is free.

Select via env:  SCHED_POLICY=fifo | priority | edf | priority_preempt

  fifo             global arrival order, ignores class (the baseline that
                   lets benchmark jobs starve production -- keep it, it
                   makes the comparison plots)
  priority         strict priority: production > research > benchmark
  edf              earliest deadline first; no-deadline jobs sort last,
                   ties broken by class priority then arrival
  priority_preempt strict priority + step-level preemption of running
                   research/benchmark jobs when production is waiting
"""

import math

from scheduling import Job, PriorityQueues


class BasePolicy:
    name = "base"
    preemptive = False

    def select(self, queues: PriorityQueues) -> Job | None:
        raise NotImplementedError


class FifoPolicy(BasePolicy):
    name = "fifo"

    def select(self, queues):
        return queues.pop_by(lambda j: j.arrival)


class PriorityPolicy(BasePolicy):
    name = "priority"

    def select(self, queues):
        return queues.pop_next()


class EdfPolicy(BasePolicy):
    name = "edf"

    def select(self, queues):
        return queues.pop_by(
            lambda j: (j.deadline if j.deadline is not None else math.inf,
                       j.priority, j.arrival)
        )


class PriorityPreemptPolicy(PriorityPolicy):
    name = "priority_preempt"
    preemptive = True


_POLICIES = {p.name: p for p in
             (FifoPolicy, PriorityPolicy, EdfPolicy, PriorityPreemptPolicy)}


def make_policy(name: str) -> BasePolicy:
    if name not in _POLICIES:
        raise ValueError(f"unknown policy '{name}'; options: {sorted(_POLICIES)}")
    return _POLICIES[name]()