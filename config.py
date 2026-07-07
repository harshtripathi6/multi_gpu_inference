"""Queue classes, SLOs, and admission quotas (Phase 2).

Priority: lower number = more important. Strict-priority dispatch in Phase 2;
Phase 3 makes the policy pluggable.
"""

QUEUE_CONFIG = {
    "production": {
        "priority": 0,
        "slo_ms": 2000,        # deadline: arrival + 2s
        "max_depth": 64,       # queued (not yet dispatched) cap -> 429
        "tenant_inflight": 8,  # per-tenant queued+running cap -> 429
        "preemptible": False,  # production is never evicted
    },
    "research": {
        "priority": 1,
        "slo_ms": None,        # best-effort
        "max_depth": 128,
        "tenant_inflight": 4,
        "preemptible": True,
    },
    "benchmark": {
        "priority": 2,
        "slo_ms": None,        # background
        "max_depth": 256,
        "tenant_inflight": 2,
        "preemptible": True,
    },
}

# Admission memory sanity check (H100 80GB, leave headroom for the pipeline).
GPU_VRAM_BUDGET_GB = 70.0