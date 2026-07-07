"""GPU telemetry via NVML (Phase 3a) -- the DCGM/NVML box in the diagram.

Background thread polls utilization / memory / temperature per GPU.
The dispatcher uses it to place work on the least-utilized free GPU
(cosmetic with one job per GPU, load-bearing once batching/MPS lands),
and /stats exposes it for the Phase 4 dashboard.

Degrades gracefully: if NVML is unavailable (no driver, pynvml missing),
available=False and the scheduler falls back to index order.
"""

import threading
import time


class GpuMonitor(threading.Thread):
    def __init__(self, interval_s: float = 1.0):
        super().__init__(daemon=True, name="gpu-monitor")
        self.interval_s = interval_s
        self.available = False
        self._lock = threading.Lock()
        self._snap = {}

    def run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)}
            self.available = True
        except Exception as e:
            print(f"[gpu-monitor] NVML unavailable ({e}); telemetry disabled")
            return

        while True:
            snap = {}
            for i, h in handles.items():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU)
                    snap[i] = {
                        "util_pct": util.gpu,
                        "mem_used_gb": round(mem.used / 1e9, 2),
                        "mem_total_gb": round(mem.total / 1e9, 2),
                        "temp_c": temp,
                        "ts": time.time(),
                    }
                except Exception:
                    snap[i] = {"error": True}
            with self._lock:
                self._snap = snap
            time.sleep(self.interval_s)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)

    def util(self, gpu_id: int) -> float:
        """Utilization for placement decisions; 0 if unknown."""
        return self.snapshot().get(gpu_id, {}).get("util_pct", 0) or 0