"""Cost + Memory Estimator (Phase 2).

Fits latency(steps) = intercept + slope * steps per (resolution, batch) cell
from calibration.json (least squares, no numpy needed). For shapes not in
the table, scales the nearest cell by pixel ratio -- crude but documented,
and diffusion cost scales close to linearly with latent pixels.

Falls back to rough heuristic constants if calibration.json is missing, so
the gateway still runs before you've calibrated (estimates are flagged
calibrated=False).
"""

import json
import os
from dataclasses import dataclass


@dataclass
class Estimate:
    gpu_ms: float
    vram_gb: float
    calibrated: bool


class CostEstimator:
    def __init__(self, path: str = "calibration.json"):
        self.cells = {}  # (pixels, batch) -> dict(slope, intercept, vram_gb)
        self.calibrated = False
        if os.path.exists(path):
            self._fit(path)

    def _fit(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)

        groups = {}  # (pixels, batch) -> list[(steps, total_ms, vram)]
        for e in data["entries"]:
            key = (e["width"] * e["height"], e["batch"])
            groups.setdefault(key, []).append(
                (e["steps"], e["total_ms_mean"], e["peak_vram_gb"])
            )

        for key, pts in groups.items():
            if len(pts) >= 2:
                slope, intercept = _least_squares(
                    [p[0] for p in pts], [p[1] for p in pts]
                )
            else:
                steps, total, _ = pts[0]
                slope, intercept = total / steps, 0.0
            self.cells[key] = {
                "slope": max(slope, 1.0),
                "intercept": max(intercept, 0.0),
                "vram_gb": max(p[2] for p in pts),
            }
        self.calibrated = len(self.cells) > 0

    def estimate(self, steps: int, width: int, height: int, batch: int = 1) -> Estimate:
        pixels = width * height
        if not self.calibrated:
            # Heuristic fallback (H100-ish SD-Turbo numbers).
            ratio = pixels / (512 * 512)
            gpu_ms = 80.0 + 35.0 * ratio * batch * steps
            vram_gb = 8.0 + 1.5 * batch * ratio
            return Estimate(round(gpu_ms, 1), round(vram_gb, 2), False)

        # Nearest cell by (pixel distance, batch distance), scale by pixel ratio.
        key = min(
            self.cells,
            key=lambda k: (abs(k[0] - pixels) / pixels, abs(k[1] - batch)),
        )
        cell = self.cells[key]
        pixel_scale = pixels / key[0]
        batch_scale = batch / key[1]
        gpu_ms = (cell["intercept"] + cell["slope"] * steps) * pixel_scale * batch_scale
        vram_gb = cell["vram_gb"] * max(pixel_scale, 1.0) * max(batch_scale, 1.0)
        return Estimate(round(gpu_ms, 1), round(vram_gb, 2), True)


def _least_squares(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return ys[0] / max(xs[0], 1), 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope, my - slope * mx