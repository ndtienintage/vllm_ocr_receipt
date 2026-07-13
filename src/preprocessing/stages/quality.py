"""S1 Quality probe — đo metric rẻ (CPU, ~5ms) TRƯỚC mọi xử lý:

  - blur (variance of Laplacian, chuẩn hoá về norm_width để so được giữa ảnh),
  - phơi sáng (brightness, highlight/shadow clip),
  - tín hiệu digital (flat_frac + bg_peak) → route DIGITAL bypass (ADR-08).

Metric ghi vào ctx.meta["quality"] — là INPUT cho gate (S5), không tự quyết
reject ở đây (blur-var một mình có false-positive cao — Phase 0 §1.2).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.contracts import PipelineContext, Route, StageReport


class QualityStage:
    name = "quality"

    def run(self, ctx: PipelineContext, report: StageReport) -> None:
        cfg = ctx.config.quality
        if not cfg.enabled:
            report.skipped_reason = "disabled"
            return

        gray = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w != cfg.norm_width:
            nh = max(1, int(h * cfg.norm_width / w))
            gray_n = cv2.resize(gray, (cfg.norm_width, nh), interpolation=cv2.INTER_AREA)
        else:
            gray_n = gray

        lap = cv2.Laplacian(gray_n, cv2.CV_64F)
        blur_lapvar = float(lap.var())
        flat_frac = float((np.abs(lap) <= 1.0).mean())
        hist = np.bincount(gray_n.ravel(), minlength=256)
        bg_peak = float(hist.max()) / float(gray_n.size)

        metrics = {
            "blur_lapvar": round(blur_lapvar, 1),
            "brightness_mean": round(float(gray_n.mean()), 1),
            "highlight_clip_frac": round(float((gray_n >= 250).mean()), 4),
            "shadow_clip_frac": round(float((gray_n <= 5).mean()), 4),
            "flat_frac": round(flat_frac, 3),
            "bg_peak": round(bg_peak, 3),
        }
        ctx.meta["quality"] = metrics
        report.decisions.update(metrics)

        if (cfg.digital_bypass
                and flat_frac >= cfg.digital_flat_min
                and bg_peak >= cfg.digital_bg_peak_min):
            ctx.route = Route.DIGITAL
            report.decisions["route"] = "digital"
        report.applied = True
