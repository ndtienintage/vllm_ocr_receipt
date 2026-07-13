"""S6 Photometric (conditional, ADR-09) — chuẩn hoá ánh sáng NHẸ:
background estimation (morphological close kernel lớn) + division đưa nền về
mức sáng đồng nhất, CLAHE nhẹ trên kênh L. KHÔNG binarize, KHÔNG sharpen.

Mặc định OFF (dataset Phase 0 không có bóng đổ nặng; minimal-intervention).
Hai đường dùng:
  1. Stage độc lập (config photometric.enabled=true) — cho ablation Phase 3.
  2. Variant trong gate (S5) cho ảnh MARGINAL — chấm điểm bằng OCR rec
     trước khi quyết định áp.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.contracts import PipelineContext, Route, StageReport


def normalize_illumination(image: np.ndarray, cfg) -> np.ndarray:
    """Khử chênh sáng nền (bóng đổ mềm) — bảo toàn texture chữ."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    k = max(15, int(min(image.shape[:2]) * cfg.bg_kernel_frac))
    k += 1 - (k % 2)  # kernel lẻ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(l_ch, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=k / 6.0)
    # division: nền → ~220 (trắng giấy, không clip highlight)
    norm = cv2.divide(l_ch, np.maximum(bg, 1), scale=220.0)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    norm = clahe.apply(norm)
    return cv2.cvtColor(cv2.merge([norm, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


class PhotometricStage:
    name = "photometric"

    def run(self, ctx: PipelineContext, report: StageReport) -> None:
        cfg = ctx.config.photometric
        if not cfg.enabled:
            report.skipped_reason = "disabled"
            return
        if ctx.route == Route.DIGITAL:
            report.skipped_reason = "digital_bypass"
            return
        if ctx.meta.get("photometric_applied_by_gate"):
            report.skipped_reason = "already_applied_by_gate_variant"
            return
        ctx.image = normalize_illumination(ctx.image, cfg)
        report.applied = True
