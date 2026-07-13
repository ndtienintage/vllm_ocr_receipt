"""Proxy metrics cho ablation (định nghĩa ở phase1_design.md §4).

Nguyên tắc đo: metric tính trên ảnh SAU KHI mô phỏng smart_resize của vLLM —
đo đúng cái model nhìn thấy, không phải cái ta ghi ra file (spec §6).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import cv2
import numpy as np


def simulate_smart_resize(image: np.ndarray, *, min_pixels: int,
                          max_pixels: int, factor: int = 32) -> np.ndarray:
    """Mô phỏng qwen-vl smart_resize: round mỗi cạnh về bội factor, ép area
    vào [min_pixels, max_pixels]. Ảnh đã fit đúng sẽ đi qua gần như nguyên."""
    h, w = image.shape[:2]
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    if (w_bar, h_bar) == (w, h):
        return image
    return cv2.resize(image, (w_bar, h_bar), interpolation=cv2.INTER_CUBIC)


def ocr_proxy_metrics(image: np.ndarray, det_model, rec_model,
                      max_boxes: int = 150) -> Dict[str, Any]:
    """PaddleOCR det+rec trên ảnh (đã simulate smart_resize) → proxy chất lượng
    đọc. OCR là OBJECTIVE FUNCTION, không phải bộ đọc."""
    out: Dict[str, Any] = {
        "n_boxes": 0, "rec_conf_mean": None, "rec_conf_median": None,
        "n_chars": 0, "text_h_at_model_px": None,
        "pixels": int(image.shape[0] * image.shape[1]),
    }
    det_res = det_model.predict(input=image)
    if not det_res:
        return out
    r0 = det_res[0]
    polys = r0.get("dt_polys") if isinstance(r0, dict) else getattr(r0, "dt_polys", None)
    if polys is None or len(polys) == 0:
        return out
    polys = np.asarray(polys, dtype=np.float32)
    out["n_boxes"] = int(len(polys))
    heights = polys[..., 1].max(axis=1) - polys[..., 1].min(axis=1)
    out["text_h_at_model_px"] = round(float(np.median(heights)), 1)

    h, w = image.shape[:2]
    crops = []
    for p in polys[:max_boxes]:
        pts = p.reshape(-1, 2)
        x1, y1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
        x2, y2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            crops.append(image[y1:y2, x1:x2])
    if not crops:
        return out
    rec_res = rec_model.predict(input=crops)
    scores, chars = [], 0
    for r in rec_res or []:
        score = r.get("rec_score") if isinstance(r, dict) else getattr(r, "rec_score", None)
        text = r.get("rec_text") if isinstance(r, dict) else getattr(r, "rec_text", "")
        if score is not None:
            scores.append(float(score))
        chars += len(text or "")
    if scores:
        out["rec_conf_mean"] = round(float(np.mean(scores)), 4)
        out["rec_conf_median"] = round(float(np.median(scores)), 4)
    out["n_chars"] = chars
    return out


def residual_skew_deg(image: np.ndarray, det_model) -> Optional[float]:
    """Góc nghiêng còn lại trên ảnh output (đo bằng det polys) — cho rotation
    correctness check."""
    from src.preprocessing import geometry
    det_res = det_model.predict(input=image)
    if not det_res:
        return None
    r0 = det_res[0]
    polys = r0.get("dt_polys") if isinstance(r0, dict) else getattr(r0, "dt_polys", None)
    if polys is None or len(polys) < 3:
        return None
    return geometry.estimate_tilt_deg(np.asarray(polys, dtype=np.float32))
