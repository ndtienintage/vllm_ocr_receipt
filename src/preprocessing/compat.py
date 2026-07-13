"""Adapter migration — drop-in cho hàm preprocessing cũ (đã gỡ bỏ):
`preprocess_image_with_meta(image, *, for_fallback_ocr=False)`.

Đây là SEAM tích hợp chính giữa orchestrator (`src.extraction.processing`) và
bộ xử lý ảnh mới (`src.preprocessing`). Giữ đúng chữ ký + meta keys mà
`processing.py` đang tiêu thụ nên orchestrator không phải đổi logic.

Bảo đảm: LUÔN trả ảnh (không bao giờ None / raise); meta luôn có
`reflow_applied`, `input_long_side`, `input_short_side`,
`median_text_height_px`, `legibility_zoom_ratio` (+ `n_columns` khi reflow).

Additive (mới so với hàm cũ): meta thêm `verdict`, `reject_reason`, `stages`
(journal), `route`, `pipeline_version`. Caller muốn reject-with-reason nên dùng
`Pipeline().run_bytes()` → `PipelineResult` trực tiếp thay vì adapter này.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.preprocessing.contracts import Purpose
from src.preprocessing.runner import Pipeline
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_PIPELINE: Optional[Pipeline] = None
_LOCK = threading.Lock()


def get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                _PIPELINE = Pipeline()
    return _PIPELINE


def _default_meta(image: np.ndarray) -> Dict[str, Any]:
    return {
        "reflow_applied": False,
        "input_long_side": int(max(image.shape[:2])),
        "input_short_side": int(min(image.shape[:2])),
        "median_text_height_px": None,
        "legibility_zoom_ratio": 1.0,
        "pipeline_version": "v2",
    }


def preprocess_image_with_meta(
    image: np.ndarray, *, for_fallback_ocr: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Drop-in cho hàm `preprocess_image_with_meta` của preprocessing cũ (đã gỡ);
    orchestrator `extraction/processing.py` gọi qua đây."""
    try:
        purpose = Purpose.FALLBACK_OCR if for_fallback_ocr else Purpose.VLM
        result = get_pipeline().run(image, purpose=purpose)
        meta = _default_meta(image)
        meta.update({
            k: v for k, v in result.meta.items()
            if k in (
                "reflow_applied", "n_columns", "input_long_side",
                "input_short_side", "median_text_height_px",
                "legibility_zoom_ratio", "legibility_capped",
                "text_h_at_output_px", "verdict", "reject_reason",
                "route", "quality", "stages", "total_ms",
            ) and v is not None
        })
        meta.setdefault("reflow_applied", False)
        meta.setdefault("legibility_zoom_ratio", 1.0)
        meta["pipeline_version"] = "v2"
        if not result.ok:
            # Compat: v1 không có khái niệm reject — trả ảnh như thường, cờ
            # verdict/reject_reason cho caller mới quyết. KHÔNG nuốt im lặng.
            logger.warning(
                "preprocessing verdict=UNREADABLE (%s) — compat path vẫn trả ảnh",
                result.reject_reason,
            )
        return result.image, meta
    except Exception as e:
        logger.error("preprocessing compat failed (%s: %s) — trả ảnh gốc",
                     type(e).__name__, e)
        meta = _default_meta(image)
        meta["pipeline_error"] = f"{type(e).__name__}: {e}"
        return image, meta
