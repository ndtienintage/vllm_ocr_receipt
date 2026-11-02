"""S0 Ingest — chuẩn hoá ảnh đã decode về BGR uint8, ghi nhận dims.

Ghi chú EXIF: decode diễn ra ở `src.utils.image_utils.load_image_from_bytes`
(cv2.imdecode IMREAD_COLOR — OpenCV 4.10, pin qua paddlex, ĐÃ áp EXIF
orientation; verify empirically ở Phase 0). Nâng cấp OpenCV → verify lại hành
vi này trước khi lên version, nếu mất thì tự transpose tại đây.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.contracts import PipelineContext, StageReport


class IngestStage:
    name = "ingest"

    def run(self, ctx: PipelineContext, report: StageReport) -> None:
        img = ctx.image
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("ingest: image rỗng hoặc không phải ndarray")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"ingest: shape không hỗ trợ {img.shape}")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        if h < 8 or w < 8:
            raise ValueError(f"ingest: ảnh quá nhỏ {w}x{h}")

        ctx.image = img
        ctx.meta["input_long_side"] = int(max(h, w))
        ctx.meta["input_short_side"] = int(min(h, w))
        report.applied = True
        report.decisions = {"width": w, "height": h}
