"""S0 Ingest — decode an toàn, chuẩn hoá về BGR uint8, ghi nhận EXIF/dims.

Ghi chú EXIF: cv2.imdecode trên OpenCV 4.10 ĐÃ áp EXIF orientation (verify
empirically ở Phase 0 — hành vi này không được document rõ giữa các version).
Unit test `test_ingest_exif_pinned` chốt hành vi; nâng cấp OpenCV làm test đỏ
→ phải xử lý EXIF thủ công tại đây trước khi lên version.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.preprocessing.contracts import PipelineContext, StageReport


def decode_image(data: bytes) -> Optional[np.ndarray]:
    """bytes (JPEG/PNG/...) → BGR ndarray. None khi hỏng/không hỗ trợ."""
    if not data:
        return None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def exif_orientation_tag(data: bytes) -> Optional[int]:
    """Đọc EXIF orientation tag (274) từ header — chỉ để audit metadata,
    KHÔNG dùng để transform (cv2 4.10 đã áp). Best-effort."""
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            return int(exif.get(274)) if exif and exif.get(274) else None
    except Exception:
        return None


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
