"""Contracts của preprocessing — mọi stage giao tiếp qua PipelineContext,
mọi quyết định ghi vào StageReport (journal). Không stage nào được nuốt lỗi
im lặng: lỗi ghi vào report, pipeline tiếp tục với ảnh hiện có.

Xem thiết kế: .claude/pharse/output/phase1_design.md (§2.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


class Verdict(str, Enum):
    READABLE = "readable"
    MARGINAL = "marginal"
    UNREADABLE = "unreadable"


class Route(str, Enum):
    PHOTO = "photo"      # ảnh chụp — đi full pipeline
    DIGITAL = "digital"  # screenshot/render — bypass det/orient/crop/gate


class Purpose(str, Enum):
    VLM = "vlm"                    # output cho Qwen3-VL (zoom + reflow cho phép)
    FALLBACK_OCR = "fallback_ocr"  # output cho PaddleOCR fallback (1 cột, không zoom)


@dataclass
class StageReport:
    """Một stage = một report. `applied=False` + `skipped_reason` khi stage
    quyết định không làm gì (điều kiện không thoả) — khác với `error`
    (stage định làm nhưng fail)."""

    name: str
    applied: bool = False
    skipped_reason: Optional[str] = None
    elapsed_ms: float = 0.0
    decisions: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "applied": self.applied,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
        if self.skipped_reason:
            d["skipped_reason"] = self.skipped_reason
        if self.decisions:
            d["decisions"] = self.decisions
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class PipelineContext:
    """State chia sẻ giữa các stage. `image` là ảnh hiện hành (BGR uint8);
    `polys` là kết quả text detection trên ảnh HIỆN HÀNH (stage nào xoay/crop
    ảnh phải transform polys tương ứng hoặc re-detect)."""

    image: np.ndarray
    config: Any                    # PipelineConfig (tránh import vòng)
    services: Any                  # Services (detectors.py)
    purpose: Purpose = Purpose.VLM
    route: Route = Route.PHOTO
    polys: Optional[np.ndarray] = None          # (N, 4, 2) float32
    verdict: Verdict = Verdict.READABLE
    reject_reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    reports: List[StageReport] = field(default_factory=list)

    def new_report(self, name: str) -> StageReport:
        r = StageReport(name=name)
        self.reports.append(r)
        return r


@dataclass
class PipelineResult:
    """Kết quả cuối. `ok=False` ⇔ verdict UNREADABLE (reject) — ảnh vẫn được
    trả (đã fit) để caller legacy dùng nếu muốn bỏ qua reject."""

    ok: bool
    image: np.ndarray
    verdict: Verdict
    reject_reason: Optional[str]
    meta: Dict[str, Any]

    @property
    def journal(self) -> Dict[str, Any]:
        return self.meta.get("stages", {})


@runtime_checkable
class Stage(Protocol):
    """Runner tạo report + đo elapsed_ms; stage chỉ ghi quyết định vào report
    và biến đổi ctx. Stage KHÔNG được nuốt exception tuỳ tiện — cứ để raise,
    runner ghi vào report.error và pipeline tiếp tục với ảnh hiện có."""

    name: str

    def run(self, ctx: PipelineContext, report: StageReport) -> None: ...
