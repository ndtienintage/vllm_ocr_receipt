"""Runner — chạy stage tuần tự theo thứ tự khai báo, đo thời gian, cô lập lỗi.

Stage lỗi KHÔNG giết pipeline (trừ ingest — không có ảnh hợp lệ thì không có
gì để tiếp tục): lỗi ghi vào report.error, pipeline chạy tiếp với ảnh hiện có.
Khác v1: không còn try/except bọc cả pipeline nuốt lỗi im lặng — journal luôn
cho biết stage nào chết, vì sao.
"""

from __future__ import annotations

import time
import numpy as np

from src.preprocessing.config import load_config
from src.preprocessing.contracts import (
    PipelineContext, PipelineResult, Purpose, Verdict,
)
from src.preprocessing.detectors import build_services
from src.preprocessing.stages import (
    DetectStage, FitStage, GateStage, IngestStage, LocalizeStage,
    OrientStage, QualityStage,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

STAGE_ORDER = [
    IngestStage, QualityStage, DetectStage, OrientStage,
    LocalizeStage, GateStage, FitStage,
]


class Pipeline:
    """Pipeline preprocessing v2. Stateless giữa các lần run (trừ model cache
    trong services) — an toàn gọi song song ở mức pipeline (services tự lock)."""

    def __init__(self) -> None:
        self.config = load_config()
        self.services = build_services(self.config)
        self.stages = [cls() for cls in STAGE_ORDER]

    def run(self, image: np.ndarray, *,
            purpose: Purpose = Purpose.VLM) -> PipelineResult:
        ctx = PipelineContext(
            image=image, config=self.config, services=self.services,
            purpose=purpose,
        )

        for i, stage in enumerate(self.stages):
            report = ctx.new_report(stage.name)
            t0 = time.perf_counter()
            try:
                stage.run(ctx, report)
            except Exception as e:
                report.error = f"{type(e).__name__}: {e}"
                if i == 0:  # ingest fail = không có ảnh hợp lệ để tiếp tục
                    raise
                logger.error("preprocessing stage %s failed: %s", stage.name, report.error)
            finally:
                report.elapsed_ms = (time.perf_counter() - t0) * 1000.0

        meta = dict(ctx.meta)
        meta["purpose"] = ctx.purpose.value
        meta["route"] = ctx.route.value
        meta["verdict"] = ctx.verdict.value
        meta["stages"] = {r.name: r.as_dict() for r in ctx.reports}
        meta["total_ms"] = round(sum(r.elapsed_ms for r in ctx.reports), 1)
        return PipelineResult(
            ok=ctx.verdict != Verdict.UNREADABLE,
            image=ctx.image,
            verdict=ctx.verdict,
            reject_reason=ctx.reject_reason,
            meta=meta,
        )
