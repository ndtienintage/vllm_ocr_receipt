"""preprocessing — Bộ xử lý ảnh hoá đơn CHÍNH, consumer là
Qwen3-VL-8B-Instruct. Stage-based, config-driven, adaptive.

Tài liệu thiết kế: .claude/pharse/output/phase1_design.md (kiến trúc + ADR).

Public API:
    from src.preprocessing import Pipeline, Purpose, load_config
    pipe = Pipeline()                      # config mặc định + Paddle services
    result = pipe.run_bytes(jpeg_bytes)    # PipelineResult(ok, image, verdict, meta)
    result = pipe.run(bgr_ndarray, purpose=Purpose.FALLBACK_OCR)

Seam tích hợp với orchestrator (`src.extraction.processing`):
    from src.preprocessing.compat import preprocess_image_with_meta
"""

from src.preprocessing.config import PipelineConfig, load_config
from src.preprocessing.contracts import (
    PipelineContext, PipelineResult, Purpose, Route, StageReport, Verdict,
)
from src.preprocessing.detectors import Services, build_services
from src.preprocessing.runner import Pipeline

__all__ = [
    "Pipeline", "PipelineConfig", "PipelineContext", "PipelineResult",
    "Purpose", "Route", "Services", "StageReport", "Verdict",
    "build_services", "load_config",
]
