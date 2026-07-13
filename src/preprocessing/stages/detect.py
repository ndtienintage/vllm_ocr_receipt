"""S2 Detect — PaddleOCR text detection chạy ĐÚNG 1 LẦN làm anchor cho mọi
quyết định sau (ADR-02). Re-detect chỉ xảy ra trong orient khi có cardinal
rotation (~19% ảnh theo Phase 0).
"""

from __future__ import annotations

from src.preprocessing.contracts import PipelineContext, Route, StageReport


class DetectStage:
    name = "detect"

    def run(self, ctx: PipelineContext, report: StageReport) -> None:
        cfg = ctx.config.detect
        if not cfg.enabled:
            report.skipped_reason = "disabled"
            return
        if ctx.route == Route.DIGITAL:
            report.skipped_reason = "digital_bypass"
            return
        if ctx.services is None or ctx.services.detector is None:
            report.skipped_reason = "no_detector_service"
            return

        polys = ctx.services.detector.detect(ctx.image)
        n = 0 if polys is None else int(len(polys))
        report.decisions["n_polys"] = n
        if polys is None or n < cfg.min_polys:
            ctx.polys = None
            report.decisions["below_min_polys"] = True
        else:
            ctx.polys = polys
        report.applied = True
