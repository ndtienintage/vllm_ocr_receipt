"""S4 Localize — crop vùng receipt theo text-density (ADR-05).

Trust gate (bài học v1, ngưỡng không mâu thuẫn với Phase 0 data): detector
under-detect → bbox không phản ánh vùng receipt thật → crop có thể CẮT items.
Aspect-inflation guard: crop full-bbox làm aspect nổ so với gốc mà poly count
thấp → degrade long-axis-only (giữ nguyên cạnh ngắn).
"""

from __future__ import annotations

from src.preprocessing import geometry
from src.preprocessing.contracts import PipelineContext, Route, StageReport


class LocalizeStage:
    name = "localize"

    def run(self, ctx: PipelineContext, report: StageReport) -> None:
        cfg = ctx.config.localize
        # Đo text height sau crop (crop chỉ translate — không đổi height);
        # ghi cả khi stage skip để gate/fit luôn có số này.
        try:
            self._run_crop(ctx, report, cfg)
        finally:
            ctx.meta["median_text_height_px"] = geometry.median_text_height(ctx.polys)

    def _run_crop(self, ctx: PipelineContext, report: StageReport, cfg) -> None:
        if not cfg.enabled:
            report.skipped_reason = "disabled"
            return
        if ctx.route == Route.DIGITAL:
            report.skipped_reason = "digital_bypass"
            return
        if ctx.polys is None:
            report.skipped_reason = "no_polys"
            return

        img = ctx.image
        h, w = img.shape[:2]
        n = int(len(ctx.polys))
        x1, y1, x2, y2 = geometry.union_bbox(ctx.polys, img.shape, 0)
        coverage = ((x2 - x1) * (y2 - y1)) / float(max(1, h * w))
        report.decisions.update({"n_polys": n, "coverage": round(coverage, 3)})

        if n < cfg.trust_min_polys or coverage < cfg.trust_min_coverage:
            report.skipped_reason = "low_trust"
            return

        bx1, by1, bx2, by2 = geometry.union_bbox(ctx.polys, img.shape, cfg.padding_px)
        orig_aspect = geometry.aspect_ratio(h, w)
        proj_aspect = geometry.aspect_ratio(by2 - by1, bx2 - bx1)
        inflation = proj_aspect / max(1e-6, orig_aspect)
        long_only = inflation > cfg.aspect_inflation_max and n < cfg.high_poly_min
        if long_only:
            if h >= w:
                bx1, bx2 = 0, w
            else:
                by1, by2 = 0, h
        report.decisions.update({
            "crop_box": [bx1, by1, bx2, by2],
            "aspect_inflation": round(inflation, 2),
            "long_axis_only": long_only,
        })
        ctx.image, ctx.polys = geometry.crop_with_polys(
            img, ctx.polys, (bx1, by1, bx2, by2))
        report.applied = True
