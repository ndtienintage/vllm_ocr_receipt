"""S5 Gate — routing READABLE / MARGINAL / UNREADABLE (ADR-07).

Verdict dựa trên OCR-EVIDENCE (polys + text height ĐẠT ĐƯỢC sau budget solver),
không phải blur-var đơn thuần (Phase 0: lapvar 115 vẫn readable — blur-var một
mình chắc chắn false-positive; đó là lý do v1 bỏ gate, nhưng bỏ hẳn thì vi
phạm spec reject-with-reason).
"""

from __future__ import annotations

from src.preprocessing import budget
from src.preprocessing.contracts import (
    PipelineContext, Purpose, Route, StageReport, Verdict,
)


class GateStage:
    name = "gate"

    def run(self, ctx: PipelineContext, report: StageReport) -> None:
        cfg = ctx.config.gate
        if not cfg.enabled:
            report.skipped_reason = "disabled"
            return
        if ctx.route == Route.DIGITAL:
            report.decisions["verdict"] = Verdict.READABLE.value
            report.skipped_reason = "digital_always_readable"
            return

        quality = ctx.meta.get("quality", {})
        blur = float(quality.get("blur_lapvar", 1e9))
        n_polys = 0 if ctx.polys is None else int(len(ctx.polys))
        text_h = ctx.meta.get("median_text_height_px")

        # Text height ĐẠT ĐƯỢC nếu đi tiếp qua budget solver (pure, rẻ)
        h, w = ctx.image.shape[:2]
        plan = budget.solve(
            w, h, text_h, ctx.config.fit,
            allow_thrift=ctx.purpose == Purpose.VLM,
            allow_reflow=ctx.purpose == Purpose.VLM and ctx.polys is not None,
        )
        text_h_achievable = plan.text_h_out
        report.decisions.update({
            "n_polys": n_polys,
            "blur_lapvar": blur,
            "text_h_achievable": (
                round(text_h_achievable, 1) if text_h_achievable else None),
        })

        verdict = Verdict.READABLE
        reason = None
        if n_polys < cfg.unreadable_min_polys:
            verdict, reason = Verdict.UNREADABLE, (
                f"no_text_detected(n_polys={n_polys}<{cfg.unreadable_min_polys})")
        elif (text_h_achievable is not None
              and text_h_achievable < cfg.unreadable_text_h
              and blur < cfg.unreadable_blur):
            verdict, reason = Verdict.UNREADABLE, (
                f"text_unrecoverable(text_h={text_h_achievable:.1f}px"
                f"<{cfg.unreadable_text_h}, blur={blur:.0f}<{cfg.unreadable_blur})")
        elif ((text_h_achievable is not None
               and text_h_achievable < cfg.marginal_text_h)
              or blur < cfg.marginal_blur):
            verdict = Verdict.MARGINAL

        if verdict == Verdict.UNREADABLE and not cfg.reject_enabled:
            report.decisions["reject_suppressed"] = reason
            verdict, reason = Verdict.MARGINAL, None

        ctx.verdict = verdict
        ctx.reject_reason = reason
        report.decisions["verdict"] = verdict.value
        if reason:
            report.decisions["reject_reason"] = reason
        report.applied = True

