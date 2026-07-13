"""S3 Orient — cardinal 0/90/180/270 (ADR-03) + fine deskew (ADR-04).

Nguyên tắc (số đo Phase 0):
  - Anchor hình học: vertical_axis_ratio trên polys THON DÀI (lọc cell vuông).
  - doc_ori chỉ được trust khi ĐỒNG THUẬN với anchor (5/5 sideways đúng);
    nói "180" trên ảnh polys-ngang phải bị textline AND-confirm (3/26 misfire).
  - Brute-force textline 2 hướng CHỈ ở nhánh mơ hồ (tiết kiệm 2× classifier
    cost so với v1 ở nhánh rõ ràng).
  - Deskew: circular MEDIAN long-axis của polys elongation ≥ 2 — robust với
    receipt dạng bảng (case `501cbb61` mean -86.8° giả), cap 45°.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.preprocessing import geometry
from src.preprocessing.contracts import PipelineContext, Route, StageReport

_DOC_LABEL_TO_QUARTERS = {"0": 0, "90": 1, "180": 2, "270": 3}


class OrientStage:
    name = "orient"

    # ── textline voting ───────────────────────────────────────────────────────

    def _vote(self, ctx: PipelineContext, rotate_q: int) -> Tuple[int, int]:
        """Vote upside/total trên top-K poly crops, crop được xoay rotate_q
        quarters CCW trước khi hỏi classifier (xoay crop nhỏ ≡ xoay toàn ảnh
        rồi crop — rẻ hơn nhiều)."""
        cfg = ctx.config.orient
        svc = ctx.services.textline if ctx.services else None
        if svc is None or ctx.polys is None:
            return 0, 0
        areas = np.array([cv2.contourArea(p.astype(np.float32)) for p in ctx.polys])
        order = np.argsort(-areas)[: cfg.textline_sample]
        upside = total = 0
        for i in order:
            crop = geometry.crop_axis_aligned(
                ctx.image, ctx.polys[int(i)], min_side_px=cfg.textline_min_side_px)
            if crop is None:
                continue
            if rotate_q % 4:
                crop, _ = geometry.rotate_cardinal(crop, None, rotate_q)
            verdict = svc.is_upside(crop)
            if verdict is None:
                continue
            upside += int(verdict)
            total += 1
        return upside, total

    # ── cardinal decision (ADR-03) ────────────────────────────────────────────

    def _orient_view(self, ctx: PipelineContext) -> np.ndarray:
        """View long-axis-only crop cho doc_ori: bỏ background dọc trục dài
        (bài học v1: raw frame nền lớn làm doc_ori misfire)."""
        img = ctx.image
        h, w = img.shape[:2]
        x1, y1, x2, y2 = geometry.union_bbox(
            ctx.polys, img.shape, ctx.config.localize.padding_px)
        if h >= w:
            x1, x2 = 0, w
        else:
            y1, y2 = 0, h
        if x2 <= x1 or y2 <= y1:
            return img
        return img[y1:y2, x1:x2]

    def _cardinal_quarters(self, ctx: PipelineContext, report: StageReport) -> int:
        cfg = ctx.config.orient
        vert_ratio = geometry.vertical_axis_ratio(
            ctx.polys, min_elongation=cfg.ratio_min_elongation)
        report.decisions["vert_ratio"] = round(vert_ratio, 2)

        doc_label: Optional[str] = None
        if ctx.services and ctx.services.doc_ori is not None:
            doc_label = ctx.services.doc_ori.label(self._orient_view(ctx))
        doc_q = _DOC_LABEL_TO_QUARTERS.get(doc_label or "", 0)
        report.decisions["doc_ori_label"] = doc_label

        sideways = vert_ratio >= cfg.vert_ratio_trigger
        has_textline = ctx.services is not None and ctx.services.textline is not None

        if sideways:
            # Nhánh RÕ: anchor rất mạnh + doc_ori đồng thuận về trục
            if vert_ratio >= cfg.vert_ratio_confident and doc_q in (1, 3):
                q = doc_q
                if has_textline:  # verify direction 1 lần (rẻ hơn brute-force)
                    up, tot = self._vote(ctx, rotate_q=q)
                    report.decisions["verify_vote"] = [up, tot]
                    if tot > 0 and up > tot / 2:
                        q = 4 - q
                        report.decisions["verify_flipped_direction"] = True
                report.decisions["cardinal_branch"] = "confident_doc"
                return q
            # Nhánh MƠ HỒ: brute-force 2 hướng bằng textline
            if has_textline:
                up_a, tot_a = self._vote(ctx, rotate_q=1)
                up_b, tot_b = self._vote(ctx, rotate_q=3)
                upright_a, upright_b = tot_a - up_a, tot_b - up_b
                report.decisions["bruteforce_upright"] = [upright_a, tot_a, upright_b, tot_b]
                report.decisions["cardinal_branch"] = "bruteforce"
                if upright_a != upright_b:
                    return 1 if upright_a > upright_b else 3
                if doc_q in (1, 3):
                    return doc_q
                return 1
            report.decisions["cardinal_branch"] = "doc_only"
            return doc_q if doc_q in (1, 3) else 1

        # Không sideways: chỉ xét 180
        if doc_q == 2:
            if has_textline:  # AND-confirm — chống 3/26 misfire Phase 0
                up, tot = self._vote(ctx, rotate_q=0)
                report.decisions["confirm_180_vote"] = [up, tot]
                report.decisions["cardinal_branch"] = "doc180_confirm"
                return 2 if (tot > 0 and up > tot / 2) else 0
            report.decisions["cardinal_branch"] = "doc180_noconfirm"
            return 2
        if (ctx.services is not None and ctx.services.doc_ori is None
                and has_textline):
            up, tot = self._vote(ctx, rotate_q=0)
            report.decisions["textline_only_vote"] = [up, tot]
            if tot > 0 and up > tot / 2:
                report.decisions["cardinal_branch"] = "textline_only_180"
                return 2
        return 0

    # ── stage entry ───────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext, report: StageReport) -> None:
        cfg = ctx.config.orient
        if not cfg.enabled:
            report.skipped_reason = "disabled"
            return
        if ctx.route == Route.DIGITAL:
            report.skipped_reason = "digital_bypass"
            return
        if ctx.polys is None:
            report.skipped_reason = "no_polys"
            return

        applied = False

        if cfg.cardinal_enabled:
            q = self._cardinal_quarters(ctx, report)
            if q % 4:
                ctx.image, ctx.polys = geometry.rotate_cardinal(ctx.image, ctx.polys, q)
                report.decisions["cardinal_quarters_ccw"] = q
                applied = True
                if (cfg.redetect_after_cardinal
                        and ctx.services is not None
                        and ctx.services.detector is not None):
                    redetected = ctx.services.detector.detect(ctx.image)
                    if (redetected is not None
                            and len(redetected) >= ctx.config.detect.min_polys):
                        report.decisions["redetect_n_polys"] = [
                            int(len(ctx.polys)), int(len(redetected))]
                        ctx.polys = redetected
        else:
            report.decisions["cardinal"] = "disabled"

        if cfg.deskew_enabled and ctx.polys is not None:
            tilt = geometry.estimate_tilt_deg(
                ctx.polys,
                min_elongation=cfg.deskew_min_elongation,
                min_samples=cfg.deskew_min_samples,
            )
            if tilt is None:
                report.decisions["deskew"] = "insufficient_samples"
            elif abs(tilt) > cfg.deskew_max_deg:
                report.decisions["deskew"] = "capped"
                report.decisions["deskew_raw_deg"] = round(tilt, 2)
            elif abs(tilt) < cfg.deskew_min_deg:
                report.decisions["deskew"] = "below_min"
            else:
                ctx.image, ctx.polys = geometry.rotate_free(ctx.image, ctx.polys, tilt)
                report.decisions["deskew_deg"] = round(tilt, 2)
                applied = True

        report.applied = applied
