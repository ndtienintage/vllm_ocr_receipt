"""S7 Fit — thực thi FitPlan của budget solver (ADR-06):
[reflow N cột nếu plan nói và tìm được split an toàn] → resize align bội 32
(INTER_CUBIC) → [pad cạnh ngắn đạt sàn min_pixels nếu cần].

Split reflow BẮT BUỘC rơi vào whitespace giữa các text row (không bao giờ cắt
qua poly); không tìm được split an toàn → re-solve không reflow, ghi lý do.
"""

from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np

from src.preprocessing import budget
from src.preprocessing.contracts import PipelineContext, Purpose, Route, StageReport


def find_safe_splits(polys: np.ndarray, height: int, n_cols: int,
                     search_frac: float) -> Optional[List[int]]:
    """Tìm n_cols-1 vạch cắt ngang nằm trong whitespace giữa các text row,
    gần vị trí chia đều nhất. None nếu bất kỳ vạch nào không tìm được."""
    intervals = sorted(
        (float(p[..., 1].min()), float(p[..., 1].max())) for p in polys)
    merged: List[List[float]] = []
    for y0, y1 in intervals:
        if merged and y0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])
    gaps: List[tuple] = []
    prev = 0.0
    for y0, y1 in merged:
        if y0 - prev >= 3.0:
            gaps.append((prev, y0))
        prev = max(prev, y1)
    if height - prev >= 3.0:
        gaps.append((prev, float(height)))

    radius = search_frac * height
    splits: List[int] = []
    for k in range(1, n_cols):
        target = height * k / n_cols
        best: Optional[float] = None
        for g0, g1 in gaps:
            cand = min(max(target, g0 + 1.0), g1 - 1.0)
            if cand <= g0 or cand >= g1:
                continue
            if abs(cand - target) > radius:
                continue
            if best is None or abs(cand - target) < abs(best - target):
                best = cand
        if best is None:
            return None
        if splits and best - splits[-1] < height / (2 * n_cols):
            return None  # 2 vạch dính nhau — cột mất cân đối nghiêm trọng
        splits.append(int(round(best)))
    return splits


def compose_columns(image: np.ndarray, splits: List[int], sep_px: int,
                    pad_color: int) -> np.ndarray:
    """Cắt theo splits, xếp các đoạn cạnh nhau (trái→phải = trên→dưới),
    ngăn bằng vạch đen, đáy cột ngắn pad màu nền."""
    h, w = image.shape[:2]
    bounds = [0] + list(splits) + [h]
    cols = [image[a:b] for a, b in zip(bounds, bounds[1:])]
    col_h = max(c.shape[0] for c in cols)
    n = len(cols)
    canvas_w = n * w + (n - 1) * sep_px
    canvas = np.full((col_h, canvas_w, 3), pad_color, dtype=np.uint8)
    x = 0
    for i, col in enumerate(cols):
        canvas[: col.shape[0], x: x + w] = col
        x += w
        if i < n - 1:
            canvas[:, x: x + sep_px] = 0
            x += sep_px
    return canvas


def target_dims(w: int, h: int, scale: float, cfg,
                *, enforce_floor: bool) -> tuple:
    """Chọn (target_w, target_h) bội block: round-NEAREST theo scale (như
    smart_resize), rồi ép trần (max_side, max_pixels) bằng cách hạ từng block,
    rồi ép sàn min_pixels bằng cách NÂNG từng block cạnh có relative-change
    nhỏ nhất. Sửa bug ablation P3: align-down 2 cạnh làm 5/26 ảnh rơi xuống
    dưới sàn ~1-5% → vLLM upscale lại (double-resample)."""
    block = cfg.block
    tw = max(block, round(w * scale / block) * block)
    th = max(block, round(h * scale / block) * block)
    while tw > cfg.max_side:
        tw -= block
    while th > cfg.max_side:
        th -= block
    while tw * th > cfg.max_pixels and (tw > block or th > block):
        if tw >= th and tw > block:
            tw -= block
        elif th > block:
            th -= block
    if enforce_floor and cfg.min_pixels > 0:
        for _ in range(256):
            if tw * th >= cfg.min_pixels:
                break
            grow = []
            if tw + block <= cfg.max_side:
                grow.append((block / tw, "w"))
            if th + block <= cfg.max_side:
                grow.append((block / th, "h"))
            if not grow:
                break  # kẹt trần cạnh cả 2 chiều → caller pad
            if min(grow)[1] == "w":
                tw += block
            else:
                th += block
    return tw, th


class FitStage:
    name = "fit"

    def run(self, ctx: PipelineContext, report: StageReport) -> None:
        cfg = ctx.config.fit
        if not cfg.enabled:
            report.skipped_reason = "disabled"
            return

        h, w = ctx.image.shape[:2]
        text_h = ctx.meta.get("median_text_height_px")
        is_vlm = ctx.purpose == Purpose.VLM
        allow_reflow = is_vlm and ctx.route == Route.PHOTO and ctx.polys is not None

        plan = budget.solve(w, h, text_h, cfg, allow_zoom=is_vlm,
                            allow_reflow=allow_reflow,
                            enforce_floor=is_vlm)
        image = ctx.image

        # ── Reflow (nếu plan nói) ────────────────────────────────────────────
        if plan.reflow_cols > 1:
            splits = find_safe_splits(
                ctx.polys, h, plan.reflow_cols, cfg.reflow_split_search_frac)
            if splits is None:
                report.decisions["reflow"] = "no_safe_split"
                plan = budget.solve(w, h, text_h, cfg, allow_zoom=is_vlm,
                                    allow_reflow=False, enforce_floor=is_vlm)
            else:
                image = compose_columns(
                    image, splits, cfg.reflow_separator_px, cfg.pad_color)
                ctx.polys = None  # polys không còn khớp canvas mới
                ctx.meta["reflow_applied"] = True
                ctx.meta["n_columns"] = plan.reflow_cols
                report.decisions["reflow"] = {
                    "n_columns": plan.reflow_cols,
                    "split_ys": splits,
                    "gain": plan.notes.get("reflow_gain"),
                }

        # ── Resize + align bội block (chống vLLM resample lần 2) ─────────────
        # target_dims tự ép trần/sàn trên dims THẬT (canvas reflow có thể lệch
        # nhẹ so với ước lượng của solver) — trần cứng không bao giờ được vượt,
        # sàn không bao giờ bị rớt do rounding.
        h2, w2 = image.shape[:2]
        target_w, target_h = target_dims(w2, h2, plan.scale, cfg,
                                         enforce_floor=is_vlm)
        if (target_w, target_h) != (w2, h2):
            image = cv2.resize(image, (target_w, target_h),
                               interpolation=cv2.INTER_CUBIC)

        actual_scale = max(target_w / w2, target_h / h2)
        # Chỉ báo zoom khi solver CHỦ ĐÍCH zoom/nâng-sàn — round-nearest
        # alignment có thể đẩy actual_scale lên ~1.0x vài % (không phải zoom).
        intentional_upscale = plan.zoom_applied or (plan.scale > 1.0 and is_vlm)
        zoom_ratio = (round(actual_scale, 2)
                      if intentional_upscale and actual_scale > 1.0 else 1.0)

        # ── Pad đạt sàn min_pixels khi trần cạnh chặn upscale ────────────────
        pad_px = 0
        h3, w3 = image.shape[:2]
        if (is_vlm and cfg.pad_to_min_pixels and cfg.min_pixels > 0
                and h3 * w3 < cfg.min_pixels):
            long3, short3 = max(h3, w3), min(h3, w3)
            needed_short = budget.align_up(
                int(math.ceil(cfg.min_pixels / long3)), cfg.block)
            pad_px = max(0, needed_short - short3)
            if pad_px > 0:
                a, b = pad_px // 2, pad_px - pad_px // 2
                color = (cfg.pad_color,) * 3
                if h3 >= w3:  # cạnh ngắn là ngang → pad trái/phải
                    image = cv2.copyMakeBorder(image, 0, 0, a, b,
                                               cv2.BORDER_CONSTANT, value=color)
                else:
                    image = cv2.copyMakeBorder(image, a, b, 0, 0,
                                               cv2.BORDER_CONSTANT, value=color)

        ctx.image = image
        ctx.meta.setdefault("reflow_applied", False)
        ctx.meta["legibility_zoom_ratio"] = zoom_ratio
        ctx.meta["text_h_at_output_px"] = (
            round(text_h * actual_scale, 1) if text_h is not None else None)
        ctx.meta["legibility_capped"] = plan.legibility_capped
        report.decisions.update({
            "plan_scale": round(plan.scale, 3),
            "actual_scale": round(actual_scale, 3),
            "out_size": [image.shape[1], image.shape[0]],
            "out_pixels": int(image.shape[0] * image.shape[1]),
            "zoom_applied": plan.zoom_applied,
            "thrift_applied": plan.thrift_applied,
            "legibility_capped": plan.legibility_capped,
            "pad_px": pad_px,
        })
        report.applied = True
