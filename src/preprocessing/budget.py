"""Pixel-budget solver (ADR-06) — pure function.

Bài toán: cho ảnh (w, h) sau crop + chiều cao chữ đo được text_h, tìm kế hoạch
(scale, reflow_cols) sao cho:
  1. Trần cứng:  area ≤ max_pixels  VÀ  cạnh dài ≤ max_side   (vLLM không phải
     resize lần 2 — chống double-resample).
  2. Sàn:        area ≥ min_pixels  (khớp floor của vLLM; nếu trần cạnh chặn
     upscale thì stage fit pad cạnh ngắn thay vì để vLLM tự nội suy).
  3. Token thrift: chữ to hơn thrift target → downscale về target (tiết kiệm
     visual token, không mất legibility).
  4. Legibility: text_h × scale < min_text_height_px → cờ `legibility_capped`
     (tín hiệu cho gate/metadata).
  5. Reflow:     CHỈ khi chứng minh bằng số — text_h_out có reflow ≥
     reflow_gain_min × text_h_out không reflow (trần cạnh dài là thứ reflow
     tháo được; trần diện tích thì không, vì reflow bảo toàn area).

Alignment bội 32 (Qwen3-VL patch16 × merge2 — verify ở Phase 0 §4) do stage fit
thực thi lúc resize; solver chỉ quyết scale/reflow.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FitPlan:
    scale: float
    text_h_out: Optional[float]
    thrift_applied: bool
    legibility_capped: bool
    reflow_cols: int = 1                  # 1 = không reflow
    reflow_gain: Optional[float] = None   # text_h_out reflow / không reflow


def _plan_for_dims(w: int, h: int, text_h: Optional[float], cfg,
                   *, allow_thrift: bool, enforce_floor: bool,
                   reflow_cols: int = 1) -> FitPlan:
    area = float(w) * float(h)
    hard_cap = min(cfg.max_side / float(max(w, h)), math.sqrt(cfg.max_pixels / area))

    thrift = bool(allow_thrift and cfg.token_thrift and cfg.thrift_target_text_h > 0
                  and text_h is not None and text_h > cfg.thrift_target_text_h)
    want = cfg.thrift_target_text_h / text_h if thrift else 1.0
    scale = min(want, hard_cap)

    # Sàn diện tích (vLLM min_pixels): nâng scale nếu còn dưới sàn, nhưng không
    # vượt trần cứng — vượt không nổi thì stage fit pad.
    if enforce_floor and cfg.min_pixels > 0 and area * scale * scale < cfg.min_pixels:
        scale = max(scale, min(math.sqrt(cfg.min_pixels / area), hard_cap))

    text_h_out = text_h * scale if text_h is not None else None
    # Legibility is an output constraint: a clear 24px line can shrink to 7px
    # on a tall receipt.
    capped = bool(text_h_out is not None and cfg.min_text_height_px > 0
                  and text_h_out < cfg.min_text_height_px * 0.98)
    return FitPlan(
        scale=scale,
        text_h_out=text_h_out,
        thrift_applied=thrift and scale < 1.0,
        legibility_capped=capped,
        reflow_cols=reflow_cols,
    )


def solve(w: int, h: int, text_h: Optional[float], cfg,
          *, allow_thrift: bool = True, allow_reflow: bool = True,
          enforce_floor: bool = True) -> FitPlan:
    """Chọn FitPlan tốt nhất cho ảnh (w, h). Xem module docstring."""
    base = _plan_for_dims(w, h, text_h, cfg, allow_thrift=allow_thrift,
                          enforce_floor=enforce_floor)

    # "Chữ đủ to" = thrift target: đã đạt thì reflow vô nghĩa.
    legibility_ref = cfg.thrift_target_text_h
    aspect = max(w, h) / max(1, min(w, h))
    reflow_worth_trying = (
        allow_reflow
        and cfg.reflow_enabled
        and h > w                                  # chỉ receipt dọc
        and aspect > cfg.reflow_aspect_trigger
        and text_h is not None and text_h > 0
        and base.text_h_out is not None
        and legibility_ref > 0
        and base.text_h_out < legibility_ref
    )
    if not reflow_worth_trying:
        return base

    best = base
    for n in range(2, cfg.reflow_max_cols + 1):
        col_h = math.ceil(h / n)
        canvas_w = n * w + (n - 1) * cfg.reflow_separator_px
        if col_h < w * cfg.reflow_min_col_aspect:
            break  # chia quá nhiều — cột bẹt hơn ngưỡng, n lớn hơn chỉ tệ hơn
        cand = _plan_for_dims(canvas_w, col_h, text_h, cfg, allow_thrift=allow_thrift,
                              enforce_floor=enforce_floor, reflow_cols=n)
        if best.text_h_out is None or (
            cand.text_h_out is not None and cand.text_h_out > best.text_h_out
        ):
            best = cand

    if (
        best.reflow_cols > 1
        and base.text_h_out > 0
        and best.text_h_out is not None
        and best.text_h_out >= base.text_h_out * cfg.reflow_gain_min
    ):
        return dataclasses.replace(
            best, reflow_gain=round(best.text_h_out / base.text_h_out, 2))
    return base


def align_up(n: int, block: int) -> int:
    return max(block, int(math.ceil(n / block)) * block)
