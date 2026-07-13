"""Pixel-budget solver (ADR-06) — thay 6 ràng buộc chồng nhau của v1 bằng một
bài toán tường minh, pure function, unit-testable.

Bài toán: cho ảnh (w, h) sau crop + chiều cao chữ đo được text_h, tìm kế hoạch
(scale, reflow_cols, pad) sao cho:
  1. Trần cứng:  area ≤ max_pixels  VÀ  cạnh dài ≤ max_side   (vLLM không phải
     resize lần 2 — chống double-resample).
  2. Sàn:        area ≥ min_pixels  (khớp floor của vLLM; nếu trần cạnh chặn
     upscale thì pad cạnh ngắn thay vì để vLLM tự nội suy).
  3. Legibility: text_h × scale ≥ target_text_h nếu đạt được trong trần;
     không đạt → cờ `legibility_capped` (tín hiệu cho gate/metadata).
  4. Reflow:     CHỈ khi chứng minh bằng số — text_h_out có reflow ≥
     reflow_gain_min × text_h_out không reflow (trần cạnh dài là thứ reflow
     tháo được; trần diện tích thì không, vì reflow bảo toàn area).
  5. Token thrift: chữ to hơn target → cho phép downscale về target (tiết kiệm
     visual token, không mất legibility).

Alignment bội 32 (Qwen3-VL patch16 × merge2 — verify ở Phase 0 §4) do stage fit
thực thi lúc resize; solver chỉ quyết scale/reflow/pad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FitPlan:
    scale: float
    reflow_cols: int                 # 1 = không reflow
    pad_to_min_pixels: bool          # cần pad cạnh ngắn sau scale để đạt sàn area
    text_h_in: Optional[float]
    text_h_out: Optional[float]
    zoom_applied: bool
    thrift_applied: bool
    legibility_capped: bool
    notes: Dict[str, Any] = field(default_factory=dict)


def _plan_for_dims(w: int, h: int, text_h: Optional[float], cfg,
                   *, allow_zoom: bool, reflow_cols: int,
                   enforce_floor: bool = True) -> FitPlan:
    area = float(w) * float(h)
    long_side = float(max(w, h))

    hard_cap = min(cfg.max_side / long_side, math.sqrt(cfg.max_pixels / area))
    zoom_cap = min(hard_cap, cfg.max_upscale)

    # Zoom (phóng quá sàn vLLM) và thrift (hạ chữ to về target) tách target
    # riêng: ablation Phase 3 cho thấy zoom-quá-sàn ÂM trên proxy OCR
    # (-0.109 conf, +31% pixel) → zoom_target mặc định 0 (OFF); thrift trung
    # tính về conf nhưng tiết kiệm 13-26% pixel → giữ.
    want = 1.0
    zoom = thrift = False
    if allow_zoom and text_h is not None and text_h > 0:
        if cfg.zoom_target_text_h > 0 and text_h < cfg.zoom_target_text_h:
            want, zoom = cfg.zoom_target_text_h / text_h, True
        elif (cfg.token_thrift and cfg.thrift_target_text_h > 0
              and text_h > cfg.thrift_target_text_h):
            want, thrift = cfg.thrift_target_text_h / text_h, True

    scale = min(want, zoom_cap) if want > 1.0 else min(want, hard_cap)

    # Sàn diện tích (vLLM min_pixels): nâng scale nếu còn dưới sàn, nhưng không
    # vượt trần cứng — vượt không nổi thì pad (stage fit thực thi).
    # enforce_floor=False cho purpose fallback_ocr: PaddleOCR có det_limit riêng,
    # sàn của vLLM không liên quan.
    pad_needed = False
    if enforce_floor and cfg.min_pixels > 0 and area * scale * scale < cfg.min_pixels:
        floor_scale = math.sqrt(cfg.min_pixels / area)
        scale = max(scale, min(floor_scale, hard_cap))
        if area * scale * scale < cfg.min_pixels * 0.999:
            pad_needed = bool(cfg.pad_to_min_pixels)

    text_h_out = text_h * scale if text_h is not None else None
    capped = bool(
        zoom and text_h_out is not None
        and text_h_out < cfg.zoom_target_text_h * 0.98
    )
    return FitPlan(
        scale=scale,
        reflow_cols=reflow_cols,
        pad_to_min_pixels=pad_needed,
        text_h_in=text_h,
        text_h_out=text_h_out,
        zoom_applied=zoom and scale > 1.0,
        thrift_applied=thrift and scale < 1.0,
        legibility_capped=capped,
        notes={
            "dims_in": [int(w), int(h)],
            "hard_cap": round(hard_cap, 3),
            "want": round(want, 3),
        },
    )


def solve(w: int, h: int, text_h: Optional[float], cfg,
          *, allow_zoom: bool = True, allow_reflow: bool = True,
          enforce_floor: bool = True) -> FitPlan:
    """Chọn FitPlan tốt nhất cho ảnh (w, h). Xem module docstring."""
    base = _plan_for_dims(w, h, text_h, cfg, allow_zoom=allow_zoom,
                          reflow_cols=1, enforce_floor=enforce_floor)

    # Ngưỡng "chữ đủ to" cho reflow: zoom target nếu bật, không thì thrift target
    legibility_ref = (cfg.zoom_target_text_h if cfg.zoom_target_text_h > 0
                      else cfg.thrift_target_text_h)
    aspect = max(w, h) / max(1, min(w, h))
    reflow_worth_trying = (
        allow_reflow
        and cfg.reflow_enabled
        and h > w                                  # chỉ receipt dọc
        and aspect > cfg.reflow_aspect_trigger
        and text_h is not None and text_h > 0
        and base.text_h_out is not None
        and legibility_ref > 0
        and base.text_h_out < legibility_ref       # đã đủ to thì reflow vô nghĩa
    )
    if not reflow_worth_trying:
        return base

    best = base
    for n in range(2, cfg.reflow_max_cols + 1):
        col_h = math.ceil(h / n)
        canvas_w = n * w + (n - 1) * cfg.reflow_separator_px
        if col_h < w * cfg.reflow_min_col_aspect:
            break  # chia quá nhiều — cột bẹt hơn ngưỡng, n lớn hơn chỉ tệ hơn
        cand = _plan_for_dims(canvas_w, col_h, text_h, cfg,
                              allow_zoom=allow_zoom, reflow_cols=n,
                              enforce_floor=enforce_floor)
        if best.text_h_out is None or (
            cand.text_h_out is not None and cand.text_h_out > best.text_h_out
        ):
            best = cand

    if (
        best.reflow_cols > 1
        and base.text_h_out is not None and base.text_h_out > 0
        and best.text_h_out is not None
        and best.text_h_out >= base.text_h_out * cfg.reflow_gain_min
    ):
        object.__setattr__(  # frozen dataclass — ghi thêm evidence cho journal
            best, "notes",
            {**best.notes, "reflow_gain": round(best.text_h_out / base.text_h_out, 2),
             "base_text_h_out": round(base.text_h_out, 1)},
        )
        return best
    return base


def align_down(n: int, block: int) -> int:
    """Round-down về bội block (không bao giờ < block)."""
    return max(block, (int(n) // block) * block)


def align_up(n: int, block: int) -> int:
    return max(block, int(math.ceil(n / block)) * block)
