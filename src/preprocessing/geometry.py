"""Hình học thuần (pure functions, không I/O, không model) — unit-testable.

Quy ước góc: long-axis angle ∈ (-90, 90], đo bằng atan2(dy, dx) trên toạ độ
ảnh (y hướng xuống). Góc rotate dương truyền cho cv2.getRotationMatrix2D và
các hàm ở đây dùng CÙNG quy ước, nên `rotate_image_free(img, angle)` với
angle = long-axis angle đo được sẽ đưa text về nằm ngang (được chốt bằng
unit test angle-recovery, không dựa vào suy luận dấu).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Poly measurements ─────────────────────────────────────────────────────────

def poly_long_axis_angle(poly: np.ndarray) -> Optional[float]:
    """Góc long-axis ∈ (-90, 90] của 1 poly ≥4 điểm (axis, không phải vector)."""
    if poly.shape[0] < 4:
        return None
    closed = np.vstack([poly, poly[:1]])
    edges = np.diff(closed, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return angle


def poly_elongation(poly: np.ndarray) -> float:
    """Tỉ lệ cạnh dài / cạnh ngắn của poly (≥ 1). Poly gần vuông (~1) là
    cell số đơn lẻ / dấu chấm — long-axis của nó vô nghĩa cho deskew."""
    if poly.shape[0] < 4:
        return 1.0
    closed = np.vstack([poly, poly[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    longest = float(lengths.max())
    shortest = float(max(lengths.min(), 1e-6))
    return longest / shortest


def poly_heights(polys: np.ndarray) -> np.ndarray:
    """Span trục y của từng poly — proxy chiều cao dòng chữ khi text đã ngang."""
    ys = polys[..., 1]
    return ys.max(axis=1) - ys.min(axis=1)


def median_text_height(polys: Optional[np.ndarray]) -> Optional[float]:
    if polys is None or len(polys) == 0:
        return None
    return float(np.median(poly_heights(polys)))


def vertical_axis_ratio(polys: np.ndarray, *, min_elongation: float = 1.5,
                        vertical_threshold_deg: float = 45.0) -> float:
    """Tỷ lệ polys THON DÀI có long-axis nghiêng về trục dọc (|angle| > 45°).

    Lọc elongation để loại cell vuông (số đơn lẻ) — nguồn nhiễu đã quan sát
    trên receipt dạng bảng (Phase 0: `501cbb61` vertical_ratio 0.53 giả).
    Trả 0.0 khi không đủ mẫu sau lọc (không đủ evidence ≠ sideways).
    """
    vert = total = 0
    for p in polys:
        if poly_elongation(p) < min_elongation:
            continue
        a = poly_long_axis_angle(p)
        if a is None:
            continue
        if abs(a) > vertical_threshold_deg:
            vert += 1
        total += 1
    if total < 3:
        return 0.0
    return vert / total


def circular_axis_median(axes_deg: List[float]) -> float:
    """Median tròn cho axes π-symmetric (2θ trick).

    Chọn phần tử mẫu minimize tổng khoảng cách tròn trong không gian 2θ —
    robust với outlier hơn circular mean (ADR-04; mean đã hỏng trên
    `501cbb61`). O(N²) chấp nhận được với N ≤ vài trăm polys.
    """
    if not axes_deg:
        return 0.0
    if len(axes_deg) == 1:
        return float(axes_deg[0])
    doubled = np.deg2rad(2.0 * np.asarray(axes_deg, dtype=np.float64))
    # khoảng cách tròn pairwise trong 2θ-space
    diff = doubled[:, None] - doubled[None, :]
    dist = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    best = int(np.argmin(dist.sum(axis=1)))
    return float(axes_deg[best])


def estimate_tilt_deg(polys: np.ndarray, *, min_elongation: float = 2.0,
                      min_samples: int = 3) -> Optional[float]:
    """Ước lượng camera-tilt: circular median long-axis của polys thon dài.

    Trả None khi không đủ mẫu tin cậy (caller skip deskew, ghi lý do).
    """
    axes = []
    for p in polys:
        if poly_elongation(p) < min_elongation:
            continue
        a = poly_long_axis_angle(p)
        if a is not None:
            axes.append(a)
    if len(axes) < min_samples:
        return None
    return circular_axis_median(axes)


# ── Transforms (image + polys giữ đồng bộ) ────────────────────────────────────

def rotate_cardinal(image: np.ndarray, polys: Optional[np.ndarray],
                    k_ccw: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Xoay k×90° CCW bằng cv2.rotate (pixel-exact, không nội suy).
    Polys transform analytically."""
    k = int(k_ccw) % 4
    if k == 0:
        return image, polys
    h, w = image.shape[:2]
    if k == 1:
        out = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif k == 2:
        out = cv2.rotate(image, cv2.ROTATE_180)
    else:
        out = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if polys is None:
        return out, None
    new = polys.copy()
    if k == 1:      # (x,y) → (y, w-1-x)
        new[..., 0] = polys[..., 1]
        new[..., 1] = (w - 1) - polys[..., 0]
    elif k == 2:    # (x,y) → (w-1-x, h-1-y)
        new[..., 0] = (w - 1) - polys[..., 0]
        new[..., 1] = (h - 1) - polys[..., 1]
    else:           # (x,y) → (h-1-y, x)
        new[..., 0] = (h - 1) - polys[..., 1]
        new[..., 1] = polys[..., 0]
    return out, new


def rotate_free(image: np.ndarray, polys: Optional[np.ndarray],
                angle_deg: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Xoay góc tự do với canvas expand + BORDER_REPLICATE + INTER_CUBIC
    (giữ nét stroke ở góc nhỏ). Polys transform qua cùng ma trận."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += (new_w - w) / 2.0
    m[1, 2] += (new_h - h) / 2.0
    rotated = cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    if polys is None:
        return rotated, None
    ones = np.ones((*polys.shape[:2], 1), dtype=np.float32)
    homog = np.concatenate([polys, ones], axis=2)
    new_polys = (homog @ m.T.astype(np.float32))
    return rotated, new_polys.astype(np.float32)


def union_bbox(polys: np.ndarray, image_shape, padding: int = 0
               ) -> Tuple[int, int, int, int]:
    """Bbox(union polys) ± padding, clip biên. Trả (x1, y1, x2, y2)."""
    h, w = image_shape[:2]
    pts = polys.reshape(-1, 2)
    x1 = max(0, int(pts[:, 0].min()) - padding)
    y1 = max(0, int(pts[:, 1].min()) - padding)
    x2 = min(w, int(math.ceil(pts[:, 0].max())) + padding)
    y2 = min(h, int(math.ceil(pts[:, 1].max())) + padding)
    return x1, y1, x2, y2


def crop_with_polys(image: np.ndarray, polys: np.ndarray, box: Tuple[int, int, int, int]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Crop theo box (x1,y1,x2,y2) + dịch polys theo offset."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return image, polys
    cropped = image[y1:y2, x1:x2]
    return cropped, polys - np.array([x1, y1], dtype=np.float32)


def crop_axis_aligned(image: np.ndarray, poly: np.ndarray,
                      min_side_px: int = 10) -> Optional[np.ndarray]:
    """Crop bbox của 1 poly (cho textline classifier). None nếu quá nhỏ."""
    h, w = image.shape[:2]
    pts = poly.reshape(-1, 2)
    x1 = max(0, int(pts[:, 0].min()))
    y1 = max(0, int(pts[:, 1].min()))
    x2 = min(w, int(pts[:, 0].max()))
    y2 = min(h, int(pts[:, 1].max()))
    if x2 - x1 < min_side_px or y2 - y1 < min_side_px:
        return None
    return image[y1:y2, x1:x2]


def aspect_ratio(h: int, w: int) -> float:
    if h <= 0 or w <= 0:
        return 1.0
    return max(h, w) / max(1, min(h, w))
