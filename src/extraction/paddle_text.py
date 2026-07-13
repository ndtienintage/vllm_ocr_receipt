"""
PaddleOCR Text Extractor — wrapper PaddleOCR PP-OCRv5 server pipeline cho
fallback text-only path.

Vai trò trong kiến trúc tổng:
  Primary path = Qwen3-VL (vision) đọc trực tiếp ảnh.
  Fallback path = paddle_text trích xuất text+bbox → text_extractor map vào
  schema qua 1 LLM call text-only KHÔNG kèm ảnh.

  Module này CHỈ chạy khi hallucination_detector flag primary result là tệ —
  KHÔNG chạy mọi request (avoid cost overhead unless needed).

Threading model (giống module preprocessing.py):
  - Singleton PaddleOCR pipeline, lazy init lần đầu predict.
  - 1 lock duy nhất quanh predict() — paddleocr nội bộ KHÔNG thread-safe.
  - Async wrapper qua asyncio.to_thread → không block event loop.

Format output cho prompt (xem format_text_block):
  `<x1>,<y1>,<x2>,<y2>|<text>` — 1 dòng / 1 OCR line, top-to-bottom.
  4 toạ độ axis-aligned (top-left + bottom-right). LLM dùng x2 để detect
  column alignment ở items rows (price/total thường align phải), y2 để biết
  line height (multi-line text vs compact).

Env vars (PADDLE_TEXT_*): khai báo + default + rationale từng knob nằm Ở MỘT
NƠI DUY NHẤT — PaddleTextConfig trong src/core/config.py. Trước đây docstring
này lặp lại bảng env và đã lệch giá trị thực tế (unclip 1.8 vs 1.2, row_overlap
0.5 vs 0.85) → không duplicate nữa.

LƯU Ý về token budget:
  format_text_block không còn cap theo char — text_extractor._build_fitted_prompt
  là single source of truth cho input-token budget. Biện pháp giảm token
  trong module này:
    1. Merge polys cùng row → giảm 30-60% line count cho receipt typical.
    2. Drop conf khỏi format + quantize bbox → giảm char/line.
  Format hiện tại emit đủ 4 toạ độ `x1,y1,x2,y2` (axis-aligned bbox). So với
  variant chỉ x1,y1, tốn thêm ~30-40% char/line nhưng cho LLM thông tin
  column-alignment (qua x2) hữu ích để map items rows chính xác. Receipt
  500+ items vẫn có thể rơi vào trim ở text_extractor — cần escalate sang
  chunking + multi-call (out of scope).
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
from typing import Any, Dict, List

import cv2
import numpy as np

from src.core.config import config as _app_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "extract_text_lines_async",
    "format_text_block",
]


_OCR_PIPELINE = None
_OCR_INIT_LOCK = threading.Lock()
_OCR_INIT_FAILED = False
_OCR_PREDICT_LOCK = threading.Lock()


# ── Lazy init ─────────────────────────────────────────────────────────────────

def _get_pipeline():
    """Lazy init full PaddleOCR pipeline (det + rec). Trả None khi init fail."""
    global _OCR_PIPELINE, _OCR_INIT_FAILED
    if _OCR_PIPELINE is not None:
        return _OCR_PIPELINE
    if _OCR_INIT_FAILED:
        return None
    with _OCR_INIT_LOCK:
        if _OCR_PIPELINE is not None:
            return _OCR_PIPELINE
        if _OCR_INIT_FAILED:
            return None
        try:
            from paddleocr import PaddleOCR
            cfg = _app_config.paddle_text
            t0 = time.perf_counter()
            _OCR_PIPELINE = PaddleOCR(
                lang=cfg.lang,
                device=cfg.device,
                text_detection_model_name=cfg.det_model,
                text_recognition_model_name=cfg.rec_model,
                use_doc_orientation_classify=cfg.use_doc_ori,
                use_doc_unwarping=cfg.use_doc_unwarping,
                use_textline_orientation=cfg.use_textline_ori,
                # Detection (DB) tuning cho receipt mờ + diacritic VN:
                text_det_thresh=cfg.det_thresh,
                text_det_box_thresh=cfg.det_box_thresh,
                text_det_unclip_ratio=cfg.det_unclip_ratio,
                text_det_limit_side_len=cfg.det_limit_side_len,
                text_det_limit_type=cfg.det_limit_type,
            )
            logger.info(
                "paddle_text pipeline ready | det=%s rec=%s lang=%s device=%s "
                "doc_ori=%s textline_ori=%s unwarp=%s | "
                "det_thresh=%.2f box_thresh=%.2f unclip=%.2f side=%d/%s | "
                "min_score=%.2f quant=%d | %.2fs",
                cfg.det_model, cfg.rec_model, cfg.lang, cfg.device,
                cfg.use_doc_ori, cfg.use_textline_ori, cfg.use_doc_unwarping,
                cfg.det_thresh, cfg.det_box_thresh, cfg.det_unclip_ratio,
                cfg.det_limit_side_len, cfg.det_limit_type,
                cfg.min_score, cfg.bbox_quant,
                time.perf_counter() - t0,
            )
            return _OCR_PIPELINE
        except Exception as e:
            _OCR_INIT_FAILED = True
            logger.warning(
                "paddle_text init failed — fallback path will be unavailable: %s: %s",
                type(e).__name__, e,
            )
            return None


# ── Predict & normalize ───────────────────────────────────────────────────────

def _predict_sync(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Decode bytes → numpy → PaddleOCR.predict → list[{text, bbox, score, y_center}].

    bbox = axis-aligned [x1, y1, x2, y2] (đơn giản, dễ feed prompt).
    Filter score < _MIN_SCORE để loại OCR rác (text bóng, shadow, viền).
    Sắp xếp top→bottom theo y_center, tie-break x trái → khớp reading order
    tự nhiên hoá đơn (header → items → totals → footer).
    """
    ocr = _get_pipeline()
    if ocr is None:
        return []

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("paddle_text: decode fail (len=%d)", len(image_bytes))
        return []

    with _OCR_PREDICT_LOCK:
        try:
            result = ocr.predict(input=image)
        except Exception as e:
            logger.warning("paddle_text predict failed: %s: %s", type(e).__name__, e)
            return []

    if not result:
        return []
    res0 = result[0]

    if isinstance(res0, dict):
        texts = list(res0.get("rec_texts", []) or [])
        scores = list(res0.get("rec_scores", []) or [])
        polys = list(res0.get("dt_polys", []) or [])
    else:
        texts = list(getattr(res0, "rec_texts", []) or [])
        scores = list(getattr(res0, "rec_scores", []) or [])
        polys = list(getattr(res0, "dt_polys", []) or [])

    cfg = _app_config.paddle_text
    lines: List[Dict[str, Any]] = []
    for text, score, poly in zip(texts, scores, polys):
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = 0.0
        if s < cfg.min_score:
            continue
        text = (text or "").strip()
        if not text:
            continue
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            continue
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append({
            "text": text,
            "bbox": [x1, y1, x2, y2],
            "score": s,
            "y_center": (y1 + y2) / 2.0,
        })

    lines.sort(key=lambda h: (h["y_center"], h["bbox"][0]))
    lines = _merge_into_rows(lines)

    if len(lines) > cfg.max_lines:
        lines = lines[:cfg.max_lines]

    return lines


def _split_row_by_col_gap(
    polys: List[Dict[str, Any]],
    col_gap_ratio: float,
) -> List[List[Dict[str, Any]]]:
    """
    Tách 1 row (đã group theo Y-overlap) thành các sub-row độc lập khi phát
    hiện gap-x lớn giữa 2 poly liên tiếp — pattern điển hình của layout 2-cột
    (vd header `LOGO     địa chỉ`, footer `Hotline    Website`).

    Threshold: gap > col_gap_ratio × median-height-của-row. Lý do dùng height
    làm scale (không dùng image-width): font height ≈ char width của receipt
    POS typical → 5× height ≈ 5 ký tự gap, gấp nhiều lần inter-token bình
    thường (1-2 ký tự) nhưng nhỏ hơn column-divide thực (10+ ký tự whitespace).

    col_gap_ratio ≤ 0 → no-op (giữ behavior cũ: gộp toàn row thành 1 line).

    Polys input được sort theo x_left trước khi quét gap. Trả list các sub-row
    (≥ 1); mỗi sub-row giữ thứ tự x_left tăng dần.
    """
    if len(polys) <= 1 or col_gap_ratio <= 0:
        return [polys]

    sorted_polys = sorted(polys, key=lambda h: h["bbox"][0])
    heights = [
        h["bbox"][3] - h["bbox"][1]
        for h in sorted_polys
        if h["bbox"][3] > h["bbox"][1]
    ]
    if not heights:
        return [sorted_polys]
    median_h = statistics.median(heights)
    threshold = col_gap_ratio * median_h

    sub_rows: List[List[Dict[str, Any]]] = [[sorted_polys[0]]]
    for prev, cur in zip(sorted_polys, sorted_polys[1:]):
        # gap = left_of_current - right_of_previous. Negative = overlap → cùng cluster.
        gap = cur["bbox"][0] - prev["bbox"][2]
        if gap > threshold:
            sub_rows.append([cur])
        else:
            sub_rows[-1].append(cur)
    return sub_rows


def _merge_into_rows(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gộp các poly cùng row vật lý thành 1 line duy nhất.

    PaddleOCR det model thường tách 1 dòng receipt thành nhiều poly nhỏ (ví dụ
    `Coca Cola` | `2` | `25,000` | `50,000` thành 4 polys). Trong text-only
    mapping path, mỗi poly = 1 dòng prompt → tốn token mà không tăng info.
    Gộp chúng vào 1 line:
      - text = join " " theo thứ tự x trái→phải.
      - bbox = union (giữ row-level x-range cho LLM detect column structure).
      - y_center = recompute từ bbox merged.
      - score = avg (defensive only — score không emit ra prompt nữa).

    Quy tắc cùng-row: 2 box cùng row nếu Y-overlap ≥ _ROW_OVERLAP × min(h1, h2).
    Vì input đã sort theo y_center top→bottom, chỉ cần check vs row cuối cùng
    (O(n) thay vì O(n²)). Receipt scan không zig-zag y → safe.

    Column-aware split: trong mỗi row đã group, nếu phát hiện gap-x lớn
    (>= _COL_GAP_RATIO × median_height) → tách thành các sub-row riêng thay
    vì merge thành 1 dòng. Tránh gộp nhầm 2-cột header `LOGO    địa chỉ` hay
    footer `Hotline    Website` thành 1 line giả. Item row có gap nhỏ vẫn
    merge bình thường.

    No-op khi ≤ 1 line.
    """
    if len(lines) <= 1:
        return lines

    cfg = _app_config.paddle_text
    rows: List[List[Dict[str, Any]]] = []
    for ln in lines:
        y1, y2 = ln["bbox"][1], ln["bbox"][3]
        h_ln = max(1, y2 - y1)
        if rows:
            last = rows[-1]
            ry1 = min(h["bbox"][1] for h in last)
            ry2 = max(h["bbox"][3] for h in last)
            h_row = max(1, ry2 - ry1)
            overlap = max(0, min(y2, ry2) - max(y1, ry1))
            if overlap >= cfg.row_overlap * min(h_ln, h_row):
                last.append(ln)
                continue
        rows.append([ln])

    merged: List[Dict[str, Any]] = []
    for row in rows:
        if len(row) == 1:
            merged.append(row[0])
            continue
        # Tách row thành các sub-row độc lập khi gap-x quá lớn (2-cột layout).
        # Sub-row chỉ chứa các poly thực sự dính nhau theo phương ngang.
        for sub in _split_row_by_col_gap(row, cfg.col_gap_ratio):
            if len(sub) == 1:
                merged.append(sub[0])
                continue
            x1 = min(h["bbox"][0] for h in sub)
            y1 = min(h["bbox"][1] for h in sub)
            x2 = max(h["bbox"][2] for h in sub)
            y2 = max(h["bbox"][3] for h in sub)
            text = " ".join(h["text"] for h in sub)
            score = sum(h["score"] for h in sub) / len(sub)
            merged.append({
                "text": text,
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "y_center": (y1 + y2) / 2.0,
            })

    # Sub-row trong cùng physical row có cùng y → restore thứ tự (y, x)
    # cho output deterministic và khớp expectation "sorted top-to-bottom by
    # y-center, tie-break by x_left" trong prompt.
    merged.sort(key=lambda h: (h["y_center"], h["bbox"][0]))
    return merged


async def extract_text_lines_async(image_bytes: bytes, *, ref: str = "N/A") -> List[Dict[str, Any]]:
    """Async wrapper. Trả [] khi disabled / init-fail / decode-fail / predict-fail
    — caller có thể an toàn fallback xuống fail_safe_receipt."""
    if not _app_config.paddle_text.enabled:
        return []
    return await asyncio.to_thread(_predict_sync, image_bytes)


# ── Format cho text-only mapping prompt ───────────────────────────────────────

_BLOCK_HEADER = "<ocr_text>\n"
_BLOCK_FOOTER = "\n</ocr_text>\n"


def format_text_block(lines: List[Dict[str, Any]]) -> str:
    """
    Format OCR lines thành block để inject vào text-only mapping prompt.

    Format dòng: `x1,y1,x2,y2|text` (axis-aligned bbox đầy đủ).
      - bbox coords được quantize chia `bbox_quant` (default 2) để tiết kiệm
        char/coord. Ratio quantize đồng nhất cho mọi coord → preserve relative
        layout (column clustering, row alignment vẫn đúng).
      - Giữ x2,y2 (right/bottom): x2 cho LLM biết right-edge alignment của
        các cột số (price/qty/total thường align phải), giúp map items rows
        sang đúng field; y2 cho biết line height (phân biệt tall multi-line
        text vs compact single line).
      - confidence không emit (đã filter ≥ `min_score` ở predict step; LLM
        không cần re-score).

    KHÔNG cap theo char ở tầng này. text_extractor._build_fitted_prompt là
    single source of truth về input-token budget — sẽ trim tail-first khi
    vẫn không fit (ví dụ: receipt 500+ items thực).

    Defensive: per-line truncate nếu text > `max_text_chars` (chỉ trigger với
    pathological output, không phải data thật).

    Trả "" nếu không có line — caller phải xử lý empty (vd: fail_safe).
    """
    if not lines:
        return ""

    cfg = _app_config.paddle_text
    formatted: List[str] = []
    for ln in lines:
        x1 = ln["bbox"][0] // cfg.bbox_quant
        y1 = ln["bbox"][1] // cfg.bbox_quant
        x2 = ln["bbox"][2] // cfg.bbox_quant
        y2 = ln["bbox"][3] // cfg.bbox_quant
        text = ln["text"].replace("\n", " ").replace("\r", " ").strip()
        if not text:
            continue
        if len(text) > cfg.max_text_chars:
            text = text[:cfg.max_text_chars].rstrip() + "…"
        formatted.append(f"{x1},{y1},{x2},{y2}|{text}")

    if not formatted:
        return ""

    return _BLOCK_HEADER + "\n".join(formatted) + _BLOCK_FOOTER
