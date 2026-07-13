"""
Phối hợp xử lý (Processing Orchestrator) — Tiền xử lý, Vision LLM, Hậu xử lý,
Hallucination-triggered Fallback.

Architecture:
  1. Semaphore acquire — bound bởi request_timeout (asyncio.wait_for ở server.py).
  2. preprocess (geometric only: detect → rotate → crop → flip → resize).
  3. Primary LLM (Qwen3-VL vision) → postprocess.
  4. Fallback trigger khi: (a) `finish_reason == "length"` — truncation signal
     structural từ vLLM, HOẶC (b) hallucination_detector phát hiện loop
     (duplicate-items run hoặc char/n-gram loop trong string fields). Khi đó:
     paddle_text → text_extractor (text-only LLM mapping) → postprocess. Trả
     kết quả fallback thay cho primary.
  5. Semaphore release tự động qua `async with`.

Cancellation propagate qua CancelledError — wait_for ở server.py phát hiện
request_timeout và raise asyncio.TimeoutError lên handler để trả HTTP 408.
"""

import asyncio
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple

import cv2

from src.core.config import config
from src.extraction.hallucination_detector import (
    dedup_consecutive_items,
    detect_hallucination,
    scrub_hallu_fields,
)
from src.extraction.llm_extractor import extract_receipt_with_llm
from src.extraction.postprocessor import postprocess_receipt
from src.extraction.text_extractor import extract_receipt_text_only
from src.preprocessing.compat import preprocess_image_with_meta
from src.utils.errors import UpstreamServiceError
from src.utils.image_utils import load_image_from_bytes
from src.utils.logging_utils import get_logger

__all__ = ["process_receipt_images"]

logger = get_logger(__name__)

# Tham số JPEG encode dùng chung cho mọi blob gửi xuống VLM/PaddleOCR.
# - QUALITY=image_jpeg_quality (default 100): tối thiểu hóa lượng tử hóa luma.
# - SAMPLING_FACTOR_444: TẮT chroma subsampling. OpenCV mặc định 4:2:0 (nửa độ
#   phân giải màu cả 2 chiều) KỂ CẢ ở q=100 → nhòe mép chữ/màu, cộng dồn với
#   double-JPEG khi input gốc đã là JPEG. 4:4:4 giữ full chroma → near-lossless
#   cho text receipt, payload chỉ tăng ~10-20%.
_JPEG_ENCODE_PARAMS = [
    int(cv2.IMWRITE_JPEG_QUALITY), config.vllm.image_jpeg_quality,
    int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
]

# Semaphore giới hạn số request xử lý đồng thời — bound trùng với --max-num-seqs
# của vLLM để backpressure xảy ra ở FastAPI thay vì queue vLLM nội bộ.
_sem = asyncio.Semaphore(config.concurrency)

# Scalar fields trên receipt — items được kiểm tra riêng (list non-empty).
# Dùng để xác định "response có ≥1 field có dữ liệu" ở stage cuối pipeline.
# Các field phụ (subtotal/currency/payment_method/receipt_code) đã bị loại khỏi
# schema ở cấp project — chỉ còn merchant, date/time, total_amount.
_RECEIPT_SCALAR_FIELDS = (
    "merchant_name", "merchant_address",
    "transaction_date", "transaction_time",
    "total_amount",
)


def _count_substantive_scalars(result: Dict[str, Any]) -> int:
    """
    Đếm số scalar field có dữ liệu hợp lệ (non-null, string non-empty sau strip).
    Dùng cho log để phân biệt "primary trả 0 field" vs "primary trả vài field"
    khi gate empty_response / fallback. Items được log riêng (len(items)).
    """
    if not isinstance(result, dict):
        return 0
    n = 0
    for key in _RECEIPT_SCALAR_FIELDS:
        v = result.get(key)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        n += 1
    return n


def _has_substantive_data(result: Dict[str, Any]) -> bool:
    """
    True nếu receipt có items non-empty HOẶC ≥1 scalar có dữ liệu hợp lệ.

    Dùng cho cả 3 gate: empty_response trigger, salvage guard, và EMPTY
    RESPONSE log cuối pipeline. (Trước đây tách `_has_any_data` riêng nhưng
    hai hàm cùng semantic — đã gộp.) Response không items và không scalar nào
    = "không trích được gì" → đủ điều kiện chạy PaddleOCR fallback.
    """
    if not isinstance(result, dict):
        return False
    items = result.get("items") or result.get("it") or []
    if isinstance(items, list) and len(items) > 0:
        return True
    return _count_substantive_scalars(result) > 0


def _process_single_image_sync(
    raw_bytes: bytes, index: int, ref: str,
) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    """
    CPU-bound: decode → preprocess (geometric only) → encode JPEG.

    Trả về (jpeg_bytes, preprocess_meta); None CHỈ KHI decode hoặc encode fail.
    meta gồm reflow_applied (caller bật reflow hint trong prompt) + các metric
    chất lượng ảnh (input_long_side, median_text_height_px — xem
    preprocess_image_with_meta docstring) cho khối `image_quality` của response.
    """
    image = load_image_from_bytes(raw_bytes)
    if image is None:
        logger.warning("[ref=%s] Image[%d] load failed (corrupt bytes).", ref, index)
        return None

    processed, meta = preprocess_image_with_meta(image)

    try:
        # q=100 + 4:4:4 (xem _JPEG_ENCODE_PARAMS): giữ chữ nhỏ sắc nét, hạn chế
        # mất mát re-encode và double-JPEG khi input gốc đã là JPEG.
        success, encoded = cv2.imencode(".jpg", processed, _JPEG_ENCODE_PARAMS)
        if not success or encoded is None:
            logger.error("[ref=%s] Image[%d] encode failed.", ref, index)
            return None
        return encoded.tobytes(), meta
    except Exception as e:
        logger.error("[ref=%s] Image[%d] encode error: %s", ref, index, e)
        return None


async def _process_single_image(
    raw_bytes: bytes, index: int, ref: str,
) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    return await asyncio.to_thread(_process_single_image_sync, raw_bytes, index, ref)


def _build_image_quality(metas: List[Dict[str, Any]], *, ref: str) -> Dict[str, Any]:
    """
    Gom metric chất lượng ảnh từ preprocess meta → khối `image_quality` gắn vào
    response. METADATA-ONLY: không can thiệp kết quả extract, chỉ cho downstream
    biết kết quả đến từ ảnh "hệ thống đọc không nổi" thay vì thiếu trong im lặng.

    Nhiều ảnh → lấy giá trị XẤU NHẤT (min): cờ phản ánh ảnh yếu nhất batch.

    Khoá trả về:
      - input_long_side,
        input_short_side      : int|None — kích thước ảnh đầu vào trước mọi xử lý.
      - low_res_input         : bool — cạnh ngắn < QUALITY_MIN_INPUT_SHORT_SIDE
                                (dấu hiệu upstream nén ảnh trước khi gửi; dùng
                                cạnh ngắn vì batch nén thực tế cap width ~1000px
                                trong khi long side vẫn 1778-2366px).
      - median_text_height_px : float|None — chiều cao chữ ở DPI nguồn sau crop.
      - low_legibility        : bool|None — chữ < QUALITY_MIN_TEXT_HEIGHT_PX;
                                None = không đo được (detect 0 poly), KHÔNG suy
                                diễn thành true/false.
      - legibility_zoom_ratio : float — tỉ lệ legibility zoom lớn nhất đã áp
                                (1.0 = không zoom). Cho downstream biết hệ
                                thống đã chủ động phóng chữ nhỏ bao nhiêu.
    """
    pcfg = config.image_quality
    long_sides = [m.get("input_long_side") for m in metas if m.get("input_long_side")]
    short_sides = [m.get("input_short_side") for m in metas if m.get("input_short_side")]
    heights = [
        m.get("median_text_height_px") for m in metas
        if m.get("median_text_height_px") is not None
    ]
    zooms = [float(m.get("legibility_zoom_ratio") or 1.0) for m in metas]
    input_long_side = min(long_sides) if long_sides else None
    input_short_side = min(short_sides) if short_sides else None
    text_height = min(heights) if heights else None
    zoom_ratio = max(zooms) if zooms else 1.0

    low_res = bool(
        pcfg.quality_min_input_short_side > 0
        and input_short_side is not None
        and input_short_side < pcfg.quality_min_input_short_side
    )
    low_legibility: Optional[bool] = None
    if pcfg.quality_min_text_height_px > 0 and text_height is not None:
        low_legibility = bool(text_height < pcfg.quality_min_text_height_px)

    if low_res:
        logger.debug(
            "[ref=%s] LOW-RES INPUT | short_side=%dpx < %dpx (long=%s) — ảnh có "
            "dấu hiệu bị downscale phía upstream trước khi gửi; chi tiết mất ở "
            "bước đó không khôi phục được phía server",
            ref, input_short_side, pcfg.quality_min_input_short_side, input_long_side,
        )
    if low_legibility:
        logger.warning(
            "[ref=%s] LOW LEGIBILITY | median_text_height=%.1fpx < %.1fpx (đo sau "
            "deskew+crop, DPI nguồn) — chữ dưới ngưỡng đọc, kết quả nhiều khả năng "
            "thiếu/sai",
            ref, text_height, pcfg.quality_min_text_height_px,
        )

    return {
        "input_long_side": input_long_side,
        "input_short_side": input_short_side,
        "low_res_input": low_res,
        "median_text_height_px": (
            round(text_height, 1) if text_height is not None else None
        ),
        "low_legibility": low_legibility,
        "legibility_zoom_ratio": zoom_ratio,
    }


def _preprocess_for_fallback_sync(raw_bytes: bytes) -> Optional[bytes]:
    """Tiền xử lý ảnh cho fallback text-only: cardinal-orient + deskew + crop,
    KHÔNG reflow, KHÔNG legibility zoom (PaddleOCR tự resize theo det_limit).
    Trả JPEG bytes; None khi decode/encode fail (caller dùng raw).

    Lý do KHÔNG dùng lại preprocessed blob của primary: blob đó có thể đã reflow
    (xếp cột cạnh nhau) → path bbox-based đọc xen kẽ cột → loạn thứ tự item.
    Lý do KHÔNG feed raw cho PaddleOCR: raw frame nền tối lớn khiến doc_ori nội
    bộ misfire 180° → OCR gương → LLM map ra items=[]. Crop bỏ background nên
    doc_ori hết cớ misfire.
    """
    image = load_image_from_bytes(raw_bytes)
    if image is None:
        return None
    processed, _ = preprocess_image_with_meta(image, for_fallback_ocr=True)
    ok, encoded = cv2.imencode(".jpg", processed, _JPEG_ENCODE_PARAMS)
    return encoded.tobytes() if ok and encoded is not None else None


async def _preprocess_for_fallback(raw_bytes: bytes, *, ref: str) -> Optional[bytes]:
    try:
        return await asyncio.to_thread(_preprocess_for_fallback_sync, raw_bytes)
    except Exception as e:
        logger.warning(
            "[ref=%s] fallback preprocess failed (%s: %s) — dùng raw bytes",
            ref, type(e).__name__, e,
        )
        return None


async def _run_pipeline_stages(
    raw_bytes_list: List[bytes], *, ref: str,
) -> Tuple[Dict[str, Any], str, int, int, Dict[str, Any]]:
    """preprocess → vision LLM → scrub → postprocess → [optional fallback ĐÚNG 1 LẦN].

    Trả (receipt, engine, prompt_tokens, completion_tokens, image_quality).
    engine ∈ {"vllm", "paddle"}. image_quality: xem _build_image_quality.

    Fallback triggers:
      (a) hallu_abort — online streaming detector cắt giữa decode.
      (b) length — truncation; guided_decoding sẽ bịa phần thiếu.
      (c) dup_items_run — offline detect_hallucination (structural loop).
      (d) vlm_upstream_error — primary raised exception.
      (e) empty_response — primary không có substantive data.

    Salvage guard: nếu fallback all-null NHƯNG primary có substantive data
    → giữ primary (engine="vllm"). Bỏ qua khi trigger="empty_response".
    """
    tasks = [_process_single_image(b, i, ref) for i, b in enumerate(raw_bytes_list)]
    results = await asyncio.gather(*tasks)
    valid = [r for r in results if r is not None]
    valid_blobs: list[bytes] = [blob for blob, _ in valid]
    # Bật reflow hint nếu BẤT KỲ ảnh nào đã chia cột (thường chỉ 1 ảnh/receipt).
    any_reflow = any(bool(m.get("reflow_applied")) for _, m in valid)

    if not valid_blobs:
        raise UpstreamServiceError("All images failed decode/encode")

    # Cờ chất lượng ảnh — tính 1 lần từ preprocess meta, trả kèm MỌI nhánh
    # (primary lẫn fallback): chất lượng ảnh nguồn không phụ thuộc engine.
    image_quality = _build_image_quality([m for _, m in valid], ref=ref)

    # Primary: Qwen3-VL vision
    result_json: Dict[str, Any] = {}
    finish_reason = "error"
    p_tok = c_tok = 0
    primary_scalars = primary_items = 0
    primary_failed = False

    try:
        result_json, finish_reason, p_tok, c_tok = await extract_receipt_with_llm(
            valid_blobs, ref=ref, reflow_applied=any_reflow,
        )
        result_json = postprocess_receipt(result_json)
        result_json, nulled = scrub_hallu_fields(result_json)
        if nulled:
            logger.warning("[ref=%s] HALLU FIELDS NULLED | fields=%s", ref, nulled)
        primary_scalars = _count_substantive_scalars(result_json)
        primary_items = len(result_json.get("items") or [])
    except asyncio.CancelledError:
        raise
    except Exception:
        primary_failed = True

    trigger: str | None = None
    if primary_failed:
        trigger = "vlm_upstream_error"
    elif finish_reason == "hallu_abort":
        trigger = "hallu:stream_abort"
    elif finish_reason == "length":
        trigger = "vlm_truncated"
    if trigger is None:
        flagged, reason = detect_hallucination(result_json)
        if flagged:
            trigger = f"hallu:{reason}"
            logger.warning(
                "[ref=%s] HALLU FLAGGED on parsed result | reason=%s | full_json=%s",
                ref, reason,
                json.dumps(result_json, ensure_ascii=False),
            )
    if trigger is None and not _has_substantive_data(result_json):
        trigger = "empty_response"

    if trigger is None:
        return result_json, "vllm", p_tok, c_tok, image_quality

    # Fallback feed ảnh ĐÃ preprocess (cardinal-orient + deskew + crop, KHÔNG
    # reflow) thay vì raw. Raw frame nền tối lớn làm doc_ori nội bộ của PaddleOCR
    # misfire 180° → OCR gương → LLM map ra items=[]. Crop bỏ background nên hết
    # cớ misfire; tắt reflow để giữ thứ tự đọc 1-cột cho path bbox-based.
    # Fail (decode/preprocess) → về raw bytes như trước.
    fallback_source = await _preprocess_for_fallback(raw_bytes_list[0], ref=ref)
    if fallback_source is None:
        fallback_source = raw_bytes_list[0]
    # empty_response + vlm_truncated: VLM đã chạy và trả data (partial khi truncated).
    # Log full primary_json để debug xem VLM extract được gì trước khi fallback.
    if trigger in ("empty_response", "vlm_truncated"):
        logger.warning(
            "[ref=%s] FALLBACK to PaddleOCR | trigger=%s | primary tokens=p%d/c%d "
            "finish=%s items=%d scalars=%d | primary_json=%s",
            ref, trigger, p_tok, c_tok, finish_reason,
            primary_items, primary_scalars,
            json.dumps(result_json, ensure_ascii=False),
        )
    else:
        logger.warning("[ref=%s] FALLBACK to PaddleOCR | trigger=%s", ref, trigger)
    fallback_json, f_p_tok, f_c_tok = await extract_receipt_text_only(fallback_source, ref=ref)
    fallback_json = postprocess_receipt(fallback_json)

    # Hallu salvage: trigger là hallu:* → primary chứa loop garbage (vd 75 item
    # cùng tên liên tiếp). Dedupe consecutive same-name runs trước salvage —
    # KHÔNG keep raw primary vì items "non-empty" của nó là loop artefact.
    # Sau dedupe, nếu primary vẫn còn ≥ 1 item hợp lệ (run đầu thường là item
    # thật trước khi model rơi loop) thì coi đó là salvage candidate; nếu rỗng
    # hoặc fallback đã substantive thì bỏ qua nhánh này.
    if trigger.startswith("hallu:"):
        items_key = "items" if "items" in result_json else "it" if "it" in result_json else None
        if items_key:
            primary_items_list = result_json.get(items_key) or []
            if isinstance(primary_items_list, list) and primary_items_list:
                deduped, n_dropped = dedup_consecutive_items(primary_items_list)
                if n_dropped > 0:
                    result_json = dict(result_json)
                    result_json[items_key] = deduped
                    primary_items = len(deduped)
                    logger.warning(
                        "[ref=%s] HALLU dedupe primary | dropped=%d kept=%d | trigger=%s",
                        ref, n_dropped, primary_items, trigger,
                    )

    # Salvage guard
    if (
        trigger != "empty_response"
        and not _has_substantive_data(fallback_json)
        and _has_substantive_data(result_json)
    ):
        logger.warning(
            "[ref=%s] WORKFLOW=PADDLE_FALLBACK salvage | fallback all-null, "
            "primary có data (items=%d scalars=%d) → GIỮ primary | trigger=%s",
            ref, primary_items, primary_scalars, trigger,
        )
        return result_json, "vllm", p_tok, c_tok, image_quality

    # Items-salvage guard (hallu:* only). Guard scalar ở trên dùng
    # _has_substantive_data(fallback) — có thể =True chỉ vì scalar phụ (date
    # _datetime_sweep match nhầm) trong khi items=0. Với hoá đơn,
    # items mới là dữ liệu chính: nếu fallback KHÔNG có item nào mà primary
    # (đã dedupe loop) còn item hợp lệ, primary đáng giữ hơn fallback rỗng.
    # Chỉ áp cho hallu:* vì run đầu trước khi model rơi loop thường là item thật.
    fallback_items = len(fallback_json.get("items") or [])
    if trigger.startswith("hallu:") and fallback_items == 0 and primary_items > 0:
        logger.warning(
            "[ref=%s] WORKFLOW=PADDLE_FALLBACK items-salvage | fallback items=0, "
            "primary-deduped items=%d (scalars=%d) → GIỮ primary | trigger=%s",
            ref, primary_items, primary_scalars, trigger,
        )
        return result_json, "vllm", p_tok, c_tok, image_quality

    return fallback_json, "paddle", f_p_tok, f_c_tok, image_quality


async def process_receipt_images(
    raw_bytes_list: List[bytes],
    *,
    reference_id: str = "N/A",
) -> Dict[str, Any]:
    """
    Pipeline end-to-end: semaphore acquire → preprocess → vision LLM → postprocess
    → [optional fallback text-only].

    KHÔNG bound timeout ở tầng này — server.py bao bằng asyncio.wait_for(request_timeout).
    Backpressure qua semaphore (capacity = config.concurrency).
    """
    ref = reference_id if reference_id and reference_id != "N/A" else f"auto-{uuid.uuid4().hex[:8]}"
    t_start = time.perf_counter()

    async with _sem:
        try:
            result, engine, p_tok, c_tok, image_quality = await _run_pipeline_stages(
                raw_bytes_list, ref=ref,
            )
            # Gắn SAU postprocess/scrub/hallu-detect (các bước đó chỉ nhìn field
            # receipt, không đụng khoá này) — response thêm khối metadata
            # `image_quality`, additive với consumer hiện có.
            if isinstance(result, dict):
                result["image_quality"] = image_quality
            elapsed = time.perf_counter() - t_start
            if not _has_substantive_data(result):
                logger.warning(
                    "[ref=%s] EMPTY RESPONSE | %.2fs | imgs=%d items=%d scalars=%d "
                    "— no field extracted after primary + fallback (xem WORKFLOW + "
                    "ALL-NULL log cùng ref)",
                    ref, elapsed, len(raw_bytes_list),
                    len(result.get("items") or []),
                    _count_substantive_scalars(result),
                )
            logger.info(
                "[ref=%s] REQ DONE | engine=%s | tokens=p%d/c%d total=%d | %.2fs | imgs=%d",
                ref, engine, p_tok, c_tok, p_tok + c_tok, elapsed, len(raw_bytes_list),
            )
            return result
        except asyncio.CancelledError:
            logger.warning(
                "[ref=%s] CANCELLED | %.2fs", ref, time.perf_counter() - t_start,
            )
            raise
        except Exception as e:
            logger.error(
                "[ref=%s] FAILED | %.2fs | %s: %s",
                ref, time.perf_counter() - t_start, type(e).__name__, e,
            )
            raise
