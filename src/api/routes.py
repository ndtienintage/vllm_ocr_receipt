"""
Router API Server cho OCR Mapping.
Endpoint tinh gọn tích hợp trực tiếp vào pipeline.

Architecture (simplified): MỘT lớp timeout (request_timeout) bao TOÀN BỘ vòng
đời request — decode + queue wait + xử lý. Vượt → HTTP 408. Backpressure dùng
asyncio.Semaphore (config.concurrency).
"""

import asyncio
import time

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.schemas.request import OCRRequest
from src.core.config import config
from src.extraction.processing import process_receipt_images
from src.utils.errors import UpstreamServiceError
from src.utils.image_utils import decode_b64, fetch_image_bytes
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR Extraction"])

MAX_IMAGE_BYTES = config.max_image_bytes

CALLBACK_TIMEOUT = 30.0
CALLBACK_RETRY_DELAYS = (1.0, 3.0)  # 3 lần thử: ngay, +1s, +3s
# Giữ strong ref tới task nền — asyncio chỉ giữ weak ref, task có thể bị GC giữa chừng.
_bg_tasks: set[asyncio.Task] = set()


async def _decode_and_process(request: OCRRequest, ref: str) -> dict:
    """Decode/fetch images → validate → run pipeline. Tách ra để wait_for bao gọn."""
    raw_images: list[bytes] = []

    try:
        if request.images_url:
            raw_images = list(
                await asyncio.gather(*[fetch_image_bytes(str(url)) for url in request.images_url])
            )
        else:
            raw_images = list(
                await asyncio.gather(*[asyncio.to_thread(decode_b64, b64) for b64 in request.images_base64])
            )

        for idx, b in enumerate(raw_images):
            if b is None or len(b) == 0:
                raise ValueError(f"Empty image payload at index {idx}")
            if len(b) > MAX_IMAGE_BYTES:
                raise ValueError(f"Image too large at index {idx}: {len(b)} bytes")

    except (asyncio.CancelledError, asyncio.TimeoutError):
        raise  # propagate cho wait_for bên ngoài
    except Exception as e:
        logger.error("[ref=%s] IMG DECODE FAILED | %s: %s", ref, type(e).__name__, e)
        raise HTTPException(
            status_code=422,
            detail="Ảnh không hợp lệ: lỗi giải mã base64 hoặc URL không tải được.",
        )

    return await process_receipt_images(raw_images, reference_id=ref)


def _http_error(e: Exception, ref: str, elapsed: float) -> HTTPException:
    """Map exception → HTTPException. Dùng chung cho cả nhánh sync (raise) và
    nhánh callback (serialize thành payload lỗi). PHẢI gọi trong except block."""
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, asyncio.TimeoutError):
        logger.error(
            "[ref=%s] REQUEST TIMEOUT | %.2fs (max=%.0fs)",
            ref, elapsed, config.request_timeout,
        )
        return HTTPException(
            status_code=408,
            detail=f"Yêu cầu vượt ngưỡng {config.request_timeout:.0f}s (actual={elapsed:.2f}s)",
        )
    if isinstance(e, UpstreamServiceError):
        return HTTPException(status_code=503, detail=f"Dịch vụ AI không khả dụng: {e}")
    logger.exception("[ref=%s] Pipeline failed unexpectedly", ref)
    return HTTPException(status_code=500, detail="Lỗi xử lý nội bộ trong quá trình trích xuất AI.")


async def _post_callback(url: str, payload: dict, ref: str) -> None:
    """POST kết quả về callback_url, retry theo CALLBACK_RETRY_DELAYS rồi bỏ cuộc."""
    async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT, follow_redirects=True) as client:
        for attempt in range(len(CALLBACK_RETRY_DELAYS) + 1):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                logger.info("[ref=%s] CALLBACK OK | HTTP %d | %s", ref, resp.status_code, url)
                return
            except Exception as e:
                logger.warning(
                    "[ref=%s] CALLBACK ATTEMPT %d FAILED | %s: %s",
                    ref, attempt + 1, type(e).__name__, e,
                )
                if attempt < len(CALLBACK_RETRY_DELAYS):
                    await asyncio.sleep(CALLBACK_RETRY_DELAYS[attempt])
    logger.error("[ref=%s] CALLBACK GIVEN UP | %s | kết quả bị mất", ref, url)


async def _process_and_callback(request: OCRRequest, ref: str) -> None:
    """Nhánh nền: chạy pipeline rồi POST kết quả (hoặc lỗi) về callback_url."""
    t_start = time.perf_counter()
    try:
        payload = await asyncio.wait_for(
            _decode_and_process(request, ref),
            timeout=config.request_timeout,
        )
    except Exception as e:
        err = _http_error(e, ref, time.perf_counter() - t_start)
        payload = {"status": "error", "status_code": err.status_code, "error": err.detail}
    payload["reference_id"] = ref
    await _post_callback(str(request.callback_url), payload, ref)


@router.post("/extract")
async def extract_receipt_data(request: OCRRequest):
    """
    Trích xuất dữ liệu hóa đơn dạng JSON.
    - Truyền images_url: danh sách 1 ảnh công khai.
    - Truyền images_base64: danh sách 1 chuỗi base64 (hỗ trợ data URI).
    - Truyền reference_id: chuỗi định danh (mặc định "N/A").
    - Truyền callback_url (tùy chọn): trả 202 ngay, kết quả POST về URL đó khi
      xong (kể cả khi lỗi: {"status":"error","status_code":...,"error":...}).

    Bao toàn bộ flow trong asyncio.wait_for(request_timeout). Vượt → 408.
    """
    ref = request.reference_id or "N/A"
    t_received = time.perf_counter()
    logger.info("[ref=%s] REQ RECEIVED | source=%s", ref, "url" if request.images_url else "base64")

    if request.callback_url:
        task = asyncio.create_task(_process_and_callback(request, ref))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
        logger.info("[ref=%s] ACCEPTED | callback=%s", ref, request.callback_url)
        return JSONResponse(
            status_code=202,
            content={"status": "accepted", "reference_id": ref},
        )

    try:
        return await asyncio.wait_for(
            _decode_and_process(request, ref),
            timeout=config.request_timeout,
        )
    except Exception as e:
        raise _http_error(e, ref, time.perf_counter() - t_received) from e
