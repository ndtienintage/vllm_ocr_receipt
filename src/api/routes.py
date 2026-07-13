"""
Router API Server cho OCR Mapping.
Endpoint tinh gọn tích hợp trực tiếp vào pipeline.

Architecture (simplified): MỘT lớp timeout (request_timeout) bao TOÀN BỘ vòng
đời request — decode + queue wait + xử lý. Vượt → HTTP 408. Backpressure dùng
asyncio.Semaphore (config.concurrency).
"""

import asyncio
import time

from fastapi import APIRouter, HTTPException

from src.schemas.request import OCRRequest
from src.core.config import config
from src.extraction.processing import process_receipt_images
from src.utils.errors import UpstreamServiceError
from src.utils.image_utils import decode_b64, fetch_image_bytes
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR Extraction"])

MAX_IMAGE_BYTES = config.max_image_bytes


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


@router.post("/extract")
async def extract_receipt_data(request: OCRRequest):
    """
    Trích xuất dữ liệu hóa đơn dạng JSON.
    - Truyền images_url: danh sách 1 ảnh công khai.
    - Truyền images_base64: danh sách 1 chuỗi base64 (hỗ trợ data URI).
    - Truyền reference_id: chuỗi định danh (mặc định "N/A").

    Bao toàn bộ flow trong asyncio.wait_for(request_timeout). Vượt → 408.
    """
    ref = request.reference_id or "N/A"
    t_received = time.perf_counter()
    logger.info("[ref=%s] REQ RECEIVED | source=%s", ref, "url" if request.images_url else "base64")

    try:
        return await asyncio.wait_for(
            _decode_and_process(request, ref),
            timeout=config.request_timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t_received
        logger.error(
            "[ref=%s] REQUEST TIMEOUT | %.2fs (max=%.0fs)",
            ref, elapsed, config.request_timeout,
        )
        raise HTTPException(
            status_code=408,
            detail=f"Yêu cầu vượt ngưỡng {config.request_timeout:.0f}s (actual={elapsed:.2f}s)",
        )
    except HTTPException:
        raise
    except UpstreamServiceError as e:
        raise HTTPException(status_code=503, detail=f"Dịch vụ AI không khả dụng: {e}")
    except Exception:
        logger.exception("[ref=%s] Pipeline failed unexpectedly", ref)
        raise HTTPException(status_code=500, detail="Lỗi xử lý nội bộ trong quá trình trích xuất AI.")
