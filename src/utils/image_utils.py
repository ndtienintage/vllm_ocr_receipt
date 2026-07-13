"""
Tiện ích I/O ảnh cho Receipt OCR — CHỈ decode/fetch input. Mọi biến đổi
hình học/quang học nằm ở `src.preprocessing` (stage-based pipeline).

- load_image_from_bytes: Giải mã bytes thô → OpenCV BGR numpy array
- fetch_image_bytes: Tải ảnh bất đồng bộ từ URL
- decode_b64: Giải mã chuỗi base64 (có/không tiền tố data URI) → bytes
"""

import base64
from typing import Optional

import cv2
import httpx
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        # KHÔNG có timeout — request_timeout (server.py wait_for) là lớp duy nhất
        # bao toàn bộ vòng đời, kể cả connect/read.
        _http_client = httpx.AsyncClient(timeout=None, follow_redirects=True)
    return _http_client


def load_image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """
    Giải mã bytes ảnh thô (JPEG/PNG/...) thành mảng numpy OpenCV BGR.
    OpenCV mặc định honor EXIF orientation. Trả về None nếu giải mã thất bại.
    """
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("cv2.imdecode returned None — unsupported format or corrupted bytes.")
        return img
    except Exception as e:
        logger.error(f"load_image_from_bytes failed: {e}")
        return None


async def fetch_image_bytes(url: str) -> bytes:
    """Tải ảnh bất đồng bộ từ URL công khai. Ném ValueError nếu thất bại."""
    client = _get_http_client()
    logger.debug(f"Downloading image from: {url}")
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
    except httpx.TimeoutException as e:
        logger.error(f"Timeout downloading image from {url}: {e}")
        raise ValueError(f"Timeout fetching image URL: {url}") from e
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code} downloading image from {url}: {e}")
        raise ValueError(f"HTTP error {e.response.status_code} fetching image URL: {url}") from e
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise ValueError(f"Failed to fetch image URL: {url}") from e


def decode_b64(b64_string: str) -> bytes:
    """
    Giải mã chuỗi base64 thành bytes thô.
    Chấp nhận base64 thuần hoặc data URI: data:image/jpeg;base64,<...>
    """
    s = b64_string.strip()
    if s.startswith("data:"):
        _, _, s = s.partition(",")
    try:
        return base64.b64decode(s)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}") from e
