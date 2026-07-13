"""
Ứng dụng FastAPI chính.

Điểm truy cập (Entry point) cho API Receipt OCR Mapping.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as ocr_router
from src.core.config import config
from src.clients.vllm import close_shared_clients, get_shared_vllm_client
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Khởi tạo Eager để loại bỏ hiện tượng trễ (latency spike) ở request đầu tiên:
    - Làm ấm (Warm) pool kết nối vLLM client
    """
    logger.info("Starting Receipt OCR API...")
    logger.info(
        "Config | concurrency=%d | request_timeout=%.0fs",
        config.concurrency, config.request_timeout,
    )

    try:
        await get_shared_vllm_client(
            base_url=config.vllm.base_url,
            model=config.vllm.model,
            api_key=config.vllm.api_key,
        )
        logger.info(
            f"vLLM client initialized — {config.vllm.model} @ {config.vllm.base_url}"
        )
    except Exception as e:
        logger.warning(f"vLLM client pre-warm failed (will retry on first request): {e}")

    yield

    logger.info("Shutting down OCR API; closing shared clients.")
    await close_shared_clients()


app = FastAPI(
    title="Receipt OCR Pipeline API",
    description="High-performance pipeline API for vLLM Vision extraction",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(ocr_router, prefix="/api")


@app.get("/")
async def root():
    """Endpoint gốc."""
    return {
        "status": "ok",
        "message": "Receipt OCR API is running",
    }


@app.get("/health")
async def health_check():
    """
    Liveness: HTTP server đang chạy + vLLM còn alive (models.list()).
    KHÔNG phát hiện GPU hang — dùng /ready cho readiness thực sự.
    """
    vllm = await get_shared_vllm_client(
        base_url=config.vllm.base_url,
        model=config.vllm.model,
        api_key=config.vllm.api_key,
    )
    vllm_ok = await vllm.health_check()
    if not vllm_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "version": "2.1.0",
                "components": {"vllm": "unreachable"},
            },
        )
    return {
        "status": "healthy",
        "version": "2.1.0",
        "components": {"vllm": "healthy"},
    }


# Cache kết quả /ready để tránh spam GPU (gọi mỗi 30s từ Docker healthcheck
# × 2 service = 4 probe/phút, không cache → 4 GPU calls lãng phí).
_READY_CACHE_TTL = 5.0
_ready_cache = {"ok": False, "expires_at": 0.0}


@app.get("/ready")
async def ready_check():
    """
    Readiness: gửi request generate 1 token với ảnh 1×1 tới vLLM → xác nhận
    toàn bộ stack GPU + vision encoder + LLM + detokenizer đang hoạt động.
    Kết quả cache 5s để không quá tải GPU.
    """
    now = time.monotonic()
    if now < _ready_cache["expires_at"]:
        if _ready_cache["ok"]:
            return {"status": "ready", "cached": True}
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "cached": True},
        )

    vllm = await get_shared_vllm_client(
        base_url=config.vllm.base_url,
        model=config.vllm.model,
        api_key=config.vllm.api_key,
    )
    ok = await vllm.ready_check(timeout=10.0)
    _ready_cache["ok"] = ok
    _ready_cache["expires_at"] = now + _READY_CACHE_TTL

    if not ok:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "cached": False},
        )
    return {"status": "ready", "cached": False}
