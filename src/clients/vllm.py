"""
Model Client cho vLLM (OpenAI-compatible).

Architecture:
- KHÔNG có timeout ở tầng này — request_timeout (asyncio.wait_for ở server.py)
  bao toàn bộ vòng đời, propagate qua CancelledError xuống đây.
- httpx Timeout(None) cho read/write — không bound, để tầng entry quyết định.
- Retry chỉ áp dụng cho lỗi JSON/format; timeout/network KHÔNG retry để cancel
  propagate sạch.
- Bỏ fallback_max_tokens — chỉ dùng MỘT ngưỡng max_tokens duy nhất; truncation chỉ log
  cảnh báo, không retry vì nới ngưỡng thường khuyếch đại hallucination.
- Probe (health/ready) GIỮ timeout riêng (5s/10s) vì là ops endpoint, không
  thuộc request path.
- Token budget: text tokens dùng POST /tokenize (exact, chat template applied),
  kết quả cache theo hash(prompt) — prompt template gần tĩnh nên hit rate ~100%.
  /tokenize fail → caller pass thẳng VLLM_MAX_TOKENS cap (không ước tính).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from json_repair import repair_json
from openai import AsyncOpenAI
from PIL import Image
from pydantic import ValidationError

from src.schemas.receipt import Receipt
from src.core.config import config as _app_config
from src.extraction.hallucination_detector import detect_streaming_hallu
from src.utils.errors import UpstreamServiceError
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    lines = [l for l in lines if not l.strip().startswith("```")]
    return "\n".join(lines).strip()


def _extract_json_substring(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()
    i = s.find("[")
    j = s.rfind("]")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()
    return s


@dataclass(frozen=True)
class VLLMCompletionResult:
    data: dict
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Raw output (đã strip code fences) — giữ lại để log preview khi parsed
    # dict all-null. Đây là path cần điều tra: JSON valid nhưng nội dung rỗng,
    # caller (extract_receipt) sẽ in preview kèm token counts để phân biệt
    # "model bỏ cuộc" vs "ảnh thực sự không có nội dung".
    raw_text: str = ""


class VLLMClient:
    """Wrapper vLLM AsyncOpenAI, không có timeout riêng (do tầng pipeline cắt)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url
        self.model = model
        self._http_client = http_client
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0)
        self._receipt_schema = Receipt.model_json_schema(by_alias=True)
        # Cache token counts per prompt hash — prompt template gần tĩnh, hit rate ~100%.
        self._token_count_cache: dict[int, int] = {}
        logger.info("VLLMClient ready | model=%s", model)

    async def close(self) -> None:
        if self._http_client is not None:
            return
        try:
            await self.client.close()
        except Exception:
            pass

    def _tokenize_url(self) -> str:
        """Derive POST /tokenize URL từ base_url (strip trailing /v1)."""
        url = self.base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        return f"{url}/tokenize"

    async def count_text_tokens(self, prompt: str) -> int:
        """
        Exact text token count qua POST /tokenize với messages format (chat
        template applied) — chính xác hơn heuristic cho English/VN mixed prompt.

        Cache kết quả theo hash(prompt): prompt template gần tĩnh (chỉ thay đổi
        năm + reflow hint) → hit rate xấp xỉ 100% sau lần đầu.

        Raise UpstreamServiceError khi /tokenize không khả dụng — caller quyết
        định hành vi (bỏ tính budget → pass VLLM_MAX_TOKENS cap thẳng).
        """
        key = hash(prompt)
        if key in self._token_count_cache:
            return self._token_count_cache[key]
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.post(
                    self._tokenize_url(),
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                count = int(resp.json()["count"])
            self._token_count_cache[key] = count
            return count
        except Exception as exc:
            raise UpstreamServiceError(f"/tokenize failed: {exc}") from exc

    async def _completion(
        self,
        *,
        messages: list,
        max_tokens: int,
        temperature: float,
        json_schema: dict,
        extra_body: dict,
        ref: str = "N/A",
        allow_streaming: bool = True,
    ) -> tuple[str, str, int, int, int]:
        """Dispatch streaming vs non-streaming dựa trên config.hallu_stream_enabled.

        Streaming path = online hallu detector chạy mỗi N chars, abort sớm khi
        decoder loop. Save 80-90% decode time cho hallu requests.
        Non-streaming = legacy behavior, dùng khi flag tắt hoặc cần debug.

        allow_streaming=False: hard override để FORCE blocking mode bất kể config.
        Dùng cho text-only fallback path — input là OCR text đã có cấu trúc,
        output JSON gần như deterministic; streaming detector dễ false-positive
        trên patterns lặp hợp lệ (multi-line header, watermark duplicate, item
        thật giống tên). Khi abort nhầm ở fallback, KHÔNG còn cấp recovery sau
        → mất hoàn toàn kết quả (chỉ còn fail_safe all-null) dù PaddleOCR đã
        tốn 2-5s GPU. Blocking ở đây an toàn hơn vì:
          - sampling deterministic (T=0, top_k=-1, top_p=1.0, rep_pen=1.0)
            khớp guided_json + structured OCR input → output gần như decodable
            ngay từ token đầu, gần như không có decoder loop.
          - max_tokens được _build_fitted_prompt cap động theo input.
          - request_timeout (300s) là bound cuối cùng.
        """
        if allow_streaming and _app_config.vllm.hallu_stream_enabled:
            return await self._completion_streaming(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                json_schema=json_schema,
                extra_body=extra_body,
                ref=ref,
            )
        return await self._completion_blocking(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_schema=json_schema,
            extra_body=extra_body,
        )

    async def _completion_blocking(
        self,
        *,
        messages: list,
        max_tokens: int,
        temperature: float,
        json_schema: dict,
        extra_body: dict,
    ) -> tuple[str, str, int, int, int]:
        merged_extra = {
            **extra_body,
            "guided_json": json_schema,
            "guided_decoding_backend": "xgrammar",
        }
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=merged_extra,
            stream=False,
        )

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"
        usage = response.usage
        prompt_tokens = int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0
        completion_tokens = int(usage.completion_tokens) if usage and usage.completion_tokens else 0
        total_tokens = int(usage.total_tokens) if usage and usage.total_tokens else 0

        return content, finish_reason, prompt_tokens, completion_tokens, total_tokens

    async def _completion_streaming(
        self,
        *,
        messages: list,
        max_tokens: int,
        temperature: float,
        json_schema: dict,
        extra_body: dict,
        ref: str,
    ) -> tuple[str, str, int, int, int]:
        """
        Streaming completion + online hallu abort.

        Pattern:
          1. Open SSE stream tới vLLM (stream=True, include_usage=True).
          2. Accumulate chunks vào buffer.
          3. Mỗi hallu_stream_interval_chars chars → chạy detect_streaming_hallu.
          4. Khi flag → break loop, close stream (save remaining decode budget).
          5. Return với finish_reason="hallu_abort" để caller short-circuit
             parse step và trigger PaddleOCR fallback.

        Trên path "happy" (không hallu): chỉ chậm hơn non-streaming ~50-100ms
        do SSE overhead + check cost. Trên path hallu: tiết kiệm 60-300s/request.

        Edge cases:
          - Stream chết giữa chừng (network error): exception propagate.
          - Usage chunk có thể không xuất hiện (vLLM build cũ) → fallback tokens=0,
            log sẽ thấy tokens=p0/c0; không phá flow.
          - asyncio.CancelledError (request_timeout) propagate sạch — không nuốt.
        """
        merged_extra = {
            **extra_body,
            "guided_json": json_schema,
            "guided_decoding_backend": "xgrammar",
        }

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=merged_extra,
            stream=True,
            stream_options={"include_usage": True},
        )

        parts: List[str] = []
        # Track length riêng để TRÁNH O(n²) string concat — `total_text += chunk`
        # rebuild buffer mỗi chunk; với 200 chunks × 100 char = 200²/2 = 20K ops
        # cho buffer 20KB, nhưng với 24KB output thật sẽ thành ~5M ops. Dùng
        # list-append (O(1) amortized) + join chỉ khi cần detect.
        total_len = 0
        finish_reason = "stop"
        prompt_tokens = completion_tokens = total_tokens = 0
        interval = max(100, _app_config.vllm.hallu_stream_interval_chars)
        next_check_at = interval
        hallu_reason: Optional[str] = None

        try:
            async for chunk in stream:
                # Choices array có thể rỗng ở chunk usage cuối (include_usage=True).
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta and getattr(delta, "content", None):
                        parts.append(delta.content)
                        total_len += len(delta.content)

                        if total_len >= next_check_at:
                            # Join 1 lần / check (~50 lần cho output 24KB) thay vì
                            # rebuild mỗi chunk. Net cost: O(N × N/interval) thay
                            # cho O(N²) — với interval=500 là cải thiện 500×.
                            hallu_reason = detect_streaming_hallu("".join(parts))
                            if hallu_reason:
                                break
                            next_check_at = total_len + interval

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                if getattr(chunk, "usage", None):
                    prompt_tokens = int(chunk.usage.prompt_tokens or 0)
                    completion_tokens = int(chunk.usage.completion_tokens or 0)
                    total_tokens = int(chunk.usage.total_tokens or 0)
        finally:
            # Best-effort close — gửi cancel xuống vLLM để free GPU slot ngay.
            try:
                await stream.close()
            except Exception:
                pass

        content = "".join(parts)

        if hallu_reason:
            finish_reason = "hallu_abort"
            logger.warning(
                "[ref=%s] LLM stream ABORTED mid-decode | hallu=%s | "
                "accumulated=%d chars tokens=p%d/c%d | raw=%s",
                ref, hallu_reason, len(content),
                prompt_tokens, completion_tokens,
                content.replace("\n", "\\n"),
            )

        return content, finish_reason, prompt_tokens, completion_tokens, total_tokens

    async def chat_json_schema(
        self,
        *,
        user_prompt: str,
        images: Optional[List[bytes]] = None,
        json_schema: dict,
        max_tokens: int,
        temperature: float = 0.0,
        max_retries: int = 2,
        extra_body: Optional[dict] = None,
        ref: str = "N/A",
        allow_streaming: bool = True,
    ) -> VLLMCompletionResult:
        """
        Gọi vLLM. Không có timeout ở đây — tầng pipeline đã bọc asyncio.wait_for.
        Retry CHỈ áp dụng cho lỗi JSON/format, KHÔNG retry lỗi timeout/network
        để tầng trên quyết định. Truncation (finish_reason='length') chỉ log warning
        — không retry với ngưỡng cao hơn vì điều đó thường tạo thêm hallucination.

        Theo Qwen3-VL OCR cookbook: KHÔNG dùng system prompt — toàn bộ chỉ thị
        nằm trong user turn cùng ảnh.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if images:
            # Per-image mm_processor_kwargs (Qwen3-VL OCR cookbook). Set ở
            # top-level content dict (NGOÀI image_url) là cách vLLM honor
            # consistently — server-level --mm-processor-kwargs bị bỏ qua trên
            # nhiều version (vLLM issue #15364, #13143).
            mm_kwargs = {
                "min_pixels": _app_config.vllm.image_min_pixels,
                "max_pixels": _app_config.vllm.image_max_pixels,
            }
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    **mm_kwargs,
                })

        messages: list[dict] = [{"role": "user", "content": content}]

        for attempt in range(1, max_retries + 1):
            t_attempt = time.perf_counter()
            try:
                raw_text, finish_reason, prompt_tokens, completion_tokens, total_tokens = (
                    await self._completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        json_schema=json_schema,
                        extra_body=extra_body or {},
                        ref=ref,
                        allow_streaming=allow_streaming,
                    )
                )
                raw_text = _strip_code_fences(raw_text)

                # Hallu abort short-circuit: stream đã cắt giữa chừng vì phát hiện
                # decoder loop. Partial JSON thường chứa PREFIX hợp lệ (vài item
                # thật được emit trước khi rơi loop) — repair để salvage. Caller
                # (processing.py) sẽ dedup consecutive duplicates và áp salvage
                # guard. Repair fail → data={} (giữ behaviour cũ).
                # Bỏ qua mọi retry — hallu là property của input image + model
                # state, retry sẽ lặp lại.
                if finish_reason == "hallu_abort":
                    partial_data: dict = {}
                    if raw_text:
                        try:
                            repaired = repair_json(raw_text, return_objects=True)
                            if isinstance(repaired, dict):
                                partial_data = repaired
                            elif isinstance(repaired, str):
                                loaded = json.loads(repaired)
                                if isinstance(loaded, dict):
                                    partial_data = loaded
                        except Exception as exc:
                            logger.warning(
                                "[ref=%s] hallu_abort partial parse failed: %s",
                                ref, exc,
                            )
                    return VLLMCompletionResult(
                        data=partial_data,
                        finish_reason="hallu_abort",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        raw_text=raw_text,
                    )

                if not raw_text:
                    continue

                try:
                    parsed = json.loads(raw_text)
                except json.JSONDecodeError:
                    # JSON parse fail = anomaly → in luôn preview để debug
                    repair_preview = raw_text[:300].replace("\n", " ")
                    try:
                        parsed = json.loads(_extract_json_substring(raw_text))
                        logger.warning(
                            "[ref=%s] LLM JSON recovered via substring extract | preview: %s",
                            ref, repair_preview,
                        )
                    except json.JSONDecodeError:
                        repaired = repair_json(raw_text, return_objects=True)
                        if isinstance(repaired, dict):
                            parsed = repaired
                        else:
                            parsed = json.loads(repaired) if isinstance(repaired, str) else {}
                        logger.warning(
                            "[ref=%s] LLM JSON recovered via json_repair | preview: %s",
                            ref, repair_preview,
                        )

                return VLLMCompletionResult(
                    data=parsed,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw_text=raw_text,
                )

            except json.JSONDecodeError as e:
                call_s = time.perf_counter() - t_attempt
                logger.warning(
                    "[ref=%s] LLM attempt %d/%d | %.2fs | invalid JSON: %s",
                    ref, attempt, max_retries, call_s, e,
                )
                if attempt >= max_retries:
                    raise UpstreamServiceError(f"vLLM invalid JSON after {max_retries} attempts: {e}") from e
                # Tiếp tục vòng lặp retry
            except Exception as e:
                msg = str(e)
                if "maximum context length" in msg or "context_length" in msg.lower():
                    n_imgs = sum(1 for c in content if c.get("type") == "image_url")
                    logger.error(
                        "[ref=%s] LLM context overflow | prompt_chars=%d images=%d "
                        "max_tokens=%d → %s",
                        ref, len(user_prompt), n_imgs, max_tokens, msg,
                    )
                raise
            except asyncio.CancelledError:
                # Do request_timeout (server.py wait_for) hoặc client hủy — không swallow, không retry
                call_s = time.perf_counter() - t_attempt
                logger.warning(
                    "[ref=%s] LLM attempt %d/%d | %.2fs | cancelled (upstream timeout/disconnect)",
                    ref, attempt, max_retries, call_s,
                )
                raise

        # Không nên đến đây
        raise UpstreamServiceError("vLLM call ended without returning a result")

    async def extract_receipt(
        self,
        *,
        user_prompt: str,
        images: Optional[List[bytes]] = None,
        max_tokens: int,
        temperature: float = 0.0,
        max_retries: int = 2,
        text_only: bool = False,
        ref: str = "N/A",
    ) -> tuple[dict, str, int, int]:
        # Sampling params theo Qwen3-VL OCR cookbook (deterministic extraction):

        extra_body: dict[str, Any] = {
            "top_p": 0.9,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "presence_penalty": 0.0,
        }

        # Streaming + online hallu detector CHỈ dùng cho vision primary.
        # text_only path: input đã structured (OCR text + bbox), output JSON
        # gần deterministic.
        result = await self.chat_json_schema(
            user_prompt=user_prompt,
            images=images,
            json_schema=self._receipt_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            extra_body=extra_body,
            ref=ref,
            allow_streaming=not text_only,
        )

        # Wrap validation để log đủ context tại NGUỒN trước khi re-raise.
        # Bubble lên llm_extractor.py mất result.data + result.raw_text — analyst
        # chỉ thấy "ValidationError: 1 validation error" mà không biết LLM trả gì.
        try:
            receipt = Receipt.model_validate(result.data)
        except ValidationError as e:
            mode = "TEXT" if text_only else "VISION"
            raw = result.raw_text or ""
            raw_preview = raw[:800].replace("\n", " ") if raw else "<EMPTY>"
            raw_more = "…" if len(raw) > 800 else ""
            # repr(parsed) chứa keys/types thực tế — phân biệt "field sai type"
            # vs "field thừa/thiếu" vs "items có element sai cấu trúc".
            parsed_preview = repr(result.data)[:600]
            parsed_more = "…" if len(repr(result.data)) > 600 else ""
            logger.error(
                "[ref=%s] LLM[%s] Receipt schema validation FAILED | "
                "errors=%r | finish=%s tokens=p%d/c%d imgs=%d | "
                "parsed[%d]: %s%s | raw[%d]: %s%s",
                ref, mode, e.errors(),
                result.finish_reason,
                result.prompt_tokens, result.completion_tokens,
                len(images) if images else 0,
                len(repr(result.data)), parsed_preview, parsed_more,
                len(raw), raw_preview, raw_more,
            )
            raise
        dumped = receipt.model_dump(by_alias=False)
        # Chỉ log khi LLM trả ALL-NULL — tín hiệu rõ ràng của fail-safe path.
        # Path bình thường giữ silent để không spam INFO khi tải cao.
        scalar_filled = sum(
            1 for k, v in dumped.items()
            if k != "items" and v is not None and (not isinstance(v, str) or v.strip())
        )
        n_items = len(dumped.get("items") or [])
        # text-only: items=0 = fail tầng mapping → log dù scalar có date
        # (date sweep hay là false-positive, che mất tín hiệu items-empty).
        # vision: giữ chặt (chỉ log khi ALL-NULL) để không spam hoá đơn hợp lệ
        # có scalar nhưng không line item.
        if n_items == 0 and (text_only or scalar_filled == 0):
            mode = "TEXT" if text_only else "VISION"
            img_count = len(images) if images else 0
            raw = result.raw_text or ""
            preview = raw[:400].replace("\n", " ") if raw else "<EMPTY>"
            more = "…" if len(raw) > 400 else ""
            logger.warning(
                "[ref=%s] LLM[%s] %s | scalars=%d items=0 finish=%s tokens=p%d/c%d imgs=%d | raw[%d]: %s%s",
                ref, mode,
                "ALL-NULL parsed" if scalar_filled == 0 else "EMPTY-ITEMS parsed",
                scalar_filled, result.finish_reason,
                result.prompt_tokens, result.completion_tokens, img_count,
                len(raw), preview, more,
            )
        return dumped, result.finish_reason, result.prompt_tokens, result.completion_tokens

    def fail_safe_receipt(self) -> dict:
        return Receipt(
            merchant_name=None,
            merchant_address=None,
            transaction_date=None,
            transaction_time=None,
            items=[],
            total_amount=None,
        ).model_dump(by_alias=False)

    async def health_check(self) -> bool:
        """Liveness check: HTTP server còn alive (không phát hiện GPU hang)."""
        try:
            models = await asyncio.wait_for(self.client.models.list(), timeout=5.0)
            return len(models.data) > 0
        except Exception:
            return False

    # Ảnh probe cho ready_check. Phải ≥ min_pixels * temporal_patch_size của
    # Qwen3-VL (mặc định 262144 * 2 = 524288 pixels) vì vLLM v0.19.x wrapper truyền
    # do_resize=False xuống image processor → ảnh nhỏ hơn không được upsample, gây
    # RuntimeError "shape ... invalid" ở patch reshape (grid_h/w = 0).
    # 768×768 = 589824 pixels — vượt ngưỡng an toàn, JPEG đặc-màu nén rất nhỏ.
    _READY_CHECK_IMG_B64: Optional[str] = None

    @classmethod
    def _get_ready_check_image_b64(cls) -> str:
        if cls._READY_CHECK_IMG_B64 is None:
            buf = io.BytesIO()
            Image.new("RGB", (768, 768), color=(0, 0, 0)).save(buf, format="JPEG", quality=20)
            cls._READY_CHECK_IMG_B64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return cls._READY_CHECK_IMG_B64

    async def ready_check(self, *, timeout: float = 10.0) -> bool:
        """
        Readiness check thực sự: gửi 1 request sinh 1 token với ảnh đặc-màu 768×768
        → xác nhận toàn bộ stack (HTTP → scheduler → GPU → vision encoder → LLM →
        detokenizer) đang hoạt động. Phát hiện CUDA hang mà health_check() không thấy.
        """
        try:
            img_b64 = self._get_ready_check_image_b64()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ok"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ]
            await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.0,
                    stream=False,
                ),
                timeout=timeout,
            )
            return True
        except Exception as e:
            logger.warning("ready_check failed: %s: %s", type(e).__name__, e)
            return False


_shared_httpx_client: Optional[httpx.AsyncClient] = None
_shared_vllm_client: Optional[VLLMClient] = None
_shared_lock = asyncio.Lock()


async def get_shared_httpx_client() -> httpx.AsyncClient:
    global _shared_httpx_client
    if _shared_httpx_client is not None:
        return _shared_httpx_client
    # KHÔNG có timeout — request_timeout (server.py wait_for) là lớp duy nhất
    # bao toàn bộ vòng đời, kể cả connect/read/write/pool.
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100, keepalive_expiry=120.0)
    _shared_httpx_client = httpx.AsyncClient(timeout=None, limits=limits, http2=False)
    return _shared_httpx_client


async def get_shared_vllm_client(
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
) -> VLLMClient:
    global _shared_vllm_client
    if _shared_vllm_client is not None:
        return _shared_vllm_client
    async with _shared_lock:
        if _shared_vllm_client is not None:
            return _shared_vllm_client
        http_client = await get_shared_httpx_client()
        _shared_vllm_client = VLLMClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            http_client=http_client,
        )
        return _shared_vllm_client


async def close_shared_clients() -> None:
    global _shared_vllm_client, _shared_httpx_client
    try:
        if _shared_vllm_client is not None:
            await _shared_vllm_client.client.close()
    finally:
        _shared_vllm_client = None
        if _shared_httpx_client is not None:
            await _shared_httpx_client.aclose()
            _shared_httpx_client = None
