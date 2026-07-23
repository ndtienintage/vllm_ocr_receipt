"""
Pydantic models cho dữ liệu Receipt — schema validation + alias mapping.

Hai mục tiêu:
  1. Cung cấp guided_json schema cho vLLM (alias keys mn/ma/td/tt/it/... giảm
     ~66% token output vs full names).
  2. Validate + coerce kiểu cho output của LLM khi parse về dict.

`populate_by_name=True` → accept cả alias ngắn lẫn tên đầy đủ khi parse,
nên downstream consumer có thể truy cập field bằng tên đầy đủ (merchant_name,
transaction_date, ...).

Validators KHÔNG check nội dung (đó là việc của prompt + hallucination_detector):
  - String: trim whitespace, cắt max_length (defensive truncate, không raise),
    null hoá nếu rỗng sau strip.
  - Numeric: dùng utils.number_utils.coerce_numeric cho VN/EN format mix.
  - Date/time: verify ISO format, trả raw khi không match (debug-friendly).
"""

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.utils.logging_utils import get_logger
from src.utils.number_utils import coerce_numeric

logger = get_logger(__name__)


# Date normalize cho _validate_date. Inline (single-use). Separator nới [\s./\-]
# khớp DATE_DMY (regex_patterns) → tránh phân kỳ với fallback path.
#   _DMY_DATE_RE: day-first DD-MM-YYYY — format model output (chuẩn VN in trên bill).
#   _YMD_DATE_RE: year-first YYYY-MM-DD (ISO) — tương thích ngược nếu model còn trả ISO.
_DMY_DATE_RE = re.compile(r"^(\d{1,2})[\s./\-](\d{1,2})[\s./\-](\d{4})$")
_YMD_DATE_RE = re.compile(r"^(\d{4})[\s./\-](\d{1,2})[\s./\-](\d{1,2})$")

# Match HH:MM hoặc HH:MM:SS ở ĐẦU chuỗi — cho phép cắt suffix lạ. Vd model
# OCR "08:52  NV:283872" có thể gộp thành "08:52:NV:283872" (15 chars > max_length
# 10) → validation fail mất TOÀN BỘ extraction. Regex chỉ extract prefix HH:MM[:SS]
# rồi drop phần còn lại.
_HHMM_PREFIX_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?")


# ── Item Model ─────────────────────────────────────────────────────────────────

class ReceiptItem(BaseModel):
    """Một mặt hàng trên hóa đơn."""
    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(None, alias="n", max_length=250)
    quantity: float | None = Field(None, alias="qty")
    price: float | None = Field(None, alias="p")
    total: float | None = Field(None, alias="t")

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: Any) -> str | None:
        """Trim + null nếu rỗng + truncate 250 ký tự (defensive vs LLM loop)."""
        if v is None:
            return None
        s = str(v).strip()
        return s[:250] if s else None

    @field_validator("quantity", mode="before")
    @classmethod
    def _coerce_quantity(cls, v: Any) -> float | None:
        """Ép float. Không reject qty âm/0 — receipt dịch vụ/voucher có thể
        emit qty=0 hoặc qty âm hợp lệ (refund row); để phía dưới (postproc/
        consumer) quyết định ý nghĩa thay vì null hoá tại schema."""
        return coerce_numeric(v, "quantity")

    @field_validator("price", mode="before")
    @classmethod
    def _coerce_price(cls, v: Any) -> float | None:
        """Ép float. Giữ âm cho dòng discount/refund."""
        return coerce_numeric(v, "price")

    @field_validator("total", mode="before")
    @classmethod
    def _coerce_total(cls, v: Any) -> float | None:
        """Ép float. Giữ âm cho dòng discount/refund."""
        return coerce_numeric(v, "total")


# ── Receipt Model ──────────────────────────────────────────────────────────────

class Receipt(BaseModel):
    """
    Dữ liệu hóa đơn hoàn chỉnh được trích xuất bởi OCR.

    Chỉ giữ các trường trọng tâm: merchant (name/address), date/time, items
    (name/qty/price/total) và `total_amount` (số tiền khách trả cuối cùng).
    Các trường phụ (subtotal, currency, payment_method, receipt_code) đã được
    loại bỏ ở cấp project để dồn attention của VLM vào trường quan trọng.
    """
    model_config = ConfigDict(populate_by_name=True)
    # Items KHAI BÁO TRƯỚC mọi field khác — guided_json (xgrammar) emit theo
    # đúng thứ tự field trong JSON Schema. Đặt `it` lên đầu để decoder lock
    # vào items trước, tránh rơi vào attractor null trên header/footer (vốn
    # đầy rule "null when X") rồi mới đến items.
    items: list[ReceiptItem] = Field(default_factory=list, alias="it", max_length=100)

    merchant_name: str | None = Field(None, alias="mn", max_length=250)
    merchant_address: str | None = Field(None, alias="ma", max_length=300)
    transaction_date: str | None = Field(None, alias="td", max_length=20)
    transaction_time: str | None = Field(None, alias="tt", max_length=10)

    @field_validator("items", mode="before")
    @classmethod
    def _coerce_items(cls, v: Any) -> list[Any]:
        """
        Lenient pre-validate cho items list — recover càng nhiều item càng tốt.

        Vấn đề: nếu CHỈ MỘT phần tử fail ReceiptItem schema (bare string, key
        lạ, type sai), toàn bộ Receipt validation fail → primary mất hết.
        Hàm này chạy mode="before" — chuyển từng phần tử về dạng dict hợp lệ
        TRƯỚC khi inner ReceiptItem validate, hoặc DROP riêng item lỗi (log).
        Không raise — luôn trả về list (có thể rỗng).

        Recovery rules:
          - None/missing → trả [] (cho LLM emit it=null thay vì it=[]).
          - Không phải list → wrap thành [] (drop input lạ).
          - Element là dict → pre-validate ReceiptItem, fail thì drop.
          - Element là str non-empty → wrap thành {"n": str}.
          - Element là None / kiểu khác → drop.
        """
        if v is None:
            return []
        if not isinstance(v, list):
            logger.warning("items: expected list, got %s — dropped", type(v).__name__)
            return []

        cleaned: list[Any] = []
        dropped = 0
        for idx, raw in enumerate(v):
            if raw is None:
                dropped += 1
                continue
            if isinstance(raw, str):
                s = raw.strip()
                if s:
                    cleaned.append({"n": s})
                else:
                    dropped += 1
                continue
            if isinstance(raw, dict):
                try:
                    ReceiptItem.model_validate(raw)
                    cleaned.append(raw)
                except ValidationError as e:
                    logger.warning(
                        "items[%d] dropped (schema fail): %s | raw=%r",
                        idx, e.errors(include_url=False), raw,
                    )
                    dropped += 1
                continue
            dropped += 1
            logger.warning(
                "items[%d] dropped (unsupported type %s): %r",
                idx, type(raw).__name__, raw,
            )

        if dropped:
            logger.warning(
                "items: %d dropped, %d kept (lenient recovery)",
                dropped, len(cleaned),
            )
        return cleaned

    total_amount: float | None = Field(None, alias="ta")

    # ── Validators: string fields ───────────────────────────────────────────────

    @field_validator("merchant_name", mode="before")
    @classmethod
    def _strip_merchant_name(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s[:250] if s else None

    @field_validator("merchant_address", mode="before")
    @classmethod
    def _strip_merchant_address(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s[:300] if s else None

    # ── Validators: numeric fields ──────────────────────────────────────────────

    @field_validator("total_amount", mode="before")
    @classmethod
    def _coerce_total_amount(cls, v: Any) -> float | None:
        """Ép float cho total_amount (VN/EN format mix)."""
        return coerce_numeric(v, "total_amount")

    # ── Validators: date/time fields ────────────────────────────────────────────

    @field_validator("transaction_date", mode="before")
    @classmethod
    def _validate_date(cls, v: Any) -> str | None:
        """Chuẩn hoá ngày giao dịch về ISO YYYY-MM-DD (output API).

        Prompt yêu cầu LLM output DD-MM-YYYY (day-first — đúng thứ tự in trên bill
        VN; model CHÉP, không tự đảo → bỏ lỗi swap ngày↔tháng). Validator đảo sang
        ISO bằng code (deterministic):
          - DD-MM-YYYY (separator -/./space): thử day-first; nếu tháng > 12 (model
            lỡ in MM-DD) → fallback đảo MM-DD.
          - YYYY-MM-DD (ISO): chuẩn hoá luôn (tương thích ngược nếu model còn trả ISO).
          - Không khớp / calendar invalid (vd 32-13-2026) → trả raw (debug-friendly).
          - None/rỗng → None.
        CHƯA xử lý (known-limit, giữ raw): tên tháng VN viết chữ ("tháng Năm"),
        năm 2 chữ số. Xem docs/field_extraction_rules.md §5.
        """
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        m = _YMD_DATE_RE.match(s)
        if m:
            try:
                return datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                ).strftime("%Y-%m-%d")
            except ValueError:
                return s
        m = _DMY_DATE_RE.match(s)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return datetime(y, mo, d).strftime("%Y-%m-%d")        # day-first (VN default)
            except ValueError:
                try:
                    return datetime(y, d, mo).strftime("%Y-%m-%d")    # fallback: model swapped → MM-DD
                except ValueError:
                    return s
        return s

    @field_validator("transaction_time", mode="before")
    @classmethod
    def _validate_time(cls, v: Any) -> str | None:
        """Extract HH:MM[:SS] prefix; cắt suffix lạ (vd model gộp "NV:283872"
        mã nhân viên vào sau time → "08:52:NV:283872" vượt max_length=10).
        Null khi rỗng. Range guard: HH≤23, MM≤59, SS≤59 (OCR rác như "25:99" → null).
        Format khác (HHhMM, "HH giờ MM") → giữ raw truncate 10 char (prompt đã
        hướng dẫn output HH:MM[:SS], đây là defensive cho edge case).
        """
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        m = _HHMM_PREFIX_RE.match(s)
        if not m:
            return s[:10]
        hh, mm = int(m.group(1)), int(m.group(2))
        ss_str = m.group(3)
        ss = int(ss_str) if ss_str else None
        if hh > 23 or mm > 59 or (ss is not None and ss > 59):
            return None
        if ss is not None:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        return f"{hh:02d}:{mm:02d}"
