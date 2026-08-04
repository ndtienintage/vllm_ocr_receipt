"""
Text-Only Receipt Extractor — fallback path khi hallucination_detector flag
primary VLM result là tệ.

Pipeline:
  preprocessed image → paddle_text (PP-OCRv5 det+rec) → text+bbox lines
  → TEXT_ONLY_PROMPT (no image attached) → LLM map vào Receipt schema
  → dict.

Khác biệt với llm_extractor (vision):
  - KHÔNG gửi ảnh kèm prompt — LLM chỉ thấy OCR text + bbox.
  - Prompt có `<input>` block mô tả bbox format + logic same-row/wrap/column.
  - Giữ nguyên Receipt schema (alias keys) — postprocessor không cần đổi.

Cùng vLLM endpoint + guided JSON. Cùng sampling params với vision path.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import config
from src.clients.vllm import VLLMClient, get_shared_vllm_client
from src.extraction import paddle_text
from src.utils.errors import UpstreamServiceError
from src.utils.logging_utils import get_logger
from src.utils.regex_patterns import (
    DATE_COMPACT,
    DATE_DMY,
    DATE_ISO,
    DATE_LABEL,
    EXPIRY_MARKERS,
    TIME_HMS,
)
from src.utils.text_utils import strip_diacritics

logger = get_logger(__name__)

__all__ = ["extract_receipt_text_only"]


# ── Safety-net: regex sweep td/tt từ Paddle lines khi LLM trả null ───────────
# Chỉ chạy khi `transaction_date` / `transaction_time` ở response = None. Quét
# từ ĐÁY ảnh lên (footer cashier line là ưu tiên cao nhất), skip dòng chứa
# expiry marker (EXPIRY_MARKERS). Match tối đa 1 lần cho mỗi field — không
# tích lũy. Toàn bộ pattern (EXPIRY_MARKERS, DATE_ISO, DATE_DMY, DATE_COMPACT,
# TIME_HMS) định nghĩa ở src/utils/regex_patterns.py.


def _norm_for_sweep(s: str) -> str:
    """Lower + strip diacritics — chuẩn hoá Paddle text trước regex match.
    Wrap quanh text_utils.strip_diacritics + .lower() vì các regex pattern
    (DATE_*, EXPIRY_MARKERS) expect lowercase ASCII (vd "ngay", "hsd")."""
    return strip_diacritics(s or "").lower()


def _try_parse_date(s: str) -> Optional[str]:
    """Try ISO → D/M/Y → compact DDMMYYYY (chỉ khi có label đứng trước)."""
    current_year = datetime.now().year

    def _check(y: int, mo: int, d: int) -> Optional[str]:
        if y < 100:
            y += 2000
        if y < 2000 or y > current_year + 1:
            return None
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None

    norm = _norm_for_sweep(s)
    m = DATE_ISO.search(norm)
    if m:
        r = _check(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if r:
            return r
    m = DATE_DMY.search(norm)
    if m:
        r = _check(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if r:
            return r
    if DATE_LABEL.search(norm):
        m = DATE_COMPACT.search(norm)
        if m:
            r = _check(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if r:
                return r
    return None


def _try_parse_time(s: str) -> Optional[str]:
    """Try HH:MM[:SS] / HHhMM. Range guard giống validate_time."""
    norm = _norm_for_sweep(s)
    m = TIME_HMS.search(norm)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    ss = int(m.group(3)) if m.group(3) is not None else 0
    if hh > 23 or mm > 59 or ss > 59:
        return None
    if m.group(3) is not None:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{hh:02d}:{mm:02d}"


def _datetime_sweep(
    receipt: Dict[str, Any],
    lines: List[Dict[str, Any]],
    *,
    ref: str,
) -> Dict[str, Any]:
    """
    In-place sweep td/tt từ Paddle lines khi LLM bỏ sót.

    Logic:
      - Sort lines theo Y giảm dần (đáy → đỉnh) — footer cashier line có
        date/time ưu tiên hơn header.
      - Skip dòng chứa expiry marker (HSD/NSX/EXP/MFG/Hạn sử dụng).
      - Lấy match đầu tiên hợp lệ cho mỗi field còn null.
    """
    need_td = receipt.get("transaction_date") in (None, "")
    need_tt = receipt.get("transaction_time") in (None, "")
    if not need_td and not need_tt:
        return receipt

    def _y_center(ln: Dict[str, Any]) -> float:
        bbox = ln.get("bbox") or [0, 0, 0, 0]
        return (bbox[1] + bbox[3]) / 2.0

    sorted_lines = sorted(lines, key=_y_center, reverse=True)

    filled: list[str] = []
    for ln in sorted_lines:
        text = ln.get("text") or ""
        if not text.strip():
            continue
        norm = _norm_for_sweep(text)
        if EXPIRY_MARKERS.search(norm):
            continue

        if need_td:
            d = _try_parse_date(text)
            if d:
                receipt["transaction_date"] = d
                need_td = False
                filled.append(f"td={d}")
        if need_tt:
            t = _try_parse_time(text)
            if t:
                receipt["transaction_time"] = t
                need_tt = False
                filled.append(f"tt={t}")

        if not need_td and not need_tt:
            break

    return receipt


# ── Prompt cho text-only mapping ─────────────────────────────────────────────
# Minimal mapping prompt: PaddleOCR lines (bbox|text) → Receipt JSON. Chỉ giữ
# schema + input format + critical correctness rules (number format, items
# shape, zero-inference). Bỏ tất cả zone/completeness/header detail — Paddle
# đã làm phần lớn việc đọc, model chỉ cần map vào schema.
#
# NGÂN SÁCH TOKEN (đo bằng tokenizer Qwen3-VL thật, xem memory
# prompt-token-measurement): template 3951 tok; + <context> năm = 4001. Input
# cap _FALLBACK_MAX_INPUT_TOKENS=9500 → còn ~5500 tok cho OCR lines trước khi
# _build_fitted_prompt phải trim đuôi. Template càng nhỏ, càng nhiều dòng OCR
# lọt vào. Nén 2026-08-04: 4942→3951 (−20%) bằng cách gộp <valid_item_gate>
# noise classes với rule 4/5 trùng nội dung (mỗi danh sách chỉ còn 1 chỗ),
# rút gọn văn phong, cắt danh sách ví dụ garbled 13→6. KHÔNG cắt cue token:
# mọi từ khoá VN/EN trong backtick/nháy đều được giữ (check tự động 230 cue).
# Blocks: <role> <input> <schema> <layout> <item_gate> <item_rules> <fields>
# <output>. Cross-ref nội bộ dùng SỐ RULE của <item_rules> — đổi thứ tự rule
# phải sửa cả tham chiếu ("per rule 2/3/4/7").
#
# KHÔNG thêm block ví dụ bbox (memory text-only-prompt-rules-not-examples):
# dạy bằng RULE, không dạy bằng demo toạ độ — token để dành cho OCR text.
# KHÔNG đặt giá trị date/time cụ thể vào prompt (memory
# no-date-time-literals-in-prompt) — chỉ dùng placeholder định dạng.

TEXT_ONLY_USER_PROMPT_TEMPLATE = """\
<role>
Map PaddleOCR lines from a Vietnamese / English receipt into ONE JSON object matching the schema. Output JSON only.
</role>

<input>
Each OCR line: `x1,y1,x2,y2|text` — (x1,y1) top-left, (x2,y2) bottom-right of the box. Sorted top→bottom (y1 asc), then left→right (x1 asc).
- SAME ROW: boxes whose [y1,y2] ranges overlap more than 50% of font height are one visual row.
- WRAPPED ROW: a name-only box followed by a row of numbers (qty/price) right-aligned (large x2) is the numeric half of that name → merge them.
- COLUMNS: distinct x-ranges on one row are separate columns (name on the left, total on the far right).
</input>

<schema>
{
 "ly":"COMPLETE"|"MISSING-HEADER"|"ITEM-ONLY"|"MISSING-MIDDLE"|"MISSING-FOOTER"|null,
 "it":[{"n":string|null,"qty":number|null,"p":number|null,"t":number|null}],
 "mn":string|null,"ma":string|null,"td":"DD-MM-YYYY"|null,"tt":"HH:MM[:SS]"|null,
 "ta":number|null
}
Emit ONLY these keys. There is no subtotal, tax, currency, payment-method, or receipt/invoice-code field — never emit them. `ly` is an internal reasoning scaffold, not customer data.
</schema>

<layout>
Classify the OCR sequence in `ly`, then apply that state's enforcement:
1. MISSING-HEADER — the first lines already carry product codes, item prices, or column headers ("SL", "Đơn giá", "Thành tiền"); no merchant name/address at the top → mn=null, ma=null.
2. ITEM-ONLY — item rows and price layouts only: no merchant brand at the top AND no totals footer at the bottom → every field except `it` MUST be null.
3. MISSING-MIDDLE — merchant header and totals footer both present, but the product body is missing, skipped, or blank → it=[].
4. MISSING-FOOTER — header and item rows present, but the sequence ends abruptly on an item row; no total label, payment method, or date follows → ta=null, td=null, tt=null.
5. COMPLETE — header (merchant) + body (items) + footer (totals/date) all present.
</layout>

<item_gate>
GATE every object before putting it in `it`: a valid SKU line names a thing a shopper bought. When the doubt is "product or noise?", the answer is DROP — this overrides any emit-when-in-doubt instinct.
- EMIT only if the row carries a product-name token AND at least one of: a recognizable VN/EN word (≥3 letters forming or closely resembling a real word, brand, or product term), OR a usable unit price / line total. Neither a real word nor a number → OCR noise → DROP.
- NOT A NAME: a line that is only digits, money, %, a date/time, a barcode / SKU / STT / MST / invoice code, a currency symbol, or one bare keyword. NEVER let it become its own `n` — DROP it, or merge it as the wrapped numeric half of the item above (rule 2).
- GARBLED → DROP: isolated spaced single letters, or a vowel-less / broken cluster with no recognizable VN/EN word — e.g. "S N", "G CA CO T", "OTEUTN", "YDCO", "PH UN KI SASE", "TG XMEN 2INI WOOD". Never invent a name to preserve a row; a number-less unreadable item is better omitted than guessed. EXCEPTION: when such a name shares a row with a usable price / line total, do not invent the name — emit `n` as the literal sentinel "ITEM BLUR" and keep the readable number.
- BODY ONLY: the row must sit between the header and the totals/footer zone. A name-like line below the totals (loyalty, thank-you) is NOT an item.
Failing the gate = never its own item. Only exception: consumption-attribute sub-lines (topping / size / sugar / ice) fold into the parent name per rule 3.

NOISE — never a product name, never merged into one:
1. PROMOTION / DISCOUNT / GIFT — `KHUYẾN MÃI`/`KM`, `CK THẺ`, `DISC NN%`/`DISCOUNT`, `Voucher`, `Tổng KM`, `Giảm giá`, any row whose amount is a negative discount; gift and bundle rows `Tặng`/`Tặng kèm`/`Quà tặng`/`Kèm theo`/`Đi kèm`, "mua … tặng …" bundles, and free-gift lines (amount 0 / "miễn phí"). DROP entirely — never an item, never merged, never kept as a price.
2. STANDALONE NUMBER / CODE — a line with no product-name word: a lone price, barcode, SKU/STT code, MST, invoice number, %, or date.
3. COLUMN-HEADER / BARE UNIT / TAG — `SL`, `Đơn giá`/`ĐG`, `Thành tiền`/`TT`, `Mặt hàng`, `Mã hàng`, `Số lượng`, `Đơn vị`; a standalone tax tag (`VAT`, `VAT 5%`, `VAT8%`, `Thuế suất`); `Giá gốc`/`Giá gôc` (the larger pre-promo price under a discounted item — never a product, never `p`/`t`, never `ta`); a bare unit word with no product attached (`Cái`, `Hộp`, `Kg`, `Lốc`, `Gói`, `Chai`, `Bịch`, `Túi`); store-info lines. When a unit word or VAT tag sits on a real product's numeric row (`1 Bộ x 79.000 79.000`, `1 21.375 VAT 5% 21.375`), parse the numbers and DROP that token — never its own item, never in `n`.
4. STRUCTURAL FOOTER — subtotal/total rows, fees (`Phí ship`, `Phụ thu`, `Phí phục vụ`, `Tip`), VAT summary (`Thuế GTGT`, `Tổng VAT`), payment (`Tiền mặt`, `Tiền thối`, `Chuyển khoản`, `Tiền khách đưa`), loyalty (`Điểm tích lũy`, `Tích điểm`, `Điểm thưởng`, `Thành viên`, `Member`, `Point`, `Cashback`), and footer metadata (Hotline, Website, `Cảm ơn quý khách`). They feed footer fields or are ignored — not items, not item sub-lines, never appended to a name.
Customer notes (`Ghi chú`/`Note`/`Yêu cầu`/`Lưu ý`/`Lời nhắn`) and order-type / dining tags (`Mang đi`/`Mang ve`/`Tại quán`/`Take away`/`Dine in`/`Giao hàng`/`Ship`) are metadata — ignore them, never an item, never merged into any `n`.
</item_gate>

<item_rules>
1. ORDER: determine `ly` first, extract `it` next to anchor the body, then parse header and footer fields.
2. SAME-ROW MERGE: a product-name box and a price box with overlapping [y1,y2] zones → one item object.
3. ATTRIBUTE MERGE (PRIORITY — fold INTO the parent SKU name, do NOT drop): a sub-line hanging off the item above that is a GENUINE CONSUMPTION ATTRIBUTE of it — topping, size, sugar level, or ice level — is not its own item. APPEND its descriptive text to the previous item's `n` with " + " in printed top→bottom order; strip trailing standalone numbers (ticket id, item count) and any add-on price, keep only words; the parent keeps its own printed `p`/`t` and never sums. Cues (± diacritics): `Topping`/`Thêm`/`Extra`/`Trân châu`/`Chân châu`/`Pudding`/`Thạch`/`Kem cheese`; `Size`/`Up size`; `Ít đường`/`Không đường`/`30% đường`/`50% đường`/`Bình thường`; `Ít đá`/`Nhiều đá`/`Không đá`/`Đá riêng`. Example: parent "Trà sữa matcha" + "Ít đường" + "Trân châu trắng" + "Ít đá" → n="Trà sữa matcha + Ít đường + Trân châu trắng + Ít đá". An ORPHAN sub-line (no product above) → DROP. Notes, order-type tags, promotions and gifts are NOT attributes — see NOISE.
4. AEON-STYLE DISCOUNT BLOCK (CRITICAL): rows whose text starts with `KHUYẾN MÃI @`, `CK THẺ … @`, or `DISC NN% @` are per-item discount detail in `@<unit_price> -<discount_amount>` notation, appearing 1-3 times AFTER an item's amount row. Drop every occurrence — do NOT create items, do NOT merge into the previous `n`. The same name repeating across many OCR lines confirms this pattern; drop them all.
5. BARCODE ROW: numeric-only lines (10-13 digits, no letters) directly below a product-name row are barcodes — skip them, not part of `n`, not their own item.
6. ROW SHAPE — map money columns by COUNT, never split one number into two:
   - name + ONE money value → that value is `t` (line total); `p`=null. ("Phở bò 90.000" → t=90000)
   - name + qty + ONE money value → parse qty; the single money is `t`, `p`=null. Do NOT split one amount across p and t. ("Cà phê 2 45.000" → qty=2, t=45000)
   - name + qty + TWO money values → first is `p` (unit price), second is `t` (line total). ("BÁNH MỲ FE'STA HOA CÚC 2 12.500 25.000" → qty=2, p=12500, t=25000). Keep the FULL printed name prefix — do not truncate at apostrophes / abbreviations.
   - a bare money number sharing the y-range of a name-only row to its LEFT belongs to that row as `t` (right-aligned total column).
   - STRIP a leading STT / line number and any VAT tag (`VAT08`, `VAT 5%`) before reading name and numbers. ("091 VAT08 Bánh mì 3 4.800 14.400" → n="Bánh mì", qty=3, p=4800, t=14400)
   - INLINE discount on ONE row (orig -disc final): `p`=orig, `t`=final, drop only the `-disc` token; never split into two items. ("Bánh 100.000 -10.000 90.000" → p=100000, t=90000)
7. WEIGHED-GOODS DECIMAL QTY (exception to the thousand-separator rule): when an item's qty token carries a decimal for measured goods (`0.704`, `0,144` kg), KEEP the decimal point in `qty` — do NOT strip it as a separator. The AEON measured-goods variant prints `<decimal_qty>  <unit_price>  VAT n%  <total>` ("0.704 51.776 VAT 5% 36.450" → qty=0.704, p=51776, t=36450); the VAT tag is metadata and the trailing `CK THẺ` / `DISC` rows drop per rule 4.
</item_rules>

<fields>
- mn: storefront / registered brand at the visual TOP of the sequence — usually the first non-trivial block, smallest `y1`, often a proper noun. STRIP leading doc-type / copy markers ("HÓA ĐƠN", "HÓA ĐƠN GTGT", "HÓA ĐƠN BÁN HÀNG", "PHIẾU THANH TOÁN", "PHIẾU TÍNH TIỀN", "BILL", "RECEIPT", "TAX INVOICE", "LIÊN 1/2/3", "COPY", "BẢN SAO"); a line containing ONLY such a marker → mn=null. Concat 2 lines ONLY when both belong to one registered name ("CÔNG TY TNHH" + "THỰC PHẨM ABC"). REJECT: address ("đường", "phố", "số nhà", "P.", "Q.", "TP.", "Tầng", "Lô"); MST / "Mã số thuế" / "Tax ID"; branch ("CN:", "Chi nhánh", "Cơ sở", "Store #"); cashier ("Thu ngân", "Nhân viên", "Cashier", "NV:", "Phục vụ"); phone / hotline (leading 0, +84, "Hotline", "Tel:"); website / email (.com / .vn / @); order / table ("Bàn", "Table", "Đơn", "Order #"); promotion labels ("KHUYẾN MÃI", "KM", "CK THẺ", "Voucher", "Giảm giá"); footer labels ("Tổng", "Tổng cộng", "Tổng thanh toán", "Thành tiền", "Phương thức thanh toán", "Tiền mặt", "Tiền thuế", "Số lượng mặt hàng"); the FIRST item-name row (price aligned right); slogans / taglines. NEVER extract an item name into `mn`. No genuine brand at the top (header cut off) → mn=null; never promote a promo / footer / column-header line into it.
- ma: full address; concat multi-line with ", ". Markers: đường, phố, phường, quận, TP., số nhà, tầng, lô. REJECT (→ null, never substitute): promotion / footer-section labels ("KHUYẾN MÃI", "PHƯƠNG THỨC THANH TOÁN", "TỔNG THANH TOÁN", "TIỀN MẶT", "Số lượng mặt hàng"); MST / tax IDs; phone / website / email; cashier lines; any item row. A cut-off address is null — never fill it with a footer label.
- td: transaction date → output "DD-MM-YYYY", day-first: COPY the printed order, do NOT reorder to year-first (a downstream step converts to ISO). REQUIRE a printed year (YYYY or YY); day+month with no printed year → null, never fill the current year. NO date string printed at all → null; do NOT manufacture one. The time (HH:MM) is NOT a date — NEVER turn clock digits into a date. Vietnamese numeric dates are DAY-first DD/MM/YYYY → keep that order; NEVER swap month↔day, including when both are ≤12. Textual "Ngày D tháng M năm Y" is explicit (ngày=day, tháng=month). A printed 2-digit year is anchored on the current year. Reject HSD / NSX / EXP / MFG expiry or manufacture dates, MST, and unrelated codes.
- ta: the SINGLE grand total the customer must pay. REQUIRES an explicit label match on the same row; OCR may drop diacritics ("TONG CONG", "TONG THANH TOAN") — still a valid match. PRIORITY when several labeled money lines coexist (the FIRST tier that appears wins, even if its number is smaller): TIER-1 = "Phải thanh toán" / "Tổng thanh toán" / "Tổng tiền thanh toán" / "Tổng cộng" / "Total"; TIER-2 = "Thành tiền" / "Tổng tiền" — ONLY when no TIER-1 label exists (on most receipts "Thành tiền" is a line-item column header, not the grand total); TENDER FALLBACK = "Tiền mặt" / "Cash" / "Chuyển khoản" / "QR Code" / "VNPay" — the payment-method amount, use for `ta` ONLY when NO TIER-1/TIER-2 label appears anywhere. Take the amount sharing the matched label's y-range — do NOT grab a stray number from a neighbouring column or the next line. A label+amount row feeds `ta` and is NEVER an item. NEVER sum `it[].t`, never copy the largest number to force-fill. HARD REJECT (never `ta`, even though they carry a number): "Tạm tính" / "Subtotal" (pre-total), "Tiền khách đưa" (cash given), "Tiền thối" / "Tiền thừa" / "Change", "Số lượng mặt hàng" / "Tổng số lượng hàng" (item COUNT, not money), "Giảm giá" / "Khuyến mãi" / "Voucher" subtotals, "Điểm" / loyalty points. No label → null.
- Numbers (VND): '.' and ',' are ALWAYS thousand separators — strip them and output an integer ("55.000" → 55000), keeping the exact digit count. Only exception: decimal qty (rule 7). Never quote numeric output.
</fields>

<output>
ONE JSON object matching the schema EXACTLY. JSON only — no prose, no markdown fence, no trailing comma.
</output>
"""


def _current_year_context() -> str:
    """`<context>` block với năm hiện tại. Anchor cho VLM khi resolve năm 2 chữ
    số đã in (DD/MM/YY → 20YY). Tách helper để
    `_build_fitted_prompt` đếm token được chính xác."""
    return (
        "<context>\n"
        f"Current year is {datetime.now().year}. Use ONLY to expand a PRINTED "
        "2-digit year (DD/MM/YY -> 20YY). Never add a year when the printed "
        "date has only day and month.\n"
        "</context>\n\n"
    )


def _build_user_prompt(text_block: str) -> str:
    """Concatenate `<context>` (year) + base template + `<ocr_text>` block.
    KHÔNG dùng str.format() vì USER prompt chứa `{` `}` literal trong JSON schema."""
    return _current_year_context() + TEXT_ONLY_USER_PROMPT_TEMPLATE + "\n" + text_block


# Mỗi vòng giữ 90% lines — softer hơn 0.80 cũ vì đếm token EXACT qua /tokenize
# (không còn heuristic 0.5 tok/char overestimate ~20%). 8 vòng × 0.9^7 ≈ 48% còn
# lại — đủ cho OCR rác mà giữ được nhiều dòng items hơn.
# Cap bumped 8000→9500 sau khi VLLM_MAX_MODEL_LEN 10k→12284. Format bbox hiện
# emit đủ 4 toạ độ (x1,y1,x2,y2) — tốn thêm token/line vs variant x1,y1 nhưng
# cho LLM column-alignment qua x2 (right-edge của price/total). Receipt 60+
# items typical vẫn fit trong cap; pathological 200+ lines có thể rơi vào trim.
# Output budget = 12284 − 9500 − 128 = 2656 tokens — đủ cho JSON 60-80 items
# (~40 tok/item).
_TRIM_RATIO = 0.90
_MAX_FIT_ITERATIONS = 8
_FALLBACK_MAX_INPUT_TOKENS = 9500


async def _build_fitted_prompt(
    lines: List[Dict[str, Any]],
    *,
    vllm: VLLMClient,
    ref: str,
) -> tuple[str, int]:
    """
    Tính (user_prompt, max_tokens) sao cho
    input_tokens ≤ _FALLBACK_MAX_INPUT_TOKENS VÀ
    input_tokens + max_tokens + safety_margin ≤ max_model_len.

    Dùng vLLM `/tokenize` (chat template applied) để đo input chính xác — đồng
    nhất với primary path trong llm_extractor._compute_max_tokens, tránh
    overestimate ~20% của heuristic char-based với prompt VN-EN mix.
    Trim OCR tail-first khi vượt cap. count_text_tokens có cache nội bộ; template
    + năm hiện tại gần tĩnh nên iter sau chỉ trả thêm phần delta OCR.

    Raises UpstreamServiceError khi:
      - /tokenize fail (caller decide: return fail_safe).
      - template alone đã > input cap (không có chỗ cho bất kỳ OCR line nào).
      - không fit nổi cả với 1 OCR line.
      - không converge sau _MAX_FIT_ITERATIONS vòng.
    """
    cfg = config.vllm
    margin = cfg.context_safety_margin
    min_out = cfg.min_output_tokens
    cap = cfg.max_tokens
    max_model_len = cfg.max_model_len

    # Input cap = min(hard cap, ngân sách input lớn nhất mà context window cho phép).
    input_cap = min(_FALLBACK_MAX_INPUT_TOKENS, max_model_len - margin - min_out)
    if input_cap < min_out:
        raise UpstreamServiceError(
            f"Fallback input cap ({input_cap}) < min_output ({min_out}) — "
            f"max_model_len={max_model_len} quá nhỏ cho text-only path"
        )

    # Include context prefix khi đo template — context được prepend trong
    # _build_user_prompt nên budget phải tính cả phần này.
    template_tokens = await vllm.count_text_tokens(
        _current_year_context() + TEXT_ONLY_USER_PROMPT_TEMPLATE
    )
    if template_tokens > input_cap:
        raise UpstreamServiceError(
            f"Text-only prompt template ({template_tokens} tokens) > input cap "
            f"({input_cap}) — không còn chỗ cho OCR block"
        )

    current_lines = lines
    for it in range(1, _MAX_FIT_ITERATIONS + 1):
        block = paddle_text.format_text_block(current_lines)
        if not block:
            raise UpstreamServiceError("format_text_block produced empty block during fit")
        prompt = _build_user_prompt(block)
        input_tokens = await vllm.count_text_tokens(prompt)

        if input_tokens <= input_cap:
            budget = max_model_len - input_tokens - margin
            chosen = min(budget, cap)
            if it > 1 or len(current_lines) < len(lines):
                logger.warning(
                    "[ref=%s] context-fit converged | iter=%d input=%d/%d max_tokens=%d "
                    "lines=%d/%d (model=%d)",
                    ref, it, input_tokens, input_cap, chosen,
                    len(current_lines), len(lines), max_model_len,
                )
            return prompt, chosen

        if len(current_lines) <= 1:
            raise UpstreamServiceError(
                f"context overflow: input={input_tokens} tokens with 1 OCR line "
                f"exceeds input cap {input_cap}"
            )
        keep = max(1, int(len(current_lines) * _TRIM_RATIO))
        if keep == len(current_lines):
            keep -= 1
        logger.warning(
            "[ref=%s] context-fit iter=%d: input=%d > cap=%d, trim OCR %d→%d lines",
            ref, it, input_tokens, input_cap,
            len(current_lines), keep,
        )
        current_lines = current_lines[:keep]

    raise UpstreamServiceError(
        f"context-fit did not converge after {_MAX_FIT_ITERATIONS} iterations"
    )


async def extract_receipt_text_only(
    image_bytes: bytes,
    *,
    ref: str = "N/A",
) -> Tuple[Dict[str, Any], int, int]:
    """
    Fallback path: PaddleOCR full extract → text-only LLM mapping.

    Trả (receipt_dict, prompt_tokens, completion_tokens).
    Trả (fail_safe_receipt(), 0, 0) khi:
      - paddle_text disable/init-fail → 0 OCR lines
      - context-fit thất bại
      - LLM JSON invalid (sau retry trong chat_json_schema)
      - Bất kỳ UpstreamServiceError / unexpected exception
    CancelledError vẫn propagate (timeout bao ngoài).
    """
    vllm = await get_shared_vllm_client(
        base_url=config.vllm.base_url,
        model=config.vllm.model,
        api_key=config.vllm.api_key,
    )

    lines = await paddle_text.extract_text_lines_async(image_bytes, ref=ref)
    if not lines:
        logger.warning("[ref=%s] FALLBACK aborted: paddle_text returned 0 lines", ref)
        return vllm.fail_safe_receipt(), 0, 0

    try:
        user_prompt, max_tokens = await _build_fitted_prompt(lines, vllm=vllm, ref=ref)
    except UpstreamServiceError as e:
        logger.warning("[ref=%s] FALLBACK context-fit failed: %s", ref, e)
        return vllm.fail_safe_receipt(), 0, 0

    try:
        receipt, _, p_tok, c_tok = await vllm.extract_receipt(
            user_prompt=user_prompt,
            images=None,
            max_tokens=max_tokens,
            temperature=config.vllm.temperature,
            max_retries=1,
            text_only=True,
            ref=ref,
        )
        receipt = _datetime_sweep(receipt, lines, ref=ref)
        # Diagnostic: Paddle có ≥3 lines nhưng LLM map ra 0 items = fail tầng
        # mapping (OCR rác? prompt drop nhầm? sampling?). In preview OCR để soi.
        # KHÔNG ràng buộc scalar field null: date _datetime_sweep match nhầm có
        # thể che mất tín hiệu items-empty (chính tín hiệu này từng làm salvage
        # guard ở processing.py vứt nhầm primary). items rỗng mới là dữ liệu
        # chính của hoá đơn.
        # Cap 10 lines × 120 chars ≈ 1.2KB — đủ thấy cấu trúc, không spam log.
        if len(lines) >= 3 and not (receipt.get("items") or []):
            filled = [
                k for k in (
                    "merchant_name", "merchant_address", "transaction_date",
                    "transaction_time", "total_amount",
                )
                if receipt.get(k) not in (None, "")
            ]
            preview_lines = []
            for ln in lines[:10]:
                txt = (ln.get("text") or "").replace("\n", " ")[:120]
                bb = ln.get("bbox") or []
                preview_lines.append(f"{bb}|{txt}")
            logger.warning(
                "[ref=%s] FALLBACK LLM[TEXT] 0 items despite %d Paddle lines | "
                "scalars_filled=%s | preview (first 10):\n%s",
                ref, len(lines), filled or "none", "\n".join(preview_lines),
            )
        return receipt, p_tok, c_tok
    except asyncio.CancelledError:
        raise
    except UpstreamServiceError as e:
        logger.warning("[ref=%s] FALLBACK upstream error: %s", ref, e)
        return vllm.fail_safe_receipt(), 0, 0
    except Exception as e:
        logger.warning(
            "[ref=%s] FALLBACK failed: %s: %s",
            ref, type(e).__name__, e,
        )
        return vllm.fail_safe_receipt(), 0, 0
