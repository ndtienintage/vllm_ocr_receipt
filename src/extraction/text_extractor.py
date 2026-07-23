"""
Text-Only Receipt Extractor — fallback path khi hallucination_detector flag
primary VLM result là tệ.

Pipeline:
  preprocessed image → paddle_text (PP-OCRv5 det+rec) → text+bbox lines
  → TEXT_ONLY_PROMPT (no image attached) → LLM map vào Receipt schema
  → dict.

Khác biệt với llm_extractor (vision):
  - KHÔNG gửi ảnh kèm prompt — LLM chỉ thấy OCR text + bbox.
  - Prompt có `<input>` block mô tả bbox format và `<final_emit>` recap policy.
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
# Minimal mapping prompt: PaddleOCR lines (x,y|text) → Receipt JSON. Chỉ giữ
# schema + input format + critical correctness rules (number format, items
# shape, zero-inference). Bỏ tất cả zone/completeness/header detail — Paddle
# đã làm phần lớn việc đọc, model chỉ cần map vào schema.

TEXT_ONLY_USER_PROMPT_TEMPLATE = """\
<role>
Map PaddleOCR lines from a Vietnamese / English receipt into ONE JSON object matching the schema. Output JSON only.
</role>

<input>
Each OCR line format: `x1,y1,x2,y2|text` 
- (x1, y1): Top-Left corner coordinates of the bounding box.
- (x2, y2): Bottom-Right corner coordinates of the bounding box.
Sorted globally top→bottom (y1 ascending), then left→right (x1 ascending).

CRITICAL COORDINATE LOGIC:
1. Same Row Check: Lines where the vertical ranges [y1, y2] overlap significantly (more than 50% of font height) belong to the SAME VISUAL ROW.
2. Text Wrapping Check: If a text block at [x1_a, y1_a, x2_a, y2_a] contains ONLY a product name, and the immediately following row at [x1_b, y1_b, x2_b, y2_b] has numbers (qty/price) aligned to the right side (large x2), they form a wrapped item row. Merge them.
3. Column Segregation: Treat distinct x-ranges [x1, x2] on the same vertical line as separate columns (e.g., SKU name on the left, total price on the far right).
</input>

<schema>
{
 "it":[{"n":string|null,"qty":number|null,"p":number|null,"t":number|null}],
 "mn":string|null,"ma":string|null,"td":"DD-MM-YYYY"|null,"tt":"HH:MM[:SS]"|null,
 "ta":number|null
}
Extract ONLY these keys. There is no subtotal, tax, currency, payment-method, or receipt/invoice-code field — never emit them.
</schema>

<extraction_discipline>
Paddle has ALREADY done the character-level OCR; your job is to MAP its lines into the schema, not to re-judge whether each character is "good enough". Be generous — extract every field whose source line is readable as printed.

1. PARTIAL EXTRACTION > BLANKET NULL. Null is per-FIELD, never per-receipt. If `mn` is unreadable but `it`/`ta`/`td` are clear, emit those clearly and only null `mn`. Returning an all-null receipt while ≥3 Paddle lines were provided is a FAILURE mode — do not do it.
2. WHEN-IN-DOUBT EMIT for items. A row with a readable product name but missing or misaligned numbers STILL gets emitted as `{"n":"name","qty":null,"p":null,"t":null}`. Missing numbers alone is NOT a reason to drop the row. This bias applies ONLY when the line reads as a genuine product name (see <valid_item_gate>) — never emit a price-only, code-only, promotion, column-header, unit-only, or footer/loyalty line as an item just to avoid an empty row. "Doubt about a noise line" resolves to DROP, not emit.
3. NULL ONLY THE UNREADABLE TOKEN, not the whole row. If the qty column is garbled but the name and total are clear, emit `{"n":"name","qty":null,"p":null,"t":<total>}`. Do not abort the row over one bad token.
4. MID-STREAM ABORT — narrow trigger only. Abort a row mid-decode ONLY when you catch yourself fabricating characters from pixels you cannot read (e.g. inventing digits to "complete" a price). A minor diacritic uncertainty, a missing accent, or a slight x-coordinate shift is NOT a trigger. BLUR SENTINEL (continue, never derail): when the NAME is genuinely unreadable garble BUT the row clearly IS one item — it carries a usable price / line-total number in the items body — emit `{"n":"ITEM BLUR","qty":<readable or null>,"p":<readable or null>,"t":<readable or null>}` and move on, instead of guessing the name or borrowing the row above. A garbled cluster with NO usable number is noise → DROP (rule 7 / gate), never "ITEM BLUR".
5. NEVER FABRICATE. Do not invent characters, digits, or merchant info that no OCR line supports. A clean per-field null beats invented content.
6. LOOP BAILOUT. If the same 6-10 character fragment ("00000", "XXXXX", "  ") repeats 3+ times inside one string field, stop, close that field with null, move on. If an item discount label (e.g. `KHUYẾN MÃI`) appears 2+ times in `it`, stop emitting items, close `it`, move to footer.
7. FRAGMENTED-OCR FLOOR (anti-flood). When the OCR is badly broken — most lines are stray letter clusters ("GOU", "OTEUTN", "S N", "H C DI A H"), lone numbers ("100", "61.5", "77.900"), or pieces that cannot be paired into a name+number item — DO NOT emit one item per fragment. A real receipt item almost always carries at least a `t` (line total). If you would produce many items where almost NONE has any of qty/p/t, that is a mapping failure: emit ONLY the few items you can read with a number or a clean product word, and leave the rest out. A short, mostly-numbered `it` (even empty `it`) beats a long list of numberless fragments — the latter is wrong output, not partial success. This does NOT override rule 2 for genuine name-only product rows on otherwise-clean OCR; it targets the garbled-dump case only.
8. NO PROSE, NO MARKDOWN FENCES. Return exactly one raw JSON object — no ```json fences, no commentary.
</extraction_discipline>

<zone_rules>
Judge which zones are present in the OCR line sequence; a missing zone nulls ONLY its own fields — never substitute from another zone:
1. MISSING HEADER: the very first lines already contain product codes, item prices, or column headers (e.g., "SL", "Đơn giá", "Thành tiền") and no merchant name/address exists at the top → set `mn = null`, `ma = null`.
2. ITEMS ONLY: the input contains exclusively item rows and price layouts — no merchant brand at the top and no financial summary footer at the bottom → all fields except `it` MUST be `null`.
3. MISSING MIDDLE: valid merchant headers at the beginning and total/payment footers at the end, but the middle product segment is missing, skipped, or blank → set `it = []`.
4. MISSING FOOTER: valid headers and item rows, but the sequence ends abruptly on an item row with no total labels, payment methods, or dates following → set `ta=null`, `td=null`, `tt=null`.
5. COMPLETE: all 3 components are present (Header + Items + Footer) → extract every field normally.
</zone_rules>

<valid_item_gate>
GATE every object before placing it in `it`. This is the FMCG / SKU filter and it OVERRIDES "WHEN-IN-DOUBT EMIT": when the doubt is "is this a PRODUCT or noise?", the answer is DROP. A valid SKU line names a thing a shopper bought; description, price tokens, promotions, and store info are NOT products.
- NAME TEST — emit only if the line carries a product-name token: letters forming a real word, brand, or SKU description (VN / EN). A line that is ONLY digits, money, %, a date/time, a barcode / SKU / STT / MST / invoice code, a currency symbol, or a single bare keyword is NOT a name → DROP it (or MERGE it as the wrapped numeric half of the item directly above, per <items_extraction_rules> §2). NEVER let such a line become its own `n`.
- REAL-WORD-OR-NUMBER GATE (decisive test against meaningless rows): emit a row as an item ONLY if it satisfies AT LEAST ONE of: (a) the name contains ≥1 recognizable Vietnamese or English word — a token of ≥3 letters that forms or closely resembles a real word / brand / product term; OR (b) the row pairs a plausible name with a usable price or line-total number. A row that has NEITHER a real word NOR any usable number is OCR noise → DROP. When in doubt whether a cluster is a garbled real product or pure noise, and it has no number, DROP.
- GARBLED-FRAGMENT REJECT: a "word" that is just isolated/spaced single letters, a vowel-less or broken cluster with no recognizable VN/EN word, is OCR noise, NOT a product name → DROP. Real examples to reject: "S N", "G CA CO T", "H C DI A H", "GOU", "OTEUTN", "PM ISN Y SSE", "Y100", "YDCO", "H C D", "PH UN KI SASE", "GOILO ZMIENG", "WES BABY VG SO", "TG XMEN 2INI WOOD". Do not emit such a cluster just to preserve a row; a number-less unreadable item is better omitted than guessed. EXCEPTION: if the garbled name shares a row with a usable price / line-total, do not invent the name — emit it as the BLUR SENTINEL "ITEM BLUR" (rule 4), keeping the readable number.
- BODY TEST — the line must sit in the items body, between the header and the totals/footer zone. A name-like line below the totals (loyalty, thank-you, footer) is NOT an item.
A line that fails the gate is NEVER emitted as its OWN item. EXCEPTION: consumption-attribute sub-lines (topping / size / sugar / ice ONLY) do not pass the gate as standalone items but ARE folded into the parent item's `n` per <items_extraction_rules> rule 3 (ATTRIBUTE MERGE). Customer notes and order-type / dining tags are metadata — ignored, never merged. Only the classes below are truly discarded.

NOISE CLASSES — never their own product name:
1. PROMOTION / DISCOUNT — `KHUYẾN MÃI`, `CK THẺ`, `DISC NN%`, `Voucher`, `Giảm giá`, or any row whose number is negative. DROP entirely (do NOT merge — see rule 4).
2. STANDALONE NUMBER / CODE — a line with no product-name word: a lone price, barcode, SKU/STT code, MST, invoice number, %, or date.
3. COLUMN-HEADER / BARE UNIT / TAG METADATA — column headers (`SL`, `Đơn giá` / `ĐG`, `Thành tiền` / `TT`, `Mặt hàng`, `Mã hàng`, `Số lượng`, `Đơn vị`); a standalone tax tag (`VAT`, `VAT 5%`, `VAT8%`, `Thuế suất`); original-price metadata (`Giá gốc` / `Giá gôc` — the larger pre-promo price printed under a discounted item; never a product, never `p`/`t`, never `ta`); a bare unit word with no product attached (`Cái`, `Hộp`, `Kg`, `Lốc`, `Gói`, `Chai`, `Bịch`, `Túi`); store-info lines. A unit word or VAT tag sitting on a real product's numeric row (e.g. `1 Bộ x 79.000 79.000`, `1 21.375 VAT 5% 21.375`) is metadata — parse the numbers, DROP the unit/VAT token; never make it its own item and never put it in `n`. (A topping / size / sugar / ice option line is NOT noise — it folds into the parent per rule 3; a customer note or order-type tag is metadata — ignore it, do not merge.)
4. FOOTER / LOYALTY / PAYMENT — totals, VAT, payment lines, and loyalty: `Điểm tích lũy`, `Tích điểm`, `Điểm thưởng`, `Thành viên`, `Member`, `Point`, `Cashback`, `Tiền khách đưa`, `Tiền thối`. Structural footer — not items, not merged.
</valid_item_gate>

<items_extraction_rules>
1. PROCESS ORDER: extract `it` first to anchor the body data, then parse header and footer fields.
2. SAME-ROW MERGE: If a product name box and a price box share overlapping [y1, y2] zones, merge them into a single item object.
3. ATTRIBUTE MERGE (PRIORITY — fold sub-lines INTO the parent SKU name, do NOT drop): a line that hangs off the item above and is a GENUINE CONSUMPTION ATTRIBUTE of that product — topping, size, sugar level, or ice level — is NOT its own item. APPEND its descriptive text to the previous item's `n` with " + ", in printed top→bottom order. Strip trailing standalone numbers (ticket id, item count) and any add-on price; keep only words (parent keeps its own printed `p`/`t`, never sum). Cues (± diacritics): toppings `Topping`/`Thêm`/`Extra`/`Trân châu`/`Chân châu`/`Pudding`/`Thạch`/`Kem cheese`; size `Size`/`Up size`; sugar `Ít đường`/`Không đường`/`30% đường`/`50% đường`/`Bình thường`; ice `Ít đá`/`Nhiều đá`/`Không đá`/`Đá riêng`. DO NOT MERGE (NOT consumption attributes — ignore them, never an item, never appended to any `n`): customer notes `Ghi chú`/`Note`/`Yêu cầu`/`Lưu ý`/`Lời nhắn`, and order-type / dining tags `Mang đi`/`Mang ve`/`Tại quán`/`Take away`/`Dine in`/`Giao hàng`/`Ship`. Promotions / discounts / gift giveaways are handled by rule 4 DROP — never merge them into `n`. Example: parent "Trà sữa matcha" + "Ít đường" + "Trân châu trắng" + "Ít đá" → n="Trà sữa matcha + Ít đường + Trân châu trắng + Ít đá". An ORPHAN sub-line (no product above) → DROP.
4. DROP vs STRUCTURAL:
   - DROP — promotions, discounts, AND gift / bundle giveaways: `KHUYẾN MÃI`/`KM`, `CK THẺ`, `DISC NN%`/`DISCOUNT`, `Voucher`, `Tổng KM`, `Giảm giá`, or any line whose amount is a negative discount; AND gift / bundle rows `Tặng`/`Tặng kèm`/`Quà tặng`/`Kèm theo`/`Đi kèm`, "mua … tặng …" bundles, or any free-gift line (amount 0 / "miễn phí") — e.g. "quà tặng: phiếu mua hàng 30.000đ mua sữa rửa mặt". Never an item, never merged into `n`, never kept as a price.
   - STRUCTURAL (NOT items, NOT merged into any name — they feed footer fields or are ignored): fees (`Phí ship`, `Phụ thu`, `Phí phục vụ`, `Tip`), VAT summary (`Thuế GTGT`, `Tổng VAT`), payment info (`Tiền mặt`, `Tiền thối`, `Chuyển khoản`), subtotal/total, footer metadata (Hotline, Website, `Cảm ơn quý khách`), loyalty/points. These are not item sub-lines, so do NOT append them to a name either.
5. AEON-STYLE DISCOUNT BLOCK (CRITICAL — drop, NEVER emit as item): rows whose text starts with `KHUYẾN MÃI @`, `CK THẺ ... @`, or `DISC NN% @` are per-item discount detail lines using `@<unit_price> -<discount_amount>` notation. They appear 1-3 times AFTER each item's amount row. Drop every occurrence; do NOT create new items, do NOT merge into the previous item's `n`. If the same name (e.g. "KHUYẾN MÃI") repeats across many OCR lines, that confirms the AEON discount-block pattern — drop them all.
6. BARCODE ROW: numeric-only OCR lines (10-13 digits, no letters) directly below a product name row are barcodes — skip them, do not include in `n`, do not emit as own item.
7. ROW SHAPE — map money columns by COUNT, never split one number into two:
   - name + ONE money value -> that value is `t` (line total); `p`=null. ("Phở bò 90.000" -> t=90000.)
   - name + qty + ONE money value -> parse qty; the single money is `t`, `p`=null. Do NOT split the one amount across p and t. ("Cà phê 2 45.000" -> qty=2, t=45000.)
   - name + qty + TWO money values -> first is `p` (unit price), second is `t` (line total). ("BÁNH MỲ FE'STA HOA CÚC 2 12.500 25.000" -> qty=2, p=12500, t=25000). Keep the FULL printed name prefix — do not truncate at apostrophes / abbreviations.
   - A bare money number sharing the same y-range as a name-only row to its LEFT belongs to that row as `t` (right-aligned total column).
   - STRIP a leading STT / line-number and any VAT tag (`VAT08`, `VAT 5%`) from the item row before reading name/numbers. ("091 VAT08 Bánh mì 3 4.800 14.400" -> n="Bánh mì", qty=3, p=4800, t=14400.)
   - INLINE discount on ONE item row (orig -disc final): `p`=orig, `t`=final, drop only the `-disc` token; never split into two items. ("Bánh 100.000 -10.000 90.000" -> p=100000, t=90000.)
8. WEIGHED-GOODS DECIMAL QTY (exception to the thousand-separator rule): when an item's qty token carries a decimal for measured goods (e.g. `0.704`, `0,144` kg), KEEP the decimal point in `qty` — do NOT strip it as a separator. The AEON measured-goods variant emits `<decimal_qty>  <unit_price>  VAT n%  <total>` (e.g. "0.704 51.776 VAT 5% 36.450" -> qty=0.704, p=51776, t=36450); the VAT tag is metadata, and the trailing `CK THẺ` / `DISC` discount rows are dropped per rule 5.
</items_extraction_rules>

<fields_and_formatting>
- mn: Storefront / registered brand at the visual TOP of the OCR sequence. Cue: usually the FIRST non-trivial text block whose `y1` is the smallest, often a proper noun. STRIP leading doc-type / copy markers: "HÓA ĐƠN", "HÓA ĐƠN GTGT", "HÓA ĐƠN BÁN HÀNG", "PHIẾU THANH TOÁN", "PHIẾU TÍNH TIỀN", "BILL", "RECEIPT", "TAX INVOICE", "LIÊN 1/2/3", "COPY", "BẢN SAO" — if the line contains ONLY one of these markers, `mn = null`. Concat 2 lines ONLY when both are part of the same registered name (e.g. "CÔNG TY TNHH" + "THỰC PHẨM ABC"). REJECT candidates that match: address (contains "đường" / "phố" / "số nhà" / "P." / "Q." / "TP." / "Tầng" / "Lô"); MST / tax ID ("MST", "Mã số thuế", "Tax ID"); branch metadata ("CN:", "Chi nhánh", "Cơ sở", "Store #"); cashier / employee ("Thu ngân", "Nhân viên", "Cashier", "NV:", "Phục vụ"); phone / hotline (starts with 0, +84, "Hotline", "Tel:"); website / email (.com / .vn / @); order / table ("Bàn", "Table", "Đơn", "Order #"); promotion / discount labels ("KHUYẾN MÃI" / "KM" / "CK THẺ" / "Voucher" / "Giảm giá"); footer-section labels ("Tổng" / "Tổng cộng" / "Tổng thanh toán" / "Thành tiền" / "Phương thức thanh toán" / "Tiền mặt" / "Tiền thuế" / "Số lượng mặt hàng"); the FIRST item-name row (price aligned right); slogans / taglines (mostly lowercase descriptive). NEVER extract an item name into `mn`. If no genuine brand survives at the top (header cut off), mn=null — never promote a promo / footer / column-header line into it.
- ma: full address; concat multi-line with ", ". Markers: đường, phố, phường, quận, TP., số nhà, tầng, lô. REJECT (output null, never substitute): promotion / footer-section labels ("KHUYẾN MÃI", "PHƯƠNG THỨC THANH TOÁN", "TỔNG THANH TOÁN", "TIỀN MẶT", "Số lượng mặt hàng"); MST / tax IDs; phone / website / email; cashier lines; any item row. A cut-off address is null — never fill it with a footer label.
- td: transaction date → output "DD-MM-YYYY" (day-first — COPY the printed order, do NOT reorder to year-first; a downstream step converts to ISO). REQUIRE a printed year: YYYY or YY. If only day+month is printed (no year), output null — never fill current year. If NO date string is printed at all, td=null — do NOT manufacture one. The time (HH:MM) is NOT a date: NEVER turn clock digits into a date. Vietnamese numeric dates are DAY-first DD/MM/YYYY → output DD-MM-YYYY in the same order. NEVER swap month↔day; if both day and month are <=12, keep the printed day-first order. Textual "Ngày D tháng M năm Y" is explicit (ngày=day, tháng=month). 2-digit printed year uses current year as anchor. Reject HSD / NSX / EXP / MFG expiry, manufacture, MST, unrelated codes.
- ta: The SINGLE grand total the customer must pay. REQUIRES AN EXPLICIT LABEL MATCH on the same row; OCR may drop diacritics ("TONG CONG", "TONG THANH TOAN") — still a valid match. PRIORITY when several labeled money lines coexist (choose the FIRST tier that appears; higher tier wins even if its number is smaller): TIER-1 = "Phải thanh toán" / "Tổng thanh toán" / "Tổng tiền thanh toán" / "Tổng cộng" / "Total"; TIER-2 = "Thành tiền" / "Tổng tiền" — use ONLY when no TIER-1 label exists (on most receipts "Thành tiền" is a line-item column header, not the grand total). TENDER FALLBACK: "Tiền mặt" / "Cash" / "Chuyển khoản" / "QR Code" / "VNPay" is the payment-method amount, NOT the total — use it for `ta` ONLY when NO TIER-1/TIER-2 total label appears anywhere on the receipt. A label+amount line feeds `ta` and is NEVER an item (e.g. "TONG CONG 2.408.260" → ta=2408260). NEVER sum `items[].t`, NEVER copy the largest number to force-fill. HARD REJECT (never `ta`, even though it carries a number): "Tạm tính" / "Subtotal" (pre-total), "Tiền khách đưa" (cash given), "Tiền thối" / "Tiền thừa" / "Change", "Số lượng mặt hàng" / "Tổng số lượng hàng" (item COUNT, not money), "Giảm giá" / "Khuyến mãi" / "Voucher" subtotals, "Điểm" / loyalty points. When the matched label and its amount are on the same row, take the amount sharing that row's y-range — do NOT grab a stray number from a neighboring column (e.g. "TONG CONG 49.000" with the real total "2.227.602" printed on the very next line: the 49.000 is a mis-aligned fragment, prefer the labeled-row amount and ignore lone column-bleed numbers). No label = `null`.
- Number Formatting (VND): '.' and ',' are ALWAYS thousand separators. Strip them and output an integer ("55.000" -> 55000). Keep the exact digit count before stripping. Never quote numeric outputs.
</fields_and_formatting>

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
