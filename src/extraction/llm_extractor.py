"""
LLM Extractor Module — Gửi ảnh hóa đơn đến Qwen3-VL qua vLLM để trích xuất JSON.

Thiết kế prompt theo Qwen3-VL OCR cookbook + Alibaba DashScope pattern:
  - KHÔNG dùng system prompt (cookbook comment-out luôn) — toàn bộ chỉ thị
    nằm trong user turn, đứng TRƯỚC ảnh.
  - Inline JSON skeleton + alias keys (khớp Receipt schema, src/schemas/receipt.py).
  - Cô đọng: chỉ giữ rule có evidence rõ về tác động accuracy.

Block order (PRIMACY / RECENCY layout — chống "lost in the middle":
Liu et al. 2023 + Anthropic prompt-structure guide). LLM chú ý mạnh nhất ở
ĐẦU và CUỐI prompt, yếu ở GIỮA (RoPE distance-decay + attention-sink). Vì model
generate theo schema order (ly → it TRƯỚC, rồi mới header/footer scalars), nên
phần CUỐI prompt là "freshest" cho ly + items — đúng 2 field user hay sai nhất.
  PRIMACY (đầu — set frame):
    1. <task>           — scope + JSON-only contract
    2. <schema>         — JSON shape (`ly` trước `it` trước header/footer scalars)
                          + reading-order roadmap (work in order, nulls don't
                          cascade) làm orientation sớm.
                          KHÔNG có <context> năm hiện tại — date/time literal bị
                          cấm trong prompt (hard rule 2026-06-08: echo→hallucination)
  MIDDLE (vùng tra-cứu — de-emphasis ít hại nhất: toàn reject-list / format
  deterministic, được examples ở cuối nhắc lại):
    3. <merchant_rules> — `mn` / `ma`: header window, brand selection, reject list
    4. <datetime>       — `td` / `tt` / `ta` reject cues
    5. <numbers>        — VND thousand-separator stripping + decimal-qty exception
  RECENCY (cuối — freshest cho ly + items, là core reasoning + 2 pain-point):
    6. <classify>       — quyết định `ly` đầu tiên + ZONE-LOCAL nulling matrix
                          (COMPLETE / MISSING-HEADER / ITEM-ONLY / MISSING-FOOTER).
                          Null cục bộ trong zone bị cắt, KHÔNG cascade.
    7. <ecommerce>      — override layout app TMĐT (Shopee/Lazada/TikTok): shop
                          name vị trí giữa màn hình, ma=địa chỉ KHÁCH→null, giá
                          gạch=giá gốc bỏ, p/t app-card
    8. <items>          — body anchor: row definition, when-in-doubt emit, BLUR
                          SENTINEL ("ITEM BLUR" marker, dropped in postprocessor),
                          ATTRIBUTE MERGE (ONLY consumption attrs — topping / size /
                          sugar / ice — folded into `n`; note / order-type ignored),
                          drop-hard (promo / discount / gift), structural footer,
                          loop bailout (items + string)
    9. <examples>       — 5 layout demos (Hyperscience plateau: 3-5 là sweet spot
                          cho instruction-tuned VLM; >5 hits attention dilution).
                          Item-centric → đặt cuối, "best example last"
   10. <output>         — type strictness + no-fence reminder (truly last)

Anti-all-null tuning:
  - reading-order roadmap (trong <schema>) + <classify> đóng "null là cause cục bộ"
  - WHEN-IN-DOUBT EMIT trong <items> bias emit khi tên row đọc được nhưng số mờ
  - ITEMS FLOOR cho ITEM-ONLY: `it` MUST NOT be empty

Path primary (VLM thuần). Khi hậu xử lý phát hiện hallucination, processing.py
sẽ thay bằng fallback text-only (paddle OCR → text_extractor.extract_receipt_text_only).

Phạm vi: hóa đơn Việt Nam (retail / F&B / khách sạn / parking / xăng dầu / dịch vụ).
"""

from typing import Any, Dict, Tuple

from src.core.config import config
from src.clients.vllm import (
    VLLMClient,
    get_shared_vllm_client,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── Prompt tuning for receipt extraction ──────────────────────────────────────
USER_PROMPT_TEMPLATE = """<task>
OCR extractor for Vietnamese/English receipts (retail, F&B, hotel, parking, gas, cinema, services, e-commerce app screenshots). Read everything printed; return ONE raw JSON object matching <schema> — no markdown fences, no prose. WYSIWYG: extract only what is printed; never infer, compute, or invent. A clean null is correct; a guessed character is wrong.
</task>

{reflow_hint}<schema>
{"ly":"COMPLETE"|"MISSING-HEADER"|"ITEM-ONLY"|"MISSING-FOOTER"|null,
 "it":[{"n":string|null,"qty":number|null,"p":number|null,"t":number|null}],
 "mn":string|null,"ma":string|null,"td":"DD-MM-YYYY"|null,"tt":"HH:MM[:SS]"|null,
 "ta":number|null}
Emit ONLY these keys (no subtotal/tax/currency/payment/receipt-code field). Work in order: ly (which zones are present) → it (your MAIN job: n=name, qty, p=unit price, t=line total) → scalars mn/ma/td/tt/ta. Each unreadable field is null on its own; nulls NEVER cascade across fields.
</schema>

<merchant_rules>
mn/ma come ONLY from the header (top zone up to the first item row / column header / total label); never from items body or footer. CROPPED-HEADER GUARD (applies even without ly): if the topmost readable line is already an item/promo/column-header with NO brand logo above → mn=null AND ma=null; never recover from a line BELOW items. A wrong value is worse than null.
mn: the storefront brand the customer recognizes (largest/boldest/centered proper noun or logo: "GO!", "WinMart", "Circle K", "Bách Hóa Xanh"). BRAND BEATS LEGAL ENTITY: if both a brand and a registered company (Cty CP / CÔNG TY CỔ PHẦN / TNHH / JSC) show, pick the brand; use the company only if no brand. Skip doc markers (HÓA ĐƠN, GTGT, PHIẾU TÍNH TIỀN, BILL, RECEIPT, TAX INVOICE, COPY). Concatenate adjacent lines only if both are one registered name ("CÔNG TY TNHH"+"THỰC PHẨM ABC"); never append slogan/address/branch/hotline/web. Payment/promo/voucher/loyalty/cashier/thank-you → null.
ma: full street address from the header, lines joined with ", ". Cues: đường/phố/số nhà/phường/quận/P./Q./TP./Tầng/Lô/ward/district/street-number. Reject (→null): promo/payment/loyalty/thank-you, MST, phone/web/email, cashier, branch-only, any item row.
</merchant_rules>

<datetime>
tt: Default is null. Time "HH:MM[:SS]"; with both check-in and check-out, prefer check-in.
td:(Most important) Default is null. Purchase/payment date, output "DD-MM-YYYY" — TRANSCRIBE printed digits in printed order (Vietnamese is day-first); do NOT reorder to year-first (downstream converts to ISO). When several dates appear pick the transaction date (cues: Ngày, Date, Ngày bán/thanh toán/lập). REQUIRES all three printed — day AND month AND year. Keep day and month in their printed positions, NEVER swap (textual "Ngày D tháng M" is explicit: ngày=day, tháng=month). If day OR month OR year is missing/unreadable, OR no date string is printed anywhere → td strictly null; NEVER hallucinate, infer, borrow, or append a missing component (incl. the current year). tt is NOT a date: never turn the clock into a date or reuse its digits as day/month/year. Reject future dates and OCR misreads.
ta: the final grand total actually paid — REQUIRES an explicit same-row label (Tổng cộng, Phải thanh toán, Tổng tiền thanh toán, Thành tiền, Total, or diacritic-free equivalents). NEVER sum it[].t or copy the largest number. Reject Tạm tính/Subtotal, Tiền khách đưa/Cash, Tiền thối/Change, Deposit, Cashback. No label → null.
</datetime>

<numbers>
VND: both '.' and ',' are thousand separators — strip, output an integer ("55.000"→55000, "1,250,000"→1250000), preserve the digit count. EXCEPT decimal-weight qty (measured goods "0,704" kg) keeps its decimal when the row has a qty column but no separate unit-price token.
</numbers>

<classify>
Decide ly from evidence at the IMAGE EDGES; a cropped zone nulls ONLY its own fields.
- COMPLETE: brand top, items mid, labeled totals bottom.
- MISSING-HEADER: topmost text is already item/promo/column-header/doc-type, no brand in the header window → mn=null AND ma=null (never substitute).
- ITEM-ONLY: tight items crop, no header AND no totals/payment → every field except it null; it MUST NOT be empty.
- MISSING-FOOTER: header+items but bottommost text is an item row, no totals/payment → ta=null (the totals zone is cut); a date/time printed in the visible header still counts for td/tt.
</classify>

<ecommerce>
APP ORDER SCREENS (Shopee/Lazada/TikTok Shop screenshots) — recognize by a phone status bar, a screen title ("Đã giao đơn hàng"/"Thông tin đơn hàng"/"Đơn hàng đã hoàn thành"/"Chi tiết đơn hàng"), a "Mall" badge, or action buttons ("Mua lại"/"Đánh giá"/"Yêu cầu hoàn tiền"). These OVERRIDE the header rules:
- mn = the SHOP name beside the shop avatar (next to a ">" chevron and/or "Mall" badge, ABOVE the product card), even if not at image top. NEVER the screen title, recipient/buyer, status bar, or button. Keep any " - …" suffix; drop "Mall"/">".
- ma = null (the printed address is the CUSTOMER's shipping address: Người nhận/Người mua/Địa chỉ nhận hàng + masked phone).
- STRUCK-THROUGH price: use ONLY the real (bolder, non-struck) price for p/t; ignore the struck giá gốc. A store-wide banner ("Giảm 237.980đ tại cửa hàng này", "Mall Giảm …") is not an item — drop it.
- qty: the "×N" badge or a printed weight; MAY be null; never fold into p/t or multiply it in.
- p is ALMOST ALWAYS null on app cards: a "×N"/"Nx" badge + exactly ONE right-aligned amount → that amount is ALWAYS t and p=null, even if qty=1; never divide t by qty. Set p ONLY if a distinct per-unit price is separately printed.
- t = the rightmost amount on the SKU's product card; every item MUST have a t.
- Variant sub-lines ("Phân loại hàng", color/size/"1 đôi") describe the SKU → APPEND a distinguishing variant to n with " + "; drop a non-informative one ("Mặc định").
</ecommerce>

<items>
Items anchor the document. Scan line by line, top to bottom.
ROW: any line sharing the Y-axis of a qty/price/total token is an item row → its text is it[].n ONLY, never mn/ma.
WHEN-IN-DOUBT EMIT: readable name but missing/misaligned numbers → still emit {"n":"name","qty":null,"p":null,"t":null}. Missing numbers alone is not a reason to drop a readable row.
BLUR SENTINEL: a line that clearly IS an item row but too blurry/faded/smudged to read without guessing → emit {"n":"ITEM BLUR","qty":null,"p":null,"t":null}. NEVER guess characters or borrow a neighbouring name. ONLY for a genuine unreadable item row — NOT structural/footer lines (drop those), NOT a readable name with missing numbers (use WHEN-IN-DOUBT EMIT).
UNIT-WORD ≠ NAME: a line that is ONLY a unit word ± a count ("Gói", "Gói 2", "2 Hộp", "Kg", "x3 Lốc") is never a name — it is the unit/qty of the product named ABOVE; merge it upward (take its qty/p/t, keep the name above).
MULTI-ROW MERGE (name line + the numeric line below = ONE item):
- GO-UNIT: NAME line + "qty unit_word x price total" OR "unit_word qty price total" ("3 Hộp x 10.000 30.000", "Gói 2 9.800 19.600") → n=NAME line; drop the unit_word and its count; qty/p/t from the numeric line.
- BARCODE: "barcode name" + "[VAT%] qty price total" → strip barcode, n=name, qty/p/t from line 2. A line that begins VATper but ALSO carries qty+price+total IS this numeric line (merge it) — NOT a structural VAT-amount line.
- DISCOUNT-NAME: "name -discount" + "barcode qty price[-disc%] total" → n=name (drop -discount and -disc%), qty/p/t from line 2.
PRICE vs TOTAL: if an item row prints only ONE amount, it is t, not p. NEVER infer p by dividing t/qty; p is null unless a separate per-unit price is printed.
ATTRIBUTE MERGE: a sub-line that is a genuine consumption attribute of the item above (topping/size/sugar/ice) → APPEND its words to the previous n with " + " in printed order; strip any add-on price and trailing ids; the parent keeps its own p/t (do NOT sum the add-on). DO NOT MERGE (metadata — ignore, never an item, never into any n): notes (Ghi chú/Note/Yêu cầu/Lưu ý/Lời nhắn), order-type tags. Orphan sub-line (no name above) → drop.
DROP HARD (never an item, never merged, regardless of recurrence, notation @unit/#unit, or diacritics):
- discount/promo: KHUYẾN MÃI/KM, CK THẺ …%, DISC%/DISCOUNT, Voucher, Giảm giá, or any per-item NEGATIVE amount (e.g. "@55.000 -10.000"). May recur 0–3× per parent — drop each.
- gift/giveaway: Tặng/Tặng kèm/Quà tặng/Kèm theo/Đi kèm, "mua…tặng…", any free row (amount 0 / miễn phí).
- INLINE discount on a single-row item ("Bánh 100.000 -10.000 90.000"): keep p=original, t=final, drop only the -discount token; do NOT split.
STRUCTURAL (never an item; feed footer or ignore): column headers, barcodes, Phí phục vụ/service fee, standalone VAT-amount lines, Giá gốc (original-price metadata), expiry/mfg dates (HSD/NSX/EXP/MFG/Hạn sử dụng/Date), subtotal/total, payment, loyalty/points, hotline/web/thank-you. An expiry line printed BETWEEN a name and its price does NOT break the item — drop it, still merge the name with the price.
LOOP BAILOUT: if the SAME n repeats on ≥4 consecutive rows all with null/0 t, you are looping — STOP and close it after the last row with a real number. EXCEPTION: "ITEM BLUR" is a deliberate marker, keep each one. Also close a string field with null if a 6–10 char fragment repeats 3× inside it. Emit only what is printed; never pad to fill the page.
</items>

<examples>
(`\\n` separates printed lines)
1. RETAIL — "091 VAT08 Bánh mì 3 4.800 14.400" → n="Bánh mì",qty=3,p=4800,t=14400 (drop STT + VAT08; qty + two numbers = p,t).
2. discount-cycle — "CỚT LET JF 300G\\n000011220688\\n1 21.375 VAT 5% 21.375\\nKHUYẾN MÃI #55.000 -10.000\\nCK THẺ AEON 5% #23.625 -2.250" → ONE item n="CỚT LET JF 300G",qty=1,p=21375,t=21375 (skip barcode, drop discount rows).
3. giá gốc — "10 B.gạo Nhật TCHT 180g\\n28.000 28.000\\n38.900" → n="B.gạo Nhật TCHT 180g",p=28000,t=28000 (equal pair = p,t; trailing larger = giá gốc, ignore).
4. expiry-mid — "Sữa chua Vinamilk\\nHSD: DD/MM/YYYY\\n2 7.000 14.000" → n="Sữa chua Vinamilk",qty=2,p=7000,t=14000 (drop middle expiry line, merge into ONE item).
5. VAT-line carries numbers — "8934...349 Sữa tươi 100per 4x180\\nVAT8% 6 367 2.200\\nGiá gốc 36.700" → n="Sữa tươi 100per 4x180",qty=6,p=367,t=2200 (VAT8per line = the numeric line; strip barcode + Giá gốc).
</examples>

<output>
Return ONE raw JSON object matching <schema> — JSON numbers, JSON null, no fences, no prose.
</output>
"""


# Hint chèn vào prompt KHI ảnh đã reflow (chia cột dọc ghép ngang qua vạch đen —
# xem src/preprocessing/stages/fit.py: compose_columns). Không reflow →
# placeholder thay bằng "" (prompt nguyên trạng).
# Đặt ngay sau <task> (primacy) để model nắm reading-mode TRƯỚC khi quét ảnh.
_REFLOW_HINT = """<layout>
This image is a tall receipt SPLIT into vertical columns separated by solid BLACK vertical bars. Read column by column: each column fully top-to-bottom, THEN the next column to its right. The black bars are separators, not content — never merge text across a bar into one line. Item order follows this column-by-column scan.
</layout>

"""


def _build_user_prompt(reflow_applied: bool = False) -> str:
    """Render prompt template (+ reflow hint nếu ảnh chia cột).

    Dùng replace thay vì str.format() vì template chứa nhiều literal `{`/`}`
    trong JSON schema và examples. KHÔNG inject năm/ngày hiện tại — date/time
    literal trong prompt bị cấm (hard rule 2026-06-08, echo→hallucination).
    """
    return USER_PROMPT_TEMPLATE.replace(
        "{reflow_hint}", _REFLOW_HINT if reflow_applied else ""
    )


async def _compute_max_tokens(
    *,
    vllm: VLLMClient,
    user_prompt: str,
) -> int:
    """
    Compute max_tokens cho primary vision path từ exact text token count.
    /tokenize fail → pass thẳng max_tokens cap (không ước tính).
    """
    cfg = config.vllm
    try:
        text_count = await vllm.count_text_tokens(user_prompt)
    except Exception:
        return cfg.max_tokens
    budget = cfg.max_model_len - text_count - cfg.context_safety_margin
    if budget < cfg.min_output_tokens:
        return cfg.min_output_tokens
    return min(cfg.max_tokens, budget)


async def extract_receipt_with_llm(
    images: list[bytes],
    *,
    ref: str = "N/A",
    reflow_applied: bool = False,
) -> Tuple[Dict[str, Any], str, int, int]:
    """
    Gửi ảnh hóa đơn đến Qwen3-VL qua vLLM.

    Trả (receipt_dict, finish_reason, prompt_tokens, completion_tokens).
    finish_reason = "length" → JSON bị cắt, caller trigger fallback PaddleOCR.

    Mọi exception (UpstreamServiceError, httpx error, ValidationError…) đều
    propagate — caller (processing.py) là nơi duy nhất quyết định fallback.
    asyncio.CancelledError cũng propagate.
    """
    if not images:
        raise ValueError(f"[ref={ref}] No images provided to LLM")

    vllm = await get_shared_vllm_client(
        base_url=config.vllm.base_url,
        model=config.vllm.model,
        api_key=config.vllm.api_key,
    )

    user_prompt = _build_user_prompt(reflow_applied=reflow_applied)
    max_tokens = await _compute_max_tokens(vllm=vllm, user_prompt=user_prompt)

    return await vllm.extract_receipt(
        user_prompt=user_prompt,
        images=images,
        max_tokens=max_tokens,
        temperature=config.vllm.temperature,
        max_retries=1,
        ref=ref,
    )
