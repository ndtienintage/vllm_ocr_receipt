"""
Hậu xử lý (Post-Processing) — cosmetic cleanup tối thiểu cho output của LLM.

Triết lý: prompt là source-of-truth duy nhất cho rule trích xuất. Module này
KHÔNG check nội dung, KHÔNG re-compute giá trị, KHÔNG chống hallucination —
đó là việc của prompt + hallucination_detector.

Việc duy nhất ở đây:
  1. _apply_cjk_strip — gỡ ký tự Hán/Nhật/Hàn khỏi text fields (LLM thỉnh
     thoảng hallucinate sang script khác cho receipt VN).
  2. _filter_excluded_items / _filter_excluded_merchant — áp pattern do user
     cung cấp trong config/exclude_*.txt.
  3. validate_and_fix_items — strip trailing punctuation trên item name.

Nếu kết quả LLM sai mà KHÔNG khớp 3 mục trên, sửa prompt — KHÔNG thêm Python guard.
"""

from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional

from src.core.config import config
from src.utils.dict_utils import first_present
from src.utils.logging_utils import get_logger
from src.utils.regex_patterns import CJK_SCRIPT, TRAILING_PUNCT, WHITESPACE
from src.utils.text_utils import normalize_for_match

logger = get_logger(__name__)


def _strip_cjk(text: str) -> str:
    """Strip CJK chars khỏi text. Dồn whitespace dư sau khi xoá. Trả "" nếu
    kết quả rỗng (caller sẽ null hoá field tương ứng)."""
    if not text:
        return text
    cleaned = CJK_SCRIPT.sub("", text)
    cleaned = WHITESPACE.sub(" ", cleaned).strip()
    return cleaned


def _strip_trailing_punct(text: str) -> str:
    """Bỏ dấu câu thừa cuối chuỗi (., ; : và whitespace)."""
    return TRAILING_PUNCT.sub("", text) if text else text


# Sentinel mà extractor prompt emit cho 1 item-row có thật nhưng quá mờ/nhoè
# để đọc (xem BLUR SENTINEL trong llm_extractor + text_extractor prompt). Nó là
# marker chống-derail (giữ nhịp sinh, chặn đoán/mượn-tên), KHÔNG phải dữ liệu
# khách → drop trước khi trả output.
BLUR_SENTINEL_NAME = normalize_for_match("ITEM BLUR")


# ── Cosmetic cleanup ──────────────────────────────────────────────────────────


def _drop_blur_sentinel(data: Dict[str, Any]) -> Dict[str, Any]:
    """Xoá các item-row là BLUR SENTINEL ("ITEM BLUR"). Match chính xác (sau
    normalize) chứ không substring — đây là marker nội bộ, không null oan tên
    sản phẩm thật. Hỗ trợ cả key "items" lẫn alias "it"."""
    for items_key in ("items", "it"):
        items = data.get(items_key)
        if not isinstance(items, list) or not items:
            continue
        kept: List[Dict[str, Any]] = []
        removed = 0
        for item in items:
            if isinstance(item, dict) and (
                normalize_for_match(first_present(item, "name", "n"))
                == BLUR_SENTINEL_NAME
            ):
                removed += 1
                continue
            kept.append(item)
        if removed:
            logger.debug("Dropped %d BLUR SENTINEL item(s)", removed)
            data[items_key] = kept
    return data


def _apply_cjk_strip(data: Dict[str, Any]) -> Dict[str, Any]:
    """Strip CJK khỏi merchant_name, merchant_address, items[].name. In-place.
    Field thành rỗng sau strip → null hoá."""
    for key in ("merchant_name", "merchant_address", "mn", "ma"):
        if key in data and isinstance(data[key], str):
            cleaned = _strip_cjk(data[key])
            data[key] = cleaned if cleaned else None

    items = data.get("items") or data.get("it")
    if not isinstance(items, list):
        return data

    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("name", "n"):
            if key in item and isinstance(item[key], str):
                cleaned = _strip_cjk(item[key])
                item[key] = cleaned if cleaned else None

    return data


def validate_and_fix_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cosmetic per-item: strip trailing punctuation trên item.name.
    Giữ giá trị gốc nếu strip raise exception (defensive)."""
    fixed_items: List[Dict[str, Any]] = []
    for item in items:
        try:
            for key in ("name", "n"):
                if key in item and isinstance(item[key], str):
                    item[key] = _strip_trailing_punct(item[key]) or item[key]
            fixed_items.append(item)
        except Exception:
            fixed_items.append(item)
    return fixed_items


# ── User-controlled pattern filters ──────────────────────────────────────────


def _filter_excluded_merchant(
    data: Dict[str, Any],
    patterns: tuple[str, ...],
) -> Dict[str, Any]:
    """Null hoá merchant_name nếu CHỨA bất kỳ substring nào trong `patterns`.
    Chỉ null mn — các field khác (items/totals/date...) giữ nguyên kể cả khi
    merchant đọc sai. Pattern được normalize cùng cách với value để match
    bất chấp case/dấu/spacing."""
    if not patterns:
        return data

    normalized_patterns = tuple(p for p in (normalize_for_match(p) for p in patterns) if p)
    if not normalized_patterns:
        return data

    for key in ("merchant_name", "mn"):
        if key not in data:
            continue
        v = data[key]
        if not isinstance(v, str) or not v.strip():
            continue
        normalized = normalize_for_match(v)
        matched = next((p for p in normalized_patterns if p in normalized), None)
        if matched is not None:
            logger.warning(
                "Null merchant_name by EXCLUDE_MERCHANT_PATTERNS: %r matched %r",
                v, matched,
            )
            data[key] = None

    return data


def _parse_canonical_entry(line: str) -> Optional[tuple[str, str]]:
    """Parse 1 entry → (alias_normalized, canonical_original).

    Hỗ trợ 2 dạng:
      - Bare: "Bách Hóa Xanh"          → alias = canonical = "Bách Hóa Xanh"
      - Map:  "gia luon re hon => GO"  → alias = "gia luon re hon", canonical = "GO"

    Dạng MAP cho phép alias là SLOGAN/cách viết tắt, canonical là brand chính
    thức (vd slogan "Giá luôn rẻ hơn" → GO). Alias được normalize cho match;
    canonical giữ NGUYÊN VĂN để emit cho client.

    Trả None nếu entry rỗng / không parse được (caller skip entry đó).
    """
    if not line:
        return None
    if "=>" in line:
        alias_raw, _, canonical_raw = line.partition("=>")
        alias = alias_raw.strip()
        canonical = canonical_raw.strip()
        if not alias or not canonical:
            return None
    else:
        alias = canonical = line.strip()
        if not alias:
            return None
    alias_norm = normalize_for_match(alias)
    if not alias_norm:
        return None
    return alias_norm, canonical


def _canonicalize_merchant_name(
    data: Dict[str, Any],
    canonical_names: tuple[str, ...],
    min_ratio: float,
) -> Dict[str, Any]:
    """Remap merchant_name về canonical name nếu match đủ chặt.

    Mỗi entry trong `canonical_names` có thể là:
      - "Bách Hóa Xanh"           — bare: alias = canonical
      - "gia luon re hon => GO"   — map: alias (slogan/biến thể) → canonical

    Match logic (xem config/merchant_canonical_names.txt — đồng bộ):
      1. Normalize cả merchant_name lẫn alias: lowercase + strip diacritics
         + strip punctuation.
      2. Substring match (alias_normalized ⊂ merchant_normalized) → match.
         Bắt "Phieu thanh toan BACH HOA XANH" → "Bách Hóa Xanh", và
         "GIA LUON RE HON KHI MUA TAI..." → "GO".
      3. Else SequenceMatcher.ratio() ≥ min_ratio → match.
         Bắt typo 1-2 ký tự trên tên dài.
      4. Entry ĐẦU TIÊN match (theo thứ tự file) được chọn — viết alias
         cụ thể / canonical phổ biến nhất lên đầu file.
    """
    if not canonical_names or min_ratio <= 0:
        return data

    parsed: list[tuple[str, str]] = []
    for entry in canonical_names:
        pair = _parse_canonical_entry(entry)
        if pair is not None:
            parsed.append(pair)
    if not parsed:
        return data

    for key in ("merchant_name", "mn"):
        if key not in data:
            continue
        v = data[key]
        if not isinstance(v, str) or not v.strip():
            continue
        normalized_v = normalize_for_match(v)
        if not normalized_v:
            continue
        match: Optional[str] = None
        match_via = ""
        match_score = 0.0
        for alias_norm, canonical in parsed:
            if alias_norm in normalized_v:
                match = canonical
                match_via = "substring"
                match_score = 1.0
                break
        if match is None:
            for alias_norm, canonical in parsed:
                ratio = SequenceMatcher(None, normalized_v, alias_norm).ratio()
                if ratio >= min_ratio:
                    match = canonical
                    match_via = "ratio"
                    match_score = ratio
                    break
        if match is not None and match != v:
            logger.warning(
                "Canonicalize merchant_name: %r → %r (via %s, score=%.2f)",
                v, match, match_via, match_score,
            )
            data[key] = match

    return data


def _filter_excluded_items(
    data: Dict[str, Any],
    patterns: tuple[str, ...],
) -> Dict[str, Any]:
    """Xoá item có name chứa bất kỳ substring nào trong `patterns` (sau
    normalize). Dùng cho user-defined deny-list (footer noise, header text
    bị nhầm thành item, etc.)."""
    if not patterns:
        return data

    items = data.get("items")
    if not isinstance(items, list) or not items:
        return data

    normalized_patterns = tuple(p for p in (normalize_for_match(p) for p in patterns) if p)
    if not normalized_patterns:
        return data

    filtered_items: List[Dict[str, Any]] = []
    removed_count = 0
    for item in items:
        if not isinstance(item, dict):
            filtered_items.append(item)
            continue
        name = normalize_for_match(first_present(item, "name", "n"))
        if name and any(p in name for p in normalized_patterns):
            removed_count += 1
            continue
        filtered_items.append(item)

    if removed_count > 0:
        logger.debug(
            "Removed %d item(s) by EXCLUDE_ITEM_PATTERNS=%s",
            removed_count, list(normalized_patterns),
        )

    data["items"] = filtered_items
    return data


# ── Public entry ──────────────────────────────────────────────────────────────


def postprocess_receipt(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cosmetic cleanup cho output của LLM. KHÔNG check nội dung.

    Apply theo thứ tự:
      1. CJK strip — chống LLM hallucinate ký tự Hán/Nhật/Hàn.
      1b. Drop BLUR SENTINEL — gỡ item-row "ITEM BLUR" (marker chống-derail của
         prompt cho row mờ), không phải dữ liệu khách.
      2. Exclude items — từ config/exclude_item_patterns.txt.
      3. Canonicalize merchant_name — remap về tên chuẩn (substring hoặc
         SequenceMatcher.ratio ≥ min_ratio). Chạy TRƯỚC exclude_merchant: tên
         thật chứa header generic (vd "PHIẾU THANH TOÁN BÁCH HÓA XANH") được
         remap về canonical ("Bách Hoá Xanh") TRƯỚC, nên exclude pattern
         "phieu thanh toan" không null oan. Placeholder thuần (không match
         canonical nào) đi qua nguyên vẹn rồi bị exclude null ở bước 4.
      4. Exclude merchant — null hoá merchant_name còn lại nếu match pattern.
      5. Trailing punct trên item.name.

    Mọi exception bên trong → log + return data như cũ (best-effort, không
    bao giờ raise lên caller).
    """
    if not data:
        return data

    try:
        data = _apply_cjk_strip(data)
        data = _drop_blur_sentinel(data)
        data = _filter_excluded_items(data, config.postprocess.exclude_item_patterns)
        data = _canonicalize_merchant_name(
            data,
            config.postprocess.merchant_canonical_names,
            config.postprocess.merchant_canonical_min_ratio,
        )
        data = _filter_excluded_merchant(data, config.postprocess.exclude_merchant_patterns)
        if "items" in data and isinstance(data["items"], list):
            data["items"] = validate_and_fix_items(data["items"])
        return data
    except Exception as e:
        logger.error("postprocess_receipt failed: %s", e)
        return data
