"""
Hallucination Detector — narrow content checks cho VLM output.

Threshold tuning qua env (định nghĩa trong src/core/config.py::HalluConfig,
đọc 1 lần khi import — không hot-reload). Xem docstring HalluConfig để hiểu
chi tiết từng tham số + trade-off.

Khuyến nghị: KHÔNG thay đổi mặc định nếu không có dataset đo trước/sau. Defaults
hiện tại đã calibrate trên hoá đơn VN — siết quá → mất kết quả hợp lệ, lỏng quá
→ decoder loop chạy hết max_tokens trước khi abort.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.core.config import config as _app_config
from src.utils.dict_utils import first_present
from src.utils.regex_patterns import (
    JSON_ITEM_N_T,
    PUNCTUATION,
    WHITESPACE,
    char_run_pattern,
    ngram_repeat_pattern,
)
from src.utils.text_utils import strip_diacritics

__all__ = [
    "detect_hallucination",
    "detect_streaming_hallu",
    "scrub_hallu_fields",
    "dedup_consecutive_items",
]


# Resolve once at import — config dataclass đã validate floor (min_val) khi parse env.
_DUP_ITEM_RUN_MIN = _app_config.hallu.dup_item_run_min
_CYCLE_MAX_PERIOD = _app_config.hallu.cycle_max_period
_CYCLE_REPEATS_MIN = _app_config.hallu.cycle_repeats_min
_CHAR_RUN_MIN = _app_config.hallu.char_run_min
_NGRAM_MAX_LEN = _app_config.hallu.ngram_max_len
_NGRAM_REPEAT_MIN = _app_config.hallu.ngram_repeat_min

# Pre-compile config-driven patterns:
#   _CHAR_RUN_RE : 1 ký tự non-digit/non-ws lặp ≥ _CHAR_RUN_MIN lần.
#   _NGRAM_RES   : cụm 2..N ký tự lặp liên tiếp ≥ _NGRAM_REPEAT_MIN lần.
_CHAR_RUN_RE = char_run_pattern(_CHAR_RUN_MIN)
_NGRAM_RES: tuple = tuple(
    ngram_repeat_pattern(n, _NGRAM_REPEAT_MIN)
    for n in range(2, _NGRAM_MAX_LEN + 1)
)

# Captured n-gram chỉ chứa ký tự JSON-syntax + literal (null/true/false) →
# pattern structural noise, không phải decoder loop. Vd receipt thưa data có
# cluster `null,null,null,null,` khi nhiều field optional cùng null; hoặc
# items[] có chuỗi `},{"n":` lặp đặc dày. Skip để giảm false-positive trên
# bulk receipt (50-100 items) — vốn JSON output có tỷ lệ syntax cao.
_JSON_STRUCTURE_ONLY_CHARS = frozenset('{}[]":,nulltrefas \n')


def _is_json_structure_ngram(captured: str) -> bool:
    """N-gram chỉ chứa ký tự JSON syntax/literal → coi như structural noise."""
    return all(ch in _JSON_STRUCTURE_ONLY_CHARS for ch in captured)


def _normalize_name(value: Any) -> str:
    """Lower + strip diacritics + bỏ punctuation + gộp whitespace. Trả "" nếu rỗng.

    Khác với text_utils.normalize_for_match: hàm này còn STRIP PUNCTUATION
    (cần thiết cho duplicate detection — "Cà Phê Đen." vs "ca phe den" phải
    cùng key, không thể giữ dấu chấm). normalize_for_match giữ punct để
    user-pattern match chính xác từng ký tự đặc biệt.
    """
    if not value:
        return ""
    no_marks = strip_diacritics(str(value)).lower()
    no_punct = PUNCTUATION.sub(" ", no_marks)
    return WHITESPACE.sub(" ", no_punct).strip()


def _normalize_total(value: Any) -> str:
    """Normalize line-total `t` → canonical string cho key so sánh dup.

    None / "" / "null" → "" (sentinel "không có total").
    Parsable number → str(float(value)) → "25000" vs "25000.0" cùng key.
    Non-numeric → lowercase-strip raw string (an toàn cho edge case).
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "null":
        return ""
    try:
        return str(float(s))
    except (TypeError, ValueError):
        return s.lower()


def _normalize_item_keys(items: List[Any]) -> List[Tuple[str, str]]:
    """Trả list (name, total) tuple per item — key cho cycle/dup detection.

    Key gồm CẢ name + line-total `t`: hai item liên tiếp cùng tên nhưng
    giá khác (vd "Coca 330ml" t=10000 rồi t=15000 do mua nhiều loại unit
    khác giá) là HỢP LỆ, không phải decoder loop.

    Non-dict / tên rỗng → ("", "") reset marker. Item dict có name nhưng
    `t` null → (name, "") — vẫn match với item khác cùng name + cùng `t`
    rỗng, nhưng KHÔNG match với item có `t` cụ thể.
    """
    out: List[Tuple[str, str]] = []
    for it in items:
        if isinstance(it, dict):
            name = _normalize_name(first_present(it, "name", "n"))
            total = _normalize_total(it.get("t"))
            out.append((name, total))
        else:
            out.append(("", ""))
    return out


def _max_cyclic_run(
    keys: List[Tuple[str, str]],
    *,
    max_period: int,
    period1_min: int,
    cycle_min: int,
) -> Tuple[int, int, str]:
    """
    Trả (period, repeats, sample_name) của cyclic loop mạnh nhất trong sequence
    KEYS (name, total) tuple, hoặc (0, 0, "") nếu sạch.

    Cycle period=P nghĩa là keys[i] == keys[i-P] liên tục nhiều vị trí.
      - P=1: cùng-(tên,total)-liên-tiếp.
      - P≥2: multi-key cyclic loop như A,B,A,B (period 2) hoặc A,B,C,A,B,C
             (period 3). Pattern này là loại loop AEON receipt — model emit
             item → discount → discount lặp đi lặp lại với cùng tên/cùng số.
             period=1 detector mù với case này vì 3 tên khác nhau xen kẽ
             reset cur_key mỗi bước → cur_run mãi = 1.

    Key (name, total) thay vì chỉ name: tránh false-positive khi hoá đơn
    hợp lệ có nhiều dòng cùng tên nhưng giá khác (multi-unit / multi-tier
    pricing). Item có name="" → reset marker bất kể total (đồng bộ với
    rule cũ).

    Threshold tách riêng cho period=1 vs period≥2:
      - period1_min (default 10): bulk receipt hợp lệ có thể có 5-9 dòng cùng
        (tên,giá) — siêu thị in nhiều dòng cùng SKU cùng giá thật.
      - cycle_min (default 3): cyclic ≥2-period gần như không tồn tại hợp lệ ở
        repeats=3 (vd A,B,A,B,A,B = 6 items lặp đúng chuỗi rất khó tự nhiên).

    Cost O(N * max_period) — với N=200 items và period=5 là 1000 ops, không
    đáng kể so với decode budget.
    """
    if not keys:
        return 0, 0, ""
    n = len(keys)
    best_period = 0
    best_repeats = 0
    best_sample = ""
    for period in range(1, max_period + 1):
        threshold = period1_min if period == 1 else cycle_min
        if n < period * threshold:
            continue
        i = period
        while i < n:
            if not keys[i][0] or keys[i] != keys[i - period]:
                i += 1
                continue
            start = i
            while i < n and keys[i][0] and keys[i] == keys[i - period]:
                i += 1
            # matches = i - start vị trí khớp keys[j-period].
            # Tổng block-repeats = matches // period + 1 (cộng seed block đầu).
            repeats = (i - start) // period + 1
            if repeats < threshold or repeats <= best_repeats:
                continue
            if period > 1:
                seed = keys[start - period : start]
                if all(s == seed[0] for s in seed):
                    continue
            best_period = period
            best_repeats = repeats
            seed_key = keys[start - period]
            best_sample = seed_key[0] or keys[start][0] or ""
    return best_period, best_repeats, best_sample


def _find_char_loop(text: str) -> Optional[str]:
    """Trả substring khớp loop pattern (char-run hoặc n-gram); None nếu sạch."""
    if not text:
        return None
    m = _CHAR_RUN_RE.search(text)
    if m:
        return m.group(0)
    # n-gram: bỏ qua 2 nhóm match
    #   (a) thuần digit: phone/mã "0000000000" hoặc zero-run hợp lệ.
    #   (b) thuần JSON syntax + null/true/false literal: bulk receipt sinh
    #       cluster `null,null,...` hoặc `},{"n":` đặc dày là pattern bình
    #       thường, không phải decoder loop.
    for ngram_re in _NGRAM_RES:
        for m in ngram_re.finditer(text):
            captured = m.group(1)
            if not any(not ch.isdigit() for ch in captured):
                continue
            if _is_json_structure_ngram(captured):
                continue
            return m.group(0)
    return None


def _iter_text_fields(result: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """Yield (path, text) cho mọi field text cần check.

    Phạm vi: merchant_name, merchant_address, items[].name. Bỏ qua các field
    có format chặt (date/time/total_amount) — chúng quá ngắn để loop có ý nghĩa
    và thường được prompt ràng buộc rõ.
    """
    for key in ("merchant_name", "mn", "merchant_address", "ma"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            yield key, v
    items = result.get("items") or result.get("it") or []
    if not isinstance(items, list):
        return
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        for key in ("name", "n"):
            v = it.get(key)
            if isinstance(v, str) and v.strip():
                yield f"items[{i}].{key}", v


# ── Streaming detector (online, during decode) ───────────────────────────────
# JSON_ITEM_N_T (xem regex_patterns): extract (name, line-total) pair từ partial
# JSON. Key dup gồm cả `t` để không flag hoá đơn hợp lệ có nhiều dòng cùng tên
# nhưng giá khác.


def detect_streaming_hallu(text: str) -> Optional[str]:
    """
    Online hallu detector cho streaming raw text — chạy mid-generation để abort
    decode loop SỚM. Khác `detect_hallucination`: input là RAW text (partial
    JSON), không phải parsed dict.

    Trả reason string khi phát hiện pattern hallu, None nếu sạch. Caller (
    llm_client._completion_streaming) dùng signal này để close stream và
    save phần decode còn lại.

    3 check, theo thứ tự rẻ → đắt:
      1. Char run: 1 ký tự non-digit/non-ws lặp ≥ _CHAR_RUN_MIN. Bắt loop
         dạng "aaaaaaaa" hoặc "............".
      2. N-gram loop: cụm 2-4 chars lặp ≥ _NGRAM_REPEAT_MIN lần liên tiếp
         (loại bỏ digit-only để tránh false-positive trên số 0000000000).
      3. Cyclic item loop: regex extract (name, total) pairs, normalize key,
         scan period 1..K. Bắt cả case period=1 (cùng (tên,giá) N lần liên
         tiếp ≥ _DUP_ITEM_RUN_MIN) lẫn period≥2 (multi-key cycle như AEON
         pattern [item → KHUYẾN MÃI → CK THẺ] ≥ _CYCLE_REPEATS_MIN lần) mà
         detector period=1 cũ mù hoàn toàn. Item cùng tên nhưng GIÁ KHÁC
         → key khác → KHÔNG tính dup (hợp lệ).

    Cost: O(n) regex scan trên toàn text. Caller gọi mỗi ~500 chars để bound
    tổng overhead. Trên 4000-char accumulated buffer: ~5-10 ms/check.
    """
    if not text:
        return None

    # Check 1: Char run
    m = _CHAR_RUN_RE.search(text)
    if m:
        return f"char_run match={m.group(0)!r}"

    # Check 2: N-gram loop
    # Skip 2 nhóm để giảm false-positive trên bulk receipt:
    #   (a) digit-only (phone, mã "0000000000")
    #   (b) JSON-structure only (`null,null,...`, `},{"n":...` cluster)
    for ngram_re in _NGRAM_RES:
        for m in ngram_re.finditer(text):
            captured = m.group(1)
            if not any(not ch.isdigit() for ch in captured):
                continue
            if _is_json_structure_ngram(captured):
                continue
            return f"ngram_loop match={m.group(0)!r}"

    # Check 3: Cyclic item-(name,total) loop (period 1..K).
    # Window TAIL only — decoder loop chỉ form ở edge trailing của generation.
    # Window size đủ chứa pattern period * repeats lớn nhất, ×2 margin cho
    # "warmup" prefix nhiễu (vài item hợp lệ trước khi vào loop). Item streaming
    # chưa decode tới field `t` sẽ KHÔNG xuất hiện trong pairs — chấp nhận
    # delay vài item edge, đổi lại tránh flag sớm khi giá chưa được emit.
    raw_pairs = JSON_ITEM_N_T.findall(text)
    if raw_pairs:
        window_size = max(_DUP_ITEM_RUN_MIN, _CYCLE_MAX_PERIOD * _CYCLE_REPEATS_MIN) * 2
        recent = [
            (_normalize_name(name), _normalize_total(total))
            for (name, total) in raw_pairs[-window_size:]
        ]
        period, repeats, sample = _max_cyclic_run(
            recent,
            max_period=_CYCLE_MAX_PERIOD,
            period1_min=_DUP_ITEM_RUN_MIN,
            cycle_min=_CYCLE_REPEATS_MIN,
        )
        if period > 0:
            kind = "dup_items_run" if period == 1 else "cyclic_items"
            return f"{kind} period={period} repeats={repeats} name={sample!r}"

    return None


def detect_hallucination(result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Flag cyclic loop trong items[] — cả period=1 (cùng (tên,giá) ≥
    _DUP_ITEM_RUN_MIN lần liên tiếp) và period≥2 (multi-key cycle ≥
    _CYCLE_REPEATS_MIN lần). Item cùng tên nhưng giá khác → key khác → KHÔNG
    flag (hợp lệ). Char/n-gram loop trên field đơn lẻ → dùng scrub_hallu_fields.

    Trả (True, reason) khi flag, ngược lại (False, None).
    """
    if not isinstance(result, dict):
        return False, None

    items = result.get("items") or result.get("it") or []
    if isinstance(items, list) and items:
        keys = _normalize_item_keys(items)
        period, repeats, sample = _max_cyclic_run(
            keys,
            max_period=_CYCLE_MAX_PERIOD,
            period1_min=_DUP_ITEM_RUN_MIN,
            cycle_min=_CYCLE_REPEATS_MIN,
        )
        if period > 0:
            kind = "dup_items_run" if period == 1 else "cyclic_items"
            return True, f"{kind} period={period} repeats={repeats} name={sample!r}"

    return False, None


def dedup_consecutive_items(items: List[Any]) -> Tuple[List[Any], int]:
    """Collapse consecutive runs of items với cùng (name, total) key thành item đầu.

    Use case: VLM loop attractor sinh ra 50+ items giống hệt nhau liên tiếp
    (vd "BIA TSINGAGERLON CHI" × 75 với cùng `t`). Sau khi detect_hallucination
    flag, caller dùng hàm này để giữ lại 1 item đại diện cho mỗi run thay vì
    discard cả list.

    Key dup đồng bộ với detect_hallucination: (normalized name, normalized t).
      - Cùng tên + cùng `t` → loop artefact, drop bản sao.
      - Cùng tên nhưng `t` khác → 2 dòng HỢP LỆ (mua nhiều loại unit khác giá
        hoặc cashier nhập tay nhiều dòng), GIỮ NGUYÊN cả hai.

    Logic:
      - Bỏ qua item không phải dict, item có name rỗng → giữ nguyên trong output.
      - 2 item liên tiếp cùng key → drop item thứ 2+ trong run.
      - Run của 1 item duy nhất (không lặp) → vẫn giữ nguyên.

    Trả (deduped_items, n_dropped).
    """
    if not items:
        return items, 0
    out: List[Any] = []
    dropped = 0
    prev_key: Optional[Tuple[str, str]] = None
    for it in items:
        if not isinstance(it, dict):
            out.append(it)
            prev_key = None
            continue
        name = _normalize_name(first_present(it, "name", "n"))
        if not name:
            out.append(it)
            prev_key = None
            continue
        key = (name, _normalize_total(it.get("t")))
        if key == prev_key:
            dropped += 1
            continue
        out.append(it)
        prev_key = key
    return out, dropped


def scrub_hallu_fields(result: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Null-out các text field có char_run / n-gram loop, giữ phần còn lại.

    Trả (cleaned_result, nulled_paths). KHÔNG sửa result in-place — return
    shallow copy với field hỏng = None / item.name = None.

    Scope: merchant_name / merchant_address, items[].name.
    Item có name loop → set name=None nhưng GIỮ item nếu còn ít nhất 1 trong
    {qty, p, t} non-null; ngược lại drop hẳn item (item rỗng vô nghĩa).

    KHÔNG xử lý duplicate_items_run — đó là structural loop, caller vẫn fallback.
    """
    if not isinstance(result, dict):
        return result, []

    cleaned = dict(result)
    nulled: List[str] = []

    for key in ("merchant_name", "mn", "merchant_address", "ma"):
        v = cleaned.get(key)
        if isinstance(v, str) and v.strip() and _find_char_loop(v):
            cleaned[key] = None
            nulled.append(key)

    items_key = "items" if "items" in cleaned else "it" if "it" in cleaned else None
    if items_key:
        items = cleaned.get(items_key) or []
        if isinstance(items, list):
            new_items = []
            for i, it in enumerate(items):
                if not isinstance(it, dict):
                    new_items.append(it)
                    continue
                name_key = "name" if "name" in it else "n" if "n" in it else None
                if name_key is None:
                    new_items.append(it)
                    continue
                v = it.get(name_key)
                if isinstance(v, str) and v.strip() and _find_char_loop(v):
                    path = f"items[{i}].{name_key}"
                    has_numeric = any(it.get(k) is not None for k in ("qty", "p", "t"))
                    if has_numeric:
                        new_it = dict(it)
                        new_it[name_key] = None
                        new_items.append(new_it)
                        nulled.append(path)
                    else:
                        nulled.append(f"{path}(dropped)")
                else:
                    new_items.append(it)
            cleaned[items_key] = new_items

    return cleaned, nulled
