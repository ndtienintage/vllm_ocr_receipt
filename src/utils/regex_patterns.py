"""
Regex patterns dùng chung trong pipeline OCR.

Tổ chức theo nhóm:
  - Text normalization: whitespace, punctuation, trailing punct.
  - Script filtering: CJK block strip cho receipt VN.
  - Receipt date/time parsing: ISO / DMY / compact DDMMYYYY / HH:MM[:SS].
  - Streaming JSON: extract item name từ partial JSON output.
  - Dynamic factories: pattern phụ thuộc runtime config (hallu detector).

Patterns phụ thuộc config (CHAR_RUN, NGRAM_REPEAT) được expose dưới dạng
factory function vì caller cần build với threshold lấy từ config.
"""

from __future__ import annotations

import re

__all__ = [
    # Static patterns
    "WHITESPACE",
    "PUNCTUATION",
    "TRAILING_PUNCT",
    "CJK_SCRIPT",
    "EXPIRY_MARKERS",
    "DATE_ISO",
    "DATE_DMY",
    "DATE_COMPACT",
    "DATE_LABEL",
    "TIME_HMS",
    "JSON_ITEM_N_T",
    # Dynamic factories
    "char_run_pattern",
    "ngram_repeat_pattern",
]


# ── Text normalization ────────────────────────────────────────────────────────

WHITESPACE = re.compile(r"\s+")
PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)
# Dấu kết câu + whitespace ở CUỐI chuỗi — dùng cho strip cosmetic.
TRAILING_PUNCT = re.compile(r"[.,;:\s]+$")


# ── Script filtering ──────────────────────────────────────────────────────────
# East Asian script blocks cần strip khỏi text field hoá đơn VN. Receipt VN
# chỉ dùng Latin + diacritics — bất kỳ ký tự CJK nào xuất hiện đều là LLM
# hallucinate. Các block:
#   U+3040..U+309F : Hiragana
#   U+30A0..U+30FF : Katakana
#   U+3400..U+4DBF : CJK Extension A
#   U+4E00..U+9FFF : CJK Unified Ideographs (chữ Hán phổ biến)
#   U+AC00..U+D7AF : Hangul (Korean)
#   U+F900..U+FAFF : CJK Compatibility Ideographs
#   U+FF00..U+FFEF : Halfwidth/Fullwidth Forms (gồm fullwidth ASCII + kana)
CJK_SCRIPT = re.compile(
    r"[぀-ゟ゠-ヿ㐀-䶿一-鿿"
    r"가-힯豈-﫿＀-￯]"
)


# ── Receipt date/time parsing (text_extractor safety-net sweep) ───────────────

# Markers loại trừ — dòng chứa các từ này KHÔNG phải transaction date/time.
EXPIRY_MARKERS = re.compile(
    r"\b(?:hsd|hsd:|nsx|nsx:|exp|exp:|mfg|mfg:|han\s*su\s*dung|expir|expiry)\b"
)
DATE_ISO = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
DATE_DMY = re.compile(r"(\d{1,2})[\s./\-](\d{1,2})[\s./\-](\d{2,4})")
# DDMMYYYY compact — chỉ tin khi có label "Ngày" / "Date" đứng trước (DATE_LABEL).
DATE_COMPACT = re.compile(r"\b(\d{2})(\d{2})(\d{4})\b")
# Label phải có TRƯỚC DATE_COMPACT để tin DDMMYYYY là transaction date.
# Norm-form expected (đã strip diacritics + lower) — match "ngay" / "date".
DATE_LABEL = re.compile(r"\bngay\b|\bdate\b")
TIME_HMS = re.compile(r"(\d{1,2})[:h](\d{2})(?::(\d{2}))?")


# ── Streaming JSON parsing (online hallu detector) ────────────────────────────
# Extract (name, line-total) pair từ partial JSON cho dup-detection key.
#
# Khoá dup phải xét CẢ tên + giá: hai item liên tiếp cùng tên nhưng `t` khác
# nhau (vd shopper mua 2 unit cùng SKU mà cashier nhập 2 dòng khác giá) là
# HỢP LỆ, không phải decoder loop.
#
# Span: từ `"n":"..."` → `"t":<num|null>`, ràng buộc `[^}]{0,200}?` để
#   - không vượt object boundary sang item kế tiếp,
#   - bound backtracking trên pathological input.
# Item streaming chưa decode đủ tới `t` sẽ KHÔNG match — chấp nhận delay
# vài item ở edge generation; bù lại loại được false-positive khi đoạn đầu
# emit name nhanh mà chưa kịp emit giá.
#
# Cap name 100 char (giữ nguyên như cũ) để chặn match toàn block khi escape
# lệch giữa decoder loop.
JSON_ITEM_N_T = re.compile(
    r'"(?:n|name)"\s*:\s*"([^"\n]{1,100})"'
    r'[^}]{0,200}?'
    r'"t"\s*:\s*(-?\d+(?:\.\d+)?|null)',
    re.UNICODE,
)


# ── Dynamic factories (config-driven) ─────────────────────────────────────────

def char_run_pattern(min_count: int) -> re.Pattern[str]:
    """
    Pattern bắt 1 ký tự non-digit/non-whitespace lặp ≥ min_count lần.

    Vd "AAAAAA", "......" — tín hiệu decoder stuck phun ký tự đơn.
    Loại digit để tránh false-positive trên số "0000000000" (phone, zero-run).
    """
    return re.compile(rf"([^\d\s])\1{{{min_count - 1},}}")


def ngram_repeat_pattern(ngram_len: int, min_repeats: int) -> re.Pattern[str]:
    """
    Pattern bắt cụm `ngram_len` ký tự bất kỳ lặp liên tiếp ≥ min_repeats lần.

    Vd "abcabc", "abcdabcdabcd" — tín hiệu decoder loop. Caller phải tự lọc
    match thuần digit (vd "0000000000" trong phone) để tránh false-positive.
    """
    return re.compile(rf"(.{{{ngram_len}}})\1{{{min_repeats - 1},}}")
