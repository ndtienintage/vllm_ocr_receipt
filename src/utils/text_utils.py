"""
Text normalization helpers cho Vietnamese OCR pipeline.

Cung cấp các primitive dùng chung giữa postprocessor + text_extractor +
hallucination_detector:
  - strip_diacritics: NFD decompose + drop combining marks; đ/Đ → d/D thủ công.
  - normalize_for_match: lower + strip diacritics + gộp whitespace.

Lưu ý semantic:
  - normalize_for_match KHÔNG strip punctuation (giữ ký tự đặc biệt để match
    chính xác user pattern). Module nào cần match-bất-chấp-punct phải tự strip
    punct sau khi gọi function này.
  - Caller nên truyền str — function defensive accept Any để dễ dùng với
    output LLM (có thể None / non-str).
"""

from __future__ import annotations

import unicodedata
from typing import Any

from src.utils.regex_patterns import WHITESPACE

__all__ = ["strip_diacritics", "normalize_for_match"]


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt + giữ Latin/punct/digit.

    Thuật toán:
      1. Replace đ/Đ → d/D thủ công (NFD không tách đ thành d + combining mark).
      2. NFD normalize decompose từng ký tự thành base + combining marks.
      3. Drop ký tự có category "Mn" (Mark, Nonspacing) = combining marks.

    Vd: "Cà Phê Đen" → "Ca Phe Den".
    Trả "" / None giữ nguyên (caller xử lý).
    """
    if not text:
        return text
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_for_match(value: Any) -> str:
    """Chuẩn hoá để substring-match bất chấp case/dấu/spacing.

    Lower + strip diacritics + gộp whitespace. Trả "" khi None/rỗng.
    Punctuation được GIỮ NGUYÊN — module nào cần match-bất-chấp-punct phải tự
    strip punct sau khi gọi function này.
    """
    if value is None:
        return ""
    text = strip_diacritics(str(value)).lower()
    return WHITESPACE.sub(" ", text).strip()
