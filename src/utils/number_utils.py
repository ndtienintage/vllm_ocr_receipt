"""
Number coercion helpers cho OCR pipeline.

Mục đích chính: ép kiểu output từ LLM/OCR (thường là str) sang float, bất chấp
các format số tiếng Việt phổ biến:
  - Thousand separator: "45.000" / "45,000" / "45 000" / "45-000" → 45000
  - Decimal (cho qty measured goods): "1,5" / "1.5" → 1.5
  - Currency suffix: "45.000 đ" / "45.000 VNĐ" / "$45.00" → 45.0 hoặc 45000
  - Group-of-3 rule: trailing ".XXX" (3 digits) là THOUSAND group, KHÔNG phải
    decimal. Đây là collision điển hình giữa format VN (45.000=45k) và format
    US (45.000=45 với 3 decimal places).

Hàm `coerce_numeric` là entry chính. Không có pure-Pydantic equivalent vì
Pydantic float coercion không hiểu format VN.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["coerce_numeric"]

# Currency suffix: đ ₫ ở cuối hoặc VND/VNĐ/USD ở đầu (case-insensitive).
_CURRENCY_SUFFIX_RE = re.compile(r"[đ₫]$")
_CURRENCY_PREFIX_RE = re.compile(r"^(VNĐ|VND|USD)\s*", re.IGNORECASE)


def coerce_numeric(value: Any, field: str = "") -> float | None:
    """
    Parse giá trị từ LLM/OCR sang float. Trả None nếu không decode được.

    Thuật toán:
      1. None / int / float → trả trực tiếp.
      2. str:
         a. Strip + bỏ khoảng trắng + bỏ currency markers.
         b. Xử lý '-' như thousand separator (vd "59-000" → "59000") chỉ khi
            mỗi nhóm sau '-' đúng 3 chữ số (tránh nhầm với dấu âm hoặc range).
         c. Xử lý ',' thông minh:
            - "1,500" (1 nhóm 3 digits sau ',') → thousand sep → "1500".
            - "1,5" (1 nhóm 1-2 digits sau ',') → decimal → "1.5".
            - Còn lại → thousand sep.
         d. Xử lý '.': nếu mọi nhóm sau '.' đều đúng 3 digits → thousand sep → bỏ '.'.
            (Vd "45.000" → "45000", "1.250.000" → "1250000").
            Còn lại giữ '.' như decimal.
         e. float() — fail → None.

    Tham số `field` chỉ để debug log future (hiện không dùng) — giữ chữ ký
    backward-compatible với caller cũ.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = value.strip().replace(" ", "")
    if not s:
        return None

    s = _CURRENCY_SUFFIX_RE.sub("", s)
    s = _CURRENCY_PREFIX_RE.sub("", s)
    s = s.strip()
    if not s:
        return None

    # Dash-as-thousand-separator: "59-000" → "59000", "1-250-000" → "1250000".
    # Bảo vệ: không động vào dấu âm ở đầu chuỗi.
    if "-" in s and not s.startswith("-"):
        dash_parts = s.split("-")
        if (
            len(dash_parts) >= 2
            and all(len(p) == 3 and p.isdigit() for p in dash_parts[1:])
            and dash_parts[0].isdigit()
        ):
            s = s.replace("-", "")

    # Comma: phân biệt thousand sep vs decimal sep dựa trên độ dài nhóm sau ','.
    if "," in s:
        parts = s.split(",")
        if len(parts) == 2 and parts[1].isdigit():
            if len(parts[1]) == 3:
                s = s.replace(",", "")  # "1,500" → "1500"
            elif len(parts[1]) <= 2:
                s = ".".join(parts)     # "1,5" → "1.5"
            else:
                s = s.replace(",", "")
        else:
            s = s.replace(",", "")

    # Dot-as-thousand-separator chỉ khi mọi nhóm sau '.' đúng 3 digits.
    # "45.000" → "45000". "1.5" giữ nguyên (1 nhóm 1 digit → không match).
    if "." in s:
        dot_parts = s.split(".")
        if all(len(p) == 3 and p.isdigit() for p in dot_parts[1:]):
            s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return None
