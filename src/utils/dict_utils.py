"""
Dict helpers dùng chung trong pipeline OCR.

Hiện chỉ có `first_present` — pattern xuất hiện ở nhiều module do `Receipt`
schema dùng `populate_by_name=True`, downstream consumer có thể thấy cả alias
ngắn (n/t/mn/ma) lẫn tên đầy đủ (name/total/merchant_name/merchant_address).
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["first_present"]


def first_present(item: Dict[str, Any], *keys: str) -> Any:
    """Trả giá trị của key đầu tiên tồn tại trong dict, None nếu không có.

    Hữu ích khi caller cần tolerate cả Pydantic alias (n) lẫn full name (name)
    do schema mở `populate_by_name=True`.
    """
    for key in keys:
        if key in item:
            return item.get(key)
    return None
