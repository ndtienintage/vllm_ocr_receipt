"""Pydantic DTO cho API Receipt OCR.

- request.py : schema đầu vào endpoint (OCRRequest).
- receipt.py : schema hoá đơn (Receipt/ReceiptItem) — vừa là guided_json cho vLLM,
               vừa validate output LLM.
"""

from src.schemas.receipt import Receipt, ReceiptItem
from src.schemas.request import OCRRequest

__all__ = ["OCRRequest", "Receipt", "ReceiptItem"]
