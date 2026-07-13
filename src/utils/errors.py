"""
Các lớp ngoại lệ tùy chỉnh cho Pipeline Receipt OCR.

Lưu ý: Timeout request (HTTP 408) raise bằng asyncio.TimeoutError trực tiếp
ở server.py — không cần custom class.
"""


class UpstreamServiceError(Exception):
    """
    Dịch vụ vLLM thượng nguồn lỗi hoặc không khả dụng.
    Tương ứng với HTTP 503 (Service Unavailable).
    """
