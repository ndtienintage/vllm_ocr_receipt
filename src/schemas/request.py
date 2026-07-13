from pydantic import BaseModel, HttpUrl, model_validator
from typing import Optional, List

MAX_IMAGES = 1


class OCRRequest(BaseModel):
    """
    Đầu vào cho endpoint trích xuất hóa đơn.
    Chỉ được phép truyền MỘT trong hai trường: images_url hoặc images_base64.
    Danh sách phải chứa đúng MAX_IMAGES ảnh.
    """
    images_url: Optional[List[HttpUrl]] = None
    images_base64: Optional[List[str]] = None
    reference_id: str = "N/A"

    @model_validator(mode="after")
    def validate_input(self):
        has_url = self.images_url is not None
        has_b64 = self.images_base64 is not None

        if not has_url and not has_b64:
            raise ValueError("Phải cung cấp một trong hai trường: images_url hoặc images_base64.")
        if has_url and has_b64:
            raise ValueError("Chỉ được phép truyền một trong hai: images_url hoặc images_base64, không phải cả hai.")

        payload = self.images_url if has_url else self.images_base64
        if len(payload) != MAX_IMAGES:
            field = "images_url" if has_url else "images_base64"
            raise ValueError(f"{field} phải chứa đúng {MAX_IMAGES} ảnh.")

        return self
