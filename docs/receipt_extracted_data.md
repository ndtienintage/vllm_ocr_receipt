# Receipt Extracted Data Schema

Schema dữ liệu trích xuất từ hóa đơn (Receipt OCR).

> Source of truth: `src/schemas/receipt.py`, `src/schemas/request.py`

---

## Request Schema

**Endpoint:** `POST /api/ocr/extract`

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `images_url` | `string[]` | Có (hoặc `images_base64`) | Danh sách **đúng 1** URL ảnh công khai |
| `images_base64` | `string[]` | Có (hoặc `images_url`) | Danh sách **đúng 1** chuỗi Base64 (hỗ trợ data URI) |
| `reference_id` | `string` | Không | ID định danh request để trace trong log (mặc định `"N/A"`) |

> Chỉ được truyền **một trong hai** `images_url` hoặc `images_base64`. Truyền cả hai hoặc không truyền → HTTP 422.
> Danh sách phải chứa **đúng 1 ảnh** — pipeline chỉ xử lý đơn ảnh mỗi request.

Ví dụ request:

```json
{
  "images_url": ["https://example.com/receipt.jpg"],
  "reference_id": "order-abc-123"
}
```

---

## Receipt — Schema response

| Trường | Kiểu | Mô tả |
|---|---|---|
| `merchant_name` | `string \| null` | Tên thương hiệu đầu hóa đơn (in đậm/lớn nhất) |
| `merchant_address` | `string \| null` | Địa chỉ chi nhánh; bỏ email/web/phone/MST |
| `transaction_date` | `string \| null` | Ngày giao dịch (YYYY-MM-DD) |
| `transaction_time` | `string \| null` | Giờ giao dịch (HH:MM hoặc HH:MM:SS, 24h) |
| `items` | `ReceiptItem[]` | Danh sách mặt hàng |
| `subtotal` | `float \| null` | Tổng tiền hàng TRƯỚC VAT/phí — nhãn: Tạm tính, Cộng tiền hàng, Subtotal |
| `total_amount` | `float \| null` | Số tiền khách trả CUỐI CÙNG (sau VAT/giảm giá) — nhãn: Tổng thanh toán, Total |
| `currency` | `string \| null` | Mã tiền tệ: `VND`, `USD`, ... |
| `payment_method` | `string \| null` | Phương thức: `CASH`, `CARD`, `TRANSFER`, `QR`, `MoMo`, `VNPay`, ... |
| `receipt_code` | `string \| null` | Mã hóa đơn (4–100 ký tự); chỉ lấy khi có tiền tố rõ (Số HĐ, Bill NO, Invoice...) |

## ReceiptItem — Schema

| Trường | Kiểu | Mô tả |
|---|---|---|
| `name` | `string \| null` | Tên sản phẩm như in trên hóa đơn (max 200 ký tự) |
| `quantity` | `float \| null` | Số lượng / khối lượng (decimal cho hàng cân kg, integer cho hàng đếm) |
| `price` | `float \| null` | Đơn giá in trên bill |
| `total` | `float \| null` | Thành tiền của dòng item |

---

## Validation Rules

### Numeric fields

- Tự động bỏ ký hiệu tiền tệ (đ, ₫, VND, VNĐ, $) và dấu phân cách nghìn
- `"1.250.000"` → `1250000`, `"1,250,000"` → `1250000`
- `"1,5"` → `1.5` (thập phân; chỉ khi phần sau dấu phẩy ≤ 2 chữ số)
- `"59-000"` → `59000` (dấu gạch ngang phân cách nghìn theo kiểu VN)
- Giá VND < 1000: nếu > 70% items có đơn giá < 1000 → nhân hết lên ×1000 (hóa đơn hiển thị theo đơn vị nghìn đồng)

### Date field (`transaction_date`)

- Output của LLM: `DD-MM-YYYY` (day-first, chép đúng thứ tự in); validator đảo sang `YYYY-MM-DD` (output API)
- Validator nhận DD-MM-YYYY (sep `-`/`.`/`/`/space) + YYYY-MM-DD (ISO, tương thích ngược); tháng>12 → fallback MM-DD
- Fallback text-only: nhận dạng `DD/MM/YYYY`, `DD.MM.YY`, `DD-MM-YYYY` (ưu tiên thứ tự VN)
- Năm 2 chữ số → +2000; năm ngoài khoảng `[2000, nămHiệntại+1]` → `null`

### Time field (`transaction_time`)

- Chỉ chấp nhận `HH:MM` hoặc `HH:MM:SS` (24h); định dạng khác → `null`

### String fields

- Phát hiện hallucination loop (1 token lặp > 50% tổng từ và ≥ 10 lần) → giữ token đầu tiên
- Loại bỏ ký tự Hán/Nhật/Hàn (CJK) khỏi `merchant_name`, `merchant_address`, `items[].name`

---

## Post-processing Pipeline

Sau khi LLM trả về JSON, pipeline thực hiện tuần tự:

1. **CJK strip** — Loại bỏ ký tự Hán/Nhật/Hàn khỏi `merchant_name`, `merchant_address`, `name` của items
2. **Summary filter** — Loại dòng item là summary/payment (Tổng thanh toán, Tiền mặt, Subtotal, ...)
3. **Exclude patterns** — Loại item chứa từ khóa trong `config/exclude_item_patterns.txt`
4. **Item fix** — Sửa lỗi đọc sai phổ biến:
   - `price == total` khi `qty > 1` → tính lại `price = total / qty`
   - `total == 0` khi `price > 0` và `qty > 0` → `total = price × qty`
5. **Totals validate** — Phát hiện `subtotal`/`total_amount` bị hoán đổi và tự sửa

---

## Mapping Alias → Tên đầy đủ

LLM sinh JSON với **alias ngắn** để tiết kiệm token. Pydantic tự động map sang tên đầy đủ trong API response.

**Receipt:**

| Alias | Tên đầy đủ |
|---|---|
| `mn` | `merchant_name` |
| `ma` | `merchant_address` |
| `td` | `transaction_date` |
| `tt` | `transaction_time` |
| `it` | `items` |
| `sub` | `subtotal` |
| `ta` | `total_amount` |
| `cur` | `currency` |
| `pm` | `payment_method` |
| `rc` | `receipt_code` |

**ReceiptItem:**

| Alias | Tên đầy đủ |
|---|---|
| `n` | `name` |
| `qty` | `quantity` |
| `p` | `price` |
| `t` | `total` |

---

## Ví dụ đầy đủ

```json
{
  "merchant_name": "Cửa hàng ABC",
  "merchant_address": "123 Nguyễn Huệ, Q.1, TP.HCM",
  "transaction_date": "2025-01-15",
  "transaction_time": "14:30",
  "receipt_code": "HD-001234",
  "currency": "VND",
  "payment_method": "CASH",
  "items": [
    {
      "name": "Cà phê sữa đá",
      "quantity": 2.0,
      "price": 35000,
      "total": 70000
    },
    {
      "name": "Bánh mì thịt",
      "quantity": 1.0,
      "price": 25000,
      "total": 25000
    }
  ],
  "subtotal": 95000,
  "total_amount": 95000
}
```
