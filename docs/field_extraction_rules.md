# Field Extraction Rules — quy tắc & cách xác định từng trường

Đặc tả **per-field** cho pipeline trích xuất hoá đơn (Qwen3-VL). Mỗi trường gồm: **định nghĩa → quy trình xác định (theo bước) → cue chấp nhận → cue từ chối (→ null) → override TMĐT (nếu có) → xử lý sau (validator/postproc) → known-limit → ví dụ**.

> **Source of truth (rule):** [`src/extraction/llm_extractor.py`](../src/extraction/llm_extractor.py) — prompt vision. Bản này là mirror người-đọc; khi prompt đổi, cập nhật file này.
> **Source of truth (kiểu + validator):** [`src/schemas/receipt.py`](../src/schemas/receipt.py).
> **Liên quan:** [prompt_design.md](prompt_design.md) (triết lý + thứ tự block), [receipt_extracted_data.md](receipt_extracted_data.md) (bảng schema API).

---

## 0. Nguyên tắc bao trùm (áp cho MỌI field)

| Nguyên tắc | Nội dung |
|---|---|
| **WYSIWYG** | Chỉ trích cái **đang in**. Không suy luận, không tính toán, không bịa. |
| **Clean null > guessed char** | Thiếu / mờ / nhoè / loé / không đọc được → `null`. Một `null` sạch là đáp án đúng; một ký tự đoán là đáp án sai. |
| **Null theo FIELD, không cascade** | Mỗi field unreadable thì null riêng nó. Một zone bị cắt **chỉ** null field của zone đó — KHÔNG lan sang field khác. |
| **Thứ tự sinh = thứ tự schema** | Model sinh `ly` → `it` → `mn` → `ma` → `td` → `tt` → `ta`. Vì vậy `ly` + `it` được quyết định trước tiên (xem [prompt_design.md](prompt_design.md) về primacy/recency). |
| **Output** | Đúng MỘT JSON object thô, không markdown fence, không prose. Số là JSON number; rỗng là JSON null. |

Schema (alias ngắn để tiết kiệm token output):

```json
{"ly": "...|null",
 "it": [{"n": "...", "qty": 0, "p": 0, "t": 0}],
 "mn": "...|null", "ma": "...|null",
 "td": "DD-MM-YYYY|null", "tt": "HH:MM[:SS]|null", "ta": 0}
```

Chỉ emit đúng các key này. **Không** có field subtotal / tax / currency / payment-method / receipt-code.

---

## 1. `ly` — layout completeness (trường GATING)

**Định nghĩa:** vùng (zone) nào của hoá đơn có mặt trong ảnh. Đây là field quyết định trước tiên, điều phối việc null các field header/footer. Sau xử lý, postprocessor **pop bỏ `ly`** trước khi trả client — nó là scaffold suy luận, không phải field client-facing.

**Quy trình xác định** — đọc bằchứng ở **RÌA ẢNH** (trên cùng / dưới cùng), rồi chọn 1 trong 4:

| Giá trị | Điều kiện | Hệ quả null |
|---|---|---|
| `COMPLETE` | Brand ở trên, item ở giữa, total có nhãn ở dưới — đủ 3 zone. | Không null ép buộc. |
| `MISSING-HEADER` | Dòng trên cùng **đã là** item / promo / column-header / chỉ có doc-type, **không** thấy brand trong header window. | **HARD: `mn=null` AND `ma=null`** (không thay thế). `td`/`tt`/`it`/`ta` vẫn emit. |
| `ITEM-ONLY` | Crop sát chỉ có items; KHÔNG header trên VÀ KHÔNG total/payment dưới. | Mọi field trừ `it` = null. `it` **KHÔNG được rỗng**. |
| `MISSING-FOOTER` | Có header + items, nhưng dòng dưới cùng là item row, không có nhãn total/payment. | `ta=td=tt=null`. `mn`/`ma`/`it` vẫn emit. |

**Nguyên tắc chốt:** điền một field thuộc zone-bị-cắt bằng một dòng sai **tệ hơn** để null.

**Validator/postproc:** `Receipt.layout` là `Literal[...]|null`. Postprocessor pop `ly` trước khi trả client.

**Known-limit:** `ly` hay bị model emit `null` thay vì `MISSING-HEADER` trên hoá đơn cắt header → ENFORCEMENT `mn=null` không bắn → footer leak vào `mn`/`ma`. Xem memory `known-limit-missing-header-footer-mn-leak`. E-commerce app screenshot thường là `COMPLETE` (có shop name + total) — xem §10.

---

## 2. `it[]` — line items (CÔNG VIỆC CHÍNH)

Items neo cả tài liệu. Quét **từng dòng, từ trên xuống**. Mỗi entry: `{n, qty, p, t}`.

### 2.0 Nhận diện & ranh giới row

- **ROW:** bất kỳ dòng nào **cùng trục Y** với một token qty / giá / thành-tiền là một item row. Text của nó thuộc `it[].n` **CHỈ** — không bao giờ là `mn`/`ma`.
- **WHEN-IN-DOUBT EMIT:** đọc được TÊN nhưng số lệch/thiếu → **vẫn** emit `{"n":"tên","qty":null,"p":null,"t":null}`. Thiếu số **không** phải lý do bỏ một row đọc được.
- **BLUR SENTINEL:** một dòng RÕ RÀNG là item row (nằm trên trục Y item / cột giá) nhưng quá mờ/nhoè để đọc mà không phải đoán → emit `{"n":"ITEM BLUR","qty":null,"p":null,"t":null}` để đánh dấu row. **KHÔNG** đoán ký tự, **KHÔNG** mượn tên hàng xóm. Đây là marker chống-derail (giữ nhịp sinh để tiếp tục các item sau); postprocessor sẽ **drop** item `n=="ITEM BLUR"` khỏi output cuối. Chỉ dùng cho item row có thật mà không đọc được — KHÔNG cho dòng structural/footer (drop những dòng đó), và KHÔNG khi tên đọc được mà chỉ thiếu số (dùng WHEN-IN-DOUBT EMIT với tên thật). *(Path text-only: chỉ emit sentinel khi row có số giá/thành-tiền dùng được; garble không-số → DROP, để chống flood.)*
- **LOOP BAILOUT:** nếu cùng một `n` lặp ≥4 row liên tiếp mà `t` đều null/0 → đang loop → DỪNG, đóng `it` sau row cuối có số thật. **NGOẠI LỆ:** sentinel "ITEM BLUR" là marker cố ý, không phải loop — giữ từng cái, không để nó kích hoạt bailout. Cũng đóng một string field bằng null nếu một đoạn 6–10 ký tự lặp 3× bên trong. Chỉ emit cái đang in; **không** độn cho đầy trang.

### 2.1 Gộp row (MERGE) — nhiều dòng = MỘT item

| Pattern | Hình dạng | Kết quả |
|---|---|---|
| **GO-UNIT** | dòng chỉ-text + "qty unit_word x price total" | `n`=dòng text; bỏ unit_word (Cái/Bộ/Gói/Lốc/Hộp/Kg/Ch/Bịch/Túi…), không đưa vào `n`. |
| **BARCODE** | "barcode tên" + "[VAT%] qty price total" | strip barcode, `n`=tên. |
| **DISCOUNT-NAME** | "tên -discount" + "barcode qty price[-disc%] total" | `n`=tên (bỏ `-discount` và đuôi `-disc%`), qty/p/t lấy từ dòng 2. |
| **ATTRIBUTE MERGE** | dòng con là **thuộc tính tiêu dùng** của item ngay trên — topping / size / sugar / ice | **APPEND** chữ của nó vào `n` trước đó bằng `" + "` theo thứ tự in; bỏ giá add-on + id đuôi; item cha **giữ nguyên** `p`/`t` của nó (KHÔNG cộng add-on). |

**Cue ATTRIBUTE MERGE (±dấu):** topping (Topping / Thêm / Extra / Trân châu / Pudding / Thạch / Kem cheese), size (Size / Up size), sugar (Ít đường / Không đường / 50% đường / Bình thường), ice (Ít đá / Nhiều đá / Không đá / Đá riêng).
**KHÔNG merge (metadata — bỏ, không phải item, không vào `n`):** note (Ghi chú / Note / Yêu cầu / Lưu ý / Lời nhắn) và order-type (Mang đi / Tại quán / Take away / Dine in / Giao hàng / Ship). Dòng con mồ côi (không có tên ở trên) → bỏ.

### 2.2 DROP HARD — KHÔNG bao giờ là item, KHÔNG merge vào `n`

Áp bất kể số lần lặp, notation (`@unit -disc` hay `#unit -disc`), hay dấu:

- **discount/promo:** KHUYẾN MÃI / KM, CK THẺ …%, DISC %/DISCOUNT, Voucher, Giảm giá, hoặc bất kỳ dòng per-item có **số ÂM** (vd "KHUYẾN MÃI @55.000 -10.000", "CK THẺ AEON 5% #016.000 -800"). Bỏ cả row; không giữ gì làm `p`/`t`. Có thể lặp 0–3× sau mỗi item cha — bỏ từng cái.
- **gift/giveaway:** Tặng / Tặng kèm / Quà tặng / Kèm theo / Đi kèm, "mua … tặng …", dòng free (0 / miễn phí). Bỏ cả row.
- **INLINE discount trên item một-dòng** ("Bánh 100.000 -10.000 90.000"): giữ `p`=gốc, `t`=cuối, chỉ bỏ token `-discount`; KHÔNG tách thành 2 item.
- **STRUCTURAL (không phải sản phẩm):** column header, barcode, Phí phục vụ/service fee, dòng VAT, **Giá gốc** (giá gốc metadata), subtotal/total, payment, loyalty/điểm, hotline/web/cảm ơn → feed footer hoặc bỏ.

### 2.3 `it.n` — tên sản phẩm

- **Định nghĩa:** tên sản phẩm như in (sau khi đã merge/strip theo §2.1).
- **Xác định:** lấy phần chữ của row; bỏ STT đầu dòng, prefix VAT, barcode, unit_word, token discount; merge attribute bằng `" + "`.
- **Từ chối:** dòng promo/gift/structural (§2.2); dòng footer.
- **TMĐT override:** append biến thể phân-biệt ("Ariel - Muỗng" vs "Ariel - Nĩa") để 2 row trùng tên không bị gộp; bỏ biến thể vô nghĩa ("Mặc định"). Xem §10.
- **Validator:** strip whitespace, **truncate 250 ký tự**, null nếu rỗng (defensive vs loop). Postproc còn strip ký tự CJK (Hán/Nhật/Hàn) khỏi `name`.

### 2.4 `it.qty` — số lượng

- **Định nghĩa:** số lượng / khối lượng đã mua. Integer cho hàng đếm; **decimal** cho hàng cân (kg).
- **Xác định:** lấy token qty trên row (hoặc dòng 2 khi MERGE). Hàng cân: qty thập phân "0,704" **giữ** dấu thập phân **khi** row có cột qty nhưng KHÔNG có token đơn-giá riêng (xem §9).
- **Validator:** `coerce_numeric` → float. **Không** reject âm/0 (dòng dịch vụ/refund có thể qty=0 hoặc âm hợp lệ).
- **⚠ Known-limit:** qty dấu-phẩy hàng cân ("0,706") đi theo nhánh đo-lường có thể tạo `items=[]`, hoặc `coerce_numeric` rescale âm thầm "0,706"→706.0. Chấp nhận như known-limit, fix ở fine-tune. Xem memory `known-limit-decimal-comma-qty`.

### 2.5 `it.p` — đơn giá &  2.6 `it.t` — thành tiền

- **`p` (đơn giá):** giá cho MỘT đơn vị.
- **`t` (thành tiền):** tổng của dòng item đó.
- **PRICE vs TOTAL (quy tắc cốt lõi):** row chỉ có **2 cột** (tên + 1 số) → số đó là **`t`**, KHÔNG phải `p`.
- **Khi qty=1 và chỉ in 1 giá:** `p = t = giá đó`.
- **Khi có 2 số bằng nhau** (đơn giá + thành tiền, qty 1): `p` = số đầu, `t` = số sau.
- **TMĐT override (PAIN-POINT) — định nghĩa lại p/t/qty trên card app:**
  - **`t` = giá ngoài cùng BÊN PHẢI** của card SKU (số tiền right-most của dòng item đó). **Mọi item BẮT BUỘC có `t`.** Khi card chỉ in **1 giá** → giá đó là **`t`** (KHÔNG phải `p`), kể cả khi có badge "xN".
  - **`p` = đơn giá MỘT đơn vị, CÓ THỂ null** — card app thường chỉ in line-total nên không có đơn-giá riêng → `p=null`; KHÔNG suy `p = t/qty`.
  - **`qty` = số lượng HOẶC khối lượng, CÓ THỂ null** — thường là badge "xN" hoặc khối lượng in; không gộp/nhân badge vào `p`/`t`.
  - Giá gốc **bị gạch ngang** cạnh giá thực → **bỏ** giá gạch, chỉ dùng giá thực (đậm). Banner "Giảm 237.980đ tại cửa hàng này" KHÔNG phải giá/item — bỏ. Xem §9–§10.
- **Validator:** `coerce_numeric` → float, **giữ âm** cho dòng discount/refund.
- **Postproc item-fix:** `p == t` khi `qty > 1` → tính lại `p = t/qty`; `t == 0` khi `p>0` và `qty>0` → `t = p×qty`. (Đây là code postproc, không phải prompt.)

---

## 3. `mn` — merchant name (tên cửa hàng)

**Định nghĩa:** thương hiệu / tên cửa hàng phát hành hoá đơn — danh tính nói **AI** bán hàng. **Header-only**: không bao giờ điền từ body items hay footer.

**Quy trình xác định:**
1. **HEADER WINDOW:** đọc vùng trên cùng cho tới item row / column header / nhãn total-payment đầu tiên.
2. **CROPPED-HEADER GUARD:** nếu dòng đọc-được trên cùng đã là item / promo / column-header, **không** có brand logo phía trên → header bị cắt → `mn=null AND ma=null`. KHÔNG khôi phục `mn` từ bất kỳ dòng nào **DƯỚI** items.
3. **Chọn brand:** danh từ riêng / logo glyph **to nhất / đậm nhất / canh giữa** ("GO!", "WinMart", "Circle K", "Bách Hóa Xanh").
4. **BRAND BEATS LEGAL ENTITY:** nếu có CẢ brand storefront LẪN dòng pháp nhân ("Cty CP / CÔNG TY CỔ PHẦN / TNHH / JSC") → chọn **brand**; chỉ fallback tên công ty khi KHÔNG có brand/logo.
5. **Concatenate** 2 dòng kề chỉ khi cùng một tên đăng ký (vd "CÔNG TY TNHH"+"THỰC PHẨM ABC"); không nối slogan.

**Chấp nhận:** brand line / logo glyph trong header window.
**Từ chối (→ null, không thay thế):**
- Doc marker đứng một mình: "HÓA ĐƠN", "GTGT", "PHIẾU THANH TOÁN", "PHIẾU TÍNH TIỀN", "BILL", "RECEIPT", "TAX INVOICE", "LIÊN 1/2/3", "COPY".
- Địa chỉ / branch / hotline / web (→ thuộc `ma` hoặc bỏ).
- Footer / non-header: payment, promo/voucher, loyalty/member, cashier, cảm ơn.

**TMĐT override (PAIN-POINT):** shop name nằm **GIỮA màn hình** cạnh avatar/logo shop, kế dấu `>` và/hoặc badge "Mall", **TRÊN** product card ("Cocoon Vietnam", "ARUAGEMSVN", "Gabby - Disney Chính Hãng") — **dù không ở đỉnh ảnh**. KHÔNG phải screen title / tên người nhận / status bar / nút. Xem §10.

**Validator:** strip, **truncate 250**, null nếu rỗng. Postproc strip CJK.
**⚠ Known-limit:** (1) postprocessor exclude pattern có thể null nhầm `mn` hợp lệ ("Phiếu thanh toán Bách Hóa Xanh") trước khi canonical rule map được — memory `postprocessor-exclude-before-canonical-bug`. (2) MISSING-HEADER footer leak — memory `known-limit-missing-header-footer-mn-leak`.

---

## 4. `ma` — merchant address (địa chỉ cửa hàng)

**Định nghĩa:** địa chỉ đường phố đầy đủ của cửa hàng, lấy từ **header**.

**Quy trình xác định:** ghép địa chỉ nhiều dòng bằng `", "`.
**Cue chấp nhận:** đường / phố / số nhà / phường / quận / P. / Q. / TP. / Tầng / Lô / tên ward-district-city / số nhà.
**Từ chối (→ null):** dòng promo/payment/loyalty/cảm ơn, MST, phone/web/email, cashier, branch-only metadata, bất kỳ item row.

**TMĐT override (QUAN TRỌNG):** trên app screenshot, địa chỉ in là **địa chỉ giao hàng của KHÁCH** ("Người nhận" / "Người mua" / "Địa chỉ nhận hàng" + SĐT che) — KHÔNG phải địa chỉ merchant → **`ma=null`**. Không copy địa chỉ giao hàng vào `ma`. Xem §10.

**Validator:** strip, **truncate 300**, null nếu rỗng. Postproc strip CJK.

---

## 5. `td` — transaction date (ngày giao dịch)

**Định nghĩa:** ngày giao dịch. LLM output **`DD-MM-YYYY`** (day-first — CHÉP đúng thứ tự in, KHÔNG tự đảo); validator đảo sang **`YYYY-MM-DD`** (output API). Lý do tách: model giỏi transcription, kém reorder → đẩy việc đảo sang code deterministic, bỏ lỗi swap ngày↔tháng.
**Xác định:** gần nhãn "Ngày" / "Date" hoặc in cạnh `tt`. Năm 2 chữ số đã in → dùng năm hiện tại làm anchor để mở rộng `YY` thành `20YY`; nếu chỉ có ngày/tháng và không có năm in trên hóa đơn → `td=null`, không tự thêm năm hiện tại.
**THỨ TỰ NGÀY (KHÔNG đảo tháng↔ngày):** ngày in dạng số là **DAY-first DD/MM/YYYY** (chuẩn VN). LLM **CHÉP nguyên thứ tự in** → `"04/06/2026"` → `"04-06-2026"` (KHÔNG đảo sang year-first). Việc đảo + disambiguation nằm ở **validator**: mặc định day-first; nếu phần tháng >12 (model lỡ in MM-DD) → fallback MM-DD. Dạng chữ "Ngày D tháng M [năm Y]" tường minh (ngày=day, tháng=month).
**Từ chối:** HSD / NSX / EXP / MFG (hạn dùng/sản xuất), năm copyright, MST, dải số SKU, mã số không liên quan.
**Validator** (`receipt.py:_validate_date`): nhận **DD-MM-YYYY** (separator `-` `.` `/` space) → đảo sang `YYYY-MM-DD`; tháng >12 → fallback MM-DD. Cũng nhận **YYYY-MM-DD** (ISO, tương thích ngược). Calendar invalid / **tên tháng VN viết chữ ("tháng Năm")** / năm 2 chữ số → trả **raw** (known-limit, không raise). Truncate 20.
**Fallback text-only:** khi LLM bỏ sót `td`, `_datetime_sweep` + `_try_parse_date` (regex `DATE_DMY`) parse theo **DMY** → trả thẳng ISO `YYYY-MM-DD` (validator passthrough ISO).
**TMĐT:** app screenshot có nhiều mốc ("Ngày đặt hàng", "Thời gian thanh toán", "Ngày giao hàng") — chọn mốc giao dịch/thanh toán; tránh ngày vận chuyển/giao.

---

## 6. `tt` — transaction time (giờ giao dịch)

**Định nghĩa:** giờ giao dịch, output **`HH:MM`** hoặc **`HH:MM:SS`** (24h).
**Xác định:** lấy token giờ trên/ cạnh ngày.
**Validator:** extract prefix `HH:MM[:SS]`, cắt suffix lạ (vd "08:52:NV:283872" → "08:52"). Range guard HH≤23, MM≤59, SS≤59 (rác "25:99" → null). Format khác (HHhMM…) → giữ raw truncate 10.

---

## 7. `ta` — total amount (số tiền cuối cùng phải trả)

**Định nghĩa:** tổng tiền khách **thực trả cuối cùng** (sau VAT/giảm giá).
**Quy trình xác định:** **BẮT BUỘC có nhãn rõ cùng dòng** — "Tổng cộng", "Phải thanh toán", "Thành tiền", "Tổng tiền thanh toán", "Total" (hoặc tương đương không dấu).
**Cấm tuyệt đối:** KHÔNG cộng `it[].t`; KHÔNG copy số lớn nhất để ép điền. **Không nhãn = null.**
**Từ chối (→ null):** Tạm tính / Subtotal (pre-total), Tiền khách đưa / Cash, Tiền thối / Change, Deposit, Cashback.
**TMĐT override:** `ta` = "Tổng:" / "Thành tiền" / "Tổng cộng". **KHÔNG** lấy "Tổng phụ" (subtotal), "Vận chuyển" (ship), "Phiếu giảm giá …" (voucher) — chúng không phải `ta` và không phải item. Xem §10.
**Validator:** `coerce_numeric` → float.

---

## 8. `<numbers>` — chuẩn hoá số VND (áp cho `qty`/`p`/`t`/`ta`)

- `.` VÀ `,` **đều** là dấu phân cách nghìn → strip, output integer: `"55.000"→55000`, `"1,250,000"→1250000`. Giữ đúng số chữ số trước khi strip.
- **NGOẠI LỆ:** qty hàng cân ("0,704" kg) **giữ** dấu thập phân khi row có cột qty nhưng KHÔNG có token đơn-giá riêng.
- **Postproc rescale (code, không phải prompt):** nếu >70% items có đơn giá <1000 → nhân tất cả ×1000 (bill hiển thị đơn vị nghìn đồng).

---

## 9. E-commerce app screenshots — bảng override gộp (§10)

App TMĐT (Shopee / Lazada / TikTok Shop) là **lớp layout riêng**. **Nhận diện** qua: status bar điện thoại ở đỉnh, screen title ("Đã giao đơn hàng" / "Thông tin đơn hàng" / "Đơn hàng đã hoàn thành" / "Chi tiết đơn hàng"), badge "Mall", nút ("Mua lại" / "Đánh giá" / "Yêu cầu hoàn tiền"). Khi nhận diện, các rule sau **GHI ĐÈ** rule header tổng quát:

| Field | Override |
|---|---|
| `mn` | Shop name cạnh avatar/logo, kế `>`/badge "Mall", **TRÊN** product card — dù không ở đỉnh ảnh. KHÔNG phải screen title / người nhận / status bar / nút. Giữ tên đầy đủ kể cả đuôi " - …"; bỏ "Mall" và ">". |
| `ma` | **= null** — địa chỉ in là của KHÁCH ("Người nhận"/"Người mua"/"Địa chỉ nhận hàng" + SĐT che). |
| `it.t` | = giá **ngoài cùng bên phải** của card SKU. **Bắt buộc có.** Card chỉ in 1 giá → giá đó là `t` (không phải `p`). |
| `it.p` | đơn giá 1 đơn vị, **có thể null** (card thường không in đơn-giá riêng → `p=null`; không suy `t/qty`). |
| `it.qty` | số lượng/khối lượng, **có thể null** (badge "xN" hoặc khối lượng). Bỏ **giá gạch ngang** (giá gốc), chỉ dùng giá thực (đậm). |
| `it.n` | Append biến thể phân-biệt ("Ariel - Muỗng" vs "Ariel - Nĩa") bằng " + "; bỏ "Mặc định". |
| `ta` | = "Tổng:"/"Thành tiền"/"Tổng cộng". KHÔNG lấy "Tổng phụ"/"Vận chuyển"/"Phiếu giảm giá". |
| (bỏ) | Banner "Giảm …đ tại cửa hàng này"/"Mall Giảm …" — promo, không phải giá/item. |

---

## 10. Ví dụ end-to-end (3 mẫu TMĐT)

**Mẫu 1 — TikTok, Cocoon Vietnam:**
```json
{"ly":"COMPLETE","mn":"Cocoon Vietnam","ma":null,
 "it":[{"n":"Minisize Tẩy da chết da đầu bồ kết Cocoon","qty":1,"p":null,"t":58650}],
 "ta":58650}
```
→ card chỉ in 1 giá bên phải → `t`=58650, `p`=null; shop name giữa màn hình (không ở đỉnh); "Giảm 237.980đ tại cửa hàng này" bỏ; `ta`=Tổng 58.650 (KHÔNG lấy Vận chuyển 26.200 / voucher).

**Mẫu 2 — TikTok, ARUAGEMSVN:**
```json
{"ly":"COMPLETE","mn":"ARUAGEMSVN","ma":null,
 "it":[{"n":"#10 Bông tai zircon thiết kế vòng","qty":1,"p":null,"t":39100}],
 "ta":39100}
```

**Mẫu 3 — Shopee, Gabby - Disney Chính Hãng (2 biến thể):**
```json
{"ly":"COMPLETE","mn":"Gabby - Disney Chính Hãng","ma":null,
 "it":[{"n":"Muỗng, Thìa Ăn Dặm Cho Bé Gabby + Ariel - Muỗng","qty":1,"p":null,"t":32000},
       {"n":"Muỗng, Thìa Ăn Dặm Cho Bé Gabby + Ariel - Nĩa","qty":1,"p":null,"t":32000}],
 "ta":53760}
```
→ mỗi card 1 giá phải → `t`=32000, `p`=null; append biến thể để 2 row trùng tên tách riêng; `ta`="Thành tiền: 53.760" (nhỏ hơn 64.000 do voucher ẩn — vẫn lấy theo nhãn, KHÔNG cộng items).

---

## 11. Bảng tra nhanh — field → quyết định 1 dòng

| Field | Lấy khi | Null khi | Validator |
|---|---|---|---|
| `ly` | luôn — đọc rìa ảnh | (không null; là enum) | Literal, postproc pop |
| `it.n` | row cùng trục Y với số; sau merge | promo/gift/structural/footer | strip, ≤250, CJK strip |
| `it.qty` | token qty / dòng-2 khi merge | không có | coerce float, giữ âm/0 |
| `it.p` | đơn giá; row 2-cột → đây là `t` | không có | coerce float, giữ âm |
| `it.t` | thành tiền dòng | không có | coerce float, giữ âm |
| `mn` | brand to/đậm/giữa header; (TMĐT: shop card) | doc-marker/footer/địa chỉ; cropped header | strip, ≤250, CJK strip |
| `ma` | địa chỉ header (đường/phường/quận…) | promo/payment/MST/phone; **TMĐT: luôn null** | strip, ≤300, CJK strip |
| `td` | gần "Ngày"/"Date" | HSD/NSX/EXP/copyright/MST/SKU | ISO verify, ≤20 |
| `tt` | token giờ | format rác/range sai | prefix HH:MM[:SS], ≤10 |
| `ta` | **có nhãn** Tổng cộng/Thành tiền/Total | không nhãn; Tạm tính/Tiền khách đưa/Tổng phụ | coerce float |

---

> **Ghi chú đồng bộ:** rule semantic ở đây dùng **verbatim chung** với [text_extractor.py](../src/extraction/text_extractor.py) (fallback text-only) — khi sửa rule một file phải sửa file kia, nếu không 2 path phân kỳ. Khác biệt hợp lệ duy nhất là theo modality (pixel vs bbox). Xem [prompt_design.md §5](prompt_design.md).
</content>
</invoke>
