# Prompt Design — Receipt OCR Pipeline

Tổng kết nghiên cứu và quyết định thiết kế prompt cho pipeline trích xuất hoá đơn tiếng Việt, dùng Qwen3-VL qua vLLM. Cả hai phase (vision primary và text-only fallback) chia sẻ **một bộ block chung verbatim** + triết lý **structural anti-hallucination** thay vì negative instructions.

> Source of truth: [src/extraction/llm_extractor.py](../src/extraction/llm_extractor.py) (Phase 1 — vision), [src/extraction/text_extractor.py](../src/extraction/text_extractor.py) (Phase 2 — text-only fallback).

> ⚠️ **Hai prompt được giữ ĐỒNG BỘ.** `<fields>`, `<merchant>`, `<classify>`, DROP-HARD list, LOOP-BAILOUT và `<numbers>` dùng wording **verbatim giống nhau** ở cả hai file. Khi sửa rule ở một file, sửa luôn file kia — nếu không hai path sẽ phân kỳ và postprocessor (reuse chung) trả output khác nhau cho cùng một hoá đơn. Khác biệt hợp lệ duy nhất là theo modality (xem §5).

---

## 1. Hai phase của pipeline

| Phase | Module | Input | Khi nào chạy |
|---|---|---|---|
| **1 — Vision** | [`llm_extractor.py`](../src/extraction/llm_extractor.py) | Ảnh + prompt (no image-OCR pre-step) | Mặc định cho mọi request |
| **2 — Text-only fallback** | [`text_extractor.py`](../src/extraction/text_extractor.py) | PaddleOCR PP-OCRv5 lines (text + bbox) + prompt (KHÔNG kèm ảnh) | Khi `hallucination_detector` flag kết quả phase 1 |

Hai prompt cùng schema (alias keys), cùng field rules, cùng sampling params, cùng block order — postprocessor reuse trực tiếp không cần phân nhánh.

---

## 2. Nguồn nghiên cứu

Các nguồn được consult khi thiết kế (sắp xếp theo mức độ thẩm quyền):

| Nguồn | Vai trò |
|---|---|
| [Alibaba DashScope — Qwen-VL-OCR docs](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr) | Prompt pattern chính thức của team Qwen-VL-OCR (verbatim phrasing for null / "?" / schema-as-description) |
| [Alibaba — Qwen Structured Output rules](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output) | Schema design rules: nullable types, field ordering, depth limit, max_tokens warning |
| [Qwen3-VL Cookbooks (ocr.ipynb, document_parsing.ipynb)](https://github.com/QwenLM/Qwen3-VL/tree/main/cookbooks) | Reference notebooks từ team Qwen |
| [Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Sampling params chính thức |
| [Unsloth Qwen3-VL run guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) | Cross-verify sampling params + serving tips |
| [vLLM Structured Outputs docs](https://docs.vllm.ai/en/latest/features/structured_outputs/) | API contract cho guided JSON, backend trade-offs |
| [vLLM Qwen3-VL serving recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html) | Serving config khuyến nghị |
| [arXiv 2408.03834 — Target Prompting for VLM IE](https://arxiv.org/abs/2408.03834) | Validated rằng structural > negative instructions; few-shot uniform |
| [HF — OCR with open VLMs (2025-26)](https://huggingface.co/blog/ocr-open-models) | Qwen3-VL "không có universal OCR prompt — experiment"; temp=0 cho deterministic |
| [Prompt-length sweet spot](https://particula.tech/blog/optimal-prompt-length-ai-performance) · [More-words-less-accuracy](https://gritdaily.com/impact-prompt-length-llm-performance/) | Sweet spot ~1.8k tok; extraction 300-800; small model degrade sớm hơn |
| [vLLM bug #18819](https://github.com/vllm-project/vllm/issues/18819) | Qwen3 structured output + `enable_thinking=False` emits invalid JSON |
| [vLLM bug #13038](https://github.com/vllm-project/vllm/issues/13038) | Qwen2.5-VL: `List[str]` schema không luôn được enforce |

---

## 3. Sáu nguyên tắc thiết kế cốt lõi

### 3.1. Per-field nullable + per-field description (Alibaba pattern)

> **Verbatim từ DashScope docs:** *"If there is no corresponding value, fill it with null."*

- Mỗi field trong schema đều `string|null` / `number|null` — không có required field.
- `<fields>` block mang format hint + reject list per-field (Alibaba "local guard"): `td` mang hint "DD-MM-YYYY" (day-first, chép thứ tự in; validator đảo sang ISO) nên item name không slip vào; `rc` REQUIRES a label; `ta` reject "tiền khách đưa"…
- **Tại sao:** Alibaba ghi rõ schema-nullable "significantly reduces hallucination rates" — required + missing evidence là combo trigger fabricate.

### 3.2. Reasoning trước data fields (Alibaba ordering rule)

> **Verbatim:** *"LLMs generate fields in the order they appear in the schema, so always place reasoning/analysis fields before the final conclusion fields."*

- `ly` (layout class) là field ĐẦU TIÊN trong schema → model "commit" verdict crop trước khi sa vào data.
- `<classify>` block đặt **trước** các block field-specific. `it` đứng ngay sau `ly` (items-first anchor — xem [receipt.py:108-113](../src/schemas/receipt.py#L108-L113), comment giải thích decoder lock vào items trước, tránh attractor null trên header/footer).

### 3.3. Zone-local nulling — không cascade

**Vấn đề từng gặp:** Model bail full-null JSON khi gặp ảnh khó (cropped header → null mn → cascade → null hết items/totals).

`<classify>` định nghĩa rõ 5 case + ma trận null cục bộ (xem §8). Câu chốt: *"A missing zone IS the correct cause for null there — that is the right answer, not a failure."* `<fields>` reinforce: *"Null is per-FIELD, NEVER per-receipt; an all-null receipt while ≥3 OCR lines were provided is a FAILURE mode."*

### 3.4. Anti-hallucination structural, không phải negative instructions

> **Từ arxiv 2408.03834:** Negative instructions ("do not hallucinate") hiệu quả thấp. Lever hiệu quả là **structural**: schema nullability + per-field descriptions + per-field reject lists.

- Reject lists **per-field** thay vì global: `td` reject HSD/NSX/EXP/MFG/© year/MST/SKU; `rc` reject phone/MST/TID/MID/NV/table; `ta` reject tiền khách đưa/change/deposit/cashback.
- `<merchant>` BRAND TEST: emit `mn` chỉ khi line pass test "removing it leaves the receipt unable to identify WHO sold the goods" + reject-category list cụ thể (address/doc-type/MST/branch/cashier/phone/web/order/slogan) + "don't fall through hunting for a substitute".
- `<items>` DROP-HARD: discount/modifier/note/fee bị drop theo label cụ thể, kèm `@unit -disc` notation + diacritic-stripped variants.

### 3.5. Concise > verbose — nhưng có sàn rule-density

> **Research:** Sweet spot prompt ~1.8k token; extraction tasks tốt nhất 300-800 tok; >2-4k token bắt đầu degrade (small model degrade sớm hơn); *"adding excessive examples commonly drops accuracy"* ([particula](https://particula.tech/blog/optimal-prompt-length-ai-performance), [gritdaily](https://gritdaily.com/impact-prompt-length-llm-performance/)).

- Đã gộp/cô đọng: vision & text dùng chung `<fields>`/`<merchant>`/`<classify>`; text gộp 3 rule-section cũ (`extraction_discipline` + `receipt_layout_rules` + `items_extraction_rules`) thành một `<items>`; examples text 11→8.
- **Quyết định có chủ đích:** prompt hiện ~2.8-3.5k token, **CAO hơn** sweet spot generic ~1.8k. Không ép xuống tiếp vì mỗi rule/example còn lại chống một failure đã ghi nhận (AEON cyclic-drop, all-null bail, promote sai mn, decimal-qty) và repo **chưa có eval harness** để bắt regression. Sàn rule-density của task này nằm trên sweet spot generic. Cắt sâu hơn cần đo accuracy trước/sau trên bộ ảnh + ground-truth.
- Few-shot phải **uniform** (XML tags/whitespace/separators giống nhau) — đây là lý do cấu trúc 2 prompt giữ song song.

### 3.6. Modality-faithful, không phải universal prompt

> **HF OCR blog:** Qwen3-VL *"isn't optimized for a single universal OCR prompt — we recommend experimenting."*

Vision tin pixel (font size, bold, vị trí); text tin bbox (y1 overlap, x2 right-edge). Hai prompt khác nhau ĐÚNG ở những cue này, giống nhau ở mọi rule semantic.

---

## 4. Kiến trúc prompt — block order

Cả 2 phase dùng cùng kiến trúc block; chỉ thêm/bớt block đặc thù modality:

| # | Block | Phase 1 vision | Phase 2 text-only | Vai trò |
|---|---|:-:|:-:|---|
| — | `<context>` | inline | prepend | Năm hiện tại — chỉ anchor năm 2 chữ số đã in, không tự thêm năm khi date thiếu năm |
| 1 | `<task>` / `<role>` | ✓ | ✓ | Scope + JSON-only + WYSIWYG (vision) / map-don't-rejudge (text) |
| 2 | `<input>` | — | ✓ | OCR bbox format `x1,y1,x2,y2\|text` + same-row/wrapped/columns logic |
| 3 | `<schema>` | ✓ | ✓ | JSON shape — alias keys, `ly` rồi `it` rồi header/footer |
| 4 | `<fields>` | ✓ | ✓ | Per-field local guard (1 dòng/field) + null discipline |
| 5 | `<merchant>` | ✓ | ✓ | BRAND TEST cho `mn` + reject-category list (**verbatim chung**) |
| 6 | `<classify>` | ✓ | ✓ | 5-case `ly` + zone-local nulling matrix (**verbatim chung**) |
| 7 | `<items>` | ✓ | ✓ | Row def, when-in-doubt emit, merge variants, read-don't-compute, DROP-HARD, loop-bailout, examples |
| 8 | `<numbers>` | ✓ | ✓ | VND thousand-sep stripping + decimal-qty exception (**verbatim chung**) |
| 9 | `<output>` | ✓ | ✓ | Type strictness, no markdown fence |
| — | OCR text block (append) | — | ✓ | Paddle lines `x,y\|text` đính sau template |

---

## 5. Per-phase khác biệt (chỉ theo modality)

| Aspect | Phase 1 (vision) | Phase 2 (text-only) |
|---|---|---|
| **Input** | Raw ảnh, KHÔNG có OCR pre-step | PaddleOCR lines + bbox, KHÔNG có ảnh |
| **Trust model** | "WYSIWYG — read only printed pixels" | "Map don't re-judge — never invent absent from OCR" |
| **`mn` brand cue** | LARGEST/boldest, centered CAPS | First block với y1 nhỏ nhất |
| **Spatial reasoning** | Pixel position, font size, bold | y1 overlap, x2 right-edge = column |
| **Examples form** | Text rows (6) | Bbox `x,y\|text` (8) |
| **Tokens budget** | Dynamic via `/tokenize` | Dynamic via `/tokenize` + OCR trim loop (0.9× / iter, max 8) |
| **`ly` enum** | 5-case (giống) | 5-case (giống) |
| **`pm`** | **AS-PRINTED** (giống) | **AS-PRINTED** (giống) |
| **Schema field order** | `ly, it, mn…` (giống) | `ly, it, mn…` (giống) |

**`pm` AS-PRINTED ở cả hai:** validator `_strip_payment_method` ([receipt.py:215](../src/schemas/receipt.py#L215)) CHỈ strip + truncate, KHÔNG canonicalize. Nếu prompt canonicalize (CASH/CARD/…) ở một path mà path kia emit raw thì cùng một hoá đơn ra hai kết quả khác nhau. Giải pháp: cả hai emit verbatim; canonicalize (nếu cần) làm ở downstream, không ở prompt.

---

## 6. Anti-patterns đã validate (KHÔNG dùng)

| Anti-pattern | Tại sao tránh | Nguồn |
|---|---|---|
| `temperature=0` cho creative OCR | Có thể amplify repetition trên dense text; model card khuyến nghị 0.7/0.8/20/presence_penalty=1.5. Nhưng deterministic extraction nhiều nguồn dùng temp thấp — project chọn thấp hơn default (xem §7) | Qwen3-VL model card / HF OCR blog (đối nghịch — cần tự đo) |
| Schema nesting > 3 levels | Degrade latency + field accuracy | Alibaba structured output |
| `enable_thinking=False` + vLLM guided JSON trên Qwen3 | Emit malformed JSON | [vLLM #18819](https://github.com/vllm-project/vllm/issues/18819) |
| Mark fields `required` khi evidence có thể missing | "Required + missing" trigger fabricate | Alibaba |
| Negative instructions "do not hallucinate" | Hiệu quả thấp; dùng structural + per-field reject | arxiv 2408.03834 |
| `max_tokens` thấp khi structured output | Truncate JSON giữa chừng | Alibaba |
| Lặp detailed rules ở nhiều block / nhiều prompt | Bloat, hurt accuracy, gây phân kỳ vision↔text | Research general |
| Canonicalize `pm` ở prompt một path | Phân kỳ output vs path kia (validator không canonicalize) | Empirical (§5) |
| Ép prompt xuống ~1.8k bằng cách bỏ rule/example | Mất rule chống failure đã biết; không có eval harness bắt regression | §3.5 |

---

## 7. Sampling params chính thức

Từ Qwen3-VL-4B-Instruct model card + Unsloth guide:

```
temperature        = 0.7
top_p              = 0.8
top_k              = 20
repetition_penalty = 1.0
presence_penalty   = 1.5
max_new_tokens     = 16384  (vision tasks)
```

Project hiện dùng `temperature` thấp hơn (cấu hình [`src/core/config.py`](../src/core/config.py)) vì OCR receipt cần determinism cao. Nhiều nguồn OCR (HF blog) dùng temp=0 cho structured extraction. Nếu thấy model bail full-null / miss items, thử nâng về 0.5-0.7.

---

## 8. 5-case classification — verbatim từ prompt

Block `<classify>` (verbatim, [llm_extractor.py](../src/extraction/llm_extractor.py), [text_extractor.py](../src/extraction/text_extractor.py) — chỉ khác cue "IMAGE EDGES" vs "OCR line SEQUENCE"):

```
COMPLETE       — brand at top, item rows in middle, labeled totals at bottom.
MISSING-HEADER — topmost content is already an item/column-header/doc-type line; brand is GONE
                 → mn=ma=null; it / td / tt / sub / ta / pm / rc still emit.
MISSING-MIDDLE — header AND footer both present, but the item zone is blank/skipped
                 → it=[]; all other fields emit per evidence.
ITEM-ONLY      — tight crop of items only; NO header AND NO totals/payment
                 → every field except `it` and `cur` = null; `it` MUST NOT be empty.
MISSING-FOOTER — header + items present, bottommost is an item row, no totals/payment labels
                 → sub=ta=pm=td=tt=rc=null; mn / ma / it still emit.

A missing zone IS the correct cause for null there — that is the right answer, not a failure.
```

5 giá trị này khớp `Receipt.layout` enum ([receipt.py:105-107](../src/schemas/receipt.py#L105-L107)). `ly` được postprocessor **pop bỏ** trước khi trả client ([postprocessor.py:295](../src/extraction/postprocessor.py#L295)) — nó là reasoning scaffold thuần, không phải field client-facing. Câu chốt cuối re-frame nulling từ "failure mode" thành "expected behavior".

---

## 9. 5-zone layout overview — quick reference

| Zone | Nội dung | Field xuất |
|---|---|---|
| 1 | Brand line (stand-alone, font lớn) | `mn` |
| 2 | Address block (đường/phố/phường/quận/TP., phone, MST, website) | `ma` |
| 3 | Doc-meta (HOÁ ĐƠN BÁN LẺ / PHIẾU TÍNH TIỀN, receipt code, cashier, date, time) | `rc`, `td`, `tt` |
| 4 | Items table (STT / Tên hàng / SL / Đơn giá / KM / Thành tiền) — **dense nhất, extract FIRST** | `it[]` |
| 5 | Totals & footer (TẠM TÍNH → VAT → TỔNG CỘNG → payment → TIỀN KHÁCH ĐƯA → thank-you) | `sub`, `ta`, `pm`, `cur` |

Zone 4 (items) là dense evidence-bearing region — extract trước khi fill mn/ma/totals để giảm bleed name-as-mn / name-as-rc.

---

## 10. Verbatim DashScope phrases dùng trong prompt

| Verbatim phrase | Vị trí trong prompt |
|---|---|
| "Read only printed pixels, never invent" | `<task>` (vision) |
| "NEVER invent data absent from the OCR lines" | `<role>` (text) |
| "a clean null is the perfect answer" / "a clean per-field null beats invented content" | `<fields>` |
| "A missing zone IS the correct cause for null" | `<classify>` |
| "payment method AS PRINTED … Do NOT translate or canonicalize" | `<fields>` (pm, cả hai) |

---

## 11. Roadmap — improvements chưa làm

| Item | Lý do chưa làm |
|---|---|
| **Eval harness** (bộ ảnh + ground-truth, đo accuracy trước/sau prompt edit) | Chưa có → hiện không thể validate prompt change định lượng; là tiền đề cho mọi đợt cắt token sâu hơn |
| Đưa `completeness{header_visible, footer_visible}` vào schema | Refactor postprocessor + Pydantic Receipt lớn — prompt-only đạt ~80% lợi ích |
| Replace single-char uncertain → "?" (DashScope verbatim) | Schema là `string\|null`, chưa support "?" placeholder |
| Per-field description vào `guided_json` JSONSchema `description` | Cần test vLLM xgrammar/outlines có honor `description` không |
| Migrate `guided_json` → `structured_outputs={"json": ...}` (vLLM ≥0.12) | API cũ vẫn work; rename khi upgrade |
| Decimal-comma qty silent-drop (VN measured goods "0,706") | Known limit — fix ở fine-tune stage (xem memory) |

---

## 12. Đọc thêm

- [Alibaba Qwen-VL-OCR docs](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr) — verbatim extraction prompt + per-field description convention.
- [Alibaba Qwen structured-output rules](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output) — nullable types, field ordering, depth limit.
- [Qwen3-VL cookbooks](https://github.com/QwenLM/Qwen3-VL/tree/main/cookbooks) — official notebooks (ocr.ipynb, document_parsing.ipynb).
- [Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) — sampling params chính thức.
- [HF — OCR with open VLMs](https://huggingface.co/blog/ocr-open-models) — current-practice prompting cho open VLM OCR.
- [Prompt-length sweet spot](https://particula.tech/blog/optimal-prompt-length-ai-performance) — token band vs accuracy.
- [vLLM structured outputs docs](https://docs.vllm.ai/en/latest/features/structured_outputs/) — guided JSON API contract.
- [Target Prompting for VLM IE (arxiv 2408.03834)](https://arxiv.org/abs/2408.03834) — structural > negative; few-shot uniform.
</content>
</invoke>
