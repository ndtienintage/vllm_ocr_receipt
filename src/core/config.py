"""
Module cấu hình cho Receipt OCR API.

Quản lý cấu hình tập trung cho vLLM OCR server, loaded từ .env.
Single source of truth — tất cả module khác import từ đây, không gọi os.getenv() trực tiếp.

Về timeout (simplified — chỉ giữ 1 lớp ở request path):
- REQUEST_TIMEOUT: lớp DUY NHẤT cho request path, bao TOÀN BỘ vòng đời request
  (decode ảnh + queue wait + preprocess + LLM + postprocess).
  Mặc định 300s. Vượt mốc → RequestTimeoutError (HTTP 408).
- Backpressure: dùng asyncio.Semaphore (capacity = GLOBAL_CONCURRENCY). Acquire
  KHÔNG có timeout riêng — chỉ bị bound bởi REQUEST_TIMEOUT bao ngoài.
- Probe timeouts (health/ready) là chuyện ops, không nằm trong request path —
  vẫn giữ riêng (5s/10s) trong llm_client.py.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root để resolve các đường dẫn tương đối (vd file pattern loại bỏ).
# config.py ở src/core/, parents[2] là root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EXCLUDE_FILE = "config/exclude_item_patterns.txt"
_DEFAULT_EXCLUDE_MERCHANT_FILE = "config/exclude_merchant_patterns.txt"
_DEFAULT_MERCHANT_CANONICAL_FILE = "config/merchant_canonical_names.txt"


def _load_exclude_patterns(path_str: str) -> tuple[str, ...]:
    """
    Đọc danh sách pattern từ file text.

    Format file:
      - Mỗi dòng 1 pattern. Hỗ trợ cả CSV trên 1 dòng (cách bằng dấu phẩy).
      - Dòng bắt đầu bằng '#' = comment, bỏ qua.
      - Dòng trống bỏ qua.

    File không tồn tại / lỗi đọc → trả tuple rỗng + cảnh báo ra stderr,
    KHÔNG ném exception (để pipeline vẫn boot được).
    """
    if not path_str:
        return ()
    path = Path(path_str)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    if not path.exists():
        print(
            f"[config] EXCLUDE_ITEM_PATTERNS file không tồn tại: {path} "
            f"— bỏ qua bước lọc theo pattern.",
            file=sys.stderr,
        )
        return ()

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[config] Không đọc được {path}: {e}", file=sys.stderr)
        return ()

    patterns: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Hỗ trợ CSV trên 1 dòng (tiện copy-paste từ env cũ)
        for token in line.split(","):
            token = token.strip()
            if token:
                patterns.append(token)
    return tuple(patterns)


@dataclass
class VLLMConfig:
    """
    Cấu hình kết nối vLLM server.

    Token budget (heuristic-based, không guess):
      - max_model_len      : phải khớp --max-model-len khi start vLLM. Config-
                             driven (override qua VLLM_MAX_MODEL_LEN env).
      - max_tokens         : UPPER CAP cho output tokens. Actual max_tokens dùng
                             trong request = min(cap, max_model_len - input_tokens
                             - context_safety_margin). Tính động trước mỗi call.
      - context_safety_margin: buffer cho chat-template overhead + heuristic
                               estimate noise (~10-20% overestimate intentional).
      - min_output_tokens  : floor — nếu budget output < min sau khi tính prompt,
                             fallback path TRIM OCR tail-first đến khi fit; primary
                             path log ERROR và để vLLM tự reject (rồi fail-safe).
    """
    base_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("VLLM_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct-FP8"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("VLLM_MAX_TOKENS", "4092")))
    temperature: float = field(default_factory=lambda: float(os.getenv("VLLM_TEMPERATURE", "0.05")))
    max_model_len: int = field(default_factory=lambda: int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")))
    context_safety_margin: int = field(default_factory=lambda: int(os.getenv("VLLM_CONTEXT_SAFETY_MARGIN", "128")))
    # Receipt JSON serialize ngắn nhất ~150 tokens; 256 đủ cho schema rỗng + vài
    # field. Vượt mức này thì response KHÔNG còn ý nghĩa → trim input thay vì gửi.
    min_output_tokens: int = field(default_factory=lambda: int(os.getenv("VLLM_MIN_OUTPUT_TOKENS", "64")))
    hallu_stream_enabled: bool = field(default_factory=lambda: os.getenv("VLLM_HALLU_STREAM_ENABLED", "true").lower() == "true")
    hallu_stream_interval_chars: int = field(default_factory=lambda: int(os.getenv("VLLM_HALLU_STREAM_INTERVAL_CHARS", "500")))

    # Image visual token budget (Qwen3-VL: 1 token = 32×32 px = 1024 px):
    #   min_pixels=1_048_576 (1024 tokens floor) — SÀN UPSCALE. Ảnh gốc thực tế
    #     chỉ ~1000-2170px; sau crop receipt hẹp còn ~0.25-0.7M px (chữ rất nhỏ).
    #     vLLM smart_resize PHÓNG TO ảnh dưới sàn này lên 1024 tok → ViT nhiều
    #     patch hơn phủ lên mỗi nét chữ → đọc chữ nhỏ tốt hơn (upscale low-res
    #     OCR là FEATURE, không phải bug; lý lẽ downscale-only chỉ áp cho client
    #     preprocess, KHÔNG áp upscale ở vLLM). Giá trị tối ưu (1.05M vs 1.5-2.0M)
    #     cần A/B trên Qwen3-VL thật — đừng hạ thấp với data low-res hiện tại.
    #   max_pixels=4_194_304 (4096 tokens cap) — TRẦN DIỆN TÍCH dùng chung cho
    #     CẢ preprocess (image_utils area cap) LẪN vLLM → 2 tầng resize đồng bộ,
    #     không double-resample. Ngân sách context cho phép 12284 − prompt(~3435)
    #     − output(4096) − margin(128) ≈ 4625 image tokens (≈4.74M px); 4.0M dùng
    #     ~86% trần, chừa headroom. KHÔNG đẩy lên 5.24M (5120 tok) — 5120+3435+
    #     4096 = 12651 > 12284 → tràn context. (Area cap chỉ kích hoạt khi ảnh gốc
    #     > ~2000px; data hiện tại ~1000px không chạm — future-proof.)
    image_min_pixels: int = field(default_factory=lambda: int(os.getenv("VLLM_IMAGE_MIN_PIXELS", "1048576")))
    image_max_pixels: int = field(default_factory=lambda: int(os.getenv("VLLM_IMAGE_MAX_PIXELS", "4194304")))
    image_jpeg_quality: int = field(default_factory=lambda: int(os.getenv("VLLM_IMAGE_JPEG_QUALITY", "100")))


@dataclass
class ImageQualityConfig:
    """Nguỡng cho khối metadata `image_quality` gắn vào response (METADATA-ONLY —
    KHÔNG can thiệp kết quả extract). Cho downstream biết ảnh thuộc loại "hệ thống
    đọc không nổi" thay vì nhận kết quả thiếu trong im lặng.

    Mọi biến đổi ảnh (resize/crop/orient/reflow/zoom) đã chuyển sang
    `src.preprocessing` (+ `defaults/default.yaml`). Class này CHỈ giữ 2 cờ chất
    lượng mà orchestrator (`extraction/processing.py::_build_image_quality`) đọc.

    quality_min_input_short_side: cạnh NGẮN ảnh đầu vào (trước xử lý) dưới ngưỡng
      → low_res_input=true (dấu hiệu upstream nén ảnh; chi tiết mất không khôi
      phục được). Dùng cạnh ngắn vì batch nén cap width ~1000px. 0 = tắt cờ.
    quality_min_text_height_px: median chiều cao poly text (đo sau deskew+crop, DPI
      nguồn) dưới ngưỡng → low_legibility=true. Default 16.0 calibrate trên batch
      24 ảnh (poly DBNet nở ~1.5-2× glyph thật). 0 = tắt cờ.
    """
    quality_min_input_short_side: int = field(default_factory=lambda: int(os.getenv("QUALITY_MIN_INPUT_SHORT_SIDE", "1200")))
    quality_min_text_height_px: float = field(default_factory=lambda: float(os.getenv("QUALITY_MIN_TEXT_HEIGHT_PX", "16.0")))


@dataclass
class PaddleTextConfig:
    """
    Cấu hình PaddleOCR cho fallback OCR path (full det+rec pipeline).

    Pipeline: ảnh raw → PaddleOCR (det+rec+orientation) → text+bbox lines →
    text-only LLM mapping vào Receipt schema. Chạy CHỈ khi VLM primary fail
    (truncated / hallucinated / empty).

    Env vars chính (xem .env để biết toàn bộ):
      PADDLE_TEXT_ENABLED        : kill-switch toàn cục, false → fallback path tắt.
      PADDLE_TEXT_REC_MODEL      : PP-OCRv5_server_rec (default) | mobile_rec.
      PADDLE_TEXT_DET_MODEL      : PP-OCRv5_server_det (default).
      PADDLE_TEXT_LANG           : "vi" cho Vietnamese.
      PADDLE_TEXT_DEVICE         : kế thừa PADDLE_DEVICE nếu không set.
      PADDLE_TEXT_USE_DOC_ORI    : doc-level orientation classifier (fallback nhận
                                    raw bytes, phải tự xoay/lật).
      PADDLE_TEXT_USE_TEXTLINE_ORI : textline orientation classifier per crop.
      PADDLE_TEXT_USE_DOC_UNWARPING : UVDoc unwarping cho ảnh cong (default false,
                                       heavy + dễ over-correct receipt phẳng).

    Detection tuning (xem src/extraction/paddle_text.py docstring cho rationale):
      PADDLE_TEXT_DET_THRESH         : default 0.2 (PaddleOCR default 0.3) — bắt text mờ.
      PADDLE_TEXT_DET_BOX_THRESH     : default 0.5 (PaddleOCR default 0.6).
      PADDLE_TEXT_DET_UNCLIP_RATIO   : default 1.2 — siết bbox bám sát text, tránh
                                        bao lan sang text row kế trên receipt nén dày
                                        (item rows cách nhau 1-2 px). PaddleOCR default
                                        1.5; 1.8 cũ bao trọn diacritic VN nhưng tạo
                                        bbox overlap giữa các rows trên receipt siêu
                                        thị 60+ items → row-merge sai.
      PADDLE_TEXT_DET_LIMIT_SIDE_LEN : default 1920 — cap longest side.
      PADDLE_TEXT_DET_LIMIT_TYPE     : "max" (cap longest) | "min" (upscale short).

    Post-processing:
      PADDLE_TEXT_MIN_SCORE      : filter ngưỡng confidence (default 0.5).
      PADDLE_TEXT_MAX_LINES      : cap số line output (default 400, chống pathological).
      PADDLE_TEXT_MAX_TEXT_CHARS : per-line text cap (default 500).
      PADDLE_TEXT_ROW_OVERLAP    : Y-overlap min để 2 poly merge thành 1 row.
                                    Default 0.85 — chỉ merge khi 2 box overlap rất cao
                                    (cùng visual row). 0.5 cũ quá lỏng trên receipt
                                    dày (rows cách 1-2 px) → gộp 5-30 dòng thật vào
                                    1 line → LLM không decode được.
      PADDLE_TEXT_COL_GAP_RATIO  : gap-x ratio (× median-height) tách 2-col layout.
      PADDLE_TEXT_BBOX_QUANT     : chia bbox coords cho hệ số (giảm token).
    """
    enabled: bool = field(default_factory=lambda: os.getenv("PADDLE_TEXT_ENABLED", "true").lower() == "true")
    rec_model: str = field(default_factory=lambda: os.getenv("PADDLE_TEXT_REC_MODEL", "PP-OCRv5_server_rec"))
    det_model: str = field(default_factory=lambda: os.getenv("PADDLE_TEXT_DET_MODEL", "PP-OCRv5_server_det"))
    lang: str = field(default_factory=lambda: os.getenv("PADDLE_TEXT_LANG", "vi"))
    device: str = field(default_factory=lambda: os.getenv("PADDLE_TEXT_DEVICE", os.getenv("PADDLE_DEVICE", "gpu")))
    use_doc_ori: bool = field(default_factory=lambda: os.getenv("PADDLE_TEXT_USE_DOC_ORI", "true").lower() == "true")
    use_textline_ori: bool = field(default_factory=lambda: os.getenv("PADDLE_TEXT_USE_TEXTLINE_ORI", "true").lower() == "true")
    use_doc_unwarping: bool = field(default_factory=lambda: os.getenv("PADDLE_TEXT_USE_DOC_UNWARPING", "false").lower() == "true")
    # Detection (DB) tuning
    det_thresh: float = field(default_factory=lambda: float(os.getenv("PADDLE_TEXT_DET_THRESH", "0.2")))
    det_box_thresh: float = field(default_factory=lambda: float(os.getenv("PADDLE_TEXT_DET_BOX_THRESH", "0.5")))
    det_unclip_ratio: float = field(default_factory=lambda: float(os.getenv("PADDLE_TEXT_DET_UNCLIP_RATIO", "1.2")))
    det_limit_side_len: int = field(default_factory=lambda: int(os.getenv("PADDLE_TEXT_DET_LIMIT_SIDE_LEN", "1920")))
    det_limit_type: str = field(default_factory=lambda: os.getenv("PADDLE_TEXT_DET_LIMIT_TYPE", "max").lower())
    # Post-processing & format
    min_score: float = field(default_factory=lambda: float(os.getenv("PADDLE_TEXT_MIN_SCORE", "0.5")))
    max_lines: int = field(default_factory=lambda: int(os.getenv("PADDLE_TEXT_MAX_LINES", "400")))
    max_text_chars: int = field(default_factory=lambda: int(os.getenv("PADDLE_TEXT_MAX_TEXT_CHARS", "500")))
    row_overlap: float = field(default_factory=lambda: float(os.getenv("PADDLE_TEXT_ROW_OVERLAP", "0.85")))
    col_gap_ratio: float = field(default_factory=lambda: float(os.getenv("PADDLE_TEXT_COL_GAP_RATIO", "5.0")))
    # max(1, ...) để chống bbox_quant=0 gây ZeroDivision khi format.
    bbox_quant: int = field(default_factory=lambda: max(1, int(os.getenv("PADDLE_TEXT_BBOX_QUANT", "2"))))


def _env_int_floored(name: str, default: int, *, min_val: int) -> int:
    """Parse int env với floor an toàn. Invalid / under-min → default + stderr warn.

    Floor (min_val) tránh edge case như RUN_MIN=1 (mọi single char match) hoặc
    NGRAM_MAX_LEN=1 (cụm 1 char trùng char-run check, dư thừa).
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        print(f"[config] {name}={raw!r} không phải int — dùng default {default}", file=sys.stderr)
        return default
    if v < min_val:
        print(f"[config] {name}={v} < min {min_val} — dùng default {default}", file=sys.stderr)
        return default
    return v


@dataclass
class HalluConfig:
    """
    Ngưỡng cho hallucination detector (online streaming + offline parsed dict).

    Dùng cho cả `detect_streaming_hallu` (online, mid-decode) lẫn
    `detect_hallucination` (offline, parsed result). Xem
    src/extraction/hallucination_detector.py docstring để hiểu trade-off.

    Env vars:
      HALLU_DUP_ITEM_RUN_MIN : số item liên tiếp cùng tên → loop signal (period=1).
                                Default 10 (was 7) — siêu thị bulk receipt và set menu
                                lễ tết có thể có 7-9 dòng cùng tên normalize hợp lệ
                                (đặc biệt khi VLM strip variant tail). 10 vẫn bắt
                                được decoder loop thật (loop thường emit cùng tên
                                hàng chục lần). Loosen 5 → false-positive cao.
      HALLU_CYCLE_MAX_PERIOD : kích thước block tối đa khi quét cyclic loop
                                period=2..N. Default 5 — đủ phủ pattern AEON
                                [item → KHUYẾN MÃI → CK THẺ] (period=3) và biến thể
                                4-5 dòng (item + barcode + qty + 2 discount).
                                Cost detector tuyến tính theo period — tăng 10 chỉ
                                +5% overhead nhưng false-positive cao trên receipt
                                có sub-block lặp hợp lệ.
      HALLU_CYCLE_REPEATS_MIN: số lần block period≥2 lặp liên tiếp tối thiểu để
                                flag loop. Default 3 — true cyclic decoder loop
                                gần như không tồn tại tự nhiên ở period≥2 (vd
                                A,B,A,B,A,B = 6 items rất khó hợp lệ trên receipt).
                                Giảm 2 quá nhạy. Tăng 4-5 tốn thêm decode budget.
      HALLU_CHAR_RUN_MIN     : số ký tự non-digit/non-ws lặp liên tiếp.
                                Default 6 — chặt đủ bỏ qua "MMMM" trong brand,
                                bắt được decoder stuck phun ký tự đơn.
      HALLU_NGRAM_MAX_LEN    : quét cụm 2..N. Default 4 — bao phủ "abcabc", "abcdabcd".
                                Tăng N → cost regex tăng tuyến tính.
      HALLU_NGRAM_REPEAT_MIN : số lần cụm n-gram lặp liên tiếp tối thiểu.
                                Default 4 — pattern "abcabcabcabc".
                                Giảm 3 quá nhạy (pattern tự nhiên bị flag).

    Floor 2 cho min_val cho mọi tham số — tránh edge case 1 ký tự trùng mọi thứ.
    """
    dup_item_run_min: int = field(default_factory=lambda: _env_int_floored("HALLU_DUP_ITEM_RUN_MIN", 10, min_val=2))
    cycle_max_period: int = field(default_factory=lambda: _env_int_floored("HALLU_CYCLE_MAX_PERIOD", 5, min_val=2))
    cycle_repeats_min: int = field(default_factory=lambda: _env_int_floored("HALLU_CYCLE_REPEATS_MIN", 3, min_val=2))
    char_run_min: int = field(default_factory=lambda: _env_int_floored("HALLU_CHAR_RUN_MIN", 6, min_val=3))
    ngram_max_len: int = field(default_factory=lambda: _env_int_floored("HALLU_NGRAM_MAX_LEN", 4, min_val=2))
    ngram_repeat_min: int = field(default_factory=lambda: _env_int_floored("HALLU_NGRAM_REPEAT_MIN", 4, min_val=2))


@dataclass
class PostprocessConfig:
    """Cấu hình hậu xử lý — danh sách văn bản loại bỏ khỏi items / merchant_name.

    Patterns được đọc từ FILE thay vì env var — dễ comment, dễ chia nhóm,
    dễ version control.

    exclude_item_patterns:
      Item có name CHỨA pattern → bị xóa khỏi items[].
      Default: config/exclude_item_patterns.txt
      Override env: EXCLUDE_ITEM_PATTERNS_FILE=/path/khac.txt

    exclude_merchant_patterns:
      merchant_name CHỨA pattern → null hoá (giữ nguyên các field khác).
      Default: config/exclude_merchant_patterns.txt (rỗng — opt-in từng pattern).
      Override env: EXCLUDE_MERCHANT_PATTERNS_FILE=/path/khac.txt

    merchant_canonical_names:
      Danh sách tên chuẩn để remap merchant_name. Khi tên VLM trả về match
      (substring sau normalize HOẶC SequenceMatcher.ratio ≥ ngưỡng) với 1
      canonical name → REPLACE bằng canonical đó. Mục đích: gom biến thể
      ("co.opmart", "Co.opMart") về 1 tên chuẩn ngay tại API.
      Default: config/merchant_canonical_names.txt (rỗng — opt-in từng tên).
      Override env: MERCHANT_CANONICAL_NAMES_FILE=/path/khac.txt

    merchant_canonical_min_ratio:
      Ngưỡng SequenceMatcher.ratio() tối thiểu để công nhận match (0.0..1.0).
      Default 0.9 (lệch 1 chữ trên 10 ký tự vẫn match). Hạ < 0.85 dễ
      false-positive trên tên ngắn. Substring match KHÔNG dùng ngưỡng này.
      Override env: MERCHANT_CANONICAL_MIN_RATIO=0.9
    """
    exclude_item_patterns: tuple[str, ...] = field(
        default_factory=lambda: _load_exclude_patterns(
            os.getenv("EXCLUDE_ITEM_PATTERNS_FILE", _DEFAULT_EXCLUDE_FILE)
        )
    )
    exclude_merchant_patterns: tuple[str, ...] = field(
        default_factory=lambda: _load_exclude_patterns(
            os.getenv("EXCLUDE_MERCHANT_PATTERNS_FILE", _DEFAULT_EXCLUDE_MERCHANT_FILE)
        )
    )
    merchant_canonical_names: tuple[str, ...] = field(
        default_factory=lambda: _load_exclude_patterns(
            os.getenv("MERCHANT_CANONICAL_NAMES_FILE", _DEFAULT_MERCHANT_CANONICAL_FILE)
        )
    )
    merchant_canonical_min_ratio: float = field(
        default_factory=lambda: float(os.getenv("MERCHANT_CANONICAL_MIN_RATIO", "0.9"))
    )


@dataclass
class AppConfig:
    """Cấu hình chính cho OCR API — single source of truth.

    Các module pipeline KHÔNG được gọi os.getenv() trực tiếp. Mọi env var phải
    được khai báo trong dataclass tương ứng + đọc qua `config.<section>.<field>`.
    """
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    image_quality: ImageQualityConfig = field(default_factory=ImageQualityConfig)
    paddle_text: PaddleTextConfig = field(default_factory=PaddleTextConfig)
    hallu: HalluConfig = field(default_factory=HalluConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    concurrency: int = field(default_factory=lambda: int(os.getenv("GLOBAL_CONCURRENCY", "24")))
    # Lớp timeout DUY NHẤT — bao toàn bộ vòng đời request (decode + queue + xử lý).
    # Vượt → RequestTimeoutError (HTTP 408). Backpressure khi quá tải dùng concurrency.
    request_timeout: float = field(default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "300.0")))
    max_image_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_BYTES", "50000000")))


# Instance cấu hình toàn cục
config = AppConfig()
