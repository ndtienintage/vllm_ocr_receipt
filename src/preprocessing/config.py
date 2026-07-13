"""Config của preprocessing (ADR-10): YAML default (defaults/default.yaml) →
env override (PP2_<SECTION>_<FIELD>) → legacy env alias (giữ tương thích
docker-compose hiện có: VLLM_IMAGE_MAX_PIXELS, VLLM_IMAGE_MIN_PIXELS,
PADDLE_DEVICE).

Mọi ngưỡng đều có mặt trong YAML — không magic number chôn trong code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

_DEFAULT_YAML = Path(__file__).parent / "defaults" / "default.yaml"


@dataclass
class ServicesConfig:
    device: str = "gpu"
    det_model: str = "PP-OCRv5_server_det"
    det_limit_side_len: int = 1920
    det_limit_type: str = "max"
    doc_ori_model: str = "PP-LCNet_x1_0_doc_ori"
    textline_model: str = "PP-LCNet_x1_0_textline_ori"
    rec_model: str = "PP-OCRv5_server_rec"
    enable_doc_ori: bool = True
    enable_textline: bool = True
    enable_rec: bool = True   # chỉ dùng cho gate variants + eval


@dataclass
class QualityConfig:
    enabled: bool = True
    norm_width: int = 1000          # chuẩn hoá blur metric về cùng thang đo
    digital_bypass: bool = True
    digital_flat_min: float = 0.40  # tỉ lệ pixel |Laplacian|≤1 (render phẳng)
    digital_bg_peak_min: float = 0.20  # tỉ lệ pixel thuộc 1 bin màu nền trội


@dataclass
class DetectConfig:
    enabled: bool = True
    min_polys: int = 3  # dưới ngưỡng → coi như không có evidence hình học


@dataclass
class OrientConfig:
    enabled: bool = True
    cardinal_enabled: bool = True
    vert_ratio_trigger: float = 0.6     # ≥ → polys nói sideways
    vert_ratio_confident: float = 0.9   # ≥ + doc_ori đồng thuận → khỏi brute-force
    ratio_min_elongation: float = 1.5   # lọc poly vuông khỏi vert_ratio
    textline_sample: int = 5
    textline_min_side_px: int = 10
    redetect_after_cardinal: bool = True
    deskew_enabled: bool = True
    deskew_min_deg: float = 0.5
    deskew_max_deg: float = 45.0        # near-90 "tilt" là artifact (Phase 0)
    deskew_min_elongation: float = 2.0
    deskew_min_samples: int = 3


@dataclass
class LocalizeConfig:
    enabled: bool = True
    padding_px: int = 20
    trust_min_polys: int = 7
    trust_min_coverage: float = 0.12
    aspect_inflation_max: float = 1.5
    high_poly_min: int = 50


@dataclass
class GateConfig:
    enabled: bool = True
    reject_enabled: bool = True
    # UNREADABLE — calibrate để 26/26 ảnh Phase 0 PASS (ảnh tệ nhất còn đọc
    # được: lapvar 115 / text_h 13). Reject chỉ dành cho case cực đoan.
    unreadable_min_polys: int = 3
    unreadable_text_h: float = 12.0     # text_h ĐẠT ĐƯỢC sau budget solver (px poly)
    unreadable_blur: float = 80.0       # lapvar @norm_width
    # MARGINAL — kích hoạt multi-variant selection
    marginal_text_h: float = 20.0
    marginal_blur: float = 250.0
    # Ablation Phase 3: photometric thua minimal 6/6 ảnh MARGINAL, tốn
    # 240ms-2.2s/ảnh → OFF mặc định; bật khi dataset có bóng đổ/loá thật.
    variants_enabled: bool = False
    variant_sample: int = 12            # số poly crop chấm điểm mỗi variant
    variant_min_gain: float = 0.02      # conf trung bình phải hơn ≥ 2 điểm % mới đổi


@dataclass
class PhotometricConfig:
    enabled: bool = False               # ADR-09: OFF mặc định, chỉ bật khi ablation thắng
    bg_kernel_frac: float = 0.125       # kernel ước lượng nền = frac × cạnh ngắn
    clahe_clip: float = 2.0
    clahe_grid: int = 8


@dataclass
class FitConfig:
    enabled: bool = True
    max_pixels: int = 4_194_304         # = VLLM_IMAGE_MAX_PIXELS (trần area vLLM)
    min_pixels: int = 1_048_576         # = VLLM_IMAGE_MIN_PIXELS (sàn area vLLM)
    max_side: int = 3584
    block: int = 32                     # Qwen3-VL patch16 × merge2 (verify Phase 0 §4)
    # Ablation Phase 3: zoom-quá-sàn ÂM trên proxy (-0.109 conf, +31% pixel)
    # → OFF mặc định (0.0 = tắt; bật lại = 32.0 khi có field-level eval).
    zoom_target_text_h: float = 0.0
    thrift_target_text_h: float = 32.0  # chữ > target → downscale (conf-neutral, -13-26% pixel)
    max_upscale: float = 2.5
    token_thrift: bool = True
    pad_to_min_pixels: bool = True
    pad_color: int = 255
    # Ablation Phase 3: reflow thua trên chính ảnh nó kích hoạt (conf -0.015,
    # chars -39%) theo proxy OCR → OFF mặc định; giữ code để re-test với VLM
    # thật (proxy không có aspect-weakness và prompt hint của VLM).
    reflow_enabled: bool = False
    reflow_aspect_trigger: float = 4.0
    reflow_gain_min: float = 1.2        # reflow phải nâng text_h_out ≥ 1.2× mới áp
    reflow_max_cols: int = 4
    reflow_min_col_aspect: float = 1.0  # cột không được bẹt hơn vuông
    reflow_separator_px: int = 10
    reflow_split_search_frac: float = 0.2  # bán kính tìm whitespace quanh split lý thuyết


@dataclass
class PipelineConfig:
    services: ServicesConfig = field(default_factory=ServicesConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    orient: OrientConfig = field(default_factory=OrientConfig)
    localize: LocalizeConfig = field(default_factory=LocalizeConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    photometric: PhotometricConfig = field(default_factory=PhotometricConfig)
    fit: FitConfig = field(default_factory=FitConfig)


# Env cũ → (section, field). Chỉ alias những khoá deploy đang set thật.
_LEGACY_ALIASES = {
    "VLLM_IMAGE_MAX_PIXELS": ("fit", "max_pixels"),
    "VLLM_IMAGE_MIN_PIXELS": ("fit", "min_pixels"),
    "PADDLE_DEVICE": ("services", "device"),
}


def _coerce(raw: str, target_type: type) -> Any:
    if target_type is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    return raw


def _apply_dict(cfg_section: Any, data: Mapping[str, Any], origin: str) -> None:
    valid = {f.name: f.type for f in fields(cfg_section)}
    for key, value in data.items():
        if key not in valid:
            raise ValueError(f"Config key không tồn tại: {origin}.{key}")
        setattr(cfg_section, key, value)


def load_config(yaml_path: Optional[str] = None,
                env: Optional[Mapping[str, str]] = None) -> PipelineConfig:
    """default.yaml → (yaml_path nếu có) → env PP2_* → legacy alias."""
    env = os.environ if env is None else env
    cfg = PipelineConfig()

    for path in filter(None, [_DEFAULT_YAML if _DEFAULT_YAML.exists() else None,
                              Path(yaml_path) if yaml_path else None]):
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for section, values in data.items():
            if not hasattr(cfg, section):
                raise ValueError(f"Config section không tồn tại: {section}")
            _apply_dict(getattr(cfg, section), values or {}, section)

    for section_field in fields(cfg):
        section = getattr(cfg, section_field.name)
        for f in fields(section):
            env_key = f"PP2_{section_field.name.upper()}_{f.name.upper()}"
            if env_key in env:
                setattr(section, f.name, _coerce(env[env_key], type(getattr(section, f.name))))

    for env_key, (section_name, field_name) in _LEGACY_ALIASES.items():
        if f"PP2_{section_name.upper()}_{field_name.upper()}" in env:
            continue  # PP2_* thắng alias
        if env_key in env:
            section = getattr(cfg, section_name)
            setattr(section, field_name,
                    _coerce(env[env_key], type(getattr(section, field_name))))
    return cfg
