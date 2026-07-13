"""Ablation harness (Phase 3) — chạy các variant trên dataset, đo proxy metrics.

Variants:
  raw          : baseline — decode → mô phỏng smart_resize vLLM (không preprocess)
  v2_full      : preprocessing config mặc định
  v2_no_orient / v2_no_localize / v2_no_zoom / v2_no_reflow / v2_no_gate
               : ablation tắt từng stage
  v2_photometric_on : bật photometric toàn cục (đo xem có đáng bật default không)

Chạy:  .venv/Scripts/python.exe -m src.preprocessing.eval.harness \
           --images images --out .claude/pharse/output/phase3_ablation.json
Ghi chú: cần GPU + paddleocr; mỗi variant × ảnh chạy det+rec để chấm điểm.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing.config import PipelineConfig, load_config
from src.preprocessing.contracts import Purpose
from src.preprocessing.detectors import build_services
from src.preprocessing.eval.metrics import ocr_proxy_metrics, simulate_smart_resize
from src.preprocessing.runner import Pipeline
from src.preprocessing.stages import decode_image


def _variant_configs(base: PipelineConfig) -> Dict[str, PipelineConfig]:
    def clone(mutate: Callable[[PipelineConfig], None]) -> PipelineConfig:
        cfg = copy.deepcopy(base)
        mutate(cfg)
        return cfg

    return {
        "v2_full": copy.deepcopy(base),
        "v2_no_orient": clone(lambda c: setattr(c.orient, "enabled", False)),
        "v2_no_localize": clone(lambda c: setattr(c.localize, "enabled", False)),
        "v2_no_zoom": clone(lambda c: setattr(c.fit, "target_text_h", 0.0)),
        "v2_no_reflow": clone(lambda c: setattr(c.fit, "reflow_enabled", False)),
        "v2_no_gate": clone(lambda c: setattr(c.gate, "enabled", False)),
        "v2_photometric_on": clone(lambda c: setattr(c.photometric, "enabled", True)),
    }


def run_harness(images_dir: str, out_path: str,
                variant_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    base_cfg = load_config()
    services = build_services(base_cfg)  # model dùng chung mọi variant

    files = sorted(p for p in Path(images_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    # Model chấm điểm (det+rec) — độc lập services của pipeline để đo được cả raw
    from paddleocr import TextDetection, TextRecognition
    det_scorer = TextDetection(model_name=base_cfg.services.det_model,
                               limit_side_len=base_cfg.services.det_limit_side_len,
                               limit_type=base_cfg.services.det_limit_type,
                               device=base_cfg.services.device)
    rec_scorer = TextRecognition(model_name=base_cfg.services.rec_model,
                                 device=base_cfg.services.device)

    pipelines = {
        name: Pipeline(cfg, services=services)
        for name, cfg in _variant_configs(base_cfg).items()
    }

    def produce(variant: str, data: bytes) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        if variant == "raw":
            img = decode_image(data)
            return img, {}
        res = pipelines[variant].run_bytes(data, purpose=Purpose.VLM)
        return res.image, {
            "verdict": res.verdict.value,
            "reject_reason": res.reject_reason,
            "route": res.meta.get("route"),
            "reflow_applied": res.meta.get("reflow_applied"),
            "legibility_zoom_ratio": res.meta.get("legibility_zoom_ratio"),
            "stage_ms": {k: v.get("elapsed_ms") for k, v in res.meta["stages"].items()},
        }

    variants = ["raw"] + list(pipelines.keys())
    if variant_filter:
        variants = [v for v in variants if v in variant_filter]

    rows: List[Dict[str, Any]] = []
    for path in files:
        data = path.read_bytes()
        for variant in variants:
            t0 = time.perf_counter()
            try:
                img, extra = produce(variant, data)
            except Exception as e:
                rows.append({"file": path.name, "variant": variant,
                             "error": f"{type(e).__name__}: {e}"})
                continue
            elapsed = (time.perf_counter() - t0) * 1000
            if img is None:
                rows.append({"file": path.name, "variant": variant,
                             "error": "decode_failed"})
                continue
            seen = simulate_smart_resize(
                img, min_pixels=base_cfg.fit.min_pixels,
                max_pixels=base_cfg.fit.max_pixels, factor=base_cfg.fit.block)
            metrics = ocr_proxy_metrics(seen, det_scorer, rec_scorer)
            metrics["resample_by_vllm"] = seen.shape[:2] != img.shape[:2]
            rows.append({"file": path.name, "variant": variant,
                         "latency_ms": round(elapsed), **metrics, **extra})
            print(f"{path.name[:14]:14s} {variant:18s} conf={metrics['rec_conf_mean']} "
                  f"boxes={metrics['n_boxes']} text_h={metrics['text_h_at_model_px']} "
                  f"{round(elapsed)}ms", flush=True)

    summary: Dict[str, Any] = {}
    for variant in variants:
        vrows = [r for r in rows if r["variant"] == variant and "error" not in r]
        if not vrows:
            continue
        confs = [r["rec_conf_mean"] for r in vrows if r.get("rec_conf_mean")]
        summary[variant] = {
            "n": len(vrows),
            "rec_conf_mean": round(float(np.mean(confs)), 4) if confs else None,
            "n_boxes_mean": round(float(np.mean([r["n_boxes"] for r in vrows])), 1),
            "n_chars_mean": round(float(np.mean([r["n_chars"] for r in vrows])), 1),
            "text_h_median": round(float(np.median(
                [r["text_h_at_model_px"] for r in vrows
                 if r.get("text_h_at_model_px")])), 1),
            "pixels_mean": round(float(np.mean([r["pixels"] for r in vrows]))),
            "latency_ms_mean": round(float(np.mean([r["latency_ms"] for r in vrows]))),
            "resampled_by_vllm": sum(1 for r in vrows if r.get("resample_by_vllm")),
        }

    result = {"summary": summary, "rows": rows}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default=".claude/pharse/output/phase3_ablation.json")
    ap.add_argument("--variants", default=None,
                    help="danh sách variant, phẩy phân cách (mặc định: tất cả)")
    args = ap.parse_args()
    run_harness(args.images, args.out,
                variant_filter=args.variants.split(",") if args.variants else None)


if __name__ == "__main__":
    main()
