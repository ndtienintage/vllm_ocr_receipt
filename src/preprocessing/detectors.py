"""Service layer — interface cho model CV + adapter PaddleOCR.

Stage KHÔNG import paddle trực tiếp: mọi model đi qua Services để (a) test
bằng fake không cần GPU, (b) lazy-init + lock (Paddle predictor không
thread-safe; GPU chia với vLLM — giữ đúng bài học v1), (c) một điểm duy nhất
parse output format PaddleOCR (dict vs object).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TextDetector(Protocol):
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Trả polys (N,4,2) float32 hoặc None khi không detect được."""
        ...


class DocOrientation(Protocol):
    def label(self, image: np.ndarray) -> Optional[str]:
        """'0' | '90' | '180' | '270' | None. Convention PaddleOCR: label = số
        độ ảnh ĐANG bị xoay CW so với upright."""
        ...


class TextlineOrientation(Protocol):
    def is_upside(self, crop: np.ndarray) -> Optional[bool]:
        """True nếu textline crop bị lộn ngược 180°. None khi không phán được."""
        ...


class TextScorer(Protocol):
    def scores(self, crops: Sequence[np.ndarray]) -> List[float]:
        """Recognition confidence [0..1] cho từng crop (objective function cho
        multi-variant — KHÔNG dùng làm bộ đọc)."""
        ...


@dataclass
class Services:
    detector: Optional[TextDetector] = None
    doc_ori: Optional[DocOrientation] = None
    textline: Optional[TextlineOrientation] = None
    scorer: Optional[TextScorer] = None


# ── Paddle adapters ───────────────────────────────────────────────────────────

def _result_field(res0, *names):
    for name in names:
        if isinstance(res0, dict):
            v = res0.get(name)
        else:
            v = getattr(res0, name, None)
        if v is not None:
            return v
    return None


class PaddleTextDetector:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._model = None
        self._init_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._failed = False

    def _get(self):
        if self._model is not None or self._failed:
            return self._model
        with self._init_lock:
            if self._model is None and not self._failed:
                try:
                    from paddleocr import TextDetection
                    self._model = TextDetection(
                        model_name=self._cfg.det_model,
                        limit_side_len=self._cfg.det_limit_side_len,
                        limit_type=self._cfg.det_limit_type,
                        device=self._cfg.device,
                    )
                    logger.info("preprocessing TextDetection init: %s/%s",
                                self._cfg.det_model, self._cfg.device)
                except Exception as e:
                    self._failed = True
                    logger.error("preprocessing TextDetection init failed: %s", e)
        return self._model

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        model = self._get()
        if model is None:
            return None
        with self._predict_lock:
            result = model.predict(input=image)
        if not result:
            return None
        polys = _result_field(result[0], "dt_polys", "polys")
        if polys is None or len(polys) == 0:
            return None
        return np.asarray(polys, dtype=np.float32)


class PaddleDocOrientation:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._model = None
        self._init_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._failed = False

    def _get(self):
        if self._model is not None or self._failed:
            return self._model
        with self._init_lock:
            if self._model is None and not self._failed:
                try:
                    from paddleocr import DocImgOrientationClassification
                    self._model = DocImgOrientationClassification(
                        model_name=self._cfg.doc_ori_model, device=self._cfg.device)
                    logger.info("preprocessing DocOrientation init: %s", self._cfg.doc_ori_model)
                except Exception as e:
                    self._failed = True
                    logger.warning("preprocessing DocOrientation init failed: %s", e)
        return self._model

    def label(self, image: np.ndarray) -> Optional[str]:
        model = self._get()
        if model is None:
            return None
        try:
            with self._predict_lock:
                result = model.predict(input=image)
            if not result:
                return None
            names = _result_field(result[0], "label_names")
            if names:
                label = str(names[0])
                for d in ("270", "180", "90", "0"):
                    if d in label:
                        return d
            ids = _result_field(result[0], "class_ids")
            if ids is not None and len(ids) > 0:
                return {0: "0", 1: "90", 2: "180", 3: "270"}.get(int(ids[0]))
            return None
        except Exception as e:
            logger.warning("doc orientation predict failed: %s", e)
            return None


class PaddleTextlineOrientation:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._model = None
        self._init_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._failed = False

    def _get(self):
        if self._model is not None or self._failed:
            return self._model
        with self._init_lock:
            if self._model is None and not self._failed:
                try:
                    from paddleocr import TextLineOrientationClassification
                    self._model = TextLineOrientationClassification(
                        model_name=self._cfg.textline_model, device=self._cfg.device)
                    logger.info("preprocessing TextlineOrientation init: %s",
                                self._cfg.textline_model)
                except Exception as e:
                    self._failed = True
                    logger.warning("preprocessing TextlineOrientation init failed: %s", e)
        return self._model

    def is_upside(self, crop: np.ndarray) -> Optional[bool]:
        model = self._get()
        if model is None:
            return None
        try:
            with self._predict_lock:
                result = model.predict(input=crop)
            if not result:
                return None
            names = _result_field(result[0], "label_names")
            if names:
                return "180" in str(names[0])
            ids = _result_field(result[0], "class_ids")
            if ids is not None and len(ids) > 0:
                return int(ids[0]) == 1
            return None
        except Exception as e:
            logger.warning("textline orientation predict failed: %s", e)
            return None


class PaddleTextScorer:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._model = None
        self._init_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._failed = False

    def _get(self):
        if self._model is not None or self._failed:
            return self._model
        with self._init_lock:
            if self._model is None and not self._failed:
                try:
                    from paddleocr import TextRecognition
                    self._model = TextRecognition(
                        model_name=self._cfg.rec_model, device=self._cfg.device)
                    logger.info("preprocessing TextRecognition init: %s", self._cfg.rec_model)
                except Exception as e:
                    self._failed = True
                    logger.warning("preprocessing TextRecognition init failed: %s", e)
        return self._model

    def scores(self, crops: Sequence[np.ndarray]) -> List[float]:
        model = self._get()
        if model is None or not crops:
            return []
        out: List[float] = []
        try:
            with self._predict_lock:
                results = model.predict(input=list(crops))
            for res in results or []:
                score = _result_field(res, "rec_score", "score")
                out.append(float(score) if score is not None else 0.0)
            return out
        except Exception as e:
            logger.warning("text scorer predict failed: %s", e)
            return []


def build_services(cfg) -> Services:
    """Services Paddle-backed theo config (model chỉ load khi dùng lần đầu)."""
    scfg = cfg.services
    return Services(
        detector=PaddleTextDetector(scfg),
        doc_ori=PaddleDocOrientation(scfg) if scfg.enable_doc_ori else None,
        textline=PaddleTextlineOrientation(scfg) if scfg.enable_textline else None,
        scorer=PaddleTextScorer(scfg) if scfg.enable_rec else None,
    )
