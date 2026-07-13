"""Các stage của preprocessing. Thứ tự chạy khai báo ở runner.STAGE_ORDER."""

from src.preprocessing.stages.ingest import IngestStage, decode_image, exif_orientation_tag
from src.preprocessing.stages.quality import QualityStage
from src.preprocessing.stages.detect import DetectStage
from src.preprocessing.stages.orient import OrientStage
from src.preprocessing.stages.localize import LocalizeStage
from src.preprocessing.stages.gate import GateStage
from src.preprocessing.stages.photometric import PhotometricStage, normalize_illumination
from src.preprocessing.stages.fit import FitStage

__all__ = [
    "IngestStage", "QualityStage", "DetectStage", "OrientStage",
    "LocalizeStage", "GateStage", "PhotometricStage", "FitStage",
    "decode_image", "exif_orientation_tag", "normalize_illumination",
]
