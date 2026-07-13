"""
Module tiện ích ghi log cho dự án Receipt OCR Mapping.

Cung cấp:
- Ghi log ra Terminal (Console) và ghi vào tệp (file)
- Xoay vòng file (Log rotation)
- Tô màu đầu ra trên console
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


class LogColors:
    """Mã màu ANSI cho màn hình Terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DEBUG = "\033[36m"
    INFO = "\033[32m"
    WARNING = "\033[33m"
    ERROR = "\033[31m"
    CRITICAL = "\033[35m"
    TIMESTAMP = "\033[90m"
    MODULE = "\033[94m"


class ColoredFormatter(logging.Formatter):
    """Trình định dạng tùy chỉnh có thêm hỗ trợ màu cho Terminal."""

    FORMATS = {
        logging.DEBUG: f"{LogColors.TIMESTAMP}%(asctime)s{LogColors.RESET} | {LogColors.DEBUG}%(levelname)-8s{LogColors.RESET} | {LogColors.MODULE}%(name)s{LogColors.RESET} | %(message)s",
        logging.INFO: f"{LogColors.TIMESTAMP}%(asctime)s{LogColors.RESET} | {LogColors.INFO}%(levelname)-8s{LogColors.RESET} | {LogColors.MODULE}%(name)s{LogColors.RESET} | %(message)s",
        logging.WARNING: f"{LogColors.TIMESTAMP}%(asctime)s{LogColors.RESET} | {LogColors.WARNING}%(levelname)-8s{LogColors.RESET} | {LogColors.MODULE}%(name)s{LogColors.RESET} | %(message)s",
        logging.ERROR: f"{LogColors.TIMESTAMP}%(asctime)s{LogColors.RESET} | {LogColors.ERROR}%(levelname)-8s{LogColors.RESET} | {LogColors.MODULE}%(name)s{LogColors.RESET} | %(message)s",
        logging.CRITICAL: f"{LogColors.TIMESTAMP}%(asctime)s{LogColors.RESET} | {LogColors.CRITICAL}%(levelname)-8s{LogColors.RESET} | {LogColors.MODULE}%(name)s{LogColors.RESET} | %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class JSONFormatter(logging.Formatter):
    """Trình định dạng JSON cho log file."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = True,
) -> logging.Logger:
    """Thiết lập logger với console + file output.

    Idempotent: nếu logger cùng `name` đã có handlers (đã setup trước đó), trả
    về ngay. Tránh re-add file handler mỗi lần `get_logger(name)` được gọi —
    RotatingFileHandler không bị deduplicate bởi stdlib, mỗi instance giữ
    riêng file descriptor.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

    if file_output:
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            log_file = f"{name.replace('.', '_')}.log"

        log_path = log_dir / log_file

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, level.upper()))

        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Lấy hoặc tạo nhanh một logger với các thuộc tính mặc định."""
    return setup_logger(name, level=level)
