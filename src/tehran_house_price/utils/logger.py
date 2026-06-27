"""
Logger setup.

Logging config از configs/logging.yaml خوانده می‌شود. اگر فایل وجود
نداشت یا load نشد، یک basicConfig ساده fallback می‌گذاریم.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from tehran_house_price.utils.paths import configs_dir, ensure_dir, logs_dir

_CONFIGURED = False


def setup_logging(config_path: Path | None = None) -> None:
    """Configure logging once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    # make sure logs/ exists before RotatingFileHandler is created
    ensure_dir(logs_dir())

    cfg_path = config_path or (configs_dir() / "logging.yaml")

    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger. Call setup_logging() first (or it will fallback)."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
