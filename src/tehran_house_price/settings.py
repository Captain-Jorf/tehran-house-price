"""
Application settings.

دو منبع داریم:
  1. configs/base.yaml  -> non-secret config (paths, model params, ...)
  2. .env               -> secrets and environment overrides

Pydantic settings کار خواندن env را راحت می‌کند. yaml را خودمان load
می‌کنیم چون pydantic-settings برای yaml به این سادگی stable نیست.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tehran_house_price.utils.paths import configs_dir, project_root


class AppSettings(BaseSettings):
    """Environment-based settings. Loaded from .env or process env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = Field(default="dev")
    log_level: str = Field(default="INFO")

    # kaggle
    kaggle_username: str | None = None
    kaggle_key: str | None = None
    kaggle_api_token: str | None = None

    # observability
    observability_enabled: bool = Field(default=True)
    request_logging_enabled: bool = Field(default=True)
    prometheus_enabled: bool = Field(default=True)
    deep_healthcheck_enabled: bool = Field(default=True)
    request_id_header_name: str = Field(default="X-Request-ID")
    health_min_disk_free_bytes: int = Field(default=100_000_000)

    # prediction logging
    prediction_logging_enabled: bool = Field(default=False)
    prediction_log_db_url: str | None = Field(default=None)

    # deployment: artifact download URLs
    # If unset, ensure_model_artifacts() is a no-op and the app relies on
    # local files (dev or docker compose volume mount).
    artifact_download_url: str | None = Field(default=None)
    artifact_metadata_download_url: str | None = Field(default=None)


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """Load configs/base.yaml as a plain dict."""
    cfg_path = path or (configs_dir() / "base.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached environment settings."""
    return AppSettings()


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Cached YAML config dict."""
    return load_yaml_config()


# convenience
ROOT = project_root()
