"""
Path helpers.

ایده ساده است: project root را یک بار پیدا کنیم و بقیه path ها را نسبت
به آن بسازیم تا فرقی نکند کد را از کجا اجرا می‌کنی.
"""

from pathlib import Path


def project_root() -> Path:
    """Return absolute path to project root."""
    # this file lives at: <root>/src/tehran_house_price/utils/paths.py
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return project_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def interim_dir() -> Path:
    return data_dir() / "interim"


def processed_dir() -> Path:
    return data_dir() / "processed"


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def logs_dir() -> Path:
    return project_root() / "logs"


def configs_dir() -> Path:
    return project_root() / "configs"


def ensure_dir(path: Path) -> Path:
    """Create directory if missing. Returns the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
