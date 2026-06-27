"""
Kaggle ingestion.

Uses kagglehub when possible. Falls back to manual instructions if
consent is required (which is the case for this particular dataset).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from tehran_house_price.settings import get_config, get_settings
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import ensure_dir, raw_dir

log = get_logger(__name__)


MANUAL_DOWNLOAD_HINT = """\
Could not download dataset automatically (likely consent required).

To proceed manually:
  1. Open: https://www.kaggle.com/datasets/{dataset}
  2. Accept terms / click Download
  3. Extract the zip into: {target}
  4. Re-run this command
"""


def _setup_kaggle_env() -> None:
    settings = get_settings()
    if settings.kaggle_api_token:
        os.environ["KAGGLE_API_TOKEN"] = settings.kaggle_api_token
        return
    if settings.kaggle_username and settings.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key
        return
    raise RuntimeError(
        "Kaggle credentials missing. set KAGGLE_API_TOKEN or "
        "KAGGLE_USERNAME + KAGGLE_KEY in .env"
    )


def _target_dir() -> Path:
    cfg = get_config()
    subdir = cfg["data"]["kaggle"]["raw_subdir"]
    return ensure_dir(raw_dir() / subdir)


def _list_csv_files(target: Path) -> list[Path]:
    return sorted(target.glob("*.csv"))


def _is_already_downloaded(target: Path) -> bool:
    return len(_list_csv_files(target)) > 0


def _copy_files(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    for item in src_dir.rglob("*"):
        if item.is_file():
            dest = dst_dir / item.name
            shutil.copy2(item, dest)
            copied.append(dest.name)
    return copied


def _clean_target(target: Path) -> None:
    for item in target.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.name != ".gitkeep":
            item.unlink()


def download_dataset(force: bool = False) -> Path:
    cfg = get_config()
    dataset_ref = cfg["data"]["kaggle"]["dataset"]
    target = _target_dir()

    has_data = _is_already_downloaded(target)
    log.info("target=%s | has_data=%s | force=%s", target, has_data, force)

    if has_data and not force:
        files = [p.name for p in _list_csv_files(target)]
        log.info("dataset already present: %s", files)
        return target

    _setup_kaggle_env()

    try:
        import kagglehub

        log.info("attempting kagglehub download for '%s'", dataset_ref)
        cache_path = Path(kagglehub.dataset_download(dataset_ref))
        log.info("kagglehub cached files at %s", cache_path)
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("403", "forbidden", "consent", "permission")):
            hint = MANUAL_DOWNLOAD_HINT.format(dataset=dataset_ref, target=target)
            log.error(hint)
            raise RuntimeError("automatic download failed - see instructions above") from e
        raise

    log.info("download ok; refreshing %s", target)
    _clean_target(target)
    copied = _copy_files(cache_path, target)
    log.info("copied %d files: %s", len(copied), copied)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Tehran house price dataset from Kaggle.")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="re-download even if files exist",
    )
    args = parser.parse_args()

    try:
        target = download_dataset(force=args.force)
        print(f"done. files are in: {target}")
        return 0
    except Exception as e:
        log.error("ingestion failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
