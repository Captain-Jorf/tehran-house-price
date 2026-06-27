"""
Kaggle ingestion.

از Kaggle API استفاده می‌کنیم تا dataset را دانلود و در data/raw/kaggle/
بنویسیم. raw data immutable است؛ هر چیزی در پیپ‌لاین cleaning بعداً
به interim می‌رود.

برای اجرا:
    python -m tehran_house_price.data.ingest_kaggle
    python -m tehran_house_price.data.ingest_kaggle --force
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer

from tehran_house_price.settings import get_config, get_settings
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import ensure_dir, raw_dir

log = get_logger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=False)


def _setup_kaggle_env() -> None:
    """
    Push kaggle credentials into os.environ before importing the kaggle module.

    Supports both auth styles:
      - new: KAGGLE_API_TOKEN (KGAT_...)
      - old: KAGGLE_USERNAME + KAGGLE_KEY (from kaggle.json)
    """
    settings = get_settings()

    if settings.kaggle_api_token:
        os.environ["KAGGLE_API_TOKEN"] = settings.kaggle_api_token
        return

    if settings.kaggle_username and settings.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key
        return

    raise RuntimeError(
        "Kaggle credentials missing. set KAGGLE_API_TOKEN (new) or "
        "KAGGLE_USERNAME + KAGGLE_KEY (old) in .env"
    )


def _target_dir() -> Path:
    cfg = get_config()
    subdir = cfg["data"]["kaggle"]["raw_subdir"]
    return ensure_dir(raw_dir() / subdir)


def _is_already_downloaded(target: Path) -> bool:
    # very simple check: any csv file present means we have something
    return any(target.glob("*.csv"))


def download_dataset(force: bool = False) -> Path:
    """
    Download the Tehran house price dataset from Kaggle into data/raw/kaggle/.

    Returns the directory where files were placed.
    """
    cfg = get_config()
    dataset_ref = cfg["data"]["kaggle"]["dataset"]
    target = _target_dir()

    if _is_already_downloaded(target) and not force:
        log.info("dataset already exists at %s. use --force to re-download", target)
        return target

    if force and target.exists():
        log.info("force mode: cleaning %s", target)
        # only delete contents, keep the directory itself
        for item in target.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    _setup_kaggle_env()

    # NOTE: importing kaggle at module top would try to authenticate at
    # import time, which is bad if creds are missing. so we do it lazily.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    log.info("downloading dataset '%s' to %s", dataset_ref, target)
    api.dataset_download_files(
        dataset_ref,
        path=str(target),
        unzip=True,
        quiet=False,
    )

    files = sorted(p.name for p in target.iterdir() if p.is_file())
    log.info("downloaded %d files: %s", len(files), files)

    return target


@app.command()
def main(
    force: bool = typer.Option(False, "--force", "-f", help="re-download even if files exist"),
) -> None:
    """CLI entrypoint for kaggle ingestion."""
    try:
        target = download_dataset(force=force)
        typer.echo(f"done. files are in: {target}")
    except Exception as e:
        log.error("ingestion failed: %s", e)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
