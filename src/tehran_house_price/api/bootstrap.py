"""Startup bootstrap for the FastAPI application.

This module ensures model artifacts are available on disk before the API
starts serving requests. In local and Docker Compose environments the
artifacts are mounted from the host. In cloud deployments (Render,
Hugging Face Spaces, etc.) the artifacts are downloaded on first startup
from public URLs configured via environment variables.

Design:
- Idempotent: if files already exist on disk, do nothing.
- Opt-in: without configured URLs, this module is a no-op.
- Defensive: download failures raise a clear error so the platform
  restarts the container rather than serving a broken API.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from tehran_house_price.api.model_loader import (
    DEFAULT_METADATA_FILENAME,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_MODELS_SUBDIR,
)
from tehran_house_price.settings import get_settings
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import project_root

logger = get_logger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 60


class ArtifactDownloadError(RuntimeError):
    """Raised when a required model artifact cannot be downloaded."""


def _default_models_dir() -> Path:
    """Return the canonical models directory under the project root."""
    return project_root() / "artifacts" / DEFAULT_MODELS_SUBDIR


def _download_file(url: str, destination: Path) -> None:
    """Download a single file from url to destination atomically.

    The download is written to a temporary sibling file first and then
    renamed, so a partial download cannot leave a corrupt artifact on disk.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    logger.info("downloading artifact | url=%s | dest=%s", url, destination)

    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise ArtifactDownloadError(
                    f"download failed with HTTP {response.status} for {url}"
                )
            with tmp_path.open("wb") as fh:
                shutil.copyfileobj(response, fh)
    except ArtifactDownloadError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise ArtifactDownloadError(f"failed to download {url}: {exc}") from exc

    tmp_path.replace(destination)
    logger.info(
        "artifact downloaded | dest=%s | size=%d bytes",
        destination,
        destination.stat().st_size,
    )


def ensure_model_artifacts(models_dir: Path | None = None) -> None:
    """Ensure model artifact files exist on disk, downloading if needed.

    Parameters
    ----------
    models_dir:
        Directory where model files should live. Defaults to
        artifacts/models under the project root.

    Behavior
    --------
    - If both files already exist, this function is a no-op.
    - If files are missing and download URLs are configured via settings,
      the files are downloaded from those URLs.
    - If files are missing and no URLs are configured, this function logs
      a warning and returns. The API startup will then fail cleanly when
      ModelService.load() cannot find the artifact.

    Raises
    ------
    ArtifactDownloadError
        If a download is attempted and fails.
    """
    settings = get_settings()
    target_dir = models_dir or _default_models_dir()

    model_path = target_dir / DEFAULT_MODEL_FILENAME
    metadata_path = target_dir / DEFAULT_METADATA_FILENAME

    targets: list[tuple[Path, str | None]] = [
        (model_path, settings.artifact_download_url),
        (metadata_path, settings.artifact_metadata_download_url),
    ]

    any_downloaded = False

    for path, url in targets:
        if path.exists():
            logger.debug("artifact already present | path=%s", path)
            continue

        if not url:
            logger.warning("artifact missing and no download URL configured | path=%s", path)
            continue

        _download_file(url, path)
        any_downloaded = True

    if any_downloaded:
        logger.info("model artifact bootstrap complete | dir=%s", target_dir)
    else:
        logger.info("model artifact bootstrap skipped | all files present or unconfigured")
