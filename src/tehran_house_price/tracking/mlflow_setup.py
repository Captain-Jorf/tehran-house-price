"""MLflow setup and run context helpers.

Design goals:
- Local file-based tracking by default (no server required).
- Tracking is optional via the ``MLFLOW_TRACKING_ENABLED`` env var,
  so training pipelines remain runnable in environments where
  MLflow is undesirable (CI smoke runs, quick local iteration).
- Standard tags (git commit, package version, environment) are
  attached automatically to every run for reproducibility.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow

from tehran_house_price import __version__ as package_version
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import project_root

log = get_logger(__name__)

DEFAULT_EXPERIMENT_NAME = "tehran_house_price"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def is_tracking_enabled() -> bool:
    """Return True when MLflow tracking is enabled via env var.

    Defaults to True so that local runs are tracked by default.
    """
    raw = os.getenv("MLFLOW_TRACKING_ENABLED", "true")
    return raw.strip().lower() in _TRUE_VALUES


def get_tracking_uri() -> str:
    """Return the MLflow tracking URI to use.

    If ``MLFLOW_TRACKING_URI`` is set, it is honored as-is.
    Otherwise, fall back to a local file store under ``<project_root>/mlruns``.
    """
    custom = os.getenv("MLFLOW_TRACKING_URI")
    if custom:
        return custom

    mlruns_path = project_root() / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    return f"file:{mlruns_path.as_posix()}"


def _git_commit_sha() -> str:
    """Return the short git commit SHA, or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root(),
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return result.stdout.strip()[:12] or "unknown"


def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT_NAME) -> str | None:
    """Configure the tracking URI and ensure the experiment exists.

    Returns the experiment id, or None if tracking is disabled.
    """
    if not is_tracking_enabled():
        log.info("mlflow tracking disabled; skipping setup")
        return None

    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    log.info("mlflow tracking uri set | uri=%s", uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        log.info(
            "mlflow experiment created | name=%s | id=%s",
            experiment_name,
            experiment_id,
        )
    else:
        experiment_id = experiment.experiment_id
        log.info(
            "mlflow experiment found | name=%s | id=%s",
            experiment_name,
            experiment_id,
        )

    mlflow.set_experiment(experiment_name)
    return experiment_id


def _default_tags() -> dict[str, str]:
    return {
        "git_commit": _git_commit_sha(),
        "package_version": package_version,
        "env": os.getenv("APP_ENV", "dev"),
        "phase": "phase5",
    }


@contextmanager
def get_run_context(
    run_name: str,
    extra_tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun | None]:
    """Start an MLflow run with standard tags.

    Yields the active run object, or None when tracking is disabled.
    Disabling tracking allows callers to use the same ``with`` block
    unconditionally.
    """
    if not is_tracking_enabled():
        log.info("mlflow tracking disabled; yielding no-op run | name=%s", run_name)
        yield None
        return

    tags = _default_tags()
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        log.info(
            "mlflow run started | name=%s | run_id=%s",
            run_name,
            run.info.run_id,
        )
        try:
            yield run
        finally:
            log.info(
                "mlflow run finished | name=%s | run_id=%s",
                run_name,
                run.info.run_id,
            )
