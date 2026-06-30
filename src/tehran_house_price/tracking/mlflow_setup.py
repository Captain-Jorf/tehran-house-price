"""mlflow setup + run context."""

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
    raw = os.getenv("MLFLOW_TRACKING_ENABLED", "true")
    return raw.strip().lower() in _TRUE_VALUES


def get_tracking_uri() -> str:
    custom = os.getenv("MLFLOW_TRACKING_URI")
    if custom:
        return custom

    mlruns_path = project_root() / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    return f"file:{mlruns_path.as_posix()}"


def _git_commit_sha() -> str:
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
    nested: bool = False,
) -> Iterator[mlflow.ActiveRun | None]:
    """start an mlflow run. pass nested=True for child runs."""
    if not is_tracking_enabled():
        log.info("mlflow tracking disabled; yielding no-op run | name=%s", run_name)
        yield None
        return

    tags = _default_tags()
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
        log.info(
            "mlflow run started | name=%s | run_id=%s | nested=%s",
            run_name,
            run.info.run_id,
            nested,
        )
        try:
            yield run
        finally:
            log.info(
                "mlflow run finished | name=%s | run_id=%s",
                run_name,
                run.info.run_id,
            )
