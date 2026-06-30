"""Safe logging helpers for MLflow runs.

These helpers wrap raw ``mlflow.*`` calls so that:

* All log calls are no-ops when tracking is disabled.
* Params are stringified and truncated to MLflow's hard limits.
* Metrics with non-finite values are silently dropped instead of crashing
  a training run.
* Artifacts are only logged when the source path exists.

The goal is that a caller can sprinkle ``log_*`` calls inside training
code without worrying about MLflow being configured or about a single
bad value bringing the whole run down.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlflow

from tehran_house_price.tracking.mlflow_setup import is_tracking_enabled
from tehran_house_price.utils.logger import get_logger

log = get_logger(__name__)

# MLflow has hard caps on param length. Keep a small safety margin.
_MAX_PARAM_VALUE_LEN = 500


def _stringify(value: Any) -> str:
    """Convert a value to string and truncate for MLflow param storage."""
    text = "" if value is None else str(value)
    if len(text) > _MAX_PARAM_VALUE_LEN:
        text = text[: _MAX_PARAM_VALUE_LEN - 3] + "..."
    return text


def log_params(params: dict[str, Any]) -> None:
    """Log a flat dict of params. Skipped silently when tracking is off."""
    if not is_tracking_enabled():
        return
    if not params:
        return

    safe = {key: _stringify(value) for key, value in params.items()}
    try:
        mlflow.log_params(safe)
    except Exception as exc:
        log.warning("mlflow log_params failed | error=%s", exc)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log numeric metrics. Non-finite values are dropped."""
    if not is_tracking_enabled():
        return
    if not metrics:
        return

    clean: dict[str, float] = {}
    for name, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            log.debug("metric skipped (not numeric) | name=%s", name)
            continue
        if not math.isfinite(number):
            log.debug("metric skipped (non-finite) | name=%s | value=%s", name, number)
            continue
        clean[name] = number

    if not clean:
        return

    try:
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:
        log.warning("mlflow log_metrics failed | error=%s", exc)


def log_artifact_file(path: str | Path, artifact_subdir: str | None = None) -> None:
    """Log a single file as an MLflow artifact.

    No-op when tracking is disabled or the file does not exist.
    """
    if not is_tracking_enabled():
        return

    file_path = Path(path)
    if not file_path.exists():
        log.debug("artifact skipped (missing) | path=%s", file_path)
        return

    try:
        mlflow.log_artifact(str(file_path), artifact_path=artifact_subdir)
    except Exception as exc:
        log.warning("mlflow log_artifact failed | path=%s | error=%s", file_path, exc)


def log_sklearn_model(model: Any, artifact_path: str = "model") -> None:
    """Log a fitted sklearn estimator (or compatible Pipeline) as an MLflow model.

    No-op when tracking is disabled.
    """
    if not is_tracking_enabled():
        return

    try:
        mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
    except Exception as exc:
        log.warning("mlflow log_sklearn_model failed | error=%s", exc)


def set_tags(tags: dict[str, str]) -> None:
    """Attach extra tags to the active run. Skipped when tracking is off."""
    if not is_tracking_enabled():
        return
    if not tags:
        return

    safe = {key: _stringify(value) for key, value in tags.items()}
    try:
        mlflow.set_tags(safe)
    except Exception as exc:
        log.warning("mlflow set_tags failed | error=%s", exc)
