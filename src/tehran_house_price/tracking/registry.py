"""mlflow model registry helpers.

simple wrappers around mlflow's register_model + transition_model_version_stage.
keeps the rest of the codebase ignorant of mlflow registry internals.
"""

from __future__ import annotations

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from tehran_house_price.tracking.mlflow_setup import (
    get_tracking_uri,
    is_tracking_enabled,
)
from tehran_house_price.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_REGISTERED_MODEL_NAME = "tehran_house_price_xgb"

# stages mlflow supports
VALID_STAGES = {"None", "Staging", "Production", "Archived"}


def register_model_from_run(
    run_id: str,
    artifact_subpath: str = "sklearn_model",
    name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> str | None:
    """register a logged sklearn model from a specific run.

    returns the new version string (e.g. "3") or None if tracking is off.
    """
    if not is_tracking_enabled():
        log.info("tracking disabled; skipping model registration")
        return None

    model_uri = f"runs:/{run_id}/{artifact_subpath}"
    log.info("registering model | uri=%s | name=%s", model_uri, name)

    try:
        result = mlflow.register_model(model_uri=model_uri, name=name)
    except MlflowException as exc:
        log.warning("model registration failed | error=%s", exc)
        return None

    log.info(
        "model registered | name=%s | version=%s",
        result.name,
        result.version,
    )
    return result.version


def promote_model(
    version: str,
    stage: str = "Staging",
    name: str = DEFAULT_REGISTERED_MODEL_NAME,
    archive_existing: bool = True,
) -> bool:
    """move a model version to a target stage.

    archive_existing=True moves any previous versions in that stage to Archived.
    returns True on success.
    """
    if not is_tracking_enabled():
        log.info("tracking disabled; skipping promotion")
        return False

    if stage not in VALID_STAGES:
        raise ValueError(f"invalid stage '{stage}'. must be one of {VALID_STAGES}")

    client = MlflowClient(tracking_uri=get_tracking_uri())

    try:
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
    except MlflowException as exc:
        log.warning("promotion failed | error=%s", exc)
        return False

    log.info(
        "model promoted | name=%s | version=%s | stage=%s",
        name,
        version,
        stage,
    )
    return True


def get_latest_version(
    name: str = DEFAULT_REGISTERED_MODEL_NAME,
    stage: str | None = None,
) -> str | None:
    """fetch the latest version of a registered model, optionally filtered by stage."""
    if not is_tracking_enabled():
        return None

    client = MlflowClient(tracking_uri=get_tracking_uri())

    try:
        if stage:
            versions = client.get_latest_versions(name=name, stages=[stage])
        else:
            versions = client.get_latest_versions(name=name)
    except MlflowException as exc:
        log.warning("could not fetch latest version | error=%s", exc)
        return None

    if not versions:
        return None
    return versions[0].version
