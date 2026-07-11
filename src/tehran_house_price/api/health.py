"""Health and readiness routes for production observability."""

from __future__ import annotations

from shutil import disk_usage
from typing import Any

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import JSONResponse

from tehran_house_price.api.model_loader import ModelService, get_model_service
from tehran_house_price.settings import ROOT, get_settings

router = APIRouter(tags=["system"])


def _build_readiness_payload(service: ModelService) -> tuple[dict[str, Any], bool]:
    """Build readiness details and final readiness status."""
    settings = get_settings()
    artifact_path = service.get_artifact_path()
    artifact_exists = artifact_path.exists() if artifact_path is not None else False
    model_loaded = service.is_loaded()

    usage = disk_usage(ROOT)
    disk_free_bytes = int(usage.free)
    disk_ok = disk_free_bytes >= settings.health_min_disk_free_bytes

    ready = model_loaded and artifact_exists and disk_ok

    payload: dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "checks": {
            "model_loaded": model_loaded,
            "artifact_exists": artifact_exists,
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "disk_ok": disk_ok,
            "disk_free_bytes": disk_free_bytes,
            "disk_min_required_bytes": settings.health_min_disk_free_bytes,
        },
    }

    return payload, ready


@router.get(
    "/health/live",
    summary="Liveness probe",
)
def health_live() -> dict[str, str]:
    """Return whether the API process is alive."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    response_model=None,
)
def health_ready(
    service: ModelService = Depends(get_model_service),
) -> JSONResponse:
    """Return whether the API is ready to serve traffic.

    Readiness currently checks:
    - model loaded in memory
    - model artifact path exists
    - enough free disk space
    """
    payload, ready = _build_readiness_payload(service)
    status_code = 200 if ready else 503
    return JSONResponse(status_code=status_code, content=payload)


def register_health_routes(app: FastAPI) -> None:
    """Attach health-related routes to the application once."""
    if getattr(app.state, "health_routes_registered", False):
        return

    settings = get_settings()
    if not settings.observability_enabled or not settings.deep_healthcheck_enabled:
        return

    app.include_router(router)
    app.state.health_routes_registered = True
