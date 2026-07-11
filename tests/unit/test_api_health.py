"""Unit tests for health and readiness endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tehran_house_price.api.health import register_health_routes
from tehran_house_price.api.model_loader import get_model_service
from tehran_house_price.settings import get_settings


class StubModelService:
    """Minimal stub for health endpoint tests."""

    def __init__(self, *, loaded: bool, artifact_path: Path | None) -> None:
        self._loaded = loaded
        self._artifact_path = artifact_path

    def is_loaded(self) -> bool:
        return self._loaded

    def get_artifact_path(self) -> Path | None:
        return self._artifact_path


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear cached settings before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _create_health_app(service: StubModelService) -> FastAPI:
    app = FastAPI()
    register_health_routes(app)
    app.dependency_overrides[get_model_service] = lambda: service
    return app


def test_health_live_returns_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The liveness endpoint should always report alive."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("DEEP_HEALTHCHECK_ENABLED", "true")

    app = _create_health_app(StubModelService(loaded=False, artifact_path=None))

    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_ready_returns_200_when_service_is_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Readiness should return 200 when all checks pass."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("DEEP_HEALTHCHECK_ENABLED", "true")
    monkeypatch.setenv("HEALTH_MIN_DISK_FREE_BYTES", "1")

    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_text("placeholder", encoding="utf-8")

    app = _create_health_app(
        StubModelService(
            loaded=True,
            artifact_path=artifact_path,
        )
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ready"
    assert payload["checks"]["model_loaded"] is True
    assert payload["checks"]["artifact_exists"] is True
    assert payload["checks"]["disk_ok"] is True


def test_health_ready_returns_503_when_model_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Readiness should fail when the model is not loaded."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("DEEP_HEALTHCHECK_ENABLED", "true")
    monkeypatch.setenv("HEALTH_MIN_DISK_FREE_BYTES", "1")

    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_text("placeholder", encoding="utf-8")

    app = _create_health_app(
        StubModelService(
            loaded=False,
            artifact_path=artifact_path,
        )
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    payload = response.json()

    assert response.status_code == 503
    assert payload["status"] == "not_ready"
    assert payload["checks"]["model_loaded"] is False
