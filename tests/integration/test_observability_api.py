"""Integration tests for observability-enabled API routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tehran_house_price.api.app import create_app
from tehran_house_price.api.model_loader import get_model_service
from tehran_house_price.settings import get_settings


class StubModelService:
    """Minimal model service stub for observability integration tests."""

    def __init__(self, artifact_path: Path) -> None:
        self._artifact_path = artifact_path

    def is_loaded(self) -> bool:
        return True

    def get_artifact_path(self) -> Path:
        return self._artifact_path


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear cached settings before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_create_app_exposes_observability_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The application factory should expose middleware, health, and metrics."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("REQUEST_LOGGING_ENABLED", "true")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "true")
    monkeypatch.setenv("DEEP_HEALTHCHECK_ENABLED", "true")
    monkeypatch.setenv("HEALTH_MIN_DISK_FREE_BYTES", "1")

    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_text("placeholder", encoding="utf-8")

    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: StubModelService(artifact_path)

    client = TestClient(app)

    version_response = client.get("/version")
    live_response = client.get("/health/live")
    ready_response = client.get("/health/ready")
    metrics_response = client.get("/metrics")

    assert version_response.status_code == 200
    assert "X-Request-ID" in version_response.headers

    assert live_response.status_code == 200
    assert live_response.json()["status"] == "alive"

    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ready"

    assert metrics_response.status_code == 200
    assert "http_requests_total" in metrics_response.text
