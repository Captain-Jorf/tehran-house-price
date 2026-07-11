"""Unit tests for API request middleware."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tehran_house_price.api.middleware import register_request_middleware
from tehran_house_price.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear cached settings before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _create_app() -> FastAPI:
    app = FastAPI()
    register_request_middleware(app)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_request_middleware_adds_request_id_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each response should include a request id header."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("REQUEST_LOGGING_ENABLED", "true")

    app = _create_app()

    with TestClient(app) as client:
        response = client.get("/ping")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]


def test_request_middleware_preserves_incoming_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a caller sends a request id, it should be propagated back."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("REQUEST_LOGGING_ENABLED", "true")

    app = _create_app()

    with TestClient(app) as client:
        response = client.get("/ping", headers={"X-Request-ID": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_request_middleware_logs_completed_requests(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Completed requests should emit a structured JSON log line."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("REQUEST_LOGGING_ENABLED", "true")

    app = _create_app()

    with (
        caplog.at_level(logging.INFO, logger="tehran_house_price.api.middleware"),
        TestClient(app) as client,
    ):
        response = client.get("/ping")

    assert response.status_code == 200

    messages = [record.message for record in caplog.records]
    assert any('"event": "http_request_completed"' in message for message in messages)
    assert any('"path": "/ping"' in message for message in messages)


def test_request_middleware_supports_disabled_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling request logging should not disable request id propagation."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("REQUEST_LOGGING_ENABLED", "false")

    app = _create_app()

    with TestClient(app) as client:
        response = client.get("/ping")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
