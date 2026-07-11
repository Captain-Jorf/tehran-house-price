"""Unit tests for Prometheus metrics integration."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tehran_house_price.api.metrics import (
    record_prediction,
    record_request_completion,
    record_request_exception,
    register_metrics_routes,
)
from tehran_house_price.api.middleware import register_request_middleware
from tehran_house_price.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear cached settings before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _create_metrics_app() -> FastAPI:
    app = FastAPI()
    register_request_middleware(app)
    register_metrics_routes(app)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        record_prediction(endpoint="/ping", model_name="test_model", count=1)
        return {"status": "ok"}

    return app


def test_metrics_endpoint_exposes_prometheus_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The metrics endpoint should expose Prometheus text format."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "true")

    app = _create_metrics_app()

    with TestClient(app) as client:
        ping_response = client.get("/ping")
        metrics_response = client.get("/metrics")

    assert ping_response.status_code == 200
    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in metrics_response.text
    assert "http_request_duration_seconds" in metrics_response.text
    assert "http_requests_in_progress" in metrics_response.text
    assert "http_request_exceptions_total" in metrics_response.text
    assert "model_predictions_total" in metrics_response.text


def test_metric_helpers_are_noops_when_metrics_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metric helpers must not raise when metrics are disabled."""
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "false")

    record_request_completion(
        method="GET",
        path="/disabled",
        status_code=200,
        duration_seconds=0.01,
    )
    record_request_exception(
        method="GET",
        path="/disabled",
        exception_type="RuntimeError",
    )
    record_prediction(
        endpoint="/disabled",
        model_name="disabled_model",
        count=1,
    )
