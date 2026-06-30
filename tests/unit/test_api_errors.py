"""Unit tests for global exception handlers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tehran_house_price.api.app import create_app
from tehran_house_price.api.model_loader import get_model_service


class _ExplodingModelService:
    """Stub that raises errors of a configurable type on predict."""

    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    def is_loaded(self) -> bool:
        return True

    def get_metadata(self) -> dict[str, Any]:
        return {}

    def get_artifact_path(self) -> Any:
        return None

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        raise self._exception


class _LoadedStubService:
    """Loaded stub that returns a constant prediction."""

    def __init__(self, constant: float = 100_000_000.0) -> None:
        self._constant = constant

    def is_loaded(self) -> bool:
        return True

    def get_metadata(self) -> dict[str, Any]:
        return {}

    def get_artifact_path(self) -> Any:
        return None

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.full(shape=len(rows), fill_value=self._constant, dtype=np.float64)


def _build_app_with_service(service: Any) -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: service
    return app


def _make_test_client(app: FastAPI) -> TestClient:
    """Build a TestClient that lets custom handlers convert 5xx errors.

    By default starlette re-raises server-side exceptions inside TestClient
    so pytest can see the original traceback. We disable that here so our
    unhandled_exception_handler gets a chance to produce a 500 response.
    """
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "district": "Pasdaran",
        "area_m2": 120.0,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }


@pytest.fixture
def value_error_client() -> Iterator[TestClient]:
    """Client whose model service raises ValueError on predict."""
    service = _ExplodingModelService(ValueError("missing required features"))
    yield _make_test_client(_build_app_with_service(service))


@pytest.fixture
def runtime_error_client() -> Iterator[TestClient]:
    """Client whose model service raises an unexpected RuntimeError."""
    service = _ExplodingModelService(RuntimeError("xgboost exploded"))
    yield _make_test_client(_build_app_with_service(service))


@pytest.fixture
def loaded_client() -> Iterator[TestClient]:
    """Client with a healthy loaded stub, used for validation tests."""
    yield _make_test_client(_build_app_with_service(_LoadedStubService()))


def test_value_error_returns_400_with_structured_payload(
    value_error_client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    response = value_error_client.post("/predict", json=valid_payload)

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "bad_request"
    assert "missing required features" in body["message"]
    assert body["status_code"] == 400
    assert body["details"] == []


def test_unhandled_exception_returns_500_with_safe_message(
    runtime_error_client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    response = runtime_error_client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_server_error"
    assert body["message"] == "An unexpected error occurred"
    assert body["status_code"] == 500
    # Critical: no traceback or internal details leak to the client.
    assert "xgboost exploded" not in body["message"]


def test_validation_error_returns_structured_details(
    loaded_client: TestClient,
) -> None:
    """422 validation errors must include per-field details."""
    bad_payload = {
        "district": "",
        "area_m2": -5,
        "rooms": -1,
        "has_parking": "yes",
        "has_storage": True,
        "has_elevator": True,
    }
    response = loaded_client.post("/predict", json=bad_payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["status_code"] == 422
    assert len(body["details"]) >= 4
    for detail in body["details"]:
        assert "field" in detail
        assert "message" in detail
        assert "type" in detail


def test_http_exception_returns_structured_payload() -> None:
    """503 from unloaded model should still follow ErrorResponse shape."""

    class _UnloadedStub:
        def is_loaded(self) -> bool:
            return False

        def get_metadata(self) -> dict[str, Any]:
            return {}

        def get_artifact_path(self) -> Any:
            return None

        def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
            raise AssertionError("predict should not be called when unloaded")

    app = _build_app_with_service(_UnloadedStub())
    client = _make_test_client(app)

    response = client.post(
        "/predict",
        json={
            "district": "Pasdaran",
            "area_m2": 120.0,
            "rooms": 2,
            "has_parking": True,
            "has_storage": True,
            "has_elevator": True,
        },
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "http_error"
    assert body["message"] == "Model is not loaded"
    assert body["status_code"] == 503


def test_404_for_unknown_route_returns_structured_payload(
    loaded_client: TestClient,
) -> None:
    """Even built-in 404s should follow the ErrorResponse shape."""
    response = loaded_client.get("/no-such-endpoint")

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "http_error"
    assert body["status_code"] == 404
