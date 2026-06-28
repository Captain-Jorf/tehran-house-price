"""Unit tests for FastAPI endpoints using TestClient.

These tests build an isolated app instance per test and inject a stub
ModelService via dependency overrides. The real model artifact is never
loaded, which keeps the suite fast and deterministic.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tehran_house_price import __version__
from tehran_house_price.api.app import create_app
from tehran_house_price.api.model_loader import get_model_service


class _StubModelService:
    """In-memory stub that mimics ModelService for endpoint tests."""

    def __init__(
        self,
        loaded: bool = True,
        constant: float = 100_000_000.0,
        metadata: dict[str, Any] | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        self._loaded = loaded
        self._constant = constant
        self._metadata = metadata or {}
        self._artifact_path = artifact_path

    def is_loaded(self) -> bool:
        return self._loaded

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_artifact_path(self) -> Path | None:
        return self._artifact_path

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.full(shape=len(rows), fill_value=self._constant, dtype=np.float64)


def _build_app_with_stub(stub: _StubModelService) -> FastAPI:
    """Create a FastAPI app with the base model service replaced by a stub.

    We only override get_model_service. The higher-level dependency
    get_loaded_model_service depends on get_model_service, so it will
    automatically pick up the stub and still run its own is_loaded check.
    """
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: stub
    return app


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """Return a valid single-prediction payload."""
    return {
        "district": "Pasdaran",
        "area_m2": 120.0,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }


@pytest.fixture
def loaded_client() -> Iterator[TestClient]:
    """TestClient backed by a stub that reports the model as loaded."""
    stub = _StubModelService(loaded=True, constant=100_000_000.0)
    app = _build_app_with_stub(stub)
    yield TestClient(app)


@pytest.fixture
def unloaded_client() -> Iterator[TestClient]:
    """TestClient backed by a stub that reports the model as not loaded."""
    stub = _StubModelService(loaded=False)
    app = _build_app_with_stub(stub)
    yield TestClient(app)


def test_health_returns_ok_when_loaded(loaded_client: TestClient) -> None:
    response = loaded_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_unloaded_state(unloaded_client: TestClient) -> None:
    response = unloaded_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_version_returns_expected_fields(loaded_client: TestClient) -> None:
    response = loaded_client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["model_name"] == "xgb_price_per_m2"
    assert body["target_name"] == "price_per_m2"
    assert body["model_loaded"] is True


def test_predict_returns_prediction(
    loaded_client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    response = loaded_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price_per_m2"] == 100_000_000.0
    assert body["predicted_total_price"] == 100_000_000.0 * valid_payload["area_m2"]
    assert body["currency"] == "toman"
    assert body["model_name"] == "xgb_price_per_m2"


def test_predict_rejects_invalid_payload(loaded_client: TestClient) -> None:
    bad_payload = {
        "district": "Pasdaran",
        "area_m2": -10,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }

    response = loaded_client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_predict_rejects_extra_fields(
    loaded_client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["unexpected"] = "field"

    response = loaded_client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_predict_returns_503_when_model_not_loaded(
    unloaded_client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    response = unloaded_client.post("/predict", json=valid_payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded"


def test_predict_batch_returns_predictions(
    loaded_client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    payload = {
        "listings": [
            valid_payload,
            {**valid_payload, "district": "Gheitarieh", "area_m2": 85.0},
        ]
    }

    response = loaded_client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["predicted_price_per_m2"] == 100_000_000.0
    assert body["predictions"][0]["predicted_total_price"] == 100_000_000.0 * 120.0
    assert body["predictions"][1]["predicted_total_price"] == 100_000_000.0 * 85.0


def test_predict_batch_rejects_empty_listings(loaded_client: TestClient) -> None:
    response = loaded_client.post("/predict/batch", json={"listings": []})

    assert response.status_code == 422


def test_predict_batch_returns_503_when_model_not_loaded(
    unloaded_client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    response = unloaded_client.post(
        "/predict/batch",
        json={"listings": [valid_payload]},
    )

    assert response.status_code == 503


def test_lifespan_loads_real_model_if_available() -> None:
    """Integration smoke test: bring up the full app and hit endpoints.

    Skipped automatically if the real model artifact is not present.
    """
    from tehran_house_price.utils.paths import project_root

    real_path = project_root() / "artifacts" / "models" / "xgb_price_per_m2.joblib"
    if not real_path.exists():
        pytest.skip("real model artifact not available")

    # Make sure the singleton starts clean so the lifespan actually loads.
    get_model_service().reset()

    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        version = client.get("/version")
        assert version.status_code == 200
        assert version.json()["model_loaded"] is True

        payload = {
            "district": "Pasdaran",
            "area_m2": 120.0,
            "rooms": 2,
            "has_parking": True,
            "has_storage": True,
            "has_elevator": True,
        }
        prediction = client.post("/predict", json=payload)
        assert prediction.status_code == 200
        body = prediction.json()
        assert body["predicted_price_per_m2"] > 0
        assert body["predicted_total_price"] > 0

    # Reset for any test that might run after.
    get_model_service().reset()
