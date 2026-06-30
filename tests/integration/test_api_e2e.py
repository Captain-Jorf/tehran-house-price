"""End-to-end integration tests for the API with the real model artifact.

These tests boot the full FastAPI app (including the lifespan handler
that loads the real model from disk) and exercise the public endpoints.
They are skipped automatically when the model artifact is missing,
which keeps the suite usable on fresh checkouts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tehran_house_price.api.app import create_app
from tehran_house_price.api.model_loader import get_model_service
from tehran_house_price.utils.paths import project_root


def _real_model_path() -> Path:
    return project_root() / "artifacts" / "models" / "xgb_price_per_m2.joblib"


pytestmark = pytest.mark.skipif(
    not _real_model_path().exists(),
    reason="real model artifact not available",
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Boot the full app with the real model and yield a TestClient."""
    get_model_service().reset()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_model_service().reset()


def test_health_reports_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_version_reports_real_artifact_path(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["model_name"] == "xgb_price_per_m2"
    assert body["artifact_path"] is not None
    assert body["artifact_path"].endswith("xgb_price_per_m2.joblib")


def test_predict_returns_realistic_value(client: TestClient) -> None:
    payload = {
        "district": "Pasdaran",
        "area_m2": 120.0,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()

    # Sanity bounds: any positive prediction within a few orders of magnitude
    # of typical Tehran prices. We intentionally keep this loose so the test
    # does not break if the model is retrained.
    assert body["predicted_price_per_m2"] > 1_000_000
    assert body["predicted_price_per_m2"] < 10_000_000_000
    assert body["predicted_total_price"] == pytest.approx(
        body["predicted_price_per_m2"] * payload["area_m2"]
    )
    assert body["currency"] == "toman"


def test_predict_batch_preserves_order(client: TestClient) -> None:
    payload = {
        "listings": [
            {
                "district": "Pasdaran",
                "area_m2": 120.0,
                "rooms": 2,
                "has_parking": True,
                "has_storage": True,
                "has_elevator": True,
            },
            {
                "district": "Gheitarieh",
                "area_m2": 80.0,
                "rooms": 1,
                "has_parking": False,
                "has_storage": False,
                "has_elevator": True,
            },
            {
                "district": "Elahieh",
                "area_m2": 200.0,
                "rooms": 4,
                "has_parking": True,
                "has_storage": True,
                "has_elevator": True,
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3

    for prediction, listing in zip(body["predictions"], payload["listings"], strict=True):
        assert prediction["predicted_price_per_m2"] > 0
        assert prediction["predicted_total_price"] == pytest.approx(
            prediction["predicted_price_per_m2"] * listing["area_m2"]
        )


def test_predict_unknown_district_does_not_crash(client: TestClient) -> None:
    """Encoder must gracefully handle unseen districts via fallback values."""
    payload = {
        "district": "A_District_That_Does_Not_Exist_12345",
        "area_m2": 100.0,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price_per_m2"] > 0


def test_invalid_payload_returns_structured_422(client: TestClient) -> None:
    bad_payload = {
        "district": "Pasdaran",
        "area_m2": -1,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }
    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["status_code"] == 422
    assert len(body["details"]) >= 1


def test_unknown_route_returns_structured_404(client: TestClient) -> None:
    response = client.get("/no-such-endpoint")

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "http_error"
    assert body["status_code"] == 404
