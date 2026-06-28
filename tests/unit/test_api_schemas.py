"""Unit tests for API request and response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tehran_house_price.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    HousePredictionRequest,
    HousePredictionResponse,
    PredictionResult,
    VersionResponse,
)


@pytest.fixture
def valid_payload() -> dict[str, object]:
    """Return a valid single-prediction payload."""
    return {
        "district": "Pasdaran",
        "area_m2": 120.0,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }


def test_house_prediction_request_accepts_valid_payload(valid_payload: dict[str, object]) -> None:
    """Valid payload should create a request model successfully."""
    request = HousePredictionRequest(**valid_payload)

    assert request.district == "Pasdaran"
    assert request.area_m2 == 120.0
    assert request.rooms == 2
    assert request.has_parking is True
    assert request.has_storage is True
    assert request.has_elevator is True


def test_house_prediction_request_strips_district_whitespace(
    valid_payload: dict[str, object],
) -> None:
    """District should be normalized by stripping whitespace."""
    valid_payload["district"] = "  Pasdaran  "

    request = HousePredictionRequest(**valid_payload)

    assert request.district == "Pasdaran"


def test_house_prediction_request_rejects_empty_district(
    valid_payload: dict[str, object],
) -> None:
    """Blank district values should be rejected."""
    valid_payload["district"] = "   "

    with pytest.raises(ValidationError):
        HousePredictionRequest(**valid_payload)


@pytest.mark.parametrize("invalid_value", ["nan", "NaN", "null", "none"])
def test_house_prediction_request_rejects_placeholder_district_values(
    valid_payload: dict[str, object],
    invalid_value: str,
) -> None:
    """Placeholder district strings should be rejected."""
    valid_payload["district"] = invalid_value

    with pytest.raises(ValidationError):
        HousePredictionRequest(**valid_payload)


def test_house_prediction_request_rejects_non_positive_area(
    valid_payload: dict[str, object],
) -> None:
    """Area must be strictly positive."""
    valid_payload["area_m2"] = 0

    with pytest.raises(ValidationError):
        HousePredictionRequest(**valid_payload)


def test_house_prediction_request_rejects_negative_rooms(
    valid_payload: dict[str, object],
) -> None:
    """Rooms cannot be negative."""
    valid_payload["rooms"] = -1

    with pytest.raises(ValidationError):
        HousePredictionRequest(**valid_payload)


def test_house_prediction_request_rejects_string_booleans(
    valid_payload: dict[str, object],
) -> None:
    """Boolean flags should require real booleans, not strings."""
    valid_payload["has_parking"] = "true"

    with pytest.raises(ValidationError):
        HousePredictionRequest(**valid_payload)


def test_house_prediction_request_rejects_extra_fields(
    valid_payload: dict[str, object],
) -> None:
    """Unexpected input fields should be rejected."""
    valid_payload["unexpected_field"] = "should_fail"

    with pytest.raises(ValidationError):
        HousePredictionRequest(**valid_payload)


def test_house_prediction_request_to_model_input(
    valid_payload: dict[str, object],
) -> None:
    """Schema should expose a model-ready row dictionary."""
    request = HousePredictionRequest(**valid_payload)

    assert request.to_model_input() == {
        "district": "Pasdaran",
        "area_m2": 120.0,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }


def test_batch_prediction_request_accepts_multiple_listings(
    valid_payload: dict[str, object],
) -> None:
    """Batch request should accept one or more listings."""
    payload = {
        "listings": [
            valid_payload,
            {
                **valid_payload,
                "district": "Gheitarieh",
                "area_m2": 85.0,
            },
        ]
    }

    request = BatchPredictionRequest(**payload)

    assert len(request.listings) == 2
    assert request.listings[0].district == "Pasdaran"
    assert request.listings[1].district == "Gheitarieh"


def test_batch_prediction_request_rejects_empty_list(valid_payload: dict[str, object]) -> None:
    """Batch request must contain at least one listing."""
    payload = {"listings": []}

    with pytest.raises(ValidationError):
        BatchPredictionRequest(**payload)


def test_batch_prediction_request_to_model_inputs(
    valid_payload: dict[str, object],
) -> None:
    """Batch schema should expose model-ready row dictionaries."""
    request = BatchPredictionRequest(
        listings=[
            HousePredictionRequest(**valid_payload),
            HousePredictionRequest(
                **{
                    **valid_payload,
                    "district": "Elahieh",
                    "area_m2": 200.0,
                }
            ),
        ]
    )

    assert request.to_model_inputs() == [
        {
            "district": "Pasdaran",
            "area_m2": 120.0,
            "rooms": 2,
            "has_parking": True,
            "has_storage": True,
            "has_elevator": True,
        },
        {
            "district": "Elahieh",
            "area_m2": 200.0,
            "rooms": 2,
            "has_parking": True,
            "has_storage": True,
            "has_elevator": True,
        },
    ]


def test_house_prediction_response_defaults() -> None:
    """Single prediction response should apply expected defaults."""
    response = HousePredictionResponse(
        predicted_price_per_m2=120_000_000.0,
        predicted_total_price=14_400_000_000.0,
    )

    assert response.currency == "toman"
    assert response.model_name == "xgb_price_per_m2"


def test_batch_prediction_response_validates_count() -> None:
    """Batch response count should match the number of predictions."""
    with pytest.raises(ValidationError):
        BatchPredictionResponse(
            predictions=[
                PredictionResult(
                    predicted_price_per_m2=100_000_000.0,
                    predicted_total_price=10_000_000_000.0,
                ),
                PredictionResult(
                    predicted_price_per_m2=120_000_000.0,
                    predicted_total_price=12_000_000_000.0,
                ),
            ],
            count=1,
        )


def test_health_response_schema() -> None:
    """Health response should serialize expected fields."""
    response = HealthResponse(model_loaded=True)

    assert response.model_dump() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_version_response_schema() -> None:
    """Version response should serialize expected fields."""
    response = VersionResponse(
        version="0.1.0",
        model_loaded=True,
        artifact_path="artifacts/models/xgb_price_per_m2.joblib",
    )

    assert response.version == "0.1.0"
    assert response.model_name == "xgb_price_per_m2"
    assert response.target_name == "price_per_m2"
    assert response.model_loaded is True
    assert response.artifact_path == "artifacts/models/xgb_price_per_m2.joblib"
