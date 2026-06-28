"""Pydantic schemas for FastAPI inference requests and responses.

These schemas define the public API contract for model serving.
They are intentionally narrower than the full dataset schema because
the deployed model only needs a small subset of features at inference time.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

DEFAULT_MODEL_NAME = "xgb_price_per_m2"
DEFAULT_TARGET_NAME = "price_per_m2"
DEFAULT_CURRENCY = "toman"

MAX_BATCH_SIZE = 1000
MAX_DISTRICT_LENGTH = 100
MAX_AREA_M2 = 10_000.0
MAX_ROOMS = 20

_INVALID_DISTRICT_VALUES = {"nan", "none", "null"}


class HousePredictionRequest(BaseModel):
    """Validated input payload for a single house price prediction."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    district: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DISTRICT_LENGTH,
        description="District or neighborhood label used by the trained model.",
        examples=["Pasdaran"],
    )
    area_m2: float = Field(
        ...,
        gt=0,
        le=MAX_AREA_M2,
        description="Property area in square meters.",
        examples=[120.0],
    )
    rooms: int = Field(
        ...,
        ge=0,
        le=MAX_ROOMS,
        description="Number of rooms. Zero is allowed for studio-style listings.",
        examples=[2],
    )
    has_parking: StrictBool = Field(
        ...,
        description="Whether the property has parking.",
        examples=[True],
    )
    has_storage: StrictBool = Field(
        ...,
        description="Whether the property has a storage room.",
        examples=[True],
    )
    has_elevator: StrictBool = Field(
        ...,
        description="Whether the property has an elevator.",
        examples=[True],
    )

    @field_validator("district")
    @classmethod
    def validate_district(cls, value: str) -> str:
        """Reject blank and placeholder district values."""
        normalized = value.strip()

        if not normalized:
            raise ValueError("district must not be empty")

        if normalized.lower() in _INVALID_DISTRICT_VALUES:
            raise ValueError("district must be a real district name, not a placeholder")

        return normalized

    def to_model_input(self) -> dict[str, Any]:
        """Return a model-ready row payload.

        This shape matches the columns expected by the persisted sklearn pipeline.
        """
        return {
            "district": self.district,
            "area_m2": self.area_m2,
            "rooms": self.rooms,
            "has_parking": self.has_parking,
            "has_storage": self.has_storage,
            "has_elevator": self.has_elevator,
        }


class BatchPredictionRequest(BaseModel):
    """Validated input payload for batch house price prediction."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    listings: list[HousePredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of house listings to score in one request.",
    )

    def to_model_inputs(self) -> list[dict[str, Any]]:
        """Return model-ready row payloads for batch inference."""
        return [listing.to_model_input() for listing in self.listings]


class PredictionResult(BaseModel):
    """Prediction payload shared by single and batch responses."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    predicted_price_per_m2: float = Field(
        ...,
        gt=0,
        description="Predicted price per square meter in Toman.",
        examples=[120_000_000.0],
    )
    predicted_total_price: float = Field(
        ...,
        gt=0,
        description="Predicted total property price in Toman.",
        examples=[14_400_000_000.0],
    )
    currency: Literal["toman"] = Field(
        default=DEFAULT_CURRENCY,
        description="Currency unit used by the model outputs.",
    )


class HousePredictionResponse(PredictionResult):
    """Response body for a single prediction request."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Identifier of the serving model.",
        examples=[DEFAULT_MODEL_NAME],
    )


class BatchPredictionResponse(BaseModel):
    """Response body for a batch prediction request."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    predictions: list[PredictionResult] = Field(
        ...,
        min_length=1,
        description="Predictions returned in the same order as the input listings.",
    )
    count: int = Field(
        ...,
        ge=1,
        description="Number of returned predictions.",
    )
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Identifier of the serving model.",
        examples=[DEFAULT_MODEL_NAME],
    )

    @model_validator(mode="after")
    def validate_count_matches_predictions(self) -> BatchPredictionResponse:
        """Ensure count matches the number of predictions."""
        if self.count != len(self.predictions):
            raise ValueError("count must match the number of predictions")
        return self


class HealthResponse(BaseModel):
    """Response body for the health endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    status: Literal["ok"] = Field(
        default="ok",
        description="High-level application health status.",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model artifact is loaded and ready for inference.",
    )


class VersionResponse(BaseModel):
    """Response body for the version endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    version: str = Field(
        ...,
        description="Application or package version.",
        examples=["0.1.0"],
    )
    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Identifier of the serving model.",
    )
    target_name: str = Field(
        default=DEFAULT_TARGET_NAME,
        description="Target predicted by the model.",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model artifact is loaded and ready for inference.",
    )
    artifact_path: str | None = Field(
        default=None,
        description="Filesystem path to the loaded model artifact, if available.",
    )
