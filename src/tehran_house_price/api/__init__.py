"""API package for model serving."""

from tehran_house_price.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    HousePredictionRequest,
    HousePredictionResponse,
    PredictionResult,
    VersionResponse,
)

__all__ = [
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "HousePredictionRequest",
    "HousePredictionResponse",
    "PredictionResult",
    "VersionResponse",
]
