"""API package for model serving."""

from tehran_house_price.api.model_loader import (
    ModelLoadError,
    ModelNotLoadedError,
    ModelService,
    get_model_service,
)
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
    "ModelLoadError",
    "ModelNotLoadedError",
    "ModelService",
    "PredictionResult",
    "VersionResponse",
    "get_model_service",
]
