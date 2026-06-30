"""API package for model serving."""

from tehran_house_price.api.app import app, create_app
from tehran_house_price.api.dependencies import get_loaded_model_service
from tehran_house_price.api.errors import register_exception_handlers
from tehran_house_price.api.model_loader import (
    ModelLoadError,
    ModelNotLoadedError,
    ModelService,
    get_model_service,
)
from tehran_house_price.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HousePredictionRequest,
    HousePredictionResponse,
    PredictionResult,
    VersionResponse,
)

__all__ = [
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "HousePredictionRequest",
    "HousePredictionResponse",
    "ModelLoadError",
    "ModelNotLoadedError",
    "ModelService",
    "PredictionResult",
    "VersionResponse",
    "app",
    "create_app",
    "get_loaded_model_service",
    "get_model_service",
    "register_exception_handlers",
]
