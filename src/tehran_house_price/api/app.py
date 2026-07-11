"""FastAPI application for Tehran house price prediction.

This module defines the public HTTP API and wires it to the in-memory
ModelService. The app is intentionally thin: validation lives in the
Pydantic schemas, inference logic lives in the model service, and the
endpoints only translate between them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Request

from tehran_house_price import __version__
from tehran_house_price.api.dependencies import get_loaded_model_service
from tehran_house_price.api.errors import register_exception_handlers
from tehran_house_price.api.health import register_health_routes
from tehran_house_price.api.metrics import record_prediction, register_metrics_routes
from tehran_house_price.api.middleware import register_request_middleware
from tehran_house_price.api.model_loader import (
    ModelLoadError,
    ModelService,
    get_model_service,
)
from tehran_house_price.api.schemas import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TARGET_NAME,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    HousePredictionRequest,
    HousePredictionResponse,
    PredictionResult,
    VersionResponse,
)
from tehran_house_price.monitoring.prediction_logger import log_prediction
from tehran_house_price.settings import get_settings
from tehran_house_price.utils.logger import get_logger

logger = get_logger(__name__)

API_TITLE = "Tehran House Price API"
API_DESCRIPTION = (
    "REST API for predicting Tehran residential property prices. "
    "Powered by an XGBoost regression pipeline trained on Kaggle data."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler.

    On startup, eagerly load the model so the first request is fast.
    On shutdown, release the model from memory.
    """
    logger.info("starting application | loading model")
    service = get_model_service()
    try:
        service.load()
    except ModelLoadError as exc:
        logger.error("model load failed at startup: %s", exc)

    yield

    logger.info("shutting down application | releasing model")
    service.reset()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    register_exception_handlers(app)
    register_request_middleware(app)

    if settings.deep_healthcheck_enabled:
        register_health_routes(app)

    if settings.prometheus_enabled:
        register_metrics_routes(app)

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Attach all HTTP routes to the given app."""

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Basic health probe and model-loaded status",
    )
    def health(
        service: ModelService = Depends(get_model_service),
    ) -> HealthResponse:
        return HealthResponse(model_loaded=service.is_loaded())

    @app.get(
        "/version",
        response_model=VersionResponse,
        tags=["system"],
        summary="Application and model metadata",
    )
    def version(
        service: ModelService = Depends(get_model_service),
    ) -> VersionResponse:
        artifact_path = service.get_artifact_path()
        return VersionResponse(
            version=__version__,
            model_name=DEFAULT_MODEL_NAME,
            target_name=DEFAULT_TARGET_NAME,
            model_loaded=service.is_loaded(),
            artifact_path=str(artifact_path) if artifact_path is not None else None,
        )

    @app.post(
        "/predict",
        response_model=HousePredictionResponse,
        tags=["inference"],
        summary="Predict price for a single house listing",
    )
    def predict(
        payload: HousePredictionRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        service: ModelService = Depends(get_loaded_model_service),
    ) -> HousePredictionResponse:
        predictions = service.predict([payload.to_model_input()])
        price_per_m2 = float(predictions[0])
        total_price = price_per_m2 * payload.area_m2

        record_prediction(
            endpoint="/predict",
            model_name=DEFAULT_MODEL_NAME,
            count=1,
        )

        background_tasks.add_task(
            log_prediction,
            request_id=getattr(request.state, "request_id", "unknown"),
            endpoint="/predict",
            model_name=DEFAULT_MODEL_NAME,
            input_data=payload.model_dump(),
            output_data={
                "predicted_price_per_m2": price_per_m2,
                "predicted_total_price": total_price,
            },
        )

        return HousePredictionResponse(
            predicted_price_per_m2=price_per_m2,
            predicted_total_price=total_price,
            model_name=DEFAULT_MODEL_NAME,
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["inference"],
        summary="Predict prices for a batch of house listings",
    )
    def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        service: ModelService = Depends(get_loaded_model_service),
    ) -> BatchPredictionResponse:
        rows = payload.to_model_inputs()
        predictions = service.predict(rows)

        results: list[PredictionResult] = []
        for listing, predicted in zip(payload.listings, predictions, strict=True):
            price_per_m2 = float(predicted)
            total_price = price_per_m2 * listing.area_m2
            results.append(
                PredictionResult(
                    predicted_price_per_m2=price_per_m2,
                    predicted_total_price=total_price,
                )
            )

        record_prediction(
            endpoint="/predict/batch",
            model_name=DEFAULT_MODEL_NAME,
            count=len(results),
        )

        background_tasks.add_task(
            log_prediction,
            request_id=getattr(request.state, "request_id", "unknown"),
            endpoint="/predict/batch",
            model_name=DEFAULT_MODEL_NAME,
            input_data=payload.model_dump(),
            output_data={"predictions": [r.model_dump() for r in results]},
        )

        return BatchPredictionResponse(
            predictions=results,
            count=len(results),
            model_name=DEFAULT_MODEL_NAME,
        )


app = create_app()
