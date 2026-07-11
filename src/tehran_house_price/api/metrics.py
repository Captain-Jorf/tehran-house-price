"""Prometheus metrics for API traffic and model inference."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from tehran_house_price.settings import get_settings
from tehran_house_price.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["system"])

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of processed HTTP requests.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of in-flight HTTP requests.",
    ["method", "path"],
)

HTTP_REQUEST_EXCEPTIONS_TOTAL = Counter(
    "http_request_exceptions_total",
    "Total number of unhandled HTTP request exceptions.",
    ["method", "path", "exception_type"],
)

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total number of model predictions served.",
    ["endpoint", "model_name"],
)


def _metrics_enabled() -> bool:
    """Return whether metrics collection is enabled."""
    settings = get_settings()
    return settings.observability_enabled and settings.prometheus_enabled


def _safe_metric_operation(operation_name: str, callback: Callable[[], None]) -> None:
    """Execute a metric update safely.

    Observability must never take down the API.
    """
    if not _metrics_enabled():
        return

    try:
        callback()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "prometheus metric update failed | operation=%s | error=%s",
            operation_name,
            exc,
        )


def record_request_completion(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record request count and latency."""
    _safe_metric_operation(
        "record_request_completion",
        lambda: HTTP_REQUESTS_TOTAL.labels(
            method=method,
            path=path,
            status_code=str(status_code),
        ).inc(),
    )
    _safe_metric_operation(
        "record_request_duration",
        lambda: HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            path=path,
        ).observe(duration_seconds),
    )


def record_request_exception(*, method: str, path: str, exception_type: str) -> None:
    """Record an unhandled request exception."""
    _safe_metric_operation(
        "record_request_exception",
        lambda: HTTP_REQUEST_EXCEPTIONS_TOTAL.labels(
            method=method,
            path=path,
            exception_type=exception_type,
        ).inc(),
    )


def record_in_progress_increment(*, method: str, path: str) -> None:
    """Increment the in-flight request gauge."""
    _safe_metric_operation(
        "record_in_progress_increment",
        lambda: HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=path).inc(),
    )


def record_in_progress_decrement(*, method: str, path: str) -> None:
    """Decrement the in-flight request gauge."""
    _safe_metric_operation(
        "record_in_progress_decrement",
        lambda: HTTP_REQUESTS_IN_PROGRESS.labels(method=method, path=path).dec(),
    )


def record_prediction(*, endpoint: str, model_name: str, count: int = 1) -> None:
    """Record served model predictions."""
    _safe_metric_operation(
        "record_prediction",
        lambda: MODEL_PREDICTIONS_TOTAL.labels(
            endpoint=endpoint,
            model_name=model_name,
        ).inc(count),
    )


@router.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus metrics endpoint",
)
def metrics() -> Response:
    """Expose Prometheus metrics in text format."""
    payload = generate_latest(REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


def register_metrics_routes(app: FastAPI) -> None:
    """Attach the metrics router to the application once."""
    if getattr(app.state, "metrics_routes_registered", False):
        return

    if not _metrics_enabled():
        return

    app.include_router(router)
    app.state.metrics_routes_registered = True
