"""HTTP middleware for request tracing and structured observability logs."""

from __future__ import annotations

import json
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from tehran_house_price.api.metrics import (
    record_in_progress_decrement,
    record_in_progress_increment,
    record_request_completion,
    record_request_exception,
)
from tehran_house_price.settings import get_settings
from tehran_house_price.utils.logger import get_logger

logger = get_logger(__name__)

_REQUEST_ID_CONTEXT: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the current request id from context, if available."""
    return _REQUEST_ID_CONTEXT.get()


def _serialize_log_event(payload: dict[str, object]) -> str:
    """Serialize a structured log event safely to JSON."""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _build_log_payload(
    *,
    event: str,
    request_id: str,
    method: str,
    path: str,
    duration_ms: float,
    status_code: int | None = None,
    client_ip: str | None = None,
    exception_type: str | None = None,
) -> dict[str, object]:
    """Create a consistent structured payload for request logs."""
    payload: dict[str, object] = {
        "event": event,
        "request_id": request_id,
        "method": method,
        "path": path,
        "duration_ms": round(duration_ms, 3),
    }

    if status_code is not None:
        payload["status_code"] = status_code
    if client_ip is not None:
        payload["client_ip"] = client_ip
    if exception_type is not None:
        payload["exception_type"] = exception_type

    return payload


def _safe_log_info(payload: dict[str, object]) -> None:
    """Log an info event without ever breaking request handling."""
    try:
        logger.info(_serialize_log_event(payload))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("structured info logging failed: %s", exc)


def _safe_log_exception(payload: dict[str, object]) -> None:
    """Log an exception event without ever breaking request handling."""
    try:
        logger.exception(_serialize_log_event(payload))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("structured exception logging failed: %s", exc)


def register_request_middleware(app: FastAPI) -> None:
    """Register request tracing middleware on the application.

    The middleware adds:
    - request id generation / propagation
    - response header injection
    - request duration timing
    - structured JSON request logs
    - Prometheus metric updates
    """
    if getattr(app.state, "request_middleware_registered", False):
        return

    settings = get_settings()
    if not settings.observability_enabled:
        return

    request_id_header_name = settings.request_id_header_name

    @app.middleware("http")
    async def request_observability_middleware(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get(request_id_header_name) or str(uuid4())
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client is not None else None

        request.state.request_id = request_id
        token = _REQUEST_ID_CONTEXT.set(request_id)

        record_in_progress_increment(method=method, path=path)
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_seconds = perf_counter() - started_at
            duration_ms = duration_seconds * 1000

            record_request_exception(
                method=method,
                path=path,
                exception_type=exc.__class__.__name__,
            )
            record_request_completion(
                method=method,
                path=path,
                status_code=500,
                duration_seconds=duration_seconds,
            )

            if settings.request_logging_enabled:
                payload = _build_log_payload(
                    event="http_request_failed",
                    request_id=request_id,
                    method=method,
                    path=path,
                    duration_ms=duration_ms,
                    status_code=500,
                    client_ip=client_ip,
                    exception_type=exc.__class__.__name__,
                )
                _safe_log_exception(payload)

            raise
        else:
            duration_seconds = perf_counter() - started_at
            duration_ms = duration_seconds * 1000

            response.headers[request_id_header_name] = request_id

            record_request_completion(
                method=method,
                path=path,
                status_code=response.status_code,
                duration_seconds=duration_seconds,
            )

            if settings.request_logging_enabled:
                payload = _build_log_payload(
                    event="http_request_completed",
                    request_id=request_id,
                    method=method,
                    path=path,
                    duration_ms=duration_ms,
                    status_code=response.status_code,
                    client_ip=client_ip,
                )
                _safe_log_info(payload)

            return response
        finally:
            record_in_progress_decrement(method=method, path=path)
            _REQUEST_ID_CONTEXT.reset(token)

    app.state.request_middleware_registered = True
