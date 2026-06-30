"""Global exception handlers for the FastAPI application.

These handlers convert raw exceptions into structured ErrorResponse
payloads so clients get a consistent error shape and the API never
leaks stack traces. Each handler also logs the error with context
to keep production debugging tractable.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tehran_house_price.api.schemas import ErrorDetail, ErrorResponse
from tehran_house_price.utils.logger import get_logger

logger = get_logger(__name__)


def _error_payload(
    error: str,
    message: str,
    status_code: int,
    details: list[ErrorDetail] | None = None,
) -> dict:
    """Build a JSON-serializable error payload."""
    response = ErrorResponse(
        error=error,
        message=message,
        status_code=status_code,
        details=details or [],
    )
    return response.model_dump()


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Convert HTTPException (both fastapi and starlette) into ErrorResponse.

    Starlette is the layer that raises 404 for unknown routes, so we
    register against StarletteHTTPException to cover both cases.
    """
    logger.warning(
        "http_exception | path=%s | status=%s | detail=%s",
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    payload = _error_payload(
        error="http_error",
        message=str(exc.detail) if exc.detail is not None else "HTTP error",
        status_code=exc.status_code,
    )
    return JSONResponse(status_code=exc.status_code, content=payload)


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Convert Pydantic validation errors into structured ErrorResponse."""
    details: list[ErrorDetail] = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        details.append(
            ErrorDetail(
                field=location or None,
                message=err.get("msg", "invalid value"),
                type=err.get("type"),
            )
        )

    logger.warning(
        "validation_error | path=%s | error_count=%d",
        request.url.path,
        len(details),
    )

    payload = _error_payload(
        error="validation_error",
        message="Request payload failed validation",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details=details,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=payload,
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Convert ValueError (typically from the model layer) into 400 response."""
    logger.warning(
        "value_error | path=%s | message=%s",
        request.url.path,
        str(exc),
    )
    payload = _error_payload(
        error="bad_request",
        message=str(exc),
        status_code=status.HTTP_400_BAD_REQUEST,
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=payload,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected errors.

    The original exception is logged with traceback, but the response
    body is intentionally generic to avoid leaking internals to clients.
    """
    logger.exception(
        "unhandled_exception | path=%s | type=%s",
        request.url.path,
        type(exc).__name__,
    )
    payload = _error_payload(
        error="internal_server_error",
        message="An unexpected error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to the given app.

    The type: ignore comments are required because Starlette types
    add_exception_handler as accepting only base Exception handlers,
    while in practice (and per FastAPI docs) specialized handlers
    are the standard pattern.
    """
    # fmt: off
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]  # noqa: E501
    app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
    # fmt: on
