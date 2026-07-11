"""Asynchronous prediction logging to PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tehran_house_price.monitoring.schemas import Base, PredictionLog
from tehran_house_price.settings import get_settings
from tehran_house_price.utils.logger import get_logger

logger = get_logger(__name__)

_engine = None
_SessionLocal = None


def _init_db() -> None:
    """Initialize the database connection once."""
    global _engine, _SessionLocal
    if _engine is not None:
        return

    settings = get_settings()
    if not settings.prediction_logging_enabled or not settings.prediction_log_db_url:
        return

    try:
        _engine = create_engine(settings.prediction_log_db_url)
        Base.metadata.create_all(bind=_engine)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        logger.info("prediction log database initialized")
    except Exception as exc:  # pragma: no cover
        logger.error("failed to initialize prediction log db: %s", exc)


def log_prediction(
    *,
    request_id: str,
    endpoint: str,
    model_name: str,
    input_data: list[dict[str, Any]] | dict[str, Any],
    output_data: list[dict[str, Any]] | dict[str, Any],
) -> None:
    """Log a prediction event to the database.

    Designed to be called as a FastAPI BackgroundTask so it
    doesn't block the API response.
    """
    settings = get_settings()
    if not settings.prediction_logging_enabled:
        return

    _init_db()

    if _SessionLocal is None:
        return

    try:
        with _SessionLocal() as session:
            log_entry = PredictionLog(
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
                endpoint=endpoint,
                model_name=model_name,
                input_data=input_data,
                output_data=output_data,
            )
            session.add(log_entry)
            session.commit()
    except Exception as exc:  # pragma: no cover
        logger.error("failed to log prediction to db: %s", exc)
