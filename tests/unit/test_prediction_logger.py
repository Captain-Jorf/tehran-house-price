"""Unit tests for prediction logging."""

from __future__ import annotations

import pytest
from tehran_house_price.monitoring.prediction_logger import log_prediction
from tehran_house_price.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear cached settings before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_log_prediction_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """It should silently return if logging is disabled."""
    monkeypatch.setenv("PREDICTION_LOGGING_ENABLED", "false")

    # This should not raise any errors and should not attempt DB connection
    log_prediction(
        request_id="test-id",
        endpoint="/predict",
        model_name="test-model",
        input_data={"area": 100},
        output_data={"price": 1000},
    )
