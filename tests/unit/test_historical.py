"""Offline dataset integrity, chronological evaluation and leakage tests."""

from pathlib import Path

import numpy as np
import pytest
from tehran_house_price.data.historical import (
    MODELS,
    backtest,
    forecast,
    load_source,
    validate_monthly,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def monthly():
    return load_source(ROOT / "data/external/tehran_monthly/source.csv")


def test_complete_five_years(monthly):
    assert len(monthly) == 60
    assert monthly.groupby(monthly.Month // 100).size().tolist() == [12] * 5


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order", "negative", "nan", "inf"])
def test_reject_bad_data(monthly, mutation):
    if mutation == "missing":
        monthly = monthly.iloc[:-1]
    elif mutation == "duplicate":
        monthly.loc[1, "Month"] = monthly.loc[0, "Month"]
    elif mutation == "order":
        monthly = monthly.iloc[::-1]
    else:
        monthly["Price"] = monthly.Price.astype(float)
        monthly.loc[0, "Price"] = {"negative": -1, "nan": np.nan, "inf": np.inf}[mutation]
    with pytest.raises(ValueError):
        validate_monthly(monthly)


def test_checksum(tmp_path):
    path = tmp_path / "tampered.csv"
    path.write_text("Month,Price\n139501,1\n")
    with pytest.raises(ValueError, match="checksum"):
        load_source(path)


def test_forecast_formulas():
    history = np.arange(1, 13, dtype=float)
    assert forecast(history, "last_month") == 12
    assert forecast(history, "drift") == 13
    assert forecast(np.exp(history / 10), "log_trend_12m") == pytest.approx(np.exp(1.3))
    with pytest.raises(ValueError):
        forecast(history, "unknown")
    with pytest.raises(ValueError):
        forecast(history[:5], "drift")


def test_chronology_and_selection(monthly):
    predictions, result = backtest(monthly)
    assert len(predictions) == 72
    assert (predictions.history_end < predictions.month_jalali).all()
    for split in ("validation", "test"):
        assert predictions[predictions.split == split].groupby("model").size().tolist() == [12] * 3
    selected = min(MODELS, key=lambda m: result["metrics"]["validation"][m]["mae"])
    assert result["selected_on_validation_mae"] == selected


def test_future_cannot_change_past_forecasts_or_selection(monthly):
    before, result_before = backtest(monthly)
    monthly.loc[48:, "Price"] *= 3
    after, result_after = backtest(monthly)
    # Includes the first test prediction, made before its actual value is known.
    np.testing.assert_array_equal(before.prediction[:39], after.prediction[:39])
    assert result_before["selected_on_validation_mae"] == result_after["selected_on_validation_mae"]


def test_reproducible(monthly):
    first, result = backtest(monthly)
    second, repeated = backtest(monthly)
    assert first.equals(second)
    assert result == repeated
