"""Recent research: provenance, honest gaps, measure isolation, time leakage."""

import json

import numpy as np
import pandas as pd
import pytest
from tehran_house_price.data.recent import (
    DATA,
    MODELS,
    SERIES,
    backtest,
    coverage_ledger,
    load_observations,
    month_index,
    month_range,
    predict,
    run,
    validate_observations,
)


@pytest.fixture
def observed():
    return load_observations()[0]


def test_recent_window_and_calendar():
    months = month_range()
    assert len(months) == 60
    assert (months[0], months[-1]) == (140006, 140505)
    assert month_index(140501) - month_index(140412) == 1
    for invalid in (140000, 140013, 140006.5, True):
        with pytest.raises(ValueError):
            month_index(invalid)


def test_actual_coverage_is_not_misrepresented(observed):
    ledger = coverage_ledger(observed)
    assert len(observed) == 43
    assert len(ledger) == 120
    assert ledger.price_per_m2_toman.notna().sum() == 43
    assert ledger.groupby("series").price_per_m2_toman.count().to_dict() == {
        SERIES[0]: 31,
        SERIES[1]: 12,
    }
    assert len(set(month_range()) - set(observed.month_jalali)) == 17
    assert ledger[ledger.month_jalali == 140401].price_per_m2_toman.isna().all()


def test_conversion_regime_is_explicit(observed):
    values = observed.set_index(["series", "month_jalali"]).price_per_m2_toman
    assert values[SERIES[0], 140006] == 31703400
    assert values[SERIES[0], 140105] == 42729900
    assert values[SERIES[0], 140303] == 85910000
    assert values[SERIES[1], 140505] == 210000000
    corrected = observed[observed.source_id == "dlearn"]
    assert corrected.quality.eq("scale_inferred_crosschecked").all()


def test_checksum_failure(tmp_path):
    (tmp_path / "observations.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_observations(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "nan",
        "inf",
        "zero",
        "future",
        "unknown_series",
        "currency",
        "evidence",
        "order",
    ],
)
def test_quality_rejects_bad_data(observed, mutation):
    if mutation == "duplicate":
        observed = pd.concat([observed, observed.iloc[[0]]])
    elif mutation in ("nan", "inf", "zero"):
        observed.loc[0, "price_per_m2_toman"] = {"nan": np.nan, "inf": np.inf, "zero": 0}[mutation]
    elif mutation == "future":
        observed.loc[0, "month_jalali"] = 140506
    elif mutation == "unknown_series":
        observed.loc[0, "series"] = "synthetic"
    elif mutation == "currency":
        observed.loc[0, "raw_unit"] = "USD"
    elif mutation == "evidence":
        observed.loc[0, "evidence"] = ""
    else:
        observed = observed.iloc[::-1]
    with pytest.raises(ValueError):
        validate_observations(observed)


def test_forecasts_reject_mixed_measures_and_future(observed):
    with pytest.raises(ValueError, match="one series"):
        predict(observed, 140506, "last_month")
    history = observed[observed.series == SERIES[0]].iloc[:6]
    with pytest.raises(ValueError, match="sorted past"):
        predict(history, 140011, "last_month")
    with pytest.raises(ValueError, match="Unknown"):
        predict(history, 140012, "unknown")


def test_calendar_gaps_not_compressed():
    months = [140101, 140102, 140103, 140105, 140107, 140108]
    history = pd.DataFrame(
        {
            "month_jalali": months,
            "series": SERIES[0],
            "price_per_m2_toman": [10, 20, 30, 50, 70, 80],
        }
    )
    assert predict(history, 140109, "calendar_drift") == pytest.approx(90)
    assert predict(history, 140109, "last_month") == 80


def test_missing_preceding_month_not_scored(observed):
    predictions, attempts, result = backtest(observed)
    cbi = attempts[attempts.series == SERIES[0]].set_index("month_jalali")
    assert cbi.loc[140202, "status"] == "previous_calendar_month_missing"
    assert cbi.loc[140212, "status"] == "previous_calendar_month_missing"
    assert all(
        month_index(t) - month_index(h) == 1
        for t, h in zip(predictions.month_jalali, predictions.history_end, strict=True)
    )
    assert result["complete_five_year_transaction_dataset"] is False
    assert result["metrics"][SERIES[0]]["test"]["last_month"]["n"] == 13
    assert result["metrics"][SERIES[1]]["test"]["last_month"]["n"] == 3


def test_future_prices_do_not_change_selection_or_past_forecasts(observed):
    before, _, result = backtest(observed)
    future = (observed.series == SERIES[1]) & (observed.month_jalali >= 140503)
    observed.loc[future, ["raw_value", "price_per_m2_toman"]] *= 3
    after, _, changed = backtest(observed)
    cutoff = (before.series == SERIES[0]) | (before.month_jalali <= 140503)
    np.testing.assert_array_equal(before.loc[cutoff, "prediction"], after.loc[cutoff, "prediction"])
    assert result["selected"] == changed["selected"]


def test_selection_uses_validation_not_test(observed):
    _, _, result = backtest(observed)
    for series in SERIES:
        expected = min(MODELS, key=lambda m: result["metrics"][series]["validation"][m]["mae"])
        assert result["selected"][series] == expected
    # In CBI test the baseline actually beats the selected model; no cherry picking.
    assert (
        result["metrics"][SERIES[0]]["test"]["last_month"]["mae"]
        < result["metrics"][SERIES[0]]["test"][result["selected"][SERIES[0]]]["mae"]
    )


def test_end_to_end_reproducible_report(tmp_path):
    result = run(DATA, tmp_path)
    files = [
        "observations.csv",
        "coverage.csv",
        "predictions.csv",
        "metrics.json",
        "evaluation_eligibility.csv",
        "metrics_table.md",
        "sources.md",
        "coverage.png",
        "prices.png",
        "backtest.png",
    ]
    first = {f: (tmp_path / f).read_bytes() for f in files}
    assert json.loads(first["metrics.json"]) == result
    assert run(DATA, tmp_path) == result
    assert {f: (tmp_path / f).read_bytes() for f in files} == first
    for name in ("coverage.png", "prices.png", "backtest.png"):
        assert first[name].startswith(b"\x89PNG")
