"""Tests for the validation layer."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from tehran_house_price.data import constants as const
from tehran_house_price.data import validate


def _good_df() -> pd.DataFrame:
    """
    Minimal valid DataFrame that should pass HouseListingSchema.

    year_built, floor, total_floors: از float استفاده می‌کنیم نه Int64.
    float64 می‌تواند NaN داشته باشد و با pandera schema سازگار است.
    """
    return pd.DataFrame(
        {
            const.LISTING_ID: ["a1", "b2"],
            const.SOURCE: ["kaggle", "kaggle"],
            const.DISTRICT: ["Vanak", "Pardis"],
            const.NEIGHBORHOOD: [None, None],
            const.AREA_M2: [85.0, 60.0],
            const.ROOMS: [2, 1],
            const.YEAR_BUILT: [float("nan"), float("nan")],
            const.FLOOR: [float("nan"), float("nan")],
            const.TOTAL_FLOORS: [float("nan"), float("nan")],
            const.HAS_ELEVATOR: [True, False],
            const.HAS_PARKING: [True, True],
            const.HAS_STORAGE: [False, True],
            const.TOTAL_PRICE: [7.5e9, 5.5e8],
            const.PRICE_PER_M2: [8.8e7, 9.2e6],
            const.PUBLISHED_AT: [pd.NaT, pd.NaT],
            const.INGESTED_AT: [
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
            ],
        }
    )


def test_validate_passes_on_clean_data():
    df = _good_df()
    passed, report = validate.validate_dataframe(df)
    assert passed is True, f"expected PASSED but got errors: {report['errors']}"
    assert report["n_rows"] == 2
    assert report["errors"] == []


def test_validate_fails_on_bad_price():
    df = _good_df()
    df.loc[0, const.TOTAL_PRICE] = 10.0  # way below min bound
    passed, report = validate.validate_dataframe(df)
    assert passed is False
    assert len(report["errors"]) > 0


def test_validate_fails_on_duplicate_listing_id():
    df = _good_df()
    df.loc[1, const.LISTING_ID] = "a1"  # duplicate
    passed, report = validate.validate_dataframe(df)
    assert passed is False


def test_validate_fails_on_bad_source():
    df = _good_df()
    df.loc[0, const.SOURCE] = "sheypoor"
    passed, report = validate.validate_dataframe(df)
    assert passed is False


def test_save_report_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "artifacts_dir", lambda: tmp_path)
    report = {"passed": True, "n_rows": 5, "errors": []}
    path = validate.save_report(report, name="test_report.json")
    assert path.exists()

    import json

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["passed"] is True
    assert loaded["n_rows"] == 5


def test_run_raises_when_parquet_missing(tmp_path):
    bogus_path = tmp_path / "doesnotexist.parquet"
    with pytest.raises(FileNotFoundError):
        validate.run(parquet_path=bogus_path)
