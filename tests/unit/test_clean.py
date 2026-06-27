"""Tests for kaggle cleaning pipeline."""

import pandas as pd

from tehran_house_price.data import clean
from tehran_house_price.data import constants as const


def _raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area": ["63", "60", "1,200", "85", "0"],
            "Room": [1, 1, 3, 2, 1],
            "Parking": ["True", "False", "True", "True", "True"],
            "Warehouse": ["True", "True", "False", "True", "False"],
            "Elevator": ["True", "True", "True", "False", "True"],
            "Address": ["Shahran", "Pardis", "Vanak", "  Sa'adat Abad  ", ""],
            "Price": [1.85e9, 5.5e8, 5.0e10, 7.5e9, 1.0e9],
            "Price(USD)": [61666.67, 18333.33, 1666666.67, 250000.0, 33333.33],
        }
    )


# ---- individual transforms ----


def test_drop_unused_columns():
    df = _raw_sample()
    out = clean.drop_unused_columns(df)
    assert "Price(USD)" not in out.columns
    assert "Area" in out.columns  # still raw, only the explicit drop happened


def test_rename_to_canonical():
    df = _raw_sample()
    out = clean.rename_to_canonical(df)
    assert const.AREA_M2 in out.columns
    assert const.TOTAL_PRICE in out.columns
    assert "Area" not in out.columns


def test_coerce_area_handles_commas():
    df = pd.DataFrame({const.AREA_M2: ["1,200", "60", "abc"]})
    out = clean.coerce_area(df)
    assert out[const.AREA_M2].iloc[0] == 1200
    assert out[const.AREA_M2].iloc[1] == 60
    assert pd.isna(out[const.AREA_M2].iloc[2])


def test_coerce_booleans():
    df = pd.DataFrame(
        {
            const.HAS_PARKING: ["True", "false", "TRUE", "no"],
            const.HAS_STORAGE: ["1", "0", "yes", "False"],
            const.HAS_ELEVATOR: ["true", "True", "y", "False"],
        }
    )
    out = clean.coerce_booleans(df)
    assert out[const.HAS_PARKING].tolist() == [True, False, True, False]
    assert out[const.HAS_STORAGE].tolist() == [True, False, True, False]
    assert out[const.HAS_ELEVATOR].tolist() == [True, True, True, False]


def test_clean_district_strips_whitespace():
    df = pd.DataFrame({const.DISTRICT: ["  Vanak  ", "Pardis", ""]})
    out = clean.clean_district(df)
    assert out[const.DISTRICT].tolist() == ["Vanak", "Pardis", ""]


def test_drop_invalid_rows_removes_bad_data():
    df = pd.DataFrame(
        {
            const.AREA_M2: [60.0, 0.0, 80.0, 5000.0],
            const.TOTAL_PRICE: [1e9, 1e9, 50.0, 1e10],  # second is 0 area, third too cheap
            const.DISTRICT: ["A", "B", "C", "D"],
        }
    )
    out = clean.drop_invalid_rows(df)
    # only first row passes all checks
    assert len(out) == 1
    assert out[const.AREA_M2].iloc[0] == 60.0


def test_add_listing_id_is_stable():
    df = pd.DataFrame(
        {
            const.SOURCE: ["kaggle", "kaggle"],
            const.DISTRICT: ["Vanak", "Vanak"],
            const.AREA_M2: [85.0, 85.0],
            const.ROOMS: [2, 2],
            const.TOTAL_PRICE: [7.5e9, 7.5e9],
        }
    )
    out = clean.add_listing_id(df)
    # same content -> same id
    assert out[const.LISTING_ID].iloc[0] == out[const.LISTING_ID].iloc[1]


def test_add_listing_id_differs_for_different_content():
    df = pd.DataFrame(
        {
            const.SOURCE: ["kaggle", "kaggle"],
            const.DISTRICT: ["Vanak", "Pardis"],
            const.AREA_M2: [85.0, 60.0],
            const.ROOMS: [2, 1],
            const.TOTAL_PRICE: [7.5e9, 5.5e8],
        }
    )
    out = clean.add_listing_id(df)
    assert out[const.LISTING_ID].iloc[0] != out[const.LISTING_ID].iloc[1]


def test_add_source_sets_constant():
    df = pd.DataFrame({"x": [1, 2]})
    out = clean.add_source(df, "kaggle")
    assert (out[const.SOURCE] == "kaggle").all()


def test_add_price_per_m2():
    df = pd.DataFrame(
        {
            const.AREA_M2: [100.0, 50.0],
            const.TOTAL_PRICE: [1e9, 1e9],
        }
    )
    out = clean.add_price_per_m2(df)
    assert out[const.PRICE_PER_M2].iloc[0] == 1e7
    assert out[const.PRICE_PER_M2].iloc[1] == 2e7


# ---- end to end ----


def test_clean_kaggle_pipeline_runs():
    raw = _raw_sample()
    out = clean.clean_kaggle(raw)

    # required columns present
    for col in (
        const.LISTING_ID,
        const.SOURCE,
        const.DISTRICT,
        const.AREA_M2,
        const.ROOMS,
        const.TOTAL_PRICE,
        const.PRICE_PER_M2,
        const.INGESTED_AT,
    ):
        assert col in out.columns

    # bad rows dropped (5000m2 not in sample, but area=0 and empty district are)
    assert len(out) < len(raw)

    # source set
    assert (out[const.SOURCE] == const.SOURCE_KAGGLE).all()

    # listing_id unique
    assert out[const.LISTING_ID].is_unique


def test_save_interim_writes_parquet(tmp_path, monkeypatch):
    df = pd.DataFrame({"a": [1, 2, 3]})
    monkeypatch.setattr(clean, "interim_dir", lambda: tmp_path)
    out = clean.save_interim(df, name="test.parquet")
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert len(loaded) == 3
