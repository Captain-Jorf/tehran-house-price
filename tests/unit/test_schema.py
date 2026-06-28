"""Tests for data contract."""

from datetime import datetime, timezone

import pandas as pd
import pytest
from pandera.errors import SchemaError
from pydantic import ValidationError
from tehran_house_price.data.schema import HouseListing, HouseListingSchema

# ----- helpers -----


def _valid_record() -> dict:
    return {
        "listing_id": "kaggle-001",
        "source": "kaggle",
        "district": "Vanak",
        "neighborhood": "Mollasadra",
        "area_m2": 85.0,
        "rooms": 2,
        "year_built": 1395,
        "floor": 3,
        "total_floors": 5,
        "has_elevator": True,
        "has_parking": True,
        "has_storage": False,
        "total_price": 7_500_000_000.0,
        "price_per_m2": 88_235_294.0,
        "published_at": datetime(2024, 1, 15, tzinfo=timezone.utc),
        "ingested_at": datetime(2024, 1, 20, tzinfo=timezone.utc),
    }


def _valid_df() -> pd.DataFrame:
    rows = [_valid_record(), {**_valid_record(), "listing_id": "kaggle-002"}]
    return pd.DataFrame(rows)


# ----- pydantic tests -----


def test_pydantic_accepts_valid_record():
    rec = HouseListing(**_valid_record())
    assert rec.source == "kaggle"
    assert rec.area_m2 == 85.0


def test_pydantic_rejects_invalid_source():
    data = _valid_record()
    data["source"] = "sheypoor"
    with pytest.raises(ValidationError):
        HouseListing(**data)


def test_pydantic_rejects_negative_area():
    data = _valid_record()
    data["area_m2"] = -10
    with pytest.raises(ValidationError):
        HouseListing(**data)


def test_pydantic_rejects_huge_area():
    data = _valid_record()
    data["area_m2"] = 99999
    with pytest.raises(ValidationError):
        HouseListing(**data)


def test_pydantic_allows_optional_fields_missing():
    data = _valid_record()
    for key in ("neighborhood", "year_built", "floor", "has_elevator"):
        data.pop(key, None)
    rec = HouseListing(**data)
    assert rec.neighborhood is None


def test_pydantic_normalizes_source_case():
    data = _valid_record()
    data["source"] = "KAGGLE"
    rec = HouseListing(**data)
    assert rec.source == "kaggle"


# ----- pandera tests -----


def test_pandera_accepts_valid_df():
    df = _valid_df()
    validated = HouseListingSchema.validate(df)
    assert len(validated) == 2


def test_pandera_rejects_duplicate_listing_id():
    df = _valid_df()
    df.loc[1, "listing_id"] = "kaggle-001"  # duplicate
    with pytest.raises(SchemaError):
        HouseListingSchema.validate(df)


def test_pandera_rejects_bad_source():
    df = _valid_df()
    df.loc[0, "source"] = "unknown"
    with pytest.raises(SchemaError):
        HouseListingSchema.validate(df)


def test_pandera_rejects_price_below_bound():
    df = _valid_df()
    df.loc[0, "total_price"] = 100.0
    with pytest.raises(SchemaError):
        HouseListingSchema.validate(df)
