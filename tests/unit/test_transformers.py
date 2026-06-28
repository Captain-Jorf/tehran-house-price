"""Unit tests for custom feature transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tehran_house_price.features import constants as fconst
from tehran_house_price.features.transformers import (
    AreaPerRoomAdder,
    BooleanToIntCaster,
    ColumnSelector,
    FrequencyEncoder,
    TargetMeanEncoder,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district": ["A", "A", "B", "B", "C"],
            "area_m2": [50.0, 80.0, 100.0, 120.0, 60.0],
            "rooms": [1, 2, 2, 3, 1],
            "has_parking": [True, False, True, True, False],
            "has_storage": [False, False, True, True, False],
            "has_elevator": [True, True, True, False, False],
        }
    )


def _toy_y() -> pd.Series:
    return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])


def test_area_per_room_adder_basic():
    df = _toy_df()
    out = AreaPerRoomAdder().fit_transform(df)
    assert fconst.FEATURE_AREA_PER_ROOM in out.columns
    assert out[fconst.FEATURE_AREA_PER_ROOM].iloc[0] == pytest.approx(50.0)
    assert out[fconst.FEATURE_AREA_PER_ROOM].iloc[1] == pytest.approx(40.0)


def test_area_per_room_adder_zero_rooms_is_safe():
    df = _toy_df()
    df.loc[0, "rooms"] = 0
    out = AreaPerRoomAdder().fit_transform(df)
    assert np.isfinite(out[fconst.FEATURE_AREA_PER_ROOM].iloc[0])


def test_frequency_encoder_basic():
    df = _toy_df()
    enc = FrequencyEncoder(column="district").fit(df)
    out = enc.transform(df)
    assert "district_freq_enc" in out.columns
    assert out["district_freq_enc"].iloc[0] == pytest.approx(2 / 5)
    assert out["district_freq_enc"].iloc[4] == pytest.approx(1 / 5)


def test_frequency_encoder_unseen_category_is_zero():
    df = _toy_df()
    enc = FrequencyEncoder(column="district").fit(df)
    new_df = pd.DataFrame({"district": ["Z"]})
    out = enc.transform(new_df)
    assert out["district_freq_enc"].iloc[0] == 0.0


def test_target_mean_encoder_uses_smoothing():
    df = _toy_df()
    y = _toy_y()
    enc = TargetMeanEncoder(column="district", smoothing=0.0).fit(df, y)
    out = enc.transform(df)
    # With smoothing=0 the encoded value equals the per-category mean
    assert out["district_target_enc"].iloc[0] == pytest.approx(15.0)
    assert out["district_target_enc"].iloc[2] == pytest.approx(35.0)
    assert out["district_target_enc"].iloc[4] == pytest.approx(50.0)


def test_target_mean_encoder_unseen_falls_back_to_global_mean():
    df = _toy_df()
    y = _toy_y()
    enc = TargetMeanEncoder(column="district").fit(df, y)
    new_df = pd.DataFrame({"district": ["UNSEEN"]})
    out = enc.transform(new_df)
    assert out["district_target_enc"].iloc[0] == pytest.approx(y.mean())


def test_target_mean_encoder_smoothing_pulls_rare_toward_global():
    df = _toy_df()
    y = _toy_y()
    # category C has 1 row -> heavy smoothing should pull it toward global mean
    enc = TargetMeanEncoder(column="district", smoothing=1000.0).fit(df, y)
    out = enc.transform(df)
    encoded_c = out["district_target_enc"].iloc[4]
    assert abs(encoded_c - y.mean()) < abs(50.0 - y.mean())


def test_boolean_to_int_caster_casts_only_specified_cols():
    df = _toy_df()
    caster = BooleanToIntCaster(columns=["has_parking", "has_storage", "has_elevator"])
    out = caster.fit_transform(df)
    assert out["has_parking"].dtype.kind in ("i", "u")
    assert out["has_storage"].dtype.kind in ("i", "u")
    assert out["has_elevator"].dtype.kind in ("i", "u")


def test_column_selector_picks_correct_columns_in_order():
    df = _toy_df()
    df["extra"] = 999
    selector = ColumnSelector(columns=["rooms", "area_m2"])
    out = selector.fit_transform(df)
    assert list(out.columns) == ["rooms", "area_m2"]


def test_column_selector_raises_on_missing_columns():
    df = _toy_df()
    selector = ColumnSelector(columns=["does_not_exist"])
    with pytest.raises(ValueError, match="missing columns"):
        selector.transform(df)


def test_target_mean_encoder_raises_if_not_fit():
    enc = TargetMeanEncoder(column="district")
    with pytest.raises(RuntimeError, match="must be fit"):
        enc.transform(_toy_df())


def test_frequency_encoder_raises_if_not_fit():
    enc = FrequencyEncoder(column="district")
    with pytest.raises(RuntimeError, match="must be fit"):
        enc.transform(_toy_df())
