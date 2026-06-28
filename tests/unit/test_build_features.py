"""Unit tests for the feature pipeline assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tehran_house_price.features import constants as fconst
from tehran_house_price.features.build_features import (
    build_feature_pipeline,
    inverse_target_for_prediction,
    transform_target_for_training,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district": ["A", "A", "B", "B", "C", "C"],
            "area_m2": [50.0, 80.0, 100.0, 120.0, 60.0, 75.0],
            "rooms": [1, 2, 2, 3, 1, 2],
            "has_parking": [True, False, True, True, False, True],
            "has_storage": [False, False, True, True, False, True],
            "has_elevator": [True, True, True, False, False, True],
        }
    )


def _toy_y() -> pd.Series:
    return pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])


def test_build_feature_pipeline_returns_correct_columns():
    pipe = build_feature_pipeline()
    X = _toy_df()
    y = _toy_y()
    out = pipe.fit_transform(X, y)
    assert list(out.columns) == fconst.FINAL_FEATURE_COLS


def test_build_feature_pipeline_all_features_numeric():
    pipe = build_feature_pipeline()
    X = _toy_df()
    y = _toy_y()
    out = pipe.fit_transform(X, y)
    for col in out.columns:
        assert np.issubdtype(out[col].dtype, np.number), f"{col} is not numeric"


def test_build_feature_pipeline_serializable_with_joblib(tmp_path):
    import joblib

    pipe = build_feature_pipeline()
    X = _toy_df()
    y = _toy_y()
    pipe.fit(X, y)

    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)

    out_original = pipe.transform(X)
    out_loaded = loaded.transform(X)
    pd.testing.assert_frame_equal(out_original, out_loaded)


def test_build_feature_pipeline_handles_unseen_district():
    pipe = build_feature_pipeline()
    X = _toy_df()
    y = _toy_y()
    pipe.fit(X, y)

    new_X = pd.DataFrame(
        {
            "district": ["UNSEEN_DISTRICT"],
            "area_m2": [90.0],
            "rooms": [2],
            "has_parking": [True],
            "has_storage": [False],
            "has_elevator": [True],
        }
    )
    out = pipe.transform(new_X)
    assert out.shape == (1, len(fconst.FINAL_FEATURE_COLS))
    assert np.isfinite(out.values).all()


def test_target_transform_and_inverse_roundtrip():
    y = pd.Series([100.0, 1_000_000.0, 50_000_000.0])
    y_log = transform_target_for_training(y)
    y_back = inverse_target_for_prediction(y_log)
    np.testing.assert_allclose(y_back, y.values, rtol=1e-9)


def test_target_transform_returns_finite_for_positive_values():
    y = pd.Series([1.0, 100.0, 1e9])
    y_log = transform_target_for_training(y)
    assert np.isfinite(y_log).all()
