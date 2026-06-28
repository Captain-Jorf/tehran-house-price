"""Unit tests for baseline regressors."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from tehran_house_price.models import baseline as baseline_mod
from tehran_house_price.models.baseline import (
    DistrictMedianBaseline,
    MeanPriceBaseline,
)


def _toy_train_val():
    train_df = pd.DataFrame(
        {
            "district": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "area_m2": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 55.0, 65.0],
            "price_per_m2": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 50.0, 70.0],
        }
    )
    val_df = pd.DataFrame(
        {
            "district": ["A", "B", "C", "UNSEEN"],
            "area_m2": [55.0, 95.0, 60.0, 75.0],
            "price_per_m2": [25.0, 250.0, 60.0, 80.0],
        }
    )
    return train_df, val_df


def test_mean_baseline_predicts_global_mean():
    train_df, val_df = _toy_train_val()
    model = MeanPriceBaseline()
    model.fit(train_df, train_df["price_per_m2"])
    preds = model.predict(val_df)
    expected = float(train_df["price_per_m2"].mean())
    assert preds.shape == (len(val_df),)
    np.testing.assert_allclose(preds, expected)


def test_mean_baseline_raises_if_not_fit():
    model = MeanPriceBaseline()
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict(pd.DataFrame({"x": [1, 2]}))


def test_district_median_baseline_predicts_per_district_median():
    train_df, val_df = _toy_train_val()
    model = DistrictMedianBaseline()
    model.fit(train_df, train_df["price_per_m2"])
    preds = model.predict(val_df)

    # expected per-district medians:
    # A: median(10,20,30)=20, B: median(100,200,300)=200, C: median(50,70)=60
    # UNSEEN: global median = median(10,20,30,100,200,300,50,70) = 60
    expected = np.array([20.0, 200.0, 60.0, 60.0])
    np.testing.assert_allclose(preds, expected)


def test_district_median_baseline_handles_unseen_with_global_median():
    train_df, val_df = _toy_train_val()
    model = DistrictMedianBaseline()
    model.fit(train_df, train_df["price_per_m2"])
    new_val = pd.DataFrame({"district": ["NEW1", "NEW2"]})
    preds = model.predict(new_val)
    global_median = float(np.median(train_df["price_per_m2"]))
    np.testing.assert_allclose(preds, [global_median, global_median])


def test_district_median_baseline_raises_if_not_fit():
    model = DistrictMedianBaseline()
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict(pd.DataFrame({"district": ["A"]}))


def test_district_median_baseline_raises_on_missing_column():
    train_df, _ = _toy_train_val()
    model = DistrictMedianBaseline(district_col="missing_col")
    with pytest.raises(ValueError, match="district column"):
        model.fit(train_df, train_df["price_per_m2"])


def test_compute_basic_metrics_correctness():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    metrics = baseline_mod._compute_basic_metrics(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(np.mean([2.0, 2.0, 3.0]))
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean([4.0, 4.0, 9.0])))
    assert metrics["mape"] == pytest.approx(np.mean([2 / 10, 2 / 20, 3 / 30]))
    assert metrics["n"] == 3


def test_compute_basic_metrics_handles_zero_in_y_true():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 12.0])
    metrics = baseline_mod._compute_basic_metrics(y_true, y_pred)
    # mape should ignore the zero-target row (nanmean)
    assert metrics["mape"] == pytest.approx(2 / 10)


def test_train_baseline_persists_model_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_mod, "artifacts_dir", lambda: tmp_path)

    train_df, val_df = _toy_train_val()
    result = baseline_mod.train_baseline(
        name="baseline_mean",
        model=MeanPriceBaseline(),
        train_df=train_df,
        val_df=val_df,
    )
    assert result.model_path.exists()
    assert result.metrics_path.exists()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["model"] == "baseline_mean"
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics


def test_train_baseline_raises_on_missing_target():
    train_df, val_df = _toy_train_val()
    train_df = train_df.drop(columns=["price_per_m2"])
    with pytest.raises(ValueError, match="target column"):
        baseline_mod.train_baseline(
            name="baseline_mean",
            model=MeanPriceBaseline(),
            train_df=train_df,
            val_df=val_df,
        )


def test_baseline_serializable_with_joblib(tmp_path):
    import joblib

    train_df, val_df = _toy_train_val()
    model = DistrictMedianBaseline()
    model.fit(train_df, train_df["price_per_m2"])
    preds_before = model.predict(val_df)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    preds_after = loaded.predict(val_df)

    np.testing.assert_allclose(preds_before, preds_after)
