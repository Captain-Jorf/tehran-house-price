"""Unit tests for the main XGBoost training module."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from tehran_house_price.models import train as train_mod
from tehran_house_price.models.train import build_full_pipeline


def _toy_dataset(n: int = 200) -> pd.DataFrame:
    """
    Small synthetic dataset that mimics the real schema:
    price_per_m2 is a noisy function of district + area.
    """
    rng = np.random.default_rng(0)
    districts = ["Vanak", "Pardis", "Shahran", "Saadat"]
    base_prices = {
        "Vanak": 60_000_000,
        "Pardis": 20_000_000,
        "Shahran": 35_000_000,
        "Saadat": 45_000_000,
    }
    rows = []
    for i in range(n):
        d = districts[i % len(districts)]
        base = base_prices[d]
        area = 50 + int(rng.integers(0, 80))
        rooms = max(1, int(area / 30))
        noise = rng.normal(0, base * 0.05)
        price_per_m2 = max(1_000_000, base + noise)
        rows.append(
            {
                "listing_id": f"id_{i:04d}",
                "district": d,
                "area_m2": float(area),
                "rooms": rooms,
                "has_parking": bool(i % 2),
                "has_storage": bool(i % 3),
                "has_elevator": bool(i % 4),
                "price_per_m2": float(price_per_m2),
            }
        )
    return pd.DataFrame(rows)


def test_log_target_regressor_predicts_in_original_scale():
    df = _toy_dataset(120)
    X = df.drop(columns=["price_per_m2", "listing_id"])
    y = df["price_per_m2"]

    pipe = build_full_pipeline(xgb_params={"n_estimators": 50, "max_depth": 3})
    pipe.fit(X, y)
    preds = pipe.predict(X)

    # predictions must be in the same order of magnitude as the target
    assert preds.shape == (len(X),)
    assert np.all(preds > 0)
    assert np.median(preds) > 1_000_000
    assert np.median(preds) < 1_000_000_000


def test_build_full_pipeline_is_serializable(tmp_path):
    import joblib

    df = _toy_dataset(120)
    X = df.drop(columns=["price_per_m2", "listing_id"])
    y = df["price_per_m2"]

    pipe = build_full_pipeline(xgb_params={"n_estimators": 50, "max_depth": 3})
    pipe.fit(X, y)
    preds_before = pipe.predict(X)

    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    preds_after = loaded.predict(X)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)


def test_build_full_pipeline_handles_unseen_district():
    df = _toy_dataset(120)
    X = df.drop(columns=["price_per_m2", "listing_id"])
    y = df["price_per_m2"]

    pipe = build_full_pipeline(xgb_params={"n_estimators": 50, "max_depth": 3})
    pipe.fit(X, y)

    new_X = pd.DataFrame(
        [
            {
                "district": "TOTALLY_NEW",
                "area_m2": 90.0,
                "rooms": 3,
                "has_parking": True,
                "has_storage": False,
                "has_elevator": True,
            }
        ]
    )
    preds = pipe.predict(new_X)
    assert preds.shape == (1,)
    assert preds[0] > 0
    assert np.isfinite(preds[0])


def test_log_target_regressor_calls_inverse_transform():
    """Sanity: log target regressor predictions are not in log space."""
    df = _toy_dataset(80)
    X = df.drop(columns=["price_per_m2", "listing_id"])
    y = df["price_per_m2"]

    pipe = build_full_pipeline(xgb_params={"n_estimators": 30, "max_depth": 3})
    pipe.fit(X, y)
    preds = pipe.predict(X)

    # If we forgot to invert, predictions would be ~log(price) (e.g. < 30).
    assert np.median(preds) > 1_000_000


def test_train_end_to_end_writes_artifacts(tmp_path, monkeypatch):
    """End-to-end train() should produce model + metadata + evaluation report."""
    from tehran_house_price.models import split as split_mod_local

    # Sandbox all writes to tmp_path
    monkeypatch.setattr(train_mod, "artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(split_mod_local, "artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr("tehran_house_price.models.evaluation.artifacts_dir", lambda: tmp_path)

    df = _toy_dataset(300)

    # Make split.run() use our toy DataFrame instead of reading from disk.
    def fake_split_run(*, val_size, seed, **_kw):
        from tehran_house_price.models.split import (
            SplitResult,
            save_split_metadata,
            split_dataframe,
        )

        train_df, val_df = split_dataframe(df, val_size=val_size, seed=seed)
        meta_path = save_split_metadata(
            train_df,
            val_df,
            seed=seed,
            val_size=val_size,
            stratify_col="district",
        )
        return SplitResult(
            train_df=train_df,
            val_df=val_df,
            seed=seed,
            val_size=val_size,
            metadata_path=meta_path,
        )

    monkeypatch.setattr(train_mod.split_mod, "run", fake_split_run)

    result = train_mod.train(
        val_size=0.2,
        seed=42,
        model_name="xgb_test",
        xgb_params={"n_estimators": 50, "max_depth": 3},
    )

    assert result.model_path.exists()
    assert result.metadata_path.exists()
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["model_name"] == "xgb_test"
    assert "hyperparameters" in meta
    assert "metrics" in meta
    assert meta["metrics"]["mape"] >= 0


def test_train_raises_when_target_missing(tmp_path, monkeypatch):
    """train() must fail loudly if processed data has no target column."""
    from tehran_house_price.models import split as split_mod_local

    monkeypatch.setattr(train_mod, "artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(split_mod_local, "artifacts_dir", lambda: tmp_path)

    df = _toy_dataset(60).drop(columns=["price_per_m2"])

    def fake_split_run(*, val_size, seed, **_kw):
        from tehran_house_price.models.split import SplitResult

        return SplitResult(
            train_df=df.iloc[: int(len(df) * (1 - val_size))].reset_index(drop=True),
            val_df=df.iloc[int(len(df) * (1 - val_size)) :].reset_index(drop=True),
            seed=seed,
            val_size=val_size,
            metadata_path=None,
        )

    monkeypatch.setattr(train_mod.split_mod, "run", fake_split_run)

    with pytest.raises(ValueError, match="target column"):
        train_mod.train(model_name="xgb_no_target")
