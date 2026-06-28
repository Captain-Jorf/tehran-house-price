"""Integration tests for the train pipeline orchestrator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from tehran_house_price.models import train_pipeline as tp


def _toy_dataset(n: int = 200) -> pd.DataFrame:
    """Synthetic dataset matching the real processed schema."""
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
                "total_price": float(price_per_m2 * area),
                "price_per_m2": float(price_per_m2),
            }
        )
    return pd.DataFrame(rows)


def _setup_sandbox(tmp_path, monkeypatch, df: pd.DataFrame) -> None:
    """
    Redirect every artifact write to tmp_path, and have split.run() use
    the in-memory toy DataFrame instead of reading a parquet file.
    """
    from tehran_house_price.models import baseline as baseline_mod
    from tehran_house_price.models import evaluation as ev
    from tehran_house_price.models import split as split_mod
    from tehran_house_price.models import train as train_mod

    monkeypatch.setattr(tp, "artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(baseline_mod, "artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(train_mod, "artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(ev, "artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(split_mod, "artifacts_dir", lambda: tmp_path)

    # Make split.run() use the in-memory toy df
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

    monkeypatch.setattr(split_mod, "run", fake_split_run)

    # Make processed_dir() point to a path holding our toy parquet so the
    # comparison loader can read it.
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed / "tehran_houses.parquet", index=False)
    monkeypatch.setattr(tp, "processed_dir", lambda: processed)


def test_run_trains_all_models_and_writes_comparison(tmp_path, monkeypatch):
    df = _toy_dataset(300)
    _setup_sandbox(tmp_path, monkeypatch, df)

    # Use a tiny xgb for speed
    from tehran_house_price.models import train as train_mod

    monkeypatch.setattr(
        train_mod,
        "DEFAULT_XGB_PARAMS",
        {**train_mod.DEFAULT_XGB_PARAMS, "n_estimators": 30, "max_depth": 3},
    )

    result = tp.run(val_size=0.2, seed=42, model_name="xgb_test")

    assert result.main_model_path is not None and result.main_model_path.exists()
    assert result.main_metadata_path is not None and result.main_metadata_path.exists()
    assert len(result.baseline_paths) == 2
    for p in result.baseline_paths:
        assert p.exists()

    assert result.comparison_path is not None and result.comparison_path.exists()
    assert result.n_models_compared >= 3  # 2 baselines + 1 main

    payload = json.loads(result.comparison_path.read_text(encoding="utf-8"))
    assert payload["n_models"] == result.n_models_compared
    assert payload["ranking_metric"] == "mape"
    assert isinstance(payload["models"], list)
    assert all("mape" in row for row in payload["models"])

    # First model in ranked list should have the smallest MAPE
    mapes = [row["mape"] for row in payload["models"]]
    assert mapes == sorted(mapes)


def test_run_skip_baselines_does_not_retrain_them(tmp_path, monkeypatch):
    df = _toy_dataset(200)
    _setup_sandbox(tmp_path, monkeypatch, df)

    # First run: train baselines and main
    from tehran_house_price.models import train as train_mod

    monkeypatch.setattr(
        train_mod,
        "DEFAULT_XGB_PARAMS",
        {**train_mod.DEFAULT_XGB_PARAMS, "n_estimators": 30, "max_depth": 3},
    )

    first = tp.run(val_size=0.2, seed=42, model_name="xgb_test")
    assert len(first.baseline_paths) == 2

    # Second run with skip_baselines: should report 0 baseline paths
    second = tp.run(val_size=0.2, seed=42, model_name="xgb_test", skip_baselines=True)
    assert second.baseline_paths == []
    # but comparison should still include previously-trained baselines + new main
    assert second.n_models_compared >= 3


def test_run_skip_main_does_not_produce_main_model(tmp_path, monkeypatch):
    df = _toy_dataset(200)
    _setup_sandbox(tmp_path, monkeypatch, df)

    result = tp.run(val_size=0.2, seed=42, skip_main=True)
    assert result.main_model_path is None
    assert result.main_metadata_path is None
    assert len(result.baseline_paths) == 2
    # comparison should still be produced from baselines alone
    assert result.comparison_path is not None
    assert result.n_models_compared == 2


def test_run_raises_when_skipping_everything(tmp_path, monkeypatch):
    df = _toy_dataset(100)
    _setup_sandbox(tmp_path, monkeypatch, df)

    with pytest.raises(ValueError, match="cannot skip both"):
        tp.run(skip_baselines=True, skip_main=True)
