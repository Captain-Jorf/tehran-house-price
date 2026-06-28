"""Unit tests for the train/val split module."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from tehran_house_price.models import split as split_mod


def _toy_df(n_per_district: int = 20) -> pd.DataFrame:
    """Build a deterministic toy DataFrame with several districts."""
    rows = []
    rid = 0
    for district in ["Vanak", "Pardis", "Shahran", "Saadat"]:
        for i in range(n_per_district):
            rows.append(
                {
                    "listing_id": f"id_{rid:04d}",
                    "district": district,
                    "area_m2": 50.0 + i,
                    "price_per_m2": 1_000_000 + 10_000 * i,
                }
            )
            rid += 1
    return pd.DataFrame(rows)


def test_split_dataframe_basic_shape():
    df = _toy_df(n_per_district=10)
    train_df, val_df = split_mod.split_dataframe(df, val_size=0.2, seed=42)
    assert len(train_df) + len(val_df) == len(df)
    assert len(val_df) == int(round(len(df) * 0.2))


def test_split_dataframe_is_deterministic():
    df = _toy_df()
    t1, v1 = split_mod.split_dataframe(df, val_size=0.2, seed=42)
    t2, v2 = split_mod.split_dataframe(df, val_size=0.2, seed=42)
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(v1, v2)


def test_split_dataframe_different_seeds_give_different_splits():
    df = _toy_df()
    _, v1 = split_mod.split_dataframe(df, val_size=0.2, seed=1)
    _, v2 = split_mod.split_dataframe(df, val_size=0.2, seed=2)
    assert set(v1["listing_id"]) != set(v2["listing_id"])


def test_split_dataframe_stratifies_districts():
    df = _toy_df(n_per_district=20)
    train_df, val_df = split_mod.split_dataframe(df, val_size=0.25, seed=42)
    train_districts = set(train_df["district"].unique())
    val_districts = set(val_df["district"].unique())
    assert train_districts == val_districts == {"Vanak", "Pardis", "Shahran", "Saadat"}


def test_split_dataframe_handles_multiple_rare_districts_merged():
    """
    Several rare districts that together form a valid bucket
    (>= 2 rows after merging) should split fine.
    """
    df = _toy_df(n_per_district=10)
    extras = pd.DataFrame(
        [
            {"listing_id": "rare_1", "district": "RareA", "area_m2": 80.0, "price_per_m2": 5e6},
            {"listing_id": "rare_2", "district": "RareB", "area_m2": 85.0, "price_per_m2": 6e6},
            {"listing_id": "rare_3", "district": "RareC", "area_m2": 90.0, "price_per_m2": 7e6},
        ]
    )
    df = pd.concat([df, extras], ignore_index=True)

    train_df, val_df = split_mod.split_dataframe(
        df,
        val_size=0.2,
        seed=42,
        min_samples_per_stratum=5,
    )
    assert len(train_df) + len(val_df) == len(df)


def test_split_dataframe_handles_single_rare_district_by_absorption():
    """
    A single rare district with only 1 row must not crash. Its row should be
    absorbed into the largest stratum so the split can still proceed.
    """
    df = _toy_df(n_per_district=10)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "listing_id": "rare_1",
                        "district": "SuperRare",
                        "area_m2": 80.0,
                        "price_per_m2": 5_000_000,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    train_df, val_df = split_mod.split_dataframe(
        df,
        val_size=0.2,
        seed=42,
        min_samples_per_stratum=5,
    )
    assert len(train_df) + len(val_df) == len(df)
    # the rare listing must end up in either train or val
    all_ids = set(train_df["listing_id"]).union(val_df["listing_id"])
    assert "rare_1" in all_ids


def test_split_dataframe_raises_on_missing_column():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(ValueError, match="stratify column"):
        split_mod.split_dataframe(df, stratify_col="district")


def test_split_dataframe_raises_on_bad_val_size():
    df = _toy_df()
    with pytest.raises(ValueError, match="val_size"):
        split_mod.split_dataframe(df, val_size=0.0)
    with pytest.raises(ValueError, match="val_size"):
        split_mod.split_dataframe(df, val_size=1.0)


def test_save_split_metadata_writes_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(split_mod, "artifacts_dir", lambda: tmp_path)

    df = _toy_df()
    train_df, val_df = split_mod.split_dataframe(df, val_size=0.25, seed=7)
    path = split_mod.save_split_metadata(
        train_df,
        val_df,
        seed=7,
        val_size=0.25,
        stratify_col="district",
    )
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["seed"] == 7
    assert data["val_size"] == 0.25
    assert data["stratify_col"] == "district"
    assert data["n_train"] == len(train_df)
    assert data["n_val"] == len(val_df)
    assert isinstance(data["train_ids_hash"], str) and len(data["train_ids_hash"]) == 16
    assert isinstance(data["val_ids_hash"], str) and len(data["val_ids_hash"]) == 16


def test_hash_ids_is_order_independent():
    s1 = pd.Series(["a", "b", "c"])
    s2 = pd.Series(["c", "a", "b"])
    assert split_mod._hash_ids(s1) == split_mod._hash_ids(s2)


def test_run_raises_when_parquet_missing(tmp_path):
    bogus = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError):
        split_mod.run(parquet_path=bogus)


def test_run_end_to_end_on_real_processed_data(monkeypatch, tmp_path):
    """If the processed parquet exists in the repo, run end-to-end."""
    monkeypatch.setattr(split_mod, "artifacts_dir", lambda: tmp_path)

    from tehran_house_price.utils.paths import processed_dir

    parquet_path = processed_dir() / "tehran_houses.parquet"
    if not parquet_path.exists():
        pytest.skip("processed parquet not present; skipping integration check")

    result = split_mod.run(val_size=0.2, seed=42, parquet_path=parquet_path)
    assert result.metadata_path is not None
    assert len(result.train_df) + len(result.val_df) > 0
    assert result.metadata_path.exists()
