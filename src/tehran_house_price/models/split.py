"""
Train / validation split for the Tehran house price dataset.

Design choices:
    - Stratified by district to keep price distribution similar in both sets.
    - Rare districts (below a threshold) are grouped into a single bucket
      for stratification only. This keeps sklearn's stratified split happy
      when a district has fewer than 2 rows.
    - If the rare bucket itself ends up too small to stratify, its rows are
      reassigned to the largest available stratum so the split never crashes.
    - Deterministic via a single seed.
    - Writes a metadata JSON so the exact split can be audited later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import artifacts_dir, ensure_dir, processed_dir

log = get_logger(__name__)

DEFAULT_SEED: int = 42
DEFAULT_VAL_SIZE: float = 0.2
DEFAULT_MIN_SAMPLES_PER_STRATUM: int = 5
RARE_BUCKET_LABEL: str = "__rare__"
MIN_STRATUM_SIZE_FOR_SPLIT: int = 2


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Indices and metadata produced by a split operation."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    seed: int
    val_size: float
    metadata_path: Path | None = None


def _build_stratify_key(
    df: pd.DataFrame,
    stratify_col: str,
    min_samples_per_stratum: int,
) -> pd.Series:
    """
    Build a stratification key that merges rare categories into one bucket.

    sklearn's train_test_split requires every stratum to have >= 2 members.
    Rare categories (n < min_samples_per_stratum) are merged into a single
    bucket. If that merged bucket itself ends up too small (< 2 members),
    its rows are reassigned to the most populated stratum so the split is
    always feasible.

    Returns a Series of stratum labels aligned with df.index.
    """
    counts = df[stratify_col].value_counts()
    rare_categories = counts[counts < min_samples_per_stratum].index

    key = df[stratify_col].where(
        ~df[stratify_col].isin(rare_categories),
        RARE_BUCKET_LABEL,
    )

    # If the rare bucket itself is too small, fold it into the largest stratum.
    key_counts = key.value_counts()
    if (
        RARE_BUCKET_LABEL in key_counts
        and key_counts[RARE_BUCKET_LABEL] < MIN_STRATUM_SIZE_FOR_SPLIT
    ):
        # pick the largest non-rare stratum as the absorbing bucket
        non_rare = key_counts.drop(RARE_BUCKET_LABEL, errors="ignore")
        if non_rare.empty:
            # pathological: every category is rare. fall back to a single bucket.
            return pd.Series(["__all__"] * len(df), index=df.index)

        absorb_label = non_rare.idxmax()
        key = key.where(key != RARE_BUCKET_LABEL, absorb_label)
        log.warning(
            "rare bucket had %d row(s); merged into largest stratum '%s'",
            int(key_counts[RARE_BUCKET_LABEL]),
            absorb_label,
        )

    return key


def _hash_ids(ids: pd.Series) -> str:
    """Order-independent stable hash of a set of listing_ids."""
    payload = "|".join(sorted(map(str, ids.tolist())))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def split_dataframe(
    df: pd.DataFrame,
    *,
    stratify_col: str = "district",
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    min_samples_per_stratum: int = DEFAULT_MIN_SAMPLES_PER_STRATUM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train and validation sets.

    Args:
        df: input DataFrame; must contain stratify_col.
        stratify_col: column to stratify on (typically 'district').
        val_size: fraction of rows to put into the validation set.
        seed: random seed for determinism.
        min_samples_per_stratum: districts with fewer rows are merged into
            a single rare bucket for stratification purposes.

    Returns:
        (train_df, val_df), both with reset indices.

    Raises:
        ValueError: if stratify_col is missing or val_size is invalid.
    """
    if stratify_col not in df.columns:
        raise ValueError(f"stratify column '{stratify_col}' not in DataFrame")
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")

    stratify_key = _build_stratify_key(df, stratify_col, min_samples_per_stratum)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=stratify_key,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    log.info(
        "split done | total=%d train=%d val=%d seed=%d val_size=%.3f",
        len(df),
        len(train_df),
        len(val_df),
        seed,
        val_size,
    )
    return train_df, val_df


def save_split_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    seed: int,
    val_size: float,
    stratify_col: str,
    id_col: str = "listing_id",
    name: str = "split_metadata.json",
) -> Path:
    """
    Persist split metadata for reproducibility and auditing.

    The metadata file is small and JSON, so it can live next to model
    artifacts and be inspected without loading the dataset.
    """
    out_dir = ensure_dir(artifacts_dir() / "splits")
    out_path = out_dir / name

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "val_size": float(val_size),
        "stratify_col": stratify_col,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "train_ids_hash": _hash_ids(train_df[id_col]) if id_col in train_df.columns else None,
        "val_ids_hash": _hash_ids(val_df[id_col]) if id_col in val_df.columns else None,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("split metadata written to %s", out_path)
    return out_path


def run(
    *,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    stratify_col: str = "district",
    parquet_path: Path | None = None,
) -> SplitResult:
    """Load processed parquet, split, write metadata. Returns SplitResult."""
    if parquet_path is None:
        parquet_path = processed_dir() / "tehran_houses.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"processed dataset not found at {parquet_path}. run build_dataset first."
        )

    log.info("loading processed data from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    train_df, val_df = split_dataframe(
        df,
        stratify_col=stratify_col,
        val_size=val_size,
        seed=seed,
    )

    metadata_path = save_split_metadata(
        train_df,
        val_df,
        seed=seed,
        val_size=val_size,
        stratify_col=stratify_col,
    )

    return SplitResult(
        train_df=train_df,
        val_df=val_df,
        seed=seed,
        val_size=val_size,
        metadata_path=metadata_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Split processed dataset into train/val.")
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stratify-col", default="district")
    args = parser.parse_args()

    try:
        result = run(
            val_size=args.val_size,
            seed=args.seed,
            stratify_col=args.stratify_col,
        )
        print(f"split done. train={len(result.train_df)} val={len(result.val_df)}")
        print(f"metadata: {result.metadata_path}")
        return 0
    except Exception as e:
        log.error("split step failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
