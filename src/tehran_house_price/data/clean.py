"""
Data cleaning for Kaggle Tehran house price dataset.

Raw CSV از Kaggle شکل خاص خودش را دارد. اینجا به canonical schema تبدیلش
می‌کنیم. هدف: یک DataFrame تمیز و parquet-ready که با HouseListingSchema
validate شود.

این کد فقط cleaning است. feature engineering نیست. آن مرحله بعد می‌آید.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tehran_house_price.data import constants as const
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import ensure_dir, interim_dir, raw_dir

log = get_logger(__name__)


# Kaggle raw column -> canonical column
KAGGLE_COLUMN_MAP: dict[str, str] = {
    "Area": const.AREA_M2,
    "Room": const.ROOMS,
    "Parking": const.HAS_PARKING,
    "Warehouse": const.HAS_STORAGE,
    "Elevator": const.HAS_ELEVATOR,
    "Address": const.DISTRICT,
    "Price": const.TOTAL_PRICE,
}

# columns to drop from raw kaggle data
KAGGLE_DROP_COLS: list[str] = ["Price(USD)"]


def load_kaggle_csv(path: Path | None = None) -> pd.DataFrame:
    """Read the kaggle housePrice.csv file as a DataFrame."""
    if path is None:
        candidates = sorted((raw_dir() / "kaggle").glob("*.csv"))
        if not candidates:
            raise FileNotFoundError("no csv files in data/raw/kaggle/. run ingest_kaggle first.")
        path = candidates[0]

    log.info("reading raw csv from %s", path)
    df = pd.read_csv(path)
    log.info("loaded shape=%s columns=%s", df.shape, list(df.columns))
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns we explicitly don't want."""
    to_drop = [c for c in KAGGLE_DROP_COLS if c in df.columns]
    if to_drop:
        log.info("dropping columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw kaggle columns to canonical names."""
    return df.rename(columns=KAGGLE_COLUMN_MAP)


def coerce_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Area sometimes comes in as a string with commas (e.g. '1,200').
    Force it to numeric. Non-parsable -> NaN (will be dropped later).
    """
    col = const.AREA_M2
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def coerce_price(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure price is numeric (float)."""
    df[const.TOTAL_PRICE] = pd.to_numeric(df[const.TOTAL_PRICE], errors="coerce")
    return df


def coerce_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """Rooms must be a non-negative int. Coerce failures -> 0."""
    df[const.ROOMS] = pd.to_numeric(df[const.ROOMS], errors="coerce").fillna(0).astype(int)
    return df


def coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """
    True/False columns in raw CSV are strings like 'True'/'False'.
    Map them to real booleans.
    """
    bool_cols = [const.HAS_ELEVATOR, const.HAS_PARKING, const.HAS_STORAGE]
    truthy = {"true", "1", "yes", "y"}
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(truthy)
    return df


def clean_district(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and drop rows with empty district."""
    col = const.DISTRICT
    df[col] = df[col].astype(str).str.strip()
    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that clearly can't be valid listings.
      - area or price NaN/zero
      - area or price out of sane bounds
      - empty district
    """
    n_before = len(df)

    df = df.dropna(subset=[const.AREA_M2, const.TOTAL_PRICE, const.DISTRICT])

    df = df[(df[const.AREA_M2] >= const.MIN_AREA_M2) & (df[const.AREA_M2] <= const.MAX_AREA_M2)]
    df = df[
        (df[const.TOTAL_PRICE] >= const.MIN_PRICE_IRR)
        & (df[const.TOTAL_PRICE] <= const.MAX_PRICE_IRR)
    ]
    df = df[df[const.DISTRICT].str.len() > 0]

    n_after = len(df)
    log.info("dropped %d invalid rows (%d -> %d)", n_before - n_after, n_before, n_after)
    return df.reset_index(drop=True)


def add_price_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price per square meter."""
    df[const.PRICE_PER_M2] = df[const.TOTAL_PRICE] / df[const.AREA_M2]
    return df


def add_source(df: pd.DataFrame, source: str = const.SOURCE_KAGGLE) -> pd.DataFrame:
    df[const.SOURCE] = source
    return df


def add_listing_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaggle data has no id. Generate a stable hash from row content.
    Stable = same row gives same id on next run, useful for dedup.
    """

    def _hash_row(row: pd.Series) -> str:
        key = (
            f"{row[const.SOURCE]}|{row[const.DISTRICT]}|{row[const.AREA_M2]}|"
            f"{row[const.ROOMS]}|{row[const.TOTAL_PRICE]}"
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    df[const.LISTING_ID] = df.apply(_hash_row, axis=1)
    return df


def add_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle has no publish date, so leave it null. Always set ingested_at."""
    now = datetime.now(timezone.utc)
    df[const.PUBLISHED_AT] = pd.NaT
    df[const.INGESTED_AT] = now
    return df


def add_missing_optional_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Schema expects these columns even if all-null for kaggle source."""
    for col in (const.NEIGHBORHOOD, const.YEAR_BUILT, const.FLOOR, const.TOTAL_FLOORS):
        if col not in df.columns:
            df[col] = pd.NA
    return df


def normalize_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finalize dtypes so the parquet output is predictable for downstream code.

    چرا اینجا و نه فقط در validate?
        validate() قبل از validation کارش را می‌کند، ولی parquet خروجی همچنان
        با dtype های اولیه ذخیره می‌شد. این باعث می‌شد ستون‌های nullable به‌صورت
        'object' ذخیره شوند. اینجا dtype های نهایی را تثبیت می‌کنیم تا
        مصرف‌کننده‌ی processed parquet با ستون‌های float تمیز کار کند، نه با
        object هایی که فقط NA دارند.

    تغییرات اعمال‌شده:
        - area_m2: float (یکدست با schema)
        - year_built, floor, total_floors: float64 (می‌تواند NaN داشته باشد)
    """
    df = df.copy()

    if const.AREA_M2 in df.columns:
        df[const.AREA_M2] = df[const.AREA_M2].astype(float)

    for col in (const.YEAR_BUILT, const.FLOOR, const.TOTAL_FLOORS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where listing_id duplicates."""
    n_before = len(df)
    df = df.drop_duplicates(subset=[const.LISTING_ID]).reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        log.info("dropped %d duplicate rows", n_before - n_after)
    return df


def clean_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for kaggle data.

    Order matters here. For example, listing_id depends on source and district,
    so add_source must run first. normalize_output_dtypes در آخر اجرا می‌شود
    تا dtype های نهایی برای ذخیره parquet تثبیت شوند.
    """
    df = drop_unused_columns(df)
    df = rename_to_canonical(df)
    df = coerce_area(df)
    df = coerce_price(df)
    df = coerce_rooms(df)
    df = coerce_booleans(df)
    df = clean_district(df)
    df = drop_invalid_rows(df)
    df = add_price_per_m2(df)
    df = add_source(df)
    df = add_listing_id(df)
    df = add_timestamps(df)
    df = add_missing_optional_cols(df)
    df = drop_duplicates(df)
    df = normalize_output_dtypes(df)
    return df


def save_interim(df: pd.DataFrame, name: str = "kaggle_clean.parquet") -> Path:
    """Save cleaned DataFrame as parquet in data/interim/."""
    out_dir = ensure_dir(interim_dir())
    out_path = out_dir / name
    df.to_parquet(out_path, index=False)
    log.info("wrote %d rows to %s", len(df), out_path)
    return out_path


def run() -> Path:
    """End-to-end: load -> clean -> save. Returns path to parquet."""
    raw = load_kaggle_csv()
    cleaned = clean_kaggle(raw)
    return save_interim(cleaned)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Clean kaggle raw csv into interim parquet.")
    parser.parse_args()
    try:
        path = run()
        print(f"done. cleaned file: {path}")
        return 0
    except Exception as e:
        log.error("cleaning failed: %s", e)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
