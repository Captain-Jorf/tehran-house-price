"""
Data validation.

Cleaned DataFrame را در برابر HouseListingSchema چک می‌کنیم. اگر داده
معتبر بود یک گزارش JSON می‌سازیم در artifacts/data_validation/.

این مرحله یک gate است: اگر داده schema را پاس نکند، نمی‌گذاریم برود به
processed/. این جلوی خراب شدن downstream را می‌گیرد.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from tehran_house_price.data import constants as const
from tehran_house_price.data.schema import HouseListingSchema
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import artifacts_dir, ensure_dir, interim_dir

log = get_logger(__name__)


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make dtypes pandera-friendly before validation.

    مشکل اصلی: ستون‌هایی که می‌توانند NaN داشته باشند (year_built, floor,
    total_floors) را نمی‌توان به numpy int64 تبدیل کرد چون int64 نمی‌تواند
    NaN داشته باشد. راه‌حل: این ستون‌ها را به float64 تبدیل می‌کنیم.
    float64 می‌تواند NaN داشته باشد و schema هم آن‌ها را float تعریف کرده.

    این تابع قبل از pandera validation صدا زده می‌شود.
    """
    df = df.copy()

    # area_m2: اگر int بود به float تبدیل کن
    if const.AREA_M2 in df.columns:
        df[const.AREA_M2] = df[const.AREA_M2].astype(float)

    # nullable columns: Int64 یا object را به float64 تبدیل کن
    # float64 می‌تواند NaN داشته باشد - pandera بدون مشکل handle می‌کند
    for col in (const.YEAR_BUILT, const.FLOOR, const.TOTAL_FLOORS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    Run pandera schema validation. Returns (is_valid, report_dict).

    Even if validation passes, the report contains basic data quality stats
    that are useful to keep around.
    """
    report: dict = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "null_counts": df.isna().sum().to_dict(),
        "passed": False,
        "errors": [],
    }

    # null_counts uses numpy types -> JSON can't always serialize. cast to int.
    report["null_counts"] = {k: int(v) for k, v in report["null_counts"].items()}

    try:
        df_norm = _normalize_dtypes(df)
        HouseListingSchema.validate(df_norm, lazy=True)
        report["passed"] = True
        log.info("schema validation passed for %d rows", len(df))

    except SchemaErrors as e:
        # lazy=True collects all errors at once
        report["passed"] = False
        report["errors"] = _extract_schema_errors(e)
        log.error("schema validation failed with %d errors", len(report["errors"]))

    except SchemaError as e:
        # eager mode fallback
        report["passed"] = False
        report["errors"] = [{"message": str(e)}]
        log.error("schema validation failed: %s", e)

    return report["passed"], report


def _extract_schema_errors(e: SchemaErrors) -> list[dict]:
    """Convert pandera SchemaErrors into JSON-friendly list of dicts."""
    out: list[dict] = []
    failure_cases = e.failure_cases
    if failure_cases is None or failure_cases.empty:
        out.append({"message": str(e)})
        return out

    for _, row in failure_cases.iterrows():
        out.append(
            {
                "column": str(row.get("column", "")),
                "check": str(row.get("check", "")),
                "failure_case": str(row.get("failure_case", "")),
                "index": int(row["index"]) if pd.notna(row.get("index")) else None,
            }
        )
    return out


def save_report(report: dict, name: str = "kaggle_validation.json") -> Path:
    """Persist validation report to artifacts/data_validation/."""
    out_dir = ensure_dir(artifacts_dir() / "data_validation")
    out_path = out_dir / name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("validation report written to %s", out_path)
    return out_path


def run(parquet_path: Path | None = None) -> tuple[bool, Path]:
    """Load cleaned parquet, validate, write report. Returns (passed, report_path)."""
    if parquet_path is None:
        parquet_path = interim_dir() / "kaggle_clean.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"cleaned data not found at {parquet_path}. run cleaning step first."
        )

    log.info("loading cleaned data from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    passed, report = validate_dataframe(df)
    report_path = save_report(report)

    return passed, report_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate cleaned dataset against schema.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with code 1 if validation fails",
    )
    args = parser.parse_args()

    try:
        passed, report_path = run()
        status = "PASSED" if passed else "FAILED"
        print(f"validation {status}. report: {report_path}")
        if args.strict and not passed:
            return 1
        return 0
    except Exception as e:
        log.error("validation step crashed: %s", e)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
