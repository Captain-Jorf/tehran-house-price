"""
Regression evaluation layer for the Tehran house price project.

This module owns all metric computation. baseline.py, the main model
trainer, and (later) monitoring will all import from here so the project
has a single canonical definition of "how we score a model".

Why a dedicated layer?
    - Consistency: every model uses the same metric implementations.
    - Reuse: Phase 5 (MLflow) and Phase 7 (monitoring) can log these
      metrics without rewriting them.
    - Testability: metric formulas live in one place that is easy to test.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import artifacts_dir, ensure_dir

log = get_logger(__name__)

DEFAULT_DISTRICT_COL: str = "district"
DEFAULT_TARGET_COL: str = "price_per_m2"
TOP_K_WORST_DISTRICTS: int = 5
MIN_DISTRICT_SAMPLES_FOR_BREAKDOWN: int = 3


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Structured evaluation output for a single model."""

    model_name: str
    global_metrics: dict[str, float]
    n_samples: int
    per_district: pd.DataFrame
    worst_districts: pd.DataFrame
    report_path: Path | None = None


def _safe_array(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=float)


def compute_global_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """
    Compute the full set of global regression metrics.

    Notes:
        - MAPE protects against y_true == 0 by ignoring those rows.
        - R^2 falls back to NaN when the variance of y_true is zero,
          which is the mathematically correct behavior.
    """
    y_true = _safe_array(y_true)
    y_pred = _safe_array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    medae = float(np.median(abs_errors))

    safe_true = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs(errors / safe_true)))

    var_y = float(np.var(y_true))
    if var_y == 0:
        r2 = float("nan")
    else:
        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot)

    return {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "mape": mape,
        "r2": r2,
    }


def compute_per_district_metrics(
    y_true: Any,
    y_pred: Any,
    districts: pd.Series,
    *,
    min_samples: int = MIN_DISTRICT_SAMPLES_FOR_BREAKDOWN,
) -> pd.DataFrame:
    """
    Compute MAE and MAPE per district. Districts with fewer than min_samples
    rows in the evaluation set are excluded because their metrics are noisy.

    Returns a DataFrame indexed by district, sorted by MAPE descending
    (worst-performing districts first).
    """
    y_true = _safe_array(y_true)
    y_pred = _safe_array(y_pred)

    if not (len(y_true) == len(y_pred) == len(districts)):
        raise ValueError("y_true, y_pred, and districts must all have the same length")

    df = pd.DataFrame(
        {
            "district": districts.values,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
    df["abs_pct_err"] = np.where(
        df["y_true"] == 0,
        np.nan,
        np.abs((df["y_true"] - df["y_pred"]) / df["y_true"]),
    )

    grouped = df.groupby("district").agg(
        n=("y_true", "size"),
        mae=("abs_err", "mean"),
        mape=("abs_pct_err", "mean"),
    )

    grouped = grouped[grouped["n"] >= min_samples]
    grouped = grouped.sort_values(by="mape", ascending=False)
    return grouped


def select_worst_districts(
    per_district: pd.DataFrame,
    *,
    top_k: int = TOP_K_WORST_DISTRICTS,
) -> pd.DataFrame:
    """Return the top_k worst-performing districts by MAPE."""
    if per_district.empty:
        return per_district
    return per_district.head(top_k).copy()


def evaluate(
    model_name: str,
    y_true: Any,
    y_pred: Any,
    *,
    districts: pd.Series | None = None,
    save: bool = False,
) -> EvaluationReport:
    """
    Build a full EvaluationReport for one model.

    Args:
        model_name: identifier used in the saved report filename.
        y_true: true target values.
        y_pred: predicted target values.
        districts: optional Series aligned with y_true for per-district breakdown.
        save: if True, persist the report as JSON.

    Returns:
        EvaluationReport with metrics, per-district table, and worst districts.
    """
    global_metrics = compute_global_metrics(y_true, y_pred)

    if districts is not None:
        per_district = compute_per_district_metrics(y_true, y_pred, districts)
        worst = select_worst_districts(per_district)
    else:
        per_district = pd.DataFrame(columns=["n", "mae", "mape"])
        worst = per_district.copy()

    report_path: Path | None = None
    if save:
        report_path = save_evaluation_report(
            model_name=model_name,
            global_metrics=global_metrics,
            per_district=per_district,
            worst=worst,
            n_samples=len(np.asarray(y_true)),
        )

    log.info(
        "evaluated '%s' | mae=%.2f rmse=%.2f mape=%.4f r2=%.4f n=%d",
        model_name,
        global_metrics["mae"],
        global_metrics["rmse"],
        global_metrics["mape"],
        global_metrics["r2"],
        len(np.asarray(y_true)),
    )

    return EvaluationReport(
        model_name=model_name,
        global_metrics=global_metrics,
        n_samples=int(len(np.asarray(y_true))),
        per_district=per_district,
        worst_districts=worst,
        report_path=report_path,
    )


def save_evaluation_report(
    *,
    model_name: str,
    global_metrics: dict[str, float],
    per_district: pd.DataFrame,
    worst: pd.DataFrame,
    n_samples: int,
    name: str | None = None,
) -> Path:
    """Persist a full evaluation report as JSON."""
    out_dir = ensure_dir(artifacts_dir() / "model_evaluation")
    fname = name or f"{model_name}_evaluation.json"
    out_path = out_dir / fname

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "n_samples": int(n_samples),
        "global_metrics": global_metrics,
        "per_district": per_district.reset_index().to_dict(orient="records"),
        "worst_districts": worst.reset_index().to_dict(orient="records"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("evaluation report written to %s", out_path)
    return out_path


def compare_models(reports: list[EvaluationReport]) -> pd.DataFrame:
    """
    Build a side-by-side comparison table of multiple evaluation reports.

    Returns:
        DataFrame with one row per model and one column per global metric,
        sorted by MAPE ascending (best model first).
    """
    if not reports:
        raise ValueError("compare_models requires at least one report")

    rows = []
    for r in reports:
        row: dict[str, Any] = {"model": r.model_name, "n_samples": r.n_samples}
        row.update(r.global_metrics)
        rows.append(row)

    out = pd.DataFrame(rows)
    if "mape" in out.columns:
        out = out.sort_values(by="mape", ascending=True).reset_index(drop=True)
    return out


def main() -> int:
    """
    CLI: evaluate the currently saved baselines and print a comparison table.

    This is a convenience entry point; it loads the val split from the
    processed parquet, loads the persisted baseline models, evaluates each,
    and writes both individual reports and a comparison report.

    Note on the import below:
        joblib pickles models by reference to their class. To unpickle
        baseline models we must have their classes available in this
        process namespace. The import has no runtime effect beyond that.
    """
    import joblib

    # Required so joblib can resolve the baseline classes at unpickle time.
    from tehran_house_price.models import baseline as _baseline  # noqa: F401
    from tehran_house_price.models import split as split_mod
    from tehran_house_price.utils.paths import processed_dir

    parser = argparse.ArgumentParser(description="Evaluate persisted baseline models on val split.")
    parser.add_argument("--val-size", type=float, default=split_mod.DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=split_mod.DEFAULT_SEED)
    args = parser.parse_args()

    try:
        models_dir = artifacts_dir() / "models"
        candidates = sorted(models_dir.glob("baseline_*.joblib"))
        if not candidates:
            log.error("no baseline models found in %s; run baseline.run first", models_dir)
            return 1

        parquet_path = processed_dir() / "tehran_houses.parquet"
        df = pd.read_parquet(parquet_path)
        _, val_df = split_mod.split_dataframe(df, val_size=args.val_size, seed=args.seed)

        y_true = val_df[DEFAULT_TARGET_COL].to_numpy(dtype=float)
        districts = val_df[DEFAULT_DISTRICT_COL]

        reports: list[EvaluationReport] = []
        for model_path in candidates:
            model = joblib.load(model_path)
            y_pred = model.predict(val_df)
            r = evaluate(
                model_name=model_path.stem,
                y_true=y_true,
                y_pred=y_pred,
                districts=districts,
                save=True,
            )
            reports.append(r)

        comparison = compare_models(reports)
        comparison_path = artifacts_dir() / "model_evaluation" / "comparison.json"
        comparison.to_json(comparison_path, orient="records", indent=2)

        print(comparison.to_string(index=False))
        print(f"\ncomparison saved to: {comparison_path}")
        return 0
    except Exception as e:
        log.error("evaluation step failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
