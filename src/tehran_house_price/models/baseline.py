"""
Baseline regressors for the Tehran house price project.

Two baselines:
    1. MeanPriceBaseline:
        predicts the global mean of the training target for every row.
        The simplest possible baseline; useful as a sanity floor.

    2. DistrictMedianBaseline:
        predicts the per-district median of the training target.
        Unseen districts fall back to the global median.

Both follow the sklearn estimator interface so they can be used inside
sklearn Pipelines and serialized with joblib alongside the production model.

Why two baselines?
    A model that cannot beat the global mean is worthless. A model that
    cannot beat a per-district median is not really using its other features.
    Together they define a clean, defensible performance floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tehran_house_price.models import split as split_mod
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import artifacts_dir, ensure_dir

log = get_logger(__name__)

DEFAULT_DISTRICT_COL: str = "district"
DEFAULT_TARGET_COL: str = "price_per_m2"


class MeanPriceBaseline(BaseEstimator, RegressorMixin):
    """Predict the global mean of y for every input row."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MeanPriceBaseline:
        self.mean_: float = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("MeanPriceBaseline must be fit before predict")
        n = len(X)
        return np.full(shape=n, fill_value=self.mean_, dtype=float)


class DistrictMedianBaseline(BaseEstimator, RegressorMixin):
    """
    Predict the per-district median of y. Unseen districts fall back to
    the global median computed during fit.
    """

    def __init__(self, district_col: str = DEFAULT_DISTRICT_COL) -> None:
        self.district_col = district_col

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DistrictMedianBaseline:
        if self.district_col not in X.columns:
            raise ValueError(f"district column '{self.district_col}' not in X")

        y_arr = np.asarray(y, dtype=float)
        df = pd.DataFrame({self.district_col: X[self.district_col].values, "_y": y_arr})

        self.global_median_: float = float(np.median(y_arr))
        self.mapping_: dict[str, float] = (
            df.groupby(self.district_col)["_y"].median().astype(float).to_dict()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "mapping_"):
            raise RuntimeError("DistrictMedianBaseline must be fit before predict")
        if self.district_col not in X.columns:
            raise ValueError(f"district column '{self.district_col}' not in X")

        return (
            X[self.district_col]
            .map(self.mapping_)
            .fillna(self.global_median_)
            .astype(float)
            .to_numpy()
        )


@dataclass(frozen=True, slots=True)
class BaselineTrainResult:
    """Outputs from training a single baseline."""

    name: str
    model_path: Path
    metrics_path: Path
    n_train: int
    n_val: int


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    A minimal set of regression metrics. The full evaluation layer comes
    in Phase 2.4; here we only need enough to compare baselines.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE: protect against zero targets (should not happen for price_per_m2,
    # but defensive coding is cheap).
    safe_true = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / safe_true)))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "n": int(len(y_true)),
    }


def _save_model(model: BaseEstimator, name: str) -> Path:
    out_dir = ensure_dir(artifacts_dir() / "models")
    path = out_dir / f"{name}.joblib"
    joblib.dump(model, path)
    log.info("saved baseline model to %s", path)
    return path


def _save_metrics(metrics: dict, name: str) -> Path:
    out_dir = ensure_dir(artifacts_dir() / "model_evaluation")
    path = out_dir / f"{name}_metrics.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": name,
        **metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("saved baseline metrics to %s", path)
    return path


def train_baseline(
    name: str,
    model: BaseEstimator,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET_COL,
) -> BaselineTrainResult:
    """Fit a baseline on train, evaluate on val, persist model and metrics."""
    if target_col not in train_df.columns or target_col not in val_df.columns:
        raise ValueError(f"target column '{target_col}' missing in train or val")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].to_numpy(dtype=float)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = _compute_basic_metrics(y_val, y_pred)

    log.info(
        "baseline '%s' | mae=%.2f rmse=%.2f mape=%.4f n_val=%d",
        name,
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["n"],
    )

    model_path = _save_model(model, name)
    metrics_path = _save_metrics(metrics, name)

    return BaselineTrainResult(
        name=name,
        model_path=model_path,
        metrics_path=metrics_path,
        n_train=len(train_df),
        n_val=len(val_df),
    )


def run(
    *,
    val_size: float = split_mod.DEFAULT_VAL_SIZE,
    seed: int = split_mod.DEFAULT_SEED,
    target_col: str = DEFAULT_TARGET_COL,
) -> list[BaselineTrainResult]:
    """Train and evaluate all baselines on the processed dataset."""
    split_result = split_mod.run(val_size=val_size, seed=seed)
    train_df = split_result.train_df
    val_df = split_result.val_df

    results: list[BaselineTrainResult] = []
    results.append(
        train_baseline(
            name="baseline_mean",
            model=MeanPriceBaseline(),
            train_df=train_df,
            val_df=val_df,
            target_col=target_col,
        )
    )
    results.append(
        train_baseline(
            name="baseline_district_median",
            model=DistrictMedianBaseline(district_col=DEFAULT_DISTRICT_COL),
            train_df=train_df,
            val_df=val_df,
            target_col=target_col,
        )
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline models and report metrics.")
    parser.add_argument("--val-size", type=float, default=split_mod.DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=split_mod.DEFAULT_SEED)
    args = parser.parse_args()

    try:
        results = run(val_size=args.val_size, seed=args.seed)
        for r in results:
            print(f"{r.name}: model={r.model_path} metrics={r.metrics_path}")
        return 0
    except Exception as e:
        log.error("baseline run failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
