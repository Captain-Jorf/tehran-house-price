"""
Main XGBoost model for the Tehran house price project.

This module produces a fully self-contained model artifact:

    Pipeline:
        feature engineering (Phase 2.1)
              |
              v
        LogTargetRegressor:
              fits XGBoost on log1p(price_per_m2)
              predicts in original price_per_m2 scale

The final artifact is one joblib file. Consumers (Phase 3 API, future
batch scoring) only need to call `.predict(raw_df)` on it.

Design choices:
    - Target is log1p-transformed so XGBoost learns on a more normal
      distribution. The wrapper applies the inverse transform automatically.
    - Pipeline + wrapper + model are pickled together so deployment never
      has to reproduce preprocessing logic in another place.
    - Metadata is saved next to the model: hyperparameters, training
      timestamp, dataset hash, metrics, and baseline comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from tehran_house_price import __version__ as package_version
from tehran_house_price.features.build_features import (
    build_feature_pipeline,
    inverse_target_for_prediction,
    transform_target_for_training,
)
from tehran_house_price.models import evaluation as ev
from tehran_house_price.models import split as split_mod
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import artifacts_dir, ensure_dir

log = get_logger(__name__)

DEFAULT_TARGET_COL: str = "price_per_m2"
DEFAULT_DISTRICT_COL: str = "district"
DEFAULT_MODEL_NAME: str = "xgb_price_per_m2"

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """
    Wrap any regressor to train on log1p(y) and predict back in original scale.

    The wrapper exists so consumers of the final pipeline never need to know
    about the log transform. predict(X) returns values in the original
    price_per_m2 scale.
    """

    def __init__(self, regressor: BaseEstimator) -> None:
        self.regressor = regressor

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogTargetRegressor:
        y_log = transform_target_for_training(y)
        self.regressor.fit(X, y_log)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_log_pred = self.regressor.predict(X)
        return inverse_target_for_prediction(np.asarray(y_log_pred, dtype=float))


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Outputs from a single training run."""

    model_name: str
    model_path: Path
    metadata_path: Path
    metrics: dict[str, float]
    n_train: int
    n_val: int


def build_full_pipeline(xgb_params: dict[str, Any] | None = None) -> Pipeline:
    """
    Build the full end-to-end pipeline: feature engineering + log-target XGB.

    The returned Pipeline is what gets persisted as the final model artifact.
    """
    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    feature_pipeline = build_feature_pipeline()
    regressor = LogTargetRegressor(regressor=XGBRegressor(**params))

    return Pipeline(
        steps=[
            ("features", feature_pipeline),
            ("model", regressor),
        ]
    )


def _split_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' missing from DataFrame")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def _save_model(model: Pipeline, name: str) -> Path:
    out_dir = ensure_dir(artifacts_dir() / "models")
    path = out_dir / f"{name}.joblib"
    joblib.dump(model, path)
    log.info("saved model to %s", path)
    return path


def _load_baseline_metrics() -> dict[str, dict[str, float]]:
    """
    Load existing baseline metrics for comparison in the model metadata.

    Returns an empty dict if baselines have not been trained yet.
    """
    out: dict[str, dict[str, float]] = {}
    eval_dir = artifacts_dir() / "model_evaluation"
    if not eval_dir.exists():
        return out

    for path in sorted(eval_dir.glob("baseline_*_evaluation.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            model_name = payload.get("model", path.stem)
            out[model_name] = payload.get("global_metrics", {})
        except Exception as e:  # pragma: no cover - defensive
            log.warning("could not parse baseline metrics from %s: %s", path, e)
    return out


def _save_metadata(
    *,
    model_name: str,
    xgb_params: dict[str, Any],
    metrics: dict[str, float],
    worst_districts: pd.DataFrame,
    split_seed: int,
    val_size: float,
    n_train: int,
    n_val: int,
    train_ids_hash: str | None,
    val_ids_hash: str | None,
) -> Path:
    """Persist training metadata next to the model artifact."""
    out_dir = ensure_dir(artifacts_dir() / "models")
    path = out_dir / f"{model_name}_metadata.json"

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": package_version,
        "model_name": model_name,
        "algorithm": "xgboost.XGBRegressor",
        "target_col": DEFAULT_TARGET_COL,
        "target_transform": "log1p / expm1",
        "hyperparameters": xgb_params,
        "split": {
            "seed": split_seed,
            "val_size": val_size,
            "n_train": n_train,
            "n_val": n_val,
            "train_ids_hash": train_ids_hash,
            "val_ids_hash": val_ids_hash,
        },
        "metrics": metrics,
        "worst_districts": worst_districts.reset_index().to_dict(orient="records"),
        "baselines": _load_baseline_metrics(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("saved model metadata to %s", path)
    return path


def train(
    *,
    val_size: float = split_mod.DEFAULT_VAL_SIZE,
    seed: int = split_mod.DEFAULT_SEED,
    model_name: str = DEFAULT_MODEL_NAME,
    xgb_params: dict[str, Any] | None = None,
    target_col: str = DEFAULT_TARGET_COL,
) -> TrainResult:
    """
    Run the full training procedure: split, fit pipeline, evaluate, persist.

    Returns a TrainResult with paths and final validation metrics.
    """
    split_result = split_mod.run(val_size=val_size, seed=seed)
    train_df = split_result.train_df
    val_df = split_result.val_df

    X_train, y_train = _split_xy(train_df, target_col)
    X_val, y_val = _split_xy(val_df, target_col)

    log.info("training '%s' | n_train=%d n_val=%d", model_name, len(train_df), len(val_df))

    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    pipeline = build_full_pipeline(xgb_params=params)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    report = ev.evaluate(
        model_name=model_name,
        y_true=y_val.to_numpy(dtype=float),
        y_pred=y_pred,
        districts=val_df[DEFAULT_DISTRICT_COL],
        save=True,
    )

    # read split metadata so we can record hashes in the model metadata
    train_ids_hash: str | None = None
    val_ids_hash: str | None = None
    if split_result.metadata_path and split_result.metadata_path.exists():
        try:
            split_meta = json.loads(split_result.metadata_path.read_text(encoding="utf-8"))
            train_ids_hash = split_meta.get("train_ids_hash")
            val_ids_hash = split_meta.get("val_ids_hash")
        except Exception as e:  # pragma: no cover - defensive
            log.warning("could not parse split metadata: %s", e)

    model_path = _save_model(pipeline, model_name)
    metadata_path = _save_metadata(
        model_name=model_name,
        xgb_params=params,
        metrics=report.global_metrics,
        worst_districts=report.worst_districts,
        split_seed=seed,
        val_size=val_size,
        n_train=len(train_df),
        n_val=len(val_df),
        train_ids_hash=train_ids_hash,
        val_ids_hash=val_ids_hash,
    )

    return TrainResult(
        model_name=model_name,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics=report.global_metrics,
        n_train=len(train_df),
        n_val=len(val_df),
    )


def _cli() -> int:
    """CLI entry point; kept separate from main() to avoid __main__ pickling issues."""
    parser = argparse.ArgumentParser(description="Train the main XGBoost model.")
    parser.add_argument("--val-size", type=float, default=split_mod.DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=split_mod.DEFAULT_SEED)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    try:
        result = train(
            val_size=args.val_size,
            seed=args.seed,
            model_name=args.model_name,
        )
        print(f"training done: {result.model_name}")
        print(f"  model:    {result.model_path}")
        print(f"  metadata: {result.metadata_path}")
        print(
            f"  metrics:  mae={result.metrics['mae']:.2f} "
            f"rmse={result.metrics['rmse']:.2f} "
            f"mape={result.metrics['mape']:.4f} "
            f"r2={result.metrics['r2']:.4f}"
        )
        return 0
    except Exception as e:
        log.error("training failed: %s", e)
        return 1


def main() -> int:
    """
    Entry point for `python -m tehran_house_price.models.train`.

    Re-imports this module under its canonical name before doing any work
    so the pickled pipeline references 'tehran_house_price.models.train'
    instead of '__main__'. See baseline.main() for the same pattern.
    """
    if __name__ == "__main__":
        from tehran_house_price.models import train as _self

        return _self._cli()
    return _cli()


if __name__ == "__main__":
    sys.exit(main())
