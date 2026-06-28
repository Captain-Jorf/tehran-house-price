"""
Train pipeline orchestrator.

Glues together the Phase 2 training steps:

  1. (optional) train baseline models
  2. train the main XGBoost model
  3. build a comparison report across all trained models
  4. return a structured TrainPipelineResult

This file is the training counterpart of build_dataset.py from Phase 1.
It exists so a single command reproduces the entire model pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from tehran_house_price.models import baseline as baseline_mod
from tehran_house_price.models import evaluation as ev
from tehran_house_price.models import split as split_mod
from tehran_house_price.models import train as train_mod
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import artifacts_dir, ensure_dir, processed_dir

log = get_logger(__name__)

DEFAULT_COMPARISON_FILENAME: str = "train_pipeline_comparison.json"


@dataclass(frozen=True, slots=True)
class TrainPipelineResult:
    """Outputs from a full training pipeline run."""

    main_model_path: Path | None
    main_metadata_path: Path | None
    baseline_paths: list[Path] = field(default_factory=list)
    comparison_path: Path | None = None
    n_models_compared: int = 0


def _load_persisted_models_for_comparison() -> list[ev.EvaluationReport]:
    """
    Load every trained model from artifacts/models/ and evaluate it on
    the canonical validation split. This guarantees apples-to-apples
    comparison even if some models were trained in earlier sessions.
    """
    # Required so joblib can resolve our custom estimator classes at unpickle time.
    from tehran_house_price.models import baseline as _baseline  # noqa: F401
    from tehran_house_price.models import train as _train  # noqa: F401

    models_dir = artifacts_dir() / "models"
    if not models_dir.exists():
        log.warning("models dir does not exist: %s", models_dir)
        return []

    candidates = sorted(models_dir.glob("*.joblib"))
    if not candidates:
        log.warning("no models found in %s", models_dir)
        return []

    parquet_path = processed_dir() / "tehran_houses.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"processed dataset not found at {parquet_path}. run build_dataset first."
        )

    df = pd.read_parquet(parquet_path)
    _, val_df = split_mod.split_dataframe(
        df,
        val_size=split_mod.DEFAULT_VAL_SIZE,
        seed=split_mod.DEFAULT_SEED,
    )

    y_true = val_df[train_mod.DEFAULT_TARGET_COL].to_numpy(dtype=float)
    districts = val_df[train_mod.DEFAULT_DISTRICT_COL]

    reports: list[ev.EvaluationReport] = []
    for model_path in candidates:
        try:
            model = joblib.load(model_path)
            y_pred = model.predict(val_df)
            report = ev.evaluate(
                model_name=model_path.stem,
                y_true=y_true,
                y_pred=y_pred,
                districts=districts,
                save=False,
            )
            reports.append(report)
        except Exception as e:
            log.warning("could not evaluate %s: %s", model_path.name, e)
    return reports


def _save_comparison(reports: list[ev.EvaluationReport]) -> Path | None:
    """Persist a comparison JSON across all models. Returns the path or None."""
    if not reports:
        log.warning("no reports to compare; skipping comparison save")
        return None

    out_dir = ensure_dir(artifacts_dir() / "model_evaluation")
    out_path = out_dir / DEFAULT_COMPARISON_FILENAME

    comparison_df = ev.compare_models(reports)
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_models": len(reports),
        "ranking_metric": "mape",
        "models": comparison_df.to_dict(orient="records"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    log.info("comparison report written to %s", out_path)
    return out_path


def run(
    *,
    skip_baselines: bool = False,
    skip_main: bool = False,
    val_size: float = split_mod.DEFAULT_VAL_SIZE,
    seed: int = split_mod.DEFAULT_SEED,
    model_name: str = train_mod.DEFAULT_MODEL_NAME,
) -> TrainPipelineResult:
    """
    Run the full Phase 2 training pipeline.

    Args:
        skip_baselines: do not retrain baselines if they already exist.
        skip_main: do not train the main XGBoost model (useful for debugging).
        val_size: validation fraction.
        seed: split seed.
        model_name: filename stem for the main model artifact.

    Returns:
        TrainPipelineResult with paths and number of compared models.

    Raises:
        ValueError: if both skip flags are True.
    """
    if skip_baselines and skip_main:
        raise ValueError("cannot skip both baselines and main model; nothing to do")

    log.info(
        "starting train pipeline | skip_baselines=%s skip_main=%s seed=%d val_size=%.3f",
        skip_baselines,
        skip_main,
        seed,
        val_size,
    )

    baseline_paths: list[Path] = []
    if not skip_baselines:
        baseline_results = baseline_mod.run(val_size=val_size, seed=seed)
        baseline_paths = [r.model_path for r in baseline_results]
        log.info("trained %d baseline model(s)", len(baseline_paths))
    else:
        log.info("skipping baseline training")

    main_model_path: Path | None = None
    main_metadata_path: Path | None = None
    if not skip_main:
        train_result = train_mod.train(
            val_size=val_size,
            seed=seed,
            model_name=model_name,
        )
        main_model_path = train_result.model_path
        main_metadata_path = train_result.metadata_path
        log.info("trained main model: %s", main_model_path)
    else:
        log.info("skipping main model training")

    reports = _load_persisted_models_for_comparison()
    comparison_path = _save_comparison(reports)

    return TrainPipelineResult(
        main_model_path=main_model_path,
        main_metadata_path=main_metadata_path,
        baseline_paths=baseline_paths,
        comparison_path=comparison_path,
        n_models_compared=len(reports),
    )


def _cli() -> int:
    """CLI entry point; kept separate from main() to avoid __main__ pickling issues."""
    parser = argparse.ArgumentParser(
        description="Run the full Phase 2 training pipeline: baselines + main model + comparison."
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="do not retrain baselines if they already exist",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="do not train the main XGBoost model",
    )
    parser.add_argument("--val-size", type=float, default=split_mod.DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=split_mod.DEFAULT_SEED)
    parser.add_argument("--model-name", default=train_mod.DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    try:
        result = run(
            skip_baselines=args.skip_baselines,
            skip_main=args.skip_main,
            val_size=args.val_size,
            seed=args.seed,
            model_name=args.model_name,
        )
        print("train pipeline done")
        print(f"  baselines:    {len(result.baseline_paths)}")
        for p in result.baseline_paths:
            print(f"    - {p}")
        print(f"  main model:   {result.main_model_path}")
        print(f"  metadata:     {result.main_metadata_path}")
        print(f"  comparison:   {result.comparison_path}")
        print(f"  models compared: {result.n_models_compared}")
        return 0
    except Exception as e:
        log.error("train pipeline failed: %s", e)
        return 1


def main() -> int:
    """
    Entry point for `python -m tehran_house_price.models.train_pipeline`.

    Re-imports this module under its canonical name so any pickled objects
    produced downstream reference the canonical module path. Same pattern
    as baseline.main() and train.main().
    """
    if __name__ == "__main__":
        from tehran_house_price.models import train_pipeline as _self

        return _self._cli()
    return _cli()


if __name__ == "__main__":
    sys.exit(main())
