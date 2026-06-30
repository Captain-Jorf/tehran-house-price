"""
train pipeline orchestrator + mlflow tracking.

parent run = train_pipeline
child runs = each baseline + main model

tracking is fully toggleable via MLFLOW_TRACKING_ENABLED.
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
from tehran_house_price.tracking import (
    get_run_context,
    is_tracking_enabled,
    log_artifact_file,
    log_metrics,
    log_params,
    log_sklearn_model,
    set_tags,
    setup_mlflow,
)
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import artifacts_dir, ensure_dir, processed_dir

log = get_logger(__name__)

DEFAULT_COMPARISON_FILENAME: str = "train_pipeline_comparison.json"
MLFLOW_EXPERIMENT_NAME: str = "tehran_house_price"


@dataclass(frozen=True, slots=True)
class TrainPipelineResult:
    main_model_path: Path | None
    main_metadata_path: Path | None
    baseline_paths: list[Path] = field(default_factory=list)
    comparison_path: Path | None = None
    n_models_compared: int = 0


# mlflow helpers (orchestration-only)


def _split_params(*, seed, val_size, n_train, n_val):
    return {
        "split.seed": seed,
        "split.val_size": val_size,
        "split.n_train": n_train,
        "split.n_val": n_val,
    }


def _common_model_params(*, model_name, algorithm, role, target_col, target_transform):
    return {
        "model_name": model_name,
        "algorithm": algorithm,
        "model_role": role,
        "target_col": target_col,
        "target_transform": target_transform,
    }


def _read_split_hashes_for(metadata_path: Path | None) -> dict[str, str]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    split_info = payload.get("split") or {}
    out: dict[str, str] = {}
    if split_info.get("train_ids_hash"):
        out["split.train_ids_hash"] = str(split_info["train_ids_hash"])
    if split_info.get("val_ids_hash"):
        out["split.val_ids_hash"] = str(split_info["val_ids_hash"])
    return out


def _evaluation_json_path_for(model_stem: str) -> Path:
    return artifacts_dir() / "model_evaluation" / f"{model_stem}_evaluation.json"


def _baseline_metrics_json_path_for(model_stem: str) -> Path:
    return artifacts_dir() / "model_evaluation" / f"{model_stem}_metrics.json"


def _worst_district_metrics(per_district: pd.DataFrame, top_k: int = 5) -> dict[str, float]:
    if per_district is None or per_district.empty:
        return {}

    head = per_district.head(top_k)
    out: dict[str, float] = {}
    for rank, (_, row) in enumerate(head.iterrows(), start=1):
        for metric in ("mae", "mape"):
            value = row.get(metric)
            if value is None:
                continue
            try:
                out[f"worst_district_{rank}_{metric}"] = float(value)
            except (TypeError, ValueError):
                continue
    return out


# comparison


def _load_persisted_models_for_comparison() -> list[ev.EvaluationReport]:
    # need these imports so joblib can resolve our custom classes
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


# per-model run loggers


def _log_baseline_runs(baseline_results, *, seed, val_size):
    if not is_tracking_enabled() or not baseline_results:
        return

    for result in baseline_results:
        algorithm = (
            "MeanPriceBaseline" if result.name == "baseline_mean" else "DistrictMedianBaseline"
        )
        with get_run_context(
            run_name=result.name,
            extra_tags={"model_role": "baseline"},
            nested=True,
        ):
            params = _common_model_params(
                model_name=result.name,
                algorithm=algorithm,
                role="baseline",
                target_col=train_mod.DEFAULT_TARGET_COL,
                target_transform="none",
            )
            params.update(
                _split_params(
                    seed=seed,
                    val_size=val_size,
                    n_train=result.n_train,
                    n_val=result.n_val,
                )
            )
            log_params(params)

            metrics_path = _baseline_metrics_json_path_for(result.name)
            if metrics_path.exists():
                try:
                    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                metrics = {key: payload[key] for key in ("mae", "rmse", "mape") if key in payload}
                log_metrics(metrics)
                log_artifact_file(metrics_path, artifact_subdir="evaluation")

            log_artifact_file(result.model_path, artifact_subdir="model")


def _log_main_run(train_result, *, seed, val_size, xgb_params):
    if not is_tracking_enabled():
        return

    with get_run_context(
        run_name=train_result.model_name,
        extra_tags={"model_role": "main"},
        nested=True,
    ):
        params = _common_model_params(
            model_name=train_result.model_name,
            algorithm="xgboost.XGBRegressor",
            role="main",
            target_col=train_mod.DEFAULT_TARGET_COL,
            target_transform="log1p / expm1",
        )
        params.update(
            _split_params(
                seed=seed,
                val_size=val_size,
                n_train=train_result.n_train,
                n_val=train_result.n_val,
            )
        )
        params.update(_read_split_hashes_for(train_result.metadata_path))
        params.update({f"hp.{key}": value for key, value in xgb_params.items()})

        log_params(params)
        log_metrics(train_result.metrics)

        eval_json = _evaluation_json_path_for(train_result.model_name)
        if eval_json.exists():
            try:
                payload = json.loads(eval_json.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            per_district_rows = payload.get("per_district") or []
            if per_district_rows:
                per_district_df = pd.DataFrame(per_district_rows).set_index("district")
                worst_metrics = _worst_district_metrics(per_district_df)
                if worst_metrics:
                    log_metrics(worst_metrics)
            log_artifact_file(eval_json, artifact_subdir="evaluation")

        log_artifact_file(train_result.metadata_path, artifact_subdir="model")
        log_artifact_file(train_result.model_path, artifact_subdir="model")

        try:
            loaded = joblib.load(train_result.model_path)
            log_sklearn_model(loaded, artifact_path="sklearn_model")
        except Exception as exc:
            log.warning("could not log sklearn model to mlflow | error=%s", exc)


def _log_parent_summary(*, comparison_path, n_models_compared, baseline_count, main_trained):
    if not is_tracking_enabled():
        return

    log_metrics(
        {
            "pipeline.n_models_compared": float(n_models_compared),
            "pipeline.n_baselines": float(baseline_count),
            "pipeline.main_trained": 1.0 if main_trained else 0.0,
        }
    )

    if comparison_path is not None and comparison_path.exists():
        log_artifact_file(comparison_path, artifact_subdir="comparison")


# public api


def run(
    *,
    skip_baselines: bool = False,
    skip_main: bool = False,
    val_size: float = split_mod.DEFAULT_VAL_SIZE,
    seed: int = split_mod.DEFAULT_SEED,
    model_name: str = train_mod.DEFAULT_MODEL_NAME,
) -> TrainPipelineResult:
    if skip_baselines and skip_main:
        raise ValueError("cannot skip both baselines and main model; nothing to do")

    log.info(
        "starting train pipeline | skip_baselines=%s skip_main=%s seed=%d val_size=%.3f",
        skip_baselines,
        skip_main,
        seed,
        val_size,
    )

    setup_mlflow(MLFLOW_EXPERIMENT_NAME)

    parent_tags = {
        "pipeline": "train_pipeline",
        "skip_baselines": str(skip_baselines).lower(),
        "skip_main": str(skip_main).lower(),
    }

    with get_run_context(run_name="train_pipeline", extra_tags=parent_tags):
        set_tags({"pipeline_seed": str(seed), "pipeline_val_size": str(val_size)})

        baseline_paths: list[Path] = []
        baseline_results: list[baseline_mod.BaselineTrainResult] = []
        if not skip_baselines:
            baseline_results = baseline_mod.run(val_size=val_size, seed=seed)
            baseline_paths = [r.model_path for r in baseline_results]
            log.info("trained %d baseline model(s)", len(baseline_paths))
            _log_baseline_runs(baseline_results, seed=seed, val_size=val_size)
        else:
            log.info("skipping baseline training")

        main_model_path: Path | None = None
        main_metadata_path: Path | None = None
        main_train_result: train_mod.TrainResult | None = None
        if not skip_main:
            main_train_result = train_mod.train(
                val_size=val_size,
                seed=seed,
                model_name=model_name,
            )
            main_model_path = main_train_result.model_path
            main_metadata_path = main_train_result.metadata_path
            log.info("trained main model: %s", main_model_path)
            _log_main_run(
                main_train_result,
                seed=seed,
                val_size=val_size,
                xgb_params=train_mod.DEFAULT_XGB_PARAMS,
            )
        else:
            log.info("skipping main model training")

        reports = _load_persisted_models_for_comparison()
        comparison_path = _save_comparison(reports)

        _log_parent_summary(
            comparison_path=comparison_path,
            n_models_compared=len(reports),
            baseline_count=len(baseline_paths),
            main_trained=main_train_result is not None,
        )

    return TrainPipelineResult(
        main_model_path=main_model_path,
        main_metadata_path=main_metadata_path,
        baseline_paths=baseline_paths,
        comparison_path=comparison_path,
        n_models_compared=len(reports),
    )


def _cli() -> int:
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
    if __name__ == "__main__":
        from tehran_house_price.models import train_pipeline as _self

        return _self._cli()
    return _cli()


if __name__ == "__main__":
    sys.exit(main())
