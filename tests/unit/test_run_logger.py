"""Tests for safe MLflow run logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tehran_house_price.tracking import run_logger


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Capture mlflow.* calls without actually hitting MLflow."""
    calls: dict[str, list[Any]] = {
        "log_params": [],
        "log_metrics": [],
        "log_artifact": [],
        "log_model": [],
        "set_tags": [],
    }

    def fake_log_params(params: dict[str, str]) -> None:
        calls["log_params"].append(params)

    def fake_log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
        calls["log_metrics"].append((dict(metrics), step))

    def fake_log_artifact(path: str, artifact_path: str | None = None) -> None:
        calls["log_artifact"].append((path, artifact_path))

    def fake_log_model(sk_model: Any, artifact_path: str = "model") -> None:
        calls["log_model"].append((sk_model, artifact_path))

    def fake_set_tags(tags: dict[str, str]) -> None:
        calls["set_tags"].append(tags)

    monkeypatch.setattr(run_logger.mlflow, "log_params", fake_log_params)
    monkeypatch.setattr(run_logger.mlflow, "log_metrics", fake_log_metrics)
    monkeypatch.setattr(run_logger.mlflow, "log_artifact", fake_log_artifact)
    monkeypatch.setattr(run_logger.mlflow.sklearn, "log_model", fake_log_model)
    monkeypatch.setattr(run_logger.mlflow, "set_tags", fake_set_tags)

    # Force tracking ON for these tests.
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")
    return calls


def test_log_params_stringifies_values(captured_calls: dict[str, list[Any]]) -> None:
    run_logger.log_params({"n_estimators": 400, "alpha": 0.1, "name": "xgb"})

    assert len(captured_calls["log_params"]) == 1
    logged = captured_calls["log_params"][0]
    assert logged == {"n_estimators": "400", "alpha": "0.1", "name": "xgb"}


def test_log_params_truncates_long_values(captured_calls: dict[str, list[Any]]) -> None:
    huge = "x" * 5000
    run_logger.log_params({"blob": huge})

    logged = captured_calls["log_params"][0]
    assert len(logged["blob"]) <= 500
    assert logged["blob"].endswith("...")


def test_log_params_noop_on_empty(captured_calls: dict[str, list[Any]]) -> None:
    run_logger.log_params({})
    assert captured_calls["log_params"] == []


def test_log_metrics_filters_non_finite(captured_calls: dict[str, list[Any]]) -> None:
    run_logger.log_metrics({"mae": 1.5, "r2": float("nan"), "rmse": float("inf"), "ok": 0.0})

    assert len(captured_calls["log_metrics"]) == 1
    logged, step = captured_calls["log_metrics"][0]
    assert logged == {"mae": 1.5, "ok": 0.0}
    assert step is None


def test_log_metrics_skips_non_numeric(captured_calls: dict[str, list[Any]]) -> None:
    run_logger.log_metrics({"mae": 1.0, "label": "abc"})  # type: ignore[dict-item]

    logged, _ = captured_calls["log_metrics"][0]
    assert logged == {"mae": 1.0}


def test_log_metrics_noop_when_all_invalid(captured_calls: dict[str, list[Any]]) -> None:
    run_logger.log_metrics({"r2": float("nan")})
    assert captured_calls["log_metrics"] == []


def test_log_artifact_file_skips_missing(
    captured_calls: dict[str, list[Any]], tmp_path: Path
) -> None:
    missing = tmp_path / "nope.json"
    run_logger.log_artifact_file(missing)
    assert captured_calls["log_artifact"] == []


def test_log_artifact_file_logs_existing(
    captured_calls: dict[str, list[Any]], tmp_path: Path
) -> None:
    real = tmp_path / "metrics.json"
    real.write_text("{}", encoding="utf-8")

    run_logger.log_artifact_file(real, artifact_subdir="reports")

    assert len(captured_calls["log_artifact"]) == 1
    path, subdir = captured_calls["log_artifact"][0]
    assert path == str(real)
    assert subdir == "reports"


def test_log_sklearn_model_passes_through(captured_calls: dict[str, list[Any]]) -> None:
    dummy = object()
    run_logger.log_sklearn_model(dummy, artifact_path="my_model")

    assert len(captured_calls["log_model"]) == 1
    model_obj, path = captured_calls["log_model"][0]
    assert model_obj is dummy
    assert path == "my_model"


def test_set_tags_stringifies(captured_calls: dict[str, list[Any]]) -> None:
    run_logger.set_tags({"phase": "phase5", "count": 3})  # type: ignore[dict-item]

    logged = captured_calls["set_tags"][0]
    assert logged == {"phase": "phase5", "count": "3"}


def test_all_helpers_noop_when_tracking_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def boom(*args: Any, **kwargs: Any) -> None:
        calls.append("called")
        raise AssertionError("mlflow should not be called when disabled")

    monkeypatch.setattr(run_logger.mlflow, "log_params", boom)
    monkeypatch.setattr(run_logger.mlflow, "log_metrics", boom)
    monkeypatch.setattr(run_logger.mlflow, "log_artifact", boom)
    monkeypatch.setattr(run_logger.mlflow.sklearn, "log_model", boom)
    monkeypatch.setattr(run_logger.mlflow, "set_tags", boom)
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "false")

    artifact = tmp_path / "a.json"
    artifact.write_text("{}", encoding="utf-8")

    run_logger.log_params({"a": 1})
    run_logger.log_metrics({"a": 1.0})
    run_logger.log_artifact_file(artifact)
    run_logger.log_sklearn_model(object())
    run_logger.set_tags({"a": "b"})

    assert calls == []
