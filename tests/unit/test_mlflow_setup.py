"""Tests for MLflow setup helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from tehran_house_price.tracking import mlflow_setup


def test_is_tracking_enabled_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_ENABLED", raising=False)
    assert mlflow_setup.is_tracking_enabled() is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off", "  no  "])
def test_is_tracking_enabled_recognized_false_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", value)
    assert mlflow_setup.is_tracking_enabled() is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_is_tracking_enabled_recognized_true_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", value)
    assert mlflow_setup.is_tracking_enabled() is True


def test_get_tracking_uri_uses_env_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.com:5000")
    assert mlflow_setup.get_tracking_uri() == "http://example.com:5000"


def test_get_tracking_uri_defaults_to_local_file_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow_setup, "project_root", lambda: tmp_path)

    uri = mlflow_setup.get_tracking_uri()

    assert uri.startswith("file:")
    assert (tmp_path / "mlruns").exists()
    assert uri.endswith("/mlruns")


def test_setup_mlflow_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "false")
    assert mlflow_setup.setup_mlflow("any_name") is None


def test_setup_mlflow_creates_and_finds_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_ENABLED", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow_setup, "project_root", lambda: tmp_path)

    experiment_name = "unit_test_exp_" + os.urandom(4).hex()

    first_id = mlflow_setup.setup_mlflow(experiment_name)
    second_id = mlflow_setup.setup_mlflow(experiment_name)

    assert first_id is not None
    assert first_id == second_id


def test_get_run_context_yields_none_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "false")
    with mlflow_setup.get_run_context("disabled_run") as run:
        assert run is None


def test_get_run_context_yields_active_run_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_ENABLED", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow_setup, "project_root", lambda: tmp_path)

    mlflow_setup.setup_mlflow("unit_test_exp_run_ctx")

    with mlflow_setup.get_run_context("active_run", extra_tags={"x": "y"}) as run:
        assert run is not None
        assert run.info.run_id


def test_git_commit_sha_returns_string() -> None:
    sha = mlflow_setup._git_commit_sha()
    assert isinstance(sha, str)
    assert sha  # not empty
