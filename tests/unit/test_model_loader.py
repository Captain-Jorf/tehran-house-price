"""Unit tests for the model loader service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from tehran_house_price.api.model_loader import (
    ModelLoadError,
    ModelNotLoadedError,
    ModelService,
    get_model_service,
)


class _ConstantRegressor(BaseEstimator, RegressorMixin):
    """Tiny deterministic regressor for loader tests."""

    def __init__(self, constant: float = 100_000_000.0) -> None:
        self.constant = constant

    def fit(self, X: pd.DataFrame, y: Any = None) -> _ConstantRegressor:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(shape=len(X), fill_value=self.constant, dtype=np.float64)


def _make_fake_pipeline(constant: float = 100_000_000.0) -> Pipeline:
    """Build a minimal sklearn Pipeline that mimics the real artifact."""
    return Pipeline(steps=[("model", _ConstantRegressor(constant=constant))])


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    """Persist a fake sklearn pipeline to a temporary joblib file."""
    pipeline = _make_fake_pipeline()
    path = tmp_path / "fake_model.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture
def fake_metadata_path(tmp_path: Path) -> Path:
    """Persist a fake metadata JSON file."""
    metadata = {
        "model_name": "fake_model",
        "target": "price_per_m2",
        "trained_at": "2025-01-01T00:00:00Z",
    }
    path = tmp_path / "fake_model_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


@pytest.fixture
def valid_row() -> dict[str, Any]:
    """Return a valid inference input row."""
    return {
        "district": "Pasdaran",
        "area_m2": 120.0,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }


def test_load_sets_loaded_state(fake_model_path: Path, fake_metadata_path: Path) -> None:
    """After load, the service should report loaded state."""
    service = ModelService()

    assert service.is_loaded() is False

    service.load(model_path=fake_model_path, metadata_path=fake_metadata_path)

    assert service.is_loaded() is True
    assert service.get_artifact_path() == fake_model_path


def test_load_reads_metadata(fake_model_path: Path, fake_metadata_path: Path) -> None:
    """Metadata should be read into memory on load."""
    service = ModelService()

    service.load(model_path=fake_model_path, metadata_path=fake_metadata_path)
    metadata = service.get_metadata()

    assert metadata["model_name"] == "fake_model"
    assert metadata["target"] == "price_per_m2"


def test_load_missing_model_raises(tmp_path: Path) -> None:
    """Loading a non-existent model file should raise ModelLoadError."""
    service = ModelService()
    missing_path = tmp_path / "does_not_exist.joblib"

    with pytest.raises(ModelLoadError):
        service.load(model_path=missing_path)


def test_load_invalid_artifact_raises(tmp_path: Path) -> None:
    """Loading a file that is not a sklearn Pipeline should raise ModelLoadError."""
    service = ModelService()
    bad_path = tmp_path / "bad_artifact.joblib"
    joblib.dump({"not": "a pipeline"}, bad_path)

    with pytest.raises(ModelLoadError):
        service.load(model_path=bad_path)


def test_load_missing_metadata_returns_empty(
    fake_model_path: Path,
    tmp_path: Path,
) -> None:
    """Missing metadata file should produce an empty metadata dict, not raise."""
    service = ModelService()
    missing_metadata = tmp_path / "missing_metadata.json"

    service.load(model_path=fake_model_path, metadata_path=missing_metadata)

    assert service.is_loaded() is True
    assert service.get_metadata() == {}


def test_load_invalid_metadata_returns_empty(
    fake_model_path: Path,
    tmp_path: Path,
) -> None:
    """Invalid JSON in metadata file should produce empty metadata, not raise."""
    service = ModelService()
    bad_metadata = tmp_path / "bad_metadata.json"
    bad_metadata.write_text("not a json", encoding="utf-8")

    service.load(model_path=fake_model_path, metadata_path=bad_metadata)

    assert service.is_loaded() is True
    assert service.get_metadata() == {}


def test_predict_without_load_raises(valid_row: dict[str, Any]) -> None:
    """Predicting before loading should raise ModelNotLoadedError."""
    service = ModelService()

    with pytest.raises(ModelNotLoadedError):
        service.predict([valid_row])


def test_predict_returns_numpy_array(
    fake_model_path: Path,
    fake_metadata_path: Path,
    valid_row: dict[str, Any],
) -> None:
    """Prediction should return a numpy float array of correct length."""
    service = ModelService()
    service.load(model_path=fake_model_path, metadata_path=fake_metadata_path)

    predictions = service.predict([valid_row, valid_row])

    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.float64
    assert predictions.shape == (2,)
    assert np.allclose(predictions, 100_000_000.0)


def test_predict_empty_rows_raises(
    fake_model_path: Path,
    fake_metadata_path: Path,
) -> None:
    """Empty input list should raise ValueError."""
    service = ModelService()
    service.load(model_path=fake_model_path, metadata_path=fake_metadata_path)

    with pytest.raises(ValueError):
        service.predict([])


def test_predict_missing_features_raises(
    fake_model_path: Path,
    fake_metadata_path: Path,
) -> None:
    """Missing required features should raise ValueError."""
    service = ModelService()
    service.load(model_path=fake_model_path, metadata_path=fake_metadata_path)

    incomplete_row = {"district": "Pasdaran", "area_m2": 100.0}

    with pytest.raises(ValueError):
        service.predict([incomplete_row])


def test_reset_unloads_model(
    fake_model_path: Path,
    fake_metadata_path: Path,
) -> None:
    """Reset should bring the service back to unloaded state."""
    service = ModelService()
    service.load(model_path=fake_model_path, metadata_path=fake_metadata_path)

    assert service.is_loaded() is True

    service.reset()

    assert service.is_loaded() is False
    assert service.get_metadata() == {}
    assert service.get_artifact_path() is None


def test_get_model_service_returns_singleton() -> None:
    """The module-level accessor should always return the same instance."""
    first = get_model_service()
    second = get_model_service()

    assert first is second


def test_load_real_model_artifact() -> None:
    """Smoke test against the real production artifact, if available.

    Skipped automatically when the artifact has not been built yet.
    """
    from tehran_house_price.utils.paths import project_root

    real_path = project_root() / "artifacts" / "models" / "xgb_price_per_m2.joblib"
    if not real_path.exists():
        pytest.skip("real model artifact not available")

    service = ModelService()
    service.load()

    assert service.is_loaded() is True
    assert service.get_artifact_path() == real_path

    row = {
        "district": "Pasdaran",
        "area_m2": 120.0,
        "rooms": 2,
        "has_parking": True,
        "has_storage": True,
        "has_elevator": True,
    }
    predictions = service.predict([row])

    assert predictions.shape == (1,)
    assert predictions[0] > 0
