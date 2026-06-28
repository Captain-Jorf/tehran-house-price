"""Model loader service for FastAPI inference.

This module manages a single in-memory copy of the trained sklearn pipeline
and its associated metadata. The model is loaded once at application startup
and reused across requests for efficient inference.

Design:
- Thread-safe singleton pattern via module-level state and a lock.
- Metadata is read from a JSON file located next to the model artifact.
- Pure load and predict interface, no FastAPI dependencies here.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import project_root

logger = get_logger(__name__)

DEFAULT_MODEL_FILENAME = "xgb_price_per_m2.joblib"
DEFAULT_METADATA_FILENAME = "xgb_price_per_m2_metadata.json"
DEFAULT_MODELS_SUBDIR = "models"

REQUIRED_FEATURES = (
    "district",
    "area_m2",
    "rooms",
    "has_parking",
    "has_storage",
    "has_elevator",
)


class ModelNotLoadedError(RuntimeError):
    """Raised when prediction is requested before the model is loaded."""


class ModelLoadError(RuntimeError):
    """Raised when the model artifact cannot be loaded from disk."""


class ModelService:
    """Singleton-style holder for the serving model and its metadata.

    The class wraps the loaded sklearn pipeline and exposes a minimal API:
    load, predict, get_metadata, and is_loaded. Thread safety is provided
    by a module-level lock used during load to prevent double-loading
    in concurrent startup scenarios.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._metadata: dict[str, Any] = {}
        self._artifact_path: Path | None = None
        self._lock = threading.Lock()

    def load(
        self,
        model_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
    ) -> None:
        """Load the model artifact and its metadata into memory.

        Parameters
        ----------
        model_path:
            Path to the joblib model file. If None, defaults to
            artifacts/models/xgb_price_per_m2.joblib under the project root.
        metadata_path:
            Path to the JSON metadata file. If None, defaults to the
            file next to the model artifact.

        Raises
        ------
        ModelLoadError
            If the artifact or metadata file is missing or unreadable.
        """
        resolved_model_path = self._resolve_model_path(model_path)
        resolved_metadata_path = self._resolve_metadata_path(metadata_path, resolved_model_path)

        with self._lock:
            if not resolved_model_path.exists():
                raise ModelLoadError(f"Model artifact not found: {resolved_model_path}")

            logger.info("loading model from %s", resolved_model_path)

            try:
                pipeline = joblib.load(resolved_model_path)
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load model from {resolved_model_path}: {exc}"
                ) from exc

            if not isinstance(pipeline, Pipeline):
                raise ModelLoadError(
                    f"Loaded object is not a sklearn Pipeline: {type(pipeline).__name__}"
                )

            metadata = self._load_metadata(resolved_metadata_path)

            self._pipeline = pipeline
            self._metadata = metadata
            self._artifact_path = resolved_model_path

            logger.info(
                "model loaded successfully | artifact=%s | metadata_keys=%s",
                resolved_model_path.name,
                list(metadata.keys()),
            )

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Run inference on a list of raw input rows.

        Parameters
        ----------
        rows:
            List of dictionaries, each containing all REQUIRED_FEATURES keys.

        Returns
        -------
        np.ndarray
            Predicted price_per_m2 values in original scale (Toman/m2).

        Raises
        ------
        ModelNotLoadedError
            If load() has not been called.
        ValueError
            If rows is empty or missing required features.
        """
        if self._pipeline is None:
            raise ModelNotLoadedError("Model is not loaded. Call load() first.")

        if not rows:
            raise ValueError("rows must not be empty")

        frame = pd.DataFrame(rows)
        missing = [feat for feat in REQUIRED_FEATURES if feat not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        predictions = self._pipeline.predict(frame)
        return np.asarray(predictions, dtype=np.float64)

    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        return self._pipeline is not None

    def get_metadata(self) -> dict[str, Any]:
        """Return a shallow copy of the loaded model metadata."""
        return dict(self._metadata)

    def get_artifact_path(self) -> Path | None:
        """Return the resolved path of the loaded artifact, or None."""
        return self._artifact_path

    def reset(self) -> None:
        """Unload the model. Primarily useful for tests."""
        with self._lock:
            self._pipeline = None
            self._metadata = {}
            self._artifact_path = None

    @staticmethod
    def _resolve_model_path(model_path: Path | str | None) -> Path:
        if model_path is not None:
            return Path(model_path)
        return project_root() / "artifacts" / DEFAULT_MODELS_SUBDIR / DEFAULT_MODEL_FILENAME

    @staticmethod
    def _resolve_metadata_path(
        metadata_path: Path | str | None,
        model_path: Path,
    ) -> Path:
        if metadata_path is not None:
            return Path(metadata_path)
        return model_path.parent / DEFAULT_METADATA_FILENAME

    @staticmethod
    def _load_metadata(metadata_path: Path) -> dict[str, Any]:
        if not metadata_path.exists():
            logger.warning("metadata file not found: %s", metadata_path)
            return {}

        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("failed to read metadata file %s: %s", metadata_path, exc)
            return {}

        if not isinstance(data, dict):
            logger.warning("metadata file does not contain a JSON object: %s", metadata_path)
            return {}

        return data


_model_service = ModelService()


def get_model_service() -> ModelService:
    """Return the process-wide ModelService singleton."""
    return _model_service
