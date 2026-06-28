"""
Build the feature engineering pipeline.

This module assembles all transformers into a single sklearn Pipeline.
The output of this pipeline is the feature matrix that the model trains on.

The pipeline is serializable with joblib and is meant to be saved as part
of the final model artifact in Phase 2.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from tehran_house_price.features import constants as fconst
from tehran_house_price.features.transformers import (
    AreaPerRoomAdder,
    BooleanToIntCaster,
    ColumnSelector,
    FrequencyEncoder,
    TargetMeanEncoder,
)
from tehran_house_price.utils.logger import get_logger

log = get_logger(__name__)


def build_feature_pipeline(
    target_enc_smoothing: float = fconst.DEFAULT_TARGET_ENC_SMOOTHING,
) -> Pipeline:
    """
    Build the feature engineering pipeline.

    Steps:
        1. Add area_per_room.
        2. Target-encode district (needs y at fit time).
        3. Frequency-encode district.
        4. Cast booleans to int.
        5. Select final ordered feature columns.

    Returns:
        sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline(
        steps=[
            ("area_per_room", AreaPerRoomAdder()),
            (
                "district_target_enc",
                TargetMeanEncoder(
                    column="district",
                    smoothing=target_enc_smoothing,
                    output_col=fconst.FEATURE_DISTRICT_TARGET_ENC,
                ),
            ),
            (
                "district_freq_enc",
                FrequencyEncoder(
                    column="district",
                    output_col=fconst.FEATURE_DISTRICT_FREQ_ENC,
                ),
            ),
            ("bool_to_int", BooleanToIntCaster(columns=fconst.BOOLEAN_INPUT_COLS)),
            ("select", ColumnSelector(columns=fconst.FINAL_FEATURE_COLS)),
        ]
    )
    return pipeline


def transform_target_for_training(y: pd.Series) -> np.ndarray:
    """
    Apply log1p to the target so the model learns on a more normal scale.

    Use inverse_target_for_prediction() to map predictions back to the
    original scale at inference time.
    """
    return np.log1p(np.asarray(y, dtype=float))


def inverse_target_for_prediction(y_log: np.ndarray) -> np.ndarray:
    """Inverse of transform_target_for_training()."""
    return np.expm1(np.asarray(y_log, dtype=float))
