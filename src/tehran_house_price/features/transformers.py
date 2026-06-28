"""
Custom sklearn transformers for the Tehran house price project.

Why custom transformers and not raw functions?
    Because they integrate cleanly into sklearn.Pipeline, are serializable
    with joblib, and let us call fit/transform with the standard interface
    that the rest of the ML stack expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tehran_house_price.features import constants as fconst


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target mean encoder for a single categorical column.

    Formula:
        encoded(c) = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)

    Where:
        n_c        = number of rows with category c
        mean_c     = mean target for category c
        global_mean = overall mean target
        smoothing  = how much we pull rare categories toward the global mean

    Notes:
        - Unknown categories at transform time fall back to the global mean.
        - This transformer assumes the input is a DataFrame with `column`
          present, and `y` is a numeric Series.
        - Use it INSIDE a CV loop to avoid leakage. The orchestrator is
          responsible for that.
    """

    def __init__(
        self,
        column: str,
        smoothing: float = fconst.DEFAULT_TARGET_ENC_SMOOTHING,
        output_col: str | None = None,
    ) -> None:
        self.column = column
        self.smoothing = smoothing
        self.output_col = output_col or f"{column}_target_enc"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TargetMeanEncoder:
        if self.column not in X.columns:
            raise ValueError(f"column '{self.column}' not in input DataFrame")

        df = pd.DataFrame({self.column: X[self.column].values, "_y": np.asarray(y)})

        self.global_mean_: float = float(df["_y"].mean())

        agg = df.groupby(self.column)["_y"].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + self.smoothing * self.global_mean_) / (
            agg["count"] + self.smoothing
        )

        self.mapping_: dict[str, float] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mapping_"):
            raise RuntimeError("TargetMeanEncoder must be fit before transform")

        encoded = X[self.column].map(self.mapping_).fillna(self.global_mean_)
        out = X.copy()
        out[self.output_col] = encoded.astype(float).values
        return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Count-based frequency encoder. Replaces category with its training
    frequency. Unknown categories at transform time get 0.
    """

    def __init__(self, column: str, output_col: str | None = None) -> None:
        self.column = column
        self.output_col = output_col or f"{column}_freq_enc"

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FrequencyEncoder:
        if self.column not in X.columns:
            raise ValueError(f"column '{self.column}' not in input DataFrame")
        counts = X[self.column].value_counts(normalize=True)
        self.mapping_: dict[str, float] = counts.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mapping_"):
            raise RuntimeError("FrequencyEncoder must be fit before transform")

        encoded = X[self.column].map(self.mapping_).fillna(0.0)
        out = X.copy()
        out[self.output_col] = encoded.astype(float).values
        return out


class AreaPerRoomAdder(BaseEstimator, TransformerMixin):
    """
    Add 'area_per_room' as a derived feature.

    rooms = 0 is treated as 1 to avoid division by zero. This is a defensive
    choice: in cleaned Kaggle data rooms >= 1, but we want the transformer
    safe by construction.
    """

    def __init__(
        self,
        area_col: str = "area_m2",
        rooms_col: str = "rooms",
        output_col: str = fconst.FEATURE_AREA_PER_ROOM,
    ) -> None:
        self.area_col = area_col
        self.rooms_col = rooms_col
        self.output_col = output_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> AreaPerRoomAdder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        safe_rooms = np.where(out[self.rooms_col] <= 0, 1, out[self.rooms_col])
        out[self.output_col] = out[self.area_col].astype(float) / safe_rooms
        return out


class BooleanToIntCaster(BaseEstimator, TransformerMixin):
    """Cast bool columns to int in-place. XGBoost prefers numeric input."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BooleanToIntCaster:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.columns:
            if c in out.columns:
                out[c] = out[c].astype(int)
        return out


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed ordered list of columns. Final step of the pipeline."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ColumnSelector:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"missing columns at transform: {missing}")
        return X[self.columns].copy()
