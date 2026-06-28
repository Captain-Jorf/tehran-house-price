"""
Feature constants.

Feature names and groups used by the feature engineering pipeline.
Centralizing these here keeps transformers and tests in sync.
"""

from __future__ import annotations

# raw input columns used as features
NUMERIC_INPUT_COLS: list[str] = [
    "area_m2",
    "rooms",
]

BOOLEAN_INPUT_COLS: list[str] = [
    "has_parking",
    "has_storage",
    "has_elevator",
]

CATEGORICAL_INPUT_COLS: list[str] = [
    "district",
]

# target column (we predict price_per_m2, not total_price)
TARGET_COL: str = "price_per_m2"

# derived feature names produced by transformers
FEATURE_DISTRICT_TARGET_ENC: str = "district_target_enc"
FEATURE_DISTRICT_FREQ_ENC: str = "district_freq_enc"
FEATURE_AREA_PER_ROOM: str = "area_per_room"

# final ordered list of features fed to the model
FINAL_FEATURE_COLS: list[str] = [
    "area_m2",
    "rooms",
    "has_parking",
    "has_storage",
    "has_elevator",
    FEATURE_DISTRICT_TARGET_ENC,
    FEATURE_DISTRICT_FREQ_ENC,
    FEATURE_AREA_PER_ROOM,
]

# default smoothing constant for target encoding
DEFAULT_TARGET_ENC_SMOOTHING: float = 10.0

# fallback value when a district is unseen at prediction time
# (will be replaced at fit time with the global target mean)
UNSEEN_CATEGORY_DEFAULT: float = 0.0
