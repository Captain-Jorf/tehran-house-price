"""MLflow-based experiment tracking layer for Tehran House Price.

This subpackage is intentionally additive: training and evaluation
work with or without MLflow. Use ``setup_mlflow()`` once at the
start of a run and ``get_run_context()`` to wrap a logical training run.
"""

from tehran_house_price.tracking.mlflow_setup import (
    DEFAULT_EXPERIMENT_NAME,
    get_run_context,
    get_tracking_uri,
    is_tracking_enabled,
    setup_mlflow,
)

__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "get_run_context",
    "get_tracking_uri",
    "is_tracking_enabled",
    "setup_mlflow",
]
