"""Unit tests for the evaluation layer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from tehran_house_price.models import evaluation as ev


def test_compute_global_metrics_basic_correctness():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])
    m = ev.compute_global_metrics(y_true, y_pred)

    assert m["mae"] == pytest.approx(np.mean([2.0, 2.0, 3.0, 4.0]))
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean([4.0, 4.0, 9.0, 16.0])))
    assert m["medae"] == pytest.approx(np.median([2.0, 2.0, 3.0, 4.0]))
    assert m["mape"] == pytest.approx(np.mean([2 / 10, 2 / 20, 3 / 30, 4 / 40]))
    assert 0.0 <= m["r2"] <= 1.0


def test_compute_global_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = ev.compute_global_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["mape"] == 0.0
    assert m["r2"] == pytest.approx(1.0)


def test_compute_global_metrics_handles_zero_target():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([1.0, 12.0, 18.0])
    m = ev.compute_global_metrics(y_true, y_pred)
    expected_mape = np.mean([2 / 10, 2 / 20])
    assert m["mape"] == pytest.approx(expected_mape)


def test_compute_global_metrics_r2_nan_when_constant_y_true():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([4.0, 6.0, 5.0])
    m = ev.compute_global_metrics(y_true, y_pred)
    assert np.isnan(m["r2"])


def test_compute_global_metrics_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        ev.compute_global_metrics(np.array([1, 2]), np.array([1, 2, 3]))


def test_compute_global_metrics_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        ev.compute_global_metrics(np.array([]), np.array([]))


def test_compute_per_district_metrics_basic():
    y_true = np.array([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
    y_pred = np.array([12.0, 18.0, 33.0, 110.0, 190.0, 320.0])
    districts = pd.Series(["A", "A", "A", "B", "B", "B"])

    out = ev.compute_per_district_metrics(y_true, y_pred, districts, min_samples=1)
    assert set(out.index) == {"A", "B"}
    assert (out["n"] == 3).all()
    # sorted by MAPE descending
    assert out.iloc[0]["mape"] >= out.iloc[1]["mape"]


def test_compute_per_district_metrics_excludes_small_groups():
    y_true = np.array([10.0, 20.0, 30.0, 100.0])
    y_pred = np.array([12.0, 18.0, 33.0, 110.0])
    districts = pd.Series(["A", "A", "A", "B"])
    out = ev.compute_per_district_metrics(y_true, y_pred, districts, min_samples=3)
    assert "A" in out.index
    assert "B" not in out.index


def test_select_worst_districts_returns_top_k():
    per_district = pd.DataFrame(
        {"n": [5, 5, 5, 5], "mae": [1, 2, 3, 4], "mape": [0.4, 0.3, 0.2, 0.1]},
        index=["A", "B", "C", "D"],
    )
    out = ev.select_worst_districts(per_district, top_k=2)
    assert list(out.index) == ["A", "B"]


def test_select_worst_districts_handles_empty():
    empty = pd.DataFrame(columns=["n", "mae", "mape"])
    out = ev.select_worst_districts(empty, top_k=5)
    assert out.empty


def test_evaluate_returns_full_report_with_districts():
    y_true = np.array([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
    y_pred = np.array([11.0, 19.0, 32.0, 105.0, 195.0, 310.0])
    districts = pd.Series(["A", "A", "A", "B", "B", "B"])

    report = ev.evaluate("toy_model", y_true, y_pred, districts=districts, save=False)
    assert report.model_name == "toy_model"
    assert report.n_samples == 6
    assert "mae" in report.global_metrics
    assert set(report.per_district.index) == {"A", "B"}
    assert not report.worst_districts.empty


def test_evaluate_without_districts_returns_empty_breakdown():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([11.0, 21.0])
    report = ev.evaluate("toy_model", y_true, y_pred, save=False)
    assert report.per_district.empty
    assert report.worst_districts.empty


def test_evaluate_save_writes_json_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "artifacts_dir", lambda: tmp_path)

    y_true = np.array([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
    y_pred = np.array([11.0, 19.0, 32.0, 105.0, 195.0, 310.0])
    districts = pd.Series(["A", "A", "A", "B", "B", "B"])

    report = ev.evaluate("toy_model", y_true, y_pred, districts=districts, save=True)
    assert report.report_path is not None
    assert report.report_path.exists()

    data = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert data["model"] == "toy_model"
    assert data["n_samples"] == 6
    assert "global_metrics" in data
    assert "per_district" in data
    assert "worst_districts" in data


def test_compare_models_sorts_by_mape_ascending():
    r1 = ev.EvaluationReport(
        model_name="model_a",
        global_metrics={"mae": 1.0, "rmse": 1.5, "medae": 0.8, "mape": 0.30, "r2": 0.5},
        n_samples=100,
        per_district=pd.DataFrame(),
        worst_districts=pd.DataFrame(),
    )
    r2 = ev.EvaluationReport(
        model_name="model_b",
        global_metrics={"mae": 0.5, "rmse": 0.8, "medae": 0.4, "mape": 0.15, "r2": 0.8},
        n_samples=100,
        per_district=pd.DataFrame(),
        worst_districts=pd.DataFrame(),
    )
    out = ev.compare_models([r1, r2])
    assert list(out["model"]) == ["model_b", "model_a"]


def test_compare_models_raises_on_empty():
    with pytest.raises(ValueError, match="at least one"):
        ev.compare_models([])
