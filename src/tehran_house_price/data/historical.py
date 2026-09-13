"""Pinned, observed monthly city-level prices; NOT individual house listings.

Run from the repository root with PYTHONPATH=src python -m
tehran_house_price.data.historical. No network or random generation is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tehran_house_price.models.evaluation import compute_global_metrics

SOURCE_SHA256 = "48e7d19787ea8738140c57df735aa18c0abdf3e0c6dd8b0d88f6ceb20355199b"
COMMIT = "ac6a6406f2c5507ab9a905859efc1b61cb4d17c1"
SOURCE_URL = (
    "https://github.com/amiralimadadi/Regression_TheranHousing/blob/"
    f"{COMMIT}/TehranHousingPriceBackground.csv"
)
MODELS = ("last_month", "drift", "log_trend_12m")


def validate_monthly(frame: pd.DataFrame) -> None:
    """Reject missing, duplicated, unordered, non-positive or non-finite data."""
    expected = [year * 100 + month for year in range(1395, 1400) for month in range(1, 13)]
    if list(frame.columns) != ["Month", "Price"] or frame["Month"].tolist() != expected:
        raise ValueError("Expected exactly 60 ordered Jalali months: 139501..139912")
    prices = frame["Price"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and positive")


def load_source(path: Path) -> pd.DataFrame:
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source checksum mismatch")
    frame = pd.read_csv(path)
    validate_monthly(frame)
    return frame


def forecast(history: np.ndarray, model: str) -> float:
    """One-step forecast using only observed history, with fixed specifications."""
    history = np.asarray(history, dtype=float)
    if len(history) < 12 or not np.isfinite(history).all() or (history <= 0).any():
        raise ValueError("At least 12 finite positive historical values are required")
    if model == "last_month":
        return float(history[-1])
    if model == "drift":
        return float(history[-1] + (history[-1] - history[0]) / (len(history) - 1))
    if model == "log_trend_12m":
        slope, intercept = np.polyfit(np.arange(12), np.log(history[-12:]), 1)
        return float(np.exp(intercept + slope * 12))
    raise ValueError(f"Unknown model: {model}")


def backtest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """36 warm-up months, 12 validation months, 12 untouched selection-test months.

    After every forecast the next actual becomes available. This is rolling
    one-month-ahead evaluation, not a fixed-origin twelve-month forecast.
    Model choice is frozen using validation MAE before test scores are examined.
    """
    validate_monthly(frame)
    prices = frame["Price"].to_numpy(dtype=float)
    rows = []
    for index in range(36, 60):
        for model in MODELS:
            rows.append(
                {
                    "month_jalali": int(frame.iloc[index]["Month"]),
                    "split": "validation" if index < 48 else "test",
                    "model": model,
                    "actual": prices[index],
                    "prediction": forecast(prices[:index], model),
                    "history_end": int(frame.iloc[index - 1]["Month"]),
                }
            )
    predictions = pd.DataFrame(rows)
    metrics = {}
    for split in ("validation", "test"):
        metrics[split] = {}
        for model in MODELS:
            subset = predictions.query("split == @split and model == @model")
            metrics[split][model] = compute_global_metrics(subset.actual, subset.prediction)
    selected = min(MODELS, key=lambda model: metrics["validation"][model]["mae"])
    return predictions, {"selected_on_validation_mae": selected, "metrics": metrics}


def plot_report(frame: pd.DataFrame, predictions: pd.DataFrame, result: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.figsize": (10, 4.5), "axes.grid": True, "grid.alpha": 0.2})
    fig, ax = plt.subplots()
    ax.plot(np.arange(60), frame.Price / 1e6, color="#136f63", linewidth=2)
    ax.axvspan(35.5, 47.5, alpha=0.15, color="orange", label="Validation: 1398")
    ax.axvspan(47.5, 59.5, alpha=0.12, color="blue", label="Test: 1399")
    ax.set(
        xticks=range(0, 60, 12),
        xticklabels=range(1395, 1400),
        xlabel="Jalali year",
        ylabel="Price per m² (million source units)",
        title="Tehran monthly observed prices — secondary source, currency unverified",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    test = predictions[predictions.split == "test"]
    actual = test[test.model == MODELS[0]]
    ax.plot(range(1, 13), actual.actual / 1e6, "k-o", label="Observed", linewidth=2)
    for model in MODELS:
        ax.plot(range(1, 13), test[test.model == model].prediction / 1e6, "--", label=model)
    ax.set(
        xlabel="Month in Jalali year 1399",
        ylabel="Million source units per m²",
        title="Rolling one-month-ahead test predictions",
        xticks=range(1, 13),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "predictions.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    x = np.arange(len(MODELS))
    for offset, split in ((-0.2, "validation"), (0.2, "test")):
        ax.bar(
            x + offset,
            [result["metrics"][split][m]["mape"] * 100 for m in MODELS],
            width=0.4,
            label=split,
        )
    ax.set(
        xticks=x, xticklabels=MODELS, ylabel="MAPE (%)", title="Forecast errors — lower is better"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=160)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data = root / "data/external/tehran_monthly"
    out = root / "docs/reports/tehran_monthly"
    out.mkdir(parents=True, exist_ok=True)
    frame = load_source(data / "source.csv")
    normalized = pd.DataFrame(
        {
            "month_jalali": frame.Month,
            "year_jalali": frame.Month // 100,
            "month_of_year": frame.Month % 100,
            "price_per_m2_source_units": frame.Price,
            "geography": "Tehran",
            "currency_status": "not_explicit_in_source",
            "source_url": SOURCE_URL,
        }
    )
    normalized.to_csv(data / "monthly.csv", index=False)
    predictions, result = backtest(frame)
    result.update(
        {
            "source_sha256": SOURCE_SHA256,
            "observations": len(frame),
            "train_months": 36,
            "validation_months": 12,
            "test_months": 12,
        }
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions.to_csv(out / "predictions.csv", index=False)
    plot_report(frame, predictions, result, out)
    rows = ["| مرحله | مدل | MAE | RMSE | MAPE (%) | R² |", "|---|---|---:|---:|---:|---:|"]
    for split in ("validation", "test"):
        for model, metrics in result["metrics"][split].items():
            rows.append(
                f"| {split} | {model} | {metrics['mae']:,.0f} | "
                f"{metrics['rmse']:,.0f} | {metrics['mape'] * 100:.2f} | "
                f"{metrics['r2']:.3f} |"
            )
    (out / "metrics_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
