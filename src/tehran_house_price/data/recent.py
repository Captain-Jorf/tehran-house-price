"""Recent Tehran price research: explicit coverage gaps and separate measures.

This does NOT train the individual-listing production model. Input is a pinned,
manual extraction of published facts. Reproduction is offline; no imputation,
source splicing, synthetic observations, or claim of complete five-year data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tehran_house_price.models.evaluation import compute_global_metrics

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/external/tehran_recent"
REPORT = ROOT / "docs/reports/tehran_recent"
SNAPSHOT_SHA256 = "48bd2c01407062f8df593dad46a426701485170338934acfc1ee1579852cc7a4"
START, END = 140006, 140505
MODELS = ("last_month", "calendar_drift", "log_trend_6obs")
SERIES = ("cbi_transaction_mean", "kilid_listing_indicator")
UNITS = {
    "toman": 1,
    "million_toman": 1e6,
    "million_irr": 1e5,
    "dlearn_pre_140105": 1,
    "dlearn_post_140105": 100,
}
TEST_START = {SERIES[0]: 140201, SERIES[1]: 140503}


def month_index(month: int) -> int:
    """Continuous Jalali month index; YYYYMM integers are NOT elapsed times."""
    if isinstance(month, bool) or not float(month).is_integer():
        raise ValueError("Invalid Jalali month")
    year, number = divmod(int(month), 100)
    if not 1 <= number <= 12 or year < 1:
        raise ValueError("Invalid Jalali month")
    return year * 12 + number - 1


def month_range(start: int = START, end: int = END) -> list[int]:
    return [
        (index // 12) * 100 + index % 12 + 1
        for index in range(month_index(start), month_index(end) + 1)
    ]


def load_observations(data_dir: Path = DATA) -> tuple[pd.DataFrame, dict]:
    path = data_dir / "observations.json"
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("Curated snapshot checksum mismatch; review provenance before updating")
    payload = json.loads(content)
    rows = []
    for record in payload["records"]:
        source = payload["sources"][record["source_id"]]
        rows.append(
            {
                **record,
                "price_per_m2_toman": record["raw_value"] * UNITS[record["raw_unit"]],
                "source_url": source["url"],
                "retrieved_on": source["retrieved_on"],
            }
        )
    frame = pd.DataFrame(rows)
    validate_observations(frame)
    return frame, payload


def validate_observations(frame: pd.DataFrame) -> None:
    required = {
        "month_jalali",
        "series",
        "geography",
        "price_per_m2_toman",
        "raw_value",
        "raw_unit",
        "source_id",
        "source_url",
        "evidence",
        "quality",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Missing observation fields")
    if frame[list(required)].isna().any().any():
        raise ValueError("Observation fields may not be null")
    if frame.duplicated(["series", "month_jalali"]).any():
        raise ValueError("Duplicate observation")
    if not frame.series.isin(SERIES).all() or not frame.geography.eq("Tehran").all():
        raise ValueError("Unsupported series/geography")
    for column in ("price_per_m2_toman", "raw_value"):
        if not np.isfinite(frame[column]).all() or (frame[column] <= 0).any():
            raise ValueError("Prices must be finite and positive")
    for row in frame.itertuples():
        month_index(row.month_jalali)
        if row.month_jalali not in month_range():
            raise ValueError("Observation outside recent completed-month window")
        if row.raw_unit not in UNITS or not np.isclose(
            row.price_per_m2_toman, row.raw_value * UNITS[row.raw_unit]
        ):
            raise ValueError("Invalid currency conversion")
        if not row.source_url.startswith("https://") or not row.evidence.strip():
            raise ValueError("Missing evidence/source URL")
    for _, group in frame.groupby("series"):
        if not group.month_jalali.is_monotonic_increasing:
            raise ValueError("Observations must be chronological within each series")


def coverage_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep missing values missing, separately for each incompatible measure."""
    index = pd.MultiIndex.from_product([SERIES, month_range()], names=["series", "month_jalali"])
    ledger = frame.set_index(["series", "month_jalali"]).reindex(index).reset_index()
    ledger["status"] = np.where(ledger.price_per_m2_toman.notna(), "collected", "not_collected")
    return ledger


def predict(history: pd.DataFrame, target_month: int, model: str) -> float:
    """Fit on past observations using actual month positions, including gaps."""
    if len(history) < 6 or history.series.nunique() != 1:
        raise ValueError("Need six observations from exactly one series")
    x = np.array([month_index(month) for month in history.month_jalali])
    y = history.price_per_m2_toman.to_numpy(dtype=float)
    target = month_index(target_month)
    if np.any(np.diff(x) <= 0) or x[-1] != target - 1:
        raise ValueError("Need sorted past history ending exactly one month before target")
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Invalid historical prices")
    if model == "last_month":
        return float(y[-1])
    if model == "calendar_drift":
        return float(y[-1] + (y[-1] - y[0]) / (x[-1] - x[0]))
    if model == "log_trend_6obs":
        # Six observations, not necessarily six consecutive months. Missing
        # calendar positions are retained rather than compressed or imputed.
        slope, intercept = np.polyfit(x[-6:] - x[-6], np.log(y[-6:]), 1)
        return float(np.exp(intercept + slope * (target - x[-6])))
    raise ValueError(f"Unknown model: {model}")


def backtest(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    validate_observations(frame)
    predictions, attempts, metrics, selected = [], [], {}, {}
    for series, group in frame.groupby("series", sort=True):
        group = group.reset_index(drop=True)
        for i, row in enumerate(group.itertuples()):
            split = "validation" if row.month_jalali < TEST_START[series] else "test"
            reason = "scored"
            if i < 6:
                reason = "warmup_less_than_six_observations"
            elif month_index(group.iloc[i - 1].month_jalali) != month_index(row.month_jalali) - 1:
                reason = "previous_calendar_month_missing"
            attempts.append(
                {
                    "series": series,
                    "month_jalali": row.month_jalali,
                    "split": split,
                    "status": reason,
                }
            )
            if reason != "scored":
                continue
            history = group.iloc[:i]
            for model in MODELS:
                predictions.append(
                    {
                        "series": series,
                        "month_jalali": row.month_jalali,
                        "split": split,
                        "model": model,
                        "history_end": int(history.iloc[-1].month_jalali),
                        "n_history": len(history),
                        "actual": row.price_per_m2_toman,
                        "prediction": predict(history, row.month_jalali, model),
                    }
                )
    forecast_frame = pd.DataFrame(predictions)
    for series in SERIES:
        metrics[series] = {}
        for split in ("validation", "test"):
            metrics[series][split] = {}
            for model in MODELS:
                subset = forecast_frame.query(
                    "series == @series and split == @split and model == @model"
                )
                if len(subset) < 2:
                    raise ValueError("Insufficient observations for validation/test metrics")
                metrics[series][split][model] = {
                    "n": len(subset),
                    **compute_global_metrics(subset.actual, subset.prediction),
                }
        selected[series] = min(MODELS, key=lambda m: metrics[series]["validation"][m]["mae"])
    result = {
        "as_of": "2026-09-12",
        "snapshot_sha256": SNAPSHOT_SHA256,
        "complete_five_year_transaction_dataset": set(month_range()).issubset(
            set(frame.loc[frame.series == SERIES[0], "month_jalali"])
        ),
        "target_months": len(month_range()),
        "observations": len(frame),
        "months_with_any_series": int(frame.month_jalali.nunique()),
        "missing_months_all_series": sorted(set(month_range()) - set(frame.month_jalali)),
        "selection_criterion": "validation_mae_per_series",
        "selected": selected,
        "metrics": metrics,
        "protocol": "retrospective one-step final-vintage; publication delays not modeled",
    }
    return forecast_frame, pd.DataFrame(attempts), result


def make_charts(
    ledger: pd.DataFrame, predictions: pd.DataFrame, result: dict, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = ["#136f63", "#b65a21"]
    labels = ["CBI transaction mean (secondary)", "Kilid listing indicator (not transactions)"]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    fig, ax = plt.subplots(figsize=(12, 3))
    grid = np.array(
        [ledger[ledger.series == s].price_per_m2_toman.notna().astype(int) for s in SERIES]
    )
    ax.imshow(grid, aspect="auto", cmap=ListedColormap(["#e5e7eb", "#136f63"]), vmin=0, vmax=1)
    ax.set(
        yticks=[0, 1],
        yticklabels=["CBI (secondary)", "Kilid indicator"],
        xticks=range(0, 60, 3),
        xticklabels=[str(m) for m in month_range()[::3]],
        title="Last 60 completed Jalali months: green = collected, gray = NOT collected",
    )
    ax.tick_params(axis="x", rotation=65)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output / "coverage.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for series, color, label in zip(SERIES, colors, labels, strict=True):
        sub = ledger[ledger.series == series]
        ax.plot(range(60), sub.price_per_m2_toman / 1e6, "o-", color=color, label=label)
    ax.set(
        xticks=range(0, 60, 6),
        xticklabels=[str(m) for m in month_range()[::6]],
        ylabel="Million toman / m²",
        xlabel="Jalali month",
        title="Published prices: gaps preserved, incompatible series NOT spliced",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "prices.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for ax, series, label in zip(axes, SERIES, labels, strict=True):
        sub = predictions[(predictions.series == series) & (predictions.split == "test")]
        selected = result["selected"][series]
        actual = sub[sub.model == selected]
        x = [month_index(m) for m in actual.month_jalali]
        # Points only: do not visually connect skipped target months.
        ax.scatter(x, actual.actual / 1e6, label="Published actual", color="black")
        for model in MODELS:
            model_sub = sub[sub.model == model]
            ax.scatter(x, model_sub.prediction / 1e6, marker="x", label=model)
        ax.set(
            xticks=x,
            xticklabels=actual.month_jalali.astype(str),
            ylabel="Million toman / m²",
            title=f"{label}; selected on validation: {selected}",
        )
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "backtest.png", dpi=160)
    plt.close(fig)


def run(data_dir: Path = DATA, output: Path = REPORT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    frame, payload = load_observations(data_dir)
    ledger = coverage_ledger(frame)
    predictions, attempts, result = backtest(frame)
    frame.to_csv(output / "observations.csv", index=False)
    ledger.to_csv(output / "coverage.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    attempts.to_csv(output / "evaluation_eligibility.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    make_charts(ledger, predictions, result, output)
    lines = [
        "| Series | Split | Model | n | MAE (toman/m²) | RMSE | MAPE (%) | R² |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for series, splits in result["metrics"].items():
        for split, models in splits.items():
            for model, m in models.items():
                lines.append(
                    f"| {series} | {split} | {model} | {m['n']} | {m['mae']:,.0f} | "
                    f"{m['rmse']:,.0f} | {100*m['mape']:.2f} | {m['r2']:.3f} |"
                )
    (output / "metrics_table.md").write_text("\n".join(lines) + "\n")
    sources = [
        "# منابع و یادداشت‌های استخراج",
        "",
        "بازیابی: 2026-09-12؛ خلاصهٔ حقایق، نه آرشیو کامل صفحات.",
    ]
    for key, source in payload["sources"].items():
        sources += [
            f"\n## {key}",
            f"[{source['title']}]({source['url']})",
            f"\nروش دسترسی: `{source['access']}`",
            f"\n{source['notes']}",
        ]
    (output / "sources.md").write_text("\n".join(sources) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=REPORT)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
