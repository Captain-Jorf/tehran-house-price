"""Offline audit of public newspaper examples, not a five-year model evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/external/tehran_listing_archive"
REPORT = ROOT / "docs/reports/tehran_listing_archive"


def load_archive(data_dir: Path = DATA) -> pd.DataFrame:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    path = data_dir / "listings.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Archive checksum mismatch")
    frame = pd.read_csv(path, dtype={"source_id": str, "age_raw": str})
    if len(frame) != manifest["row_count"]:
        raise ValueError("Row count mismatch")
    if frame.duplicated(["source_id", "source_row"]).any():
        raise ValueError("Duplicate source row")
    if not frame.measure.eq("asking_price").all():
        raise ValueError("Incompatible price measure")
    for source_id, source in manifest["sources"].items():
        group = frame[frame.source_id == source_id]
        if (
            len(group) != source["rows"]
            or not group.publication_date_jalali.eq(source["publication_date_jalali"]).all()
        ):
            raise ValueError("Source/date mismatch")
    if not (frame.price_per_m2_toman == (frame.price_per_m2_million_toman_raw * 1e6).round()).all():
        raise ValueError("Currency conversion mismatch")
    if not (frame.area_m2.gt(0) & frame.price_per_m2_toman.gt(0)).all():
        raise ValueError("Invalid price/area")
    new = frame.age_raw.eq("نوساز")
    if not frame.loc[new, "age_years"].isna().all():
        raise ValueError("New-build numeric age is unknown")
    if not frame.loc[~new, "age_years"].eq(frame.loc[~new, "age_raw"].astype(int)).all():
        raise ValueError("Age transcription mismatch")
    return frame


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = load_archive()
    REPORT.mkdir(parents=True, exist_ok=True)
    counts = frame.groupby("publication_date_jalali").size()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(counts.index, counts.values, color="#247b86")
    ax.set(
        title="Public asking-price examples: three snapshots, NOT five-year coverage",
        xlabel="Publication date (Jalali)",
        ylabel="Published rows",
    )
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha="center")
    ax.set_ylim(0, 65)
    fig.tight_layout()
    fig.savefig(REPORT / "sample_counts.png", dpi=150)
    plt.close(fig)
    print(counts.to_string())


if __name__ == "__main__":
    main()
