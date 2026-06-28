"""Integration tests for the dataset build orchestrator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tehran_house_price.data import build_dataset


def _write_dummy_parquet(path: Path) -> None:
    """Write a tiny parquet file for orchestration tests."""
    df = pd.DataFrame({"listing_id": ["a1"], "total_price": [1.0]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_build_dataset_promotes_validated_interim_file(tmp_path, monkeypatch):
    """Successful build should copy interim parquet into processed/."""
    raw_target = tmp_path / "data" / "raw" / "kaggle"
    processed_base = tmp_path / "data" / "processed"
    interim_path = tmp_path / "data" / "interim" / "kaggle_clean.parquet"
    report_path = tmp_path / "artifacts" / "data_validation" / "kaggle_validation.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('{"passed": true, "errors": []}', encoding="utf-8")

    def fake_download_dataset(force: bool = False) -> Path:
        raw_target.mkdir(parents=True, exist_ok=True)
        csv_content = (
            "Area,Room,Parking,Warehouse,Elevator,Address,Price\n"
            "85,2,True,False,True,Vanak,7500000000\n"
        )
        (raw_target / "housePrice.csv").write_text(csv_content, encoding="utf-8")
        return raw_target

    def fake_clean_run() -> Path:
        _write_dummy_parquet(interim_path)
        return interim_path

    def fake_validate_run(parquet_path: Path | None = None) -> tuple[bool, Path]:
        assert parquet_path == interim_path
        return True, report_path

    monkeypatch.setattr(build_dataset.ingest_kaggle, "download_dataset", fake_download_dataset)
    monkeypatch.setattr(build_dataset.clean, "run", fake_clean_run)
    monkeypatch.setattr(build_dataset.validate, "run", fake_validate_run)
    monkeypatch.setattr(build_dataset, "processed_dir", lambda: processed_base)
    monkeypatch.setattr(
        build_dataset,
        "get_config",
        lambda: {
            "data": {"kaggle": {"raw_subdir": "kaggle"}},
            "processed": {
                "output_filename": "tehran_houses.parquet",
                "format": "parquet",
            },
        },
    )

    result = build_dataset.build_dataset()

    expected_processed = processed_base / "tehran_houses.parquet"

    assert result.raw_target_dir == raw_target
    assert result.interim_path == interim_path
    assert result.validation_report_path == report_path
    assert result.processed_path == expected_processed
    assert expected_processed.exists()

    actual = pd.read_parquet(expected_processed)
    assert len(actual) == 1
    assert list(actual.columns) == ["listing_id", "total_price"]


def test_build_dataset_raises_when_validation_fails(tmp_path, monkeypatch):
    """Build should fail fast and not promote data when validation fails."""
    raw_base = tmp_path / "data" / "raw"
    processed_base = tmp_path / "data" / "processed"
    interim_path = tmp_path / "data" / "interim" / "kaggle_clean.parquet"
    report_path = tmp_path / "artifacts" / "data_validation" / "kaggle_validation.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        '{"passed": false, "errors": [{"message": "bad data"}]}', encoding="utf-8"
    )

    def fake_clean_run() -> Path:
        _write_dummy_parquet(interim_path)
        return interim_path

    def fake_validate_run(parquet_path: Path | None = None) -> tuple[bool, Path]:
        assert parquet_path == interim_path
        return False, report_path

    monkeypatch.setattr(build_dataset.clean, "run", fake_clean_run)
    monkeypatch.setattr(build_dataset.validate, "run", fake_validate_run)
    monkeypatch.setattr(build_dataset, "processed_dir", lambda: processed_base)
    monkeypatch.setattr(build_dataset, "raw_dir", lambda: raw_base)
    monkeypatch.setattr(
        build_dataset,
        "get_config",
        lambda: {
            "data": {"kaggle": {"raw_subdir": "kaggle"}},
            "processed": {
                "output_filename": "tehran_houses.parquet",
                "format": "parquet",
            },
        },
    )

    with pytest.raises(RuntimeError, match="dataset validation failed"):
        build_dataset.build_dataset(skip_ingest=True)

    expected_processed = processed_base / "tehran_houses.parquet"
    assert not expected_processed.exists()
