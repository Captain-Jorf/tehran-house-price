"""
Dataset build orchestration.

This module glues together the Phase 1 data pipeline:

  1. ingest raw data
  2. clean raw data into interim parquet
  3. validate the interim dataset against the schema
  4. promote the validated dataset into data/processed/

Why keep this as a separate orchestrator?
- each step remains independently runnable for debugging
- the project still gets a single entrypoint for reproducible builds
- this structure is easy to extend later when we merge Kaggle + Divar
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from tehran_house_price.data import clean, ingest_kaggle, validate
from tehran_house_price.settings import get_config
from tehran_house_price.utils.logger import get_logger
from tehran_house_price.utils.paths import ensure_dir, processed_dir, raw_dir

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Paths produced by a successful dataset build."""

    raw_target_dir: Path
    interim_path: Path
    validation_report_path: Path
    processed_path: Path


def _resolve_kaggle_raw_target() -> Path:
    """Return the configured raw target directory for Kaggle files."""
    cfg = get_config()
    subdir = cfg["data"]["kaggle"]["raw_subdir"]
    return ensure_dir(raw_dir() / subdir)


def _resolve_processed_output_path(output_name: str | None = None) -> Path:
    """
    Resolve the final processed dataset path.

    We keep Phase 1 simple and only support parquet output for now.
    """
    cfg = get_config()
    processed_cfg = cfg.get("processed", {})
    output_format = processed_cfg.get("format", "parquet")

    if output_format != "parquet":
        raise ValueError(
            f"unsupported processed format '{output_format}'. "
            "only 'parquet' is supported right now."
        )

    filename = output_name or processed_cfg.get("output_filename", "tehran_houses.parquet")

    if not filename.endswith(".parquet"):
        raise ValueError("processed output filename must end with '.parquet'")

    return ensure_dir(processed_dir()) / filename


def promote_dataset(interim_path: Path, output_name: str | None = None) -> Path:
    """
    Copy the validated interim dataset into data/processed/.

    We copy instead of move because interim data is still useful for debugging,
    auditability, and step-by-step reruns.
    """
    if not interim_path.exists():
        raise FileNotFoundError(f"interim dataset not found: {interim_path}")

    processed_path = _resolve_processed_output_path(output_name=output_name)
    shutil.copy2(interim_path, processed_path)

    log.info("promoted dataset from %s to %s", interim_path, processed_path)
    return processed_path


def build_dataset(
    *,
    force_ingest: bool = False,
    skip_ingest: bool = False,
    output_name: str | None = None,
) -> BuildResult:
    """
    Run the full Phase 1 data build.

    Args:
        force_ingest:
            Re-download Kaggle files even if raw CSVs already exist.
        skip_ingest:
            Skip the ingestion step and reuse files already present in data/raw/.
        output_name:
            Optional override for processed output filename.

    Returns:
        BuildResult with all important output paths.

    Raises:
        RuntimeError:
            If validation fails.
        FileNotFoundError:
            If a required intermediate file is missing.
        ValueError:
            If config/output settings are invalid.
    """
    log.info(
        "starting dataset build | force_ingest=%s | skip_ingest=%s | output_name=%s",
        force_ingest,
        skip_ingest,
        output_name,
    )

    if force_ingest and skip_ingest:
        raise ValueError("force_ingest=True cannot be used together with skip_ingest=True")

    if skip_ingest:
        raw_target = _resolve_kaggle_raw_target()
        log.info("skipping ingest step; expecting raw files under %s", raw_target)
    else:
        raw_target = ingest_kaggle.download_dataset(force=force_ingest)

    interim_path = clean.run()
    log.info("clean step completed: %s", interim_path)

    passed, report_path = validate.run(parquet_path=interim_path)
    if not passed:
        raise RuntimeError(f"dataset validation failed. see report: {report_path}")

    processed_path = promote_dataset(interim_path, output_name=output_name)

    result = BuildResult(
        raw_target_dir=raw_target,
        interim_path=interim_path,
        validation_report_path=report_path,
        processed_path=processed_path,
    )

    log.info("dataset build completed successfully: %s", result)
    return result


def main() -> int:
    """CLI entrypoint for the full data build."""
    parser = argparse.ArgumentParser(
        description="Run the full Phase 1 dataset pipeline: ingest -> clean -> validate -> promote."
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="re-download Kaggle files even if raw CSVs already exist",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="skip ingestion and reuse files already present in data/raw/",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="override processed output filename (must end with .parquet)",
    )
    args = parser.parse_args()

    try:
        result = build_dataset(
            force_ingest=args.force_ingest,
            skip_ingest=args.skip_ingest,
            output_name=args.output_name,
        )
        print("dataset build PASSED")
        print(f"raw target: {result.raw_target_dir}")
        print(f"interim: {result.interim_path}")
        print(f"validation report: {result.validation_report_path}")
        print(f"processed: {result.processed_path}")
        return 0
    except Exception as e:
        log.error("dataset build failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
