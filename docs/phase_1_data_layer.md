## بازبینی نتایج

عالی، همه چیز کاملاً درست شد:

### dtype های نهایی
```text
area_m2                  float64    ✓ (قبلاً int64 بود)
year_built               float64    ✓ (قبلاً object بود)
floor                    float64    ✓ (قبلاً object بود)
total_floors             float64    ✓ (قبلاً object بود)
```

این یعنی parquet خروجی الان کاملاً production-quality است و Phase 2 می‌تواند بدون normalization اضافی روی آن کار کند.

### تست‌ها
```text
39 passed in 1.79s
```
هیچ regression. عالی.

### Commit history
```text
cf7e6cc  phase 1: normalize output dtypes in clean step for consistent parquet
7a6ff1a  phase 1: add dataset build orchestrator with integration test
1d43f63  phase 1: fix pandera validation - use float for nullable int columns
2b09341  phase 1: data cleaning for kaggle source
a509de0  phase 1: fix logger path resolution + add kaggle ingestion
20b8d0d  phase 1: add data contract (pandera + pydantic schema)
62afc82  phase 1: foundation - settings, logger, paths, configs, deps
df656fc  phase 0: initial project scaffold
```

این یک تاریخچه commit بسیار تمیز و قابل دفاع است.

---

## آخرین کار Phase 1: مستندسازی

حالا می‌رویم سراغ `docs/phase_1_data_layer.md`. این فایل از نظر مصاحبه و portfolio خیلی مهم است چون نشان می‌دهد فقط کد ننوشته‌ای، بلکه design را هم مستند کرده‌ای.

---

## دستور ساخت فایل docs

```powershell
New-Item -Path "docs\phase_1_data_layer.md" -ItemType File -Force | Out-Null
```

(اگر از قبل خالی هست، این `-Force` فقط مطمئن می‌شود که فایل وجود دارد.)

---

## فایل کامل: `docs/phase_1_data_layer.md`

این محتوا را داخل فایل بریز:

```markdown
# Phase 1 - Data Layer

This document describes the data layer of the Tehran House Price project:
where the data comes from, how it is cleaned, what schema it follows,
how it is validated, and how the full pipeline runs end-to-end.

---

## 1. Goal

Build a reproducible, testable, and validated dataset that downstream phases
(ML training, API serving) can rely on without re-doing any data wrangling.

A clean data layer is the foundation for everything else. If this layer is
fragile, every model, every API call, and every deployment will be fragile too.

---

## 2. Data Sources

### 2.1. Kaggle (current)

- Dataset: `mokar2001/house-price-tehran-iran`
- Format: single CSV (`housePrice.csv`)
- Size: 3479 raw rows, 8 columns
- Storage: `data/raw/kaggle/housePrice.csv`

The dataset requires user consent on Kaggle. The Kaggle API returns `403`
without consent, so we support both `kagglehub` automatic download and a
manual fallback path. The manual fallback is the path that actually works
for this specific dataset.

### 2.2. Divar (planned)

- Will be ingested via scraping later.
- Will live under `data/raw/divar/`.
- Phase 1 reserves the canonical columns Divar will provide
  (`neighborhood`, `year_built`, `floor`, `total_floors`, `published_at`).

---

## 3. Architecture

The data layer is split into four independent, runnable modules:

```
ingest_kaggle.py   -> downloads or verifies raw data
clean.py           -> normalizes raw data into the canonical schema
validate.py        -> validates the cleaned data against the schema
build_dataset.py   -> orchestrates the full pipeline and promotes output
```

Each module exposes:

- a `run()` function for programmatic use
- a CLI `main()` entrypoint (argparse) for terminal use

This makes every step debuggable in isolation while still being callable
from a single end-to-end build.

---

## 4. Storage Layout

```
data/
├── raw/                 # immutable input (gitignored)
│   ├── kaggle/
│   │   └── housePrice.csv
│   └── divar/           # reserved for future scraping
├── interim/             # cleaned but not yet promoted (gitignored)
│   └── kaggle_clean.parquet
└── processed/           # validated, downstream-ready (gitignored)
    └── tehran_houses.parquet
```

Why three layers:

- `raw/`: never modified, traceable input
- `interim/`: cleaning output kept for debugging and audit
- `processed/`: only files that passed validation

---

## 5. Canonical Schema

All data, regardless of source, is normalized into the canonical schema.

| Column        | Type            | Nullable | Notes                                    |
|---------------|-----------------|----------|------------------------------------------|
| listing_id    | str             | No       | sha1 hash of source/district/area/rooms/price |
| source        | str             | No       | one of {"kaggle", "divar"}               |
| district      | str             | No       | district name                            |
| neighborhood  | str             | Yes      | not available in Kaggle                  |
| area_m2       | float           | No       | apartment area in square meters          |
| rooms         | int             | No       | number of rooms                          |
| year_built    | float           | Yes      | float to allow NaN                       |
| floor         | float           | Yes      | float to allow NaN                       |
| total_floors  | float           | Yes      | float to allow NaN                       |
| has_elevator  | bool            | Yes      |                                          |
| has_parking   | bool            | Yes      |                                          |
| has_storage   | bool            | Yes      |                                          |
| total_price   | float           | No       | in Toman (verified from Price/USD ratio) |
| price_per_m2  | float           | Yes      | total_price / area_m2                    |
| published_at  | datetime        | Yes      | not available in Kaggle                  |
| ingested_at   | datetime (UTC)  | No       | set during cleaning                      |

### 5.1. Why float for nullable numeric columns

`year_built`, `floor`, and `total_floors` are defined as `float` instead of
`int` in the DataFrame schema. The reason is purely technical:

- numpy `int64` cannot hold `NaN`
- pandas nullable `Int64` is not interchangeable with pandera's `int` coercion
- using `float` with `NaN` is the simplest and most portable choice for a
  DataFrame-level schema

The Pydantic record-level schema still defines them as `int | None`, which is
correct at the API layer because individual records do not have the same
NaN-vs-None ambiguity as columns.

---

## 6. Two-Level Validation

Two complementary validation layers were introduced:

### 6.1. DataFrame validation (pandera)

- File: `src/tehran_house_price/data/schema.py` -> `HouseListingSchema`
- Applied to the cleaned DataFrame before promoting to `processed/`
- Checks dtypes, bounds, uniqueness of `listing_id`, and allowed sources

### 6.2. Record validation (pydantic)

- File: `src/tehran_house_price/data/schema.py` -> `HouseListing`
- Used for individual records (e.g. future FastAPI requests and responses)
- Standard choice for API validation

This split is intentional: pandera is best for tabular data, pydantic is
the standard for API I/O.

---

## 7. Cleaning Pipeline

Implemented in `src/tehran_house_price/data/clean.py`.

Steps, in order:

1. Drop unused columns (`Price(USD)`)
2. Rename raw columns to canonical names
3. Coerce `area_m2` to numeric
4. Coerce `total_price` to numeric
5. Coerce `rooms` to int
6. Coerce boolean strings ("True"/"False") to real booleans
7. Strip whitespace in `district`
8. Drop rows with invalid area/price/district
9. Compute `price_per_m2`
10. Add `source = "kaggle"`
11. Generate stable `listing_id` (sha1 hash of content)
12. Add `ingested_at` (UTC); `published_at` is null for Kaggle
13. Add missing optional columns expected by the schema
14. Drop duplicates by `listing_id`
15. Normalize output dtypes for clean parquet storage

Order matters. For example, `listing_id` depends on `source` and `district`,
so `add_source` must run before `add_listing_id`.

### 7.1. Results on the Kaggle dataset

| Step                  | Count |
|-----------------------|-------|
| Raw rows              | 3479  |
| Dropped (invalid)     | 8     |
| Dropped (duplicates)  | 236   |
| Final cleaned rows    | 3235  |

### 7.2. Output dtypes in `tehran_houses.parquet`

```
area_m2          float64
rooms            int32
has_parking      bool
has_storage      bool
has_elevator     bool
district         object
total_price      float64
price_per_m2     float64
source           object
listing_id       object
published_at     datetime64[ns]
ingested_at      datetime64[us, UTC]
neighborhood     object
year_built       float64
floor            float64
total_floors     float64
```

No accidental `object` dtype for nullable numeric columns. The parquet file
is now safe to consume directly in Phase 2.

---

## 8. Validation Layer

Implemented in `src/tehran_house_price/data/validate.py`.

Behavior:

- Loads the cleaned parquet
- Normalizes dtypes defensively before validating
- Runs `HouseListingSchema.validate(df, lazy=True)`
- Writes a JSON report to `artifacts/data_validation/kaggle_validation.json`
- Returns a boolean `passed` flag

The report contains row count, column list, null counts per column, and a
summary of any schema violations. It is small enough to be inspected by hand
and machine-readable for CI integration.

### 8.1. Known issue that was fixed during Phase 1

Initially, pandera produced ~9700 errors because nullable numeric columns
(`year_built`, `floor`, `total_floors`) were defined as `Series[int]` while
the underlying pandas data could not be coerced to numpy `int64` (which
cannot hold `NaN`).

Resolution:

- Changed schema types to `Series[float]` for these columns.
- Added `_normalize_dtypes()` in `validate.py` as a defensive safety net.
- Added `normalize_output_dtypes()` in `clean.py` so the parquet file is
  written with the correct dtypes in the first place.

After the fix:

- Validation: PASSED for all 3235 cleaned rows.
- Test suite: 39 / 39 passing.

---

## 9. Build Orchestration

Implemented in `src/tehran_house_price/data/build_dataset.py`.

The orchestrator runs the full pipeline:

1. `ingest_kaggle.download_dataset()` — verify or download raw input.
2. `clean.run()` — produce `data/interim/kaggle_clean.parquet`.
3. `validate.run()` — produce JSON validation report.
4. `promote_dataset()` — copy validated interim file to `data/processed/`.

Key behaviors:

- Returns a `BuildResult` dataclass with all important output paths.
- Raises `RuntimeError` if validation fails — the pipeline never silently
  promotes invalid data.
- Copies interim to processed instead of moving, so debugging and reruns
  remain possible.
- Supports two CLI flags:
  - `--force-ingest`: re-download raw data even if already present.
  - `--skip-ingest`: reuse existing raw files (useful when offline).
- Supports `--output-name` to override the processed filename.

---

## 10. How to Run

All commands assume the virtual environment `thpenv` is active and the
package is installed in editable mode (`pip install -e .`).

### 10.1. Run the full pipeline

```powershell
python -m tehran_house_price.data.build_dataset
10.2. Run individual steps
PowerShell

python -m tehran_house_price.data.ingest_kaggle
python -m tehran_house_price.data.clean
python -m tehran_house_price.data.validate
10.3. Useful flags
PowerShell

python -m tehran_house_price.data.build_dataset --skip-ingest
python -m tehran_house_price.data.build_dataset --force-ingest
python -m tehran_house_price.data.ingest_kaggle --force
python -m tehran_house_price.data.validate --strict
10.4. Run all tests
PowerShell

pytest tests/ -v
11. Tests
Test layout:

text

tests/
├── integration/
│   └── test_build_dataset.py   # full pipeline orchestration
└── unit/
    ├── test_clean.py            # 12 tests: each cleaning step + end-to-end
    ├── test_ingest_kaggle.py    # ingestion behavior
    ├── test_logger.py           # logger setup
    ├── test_schema.py           # pandera + pydantic schema
    ├── test_smoke.py            # version, config, paths
    └── test_validate.py         # validation behavior + report writing
Current status: 39 tests, all passing.

Integration tests use monkeypatch to isolate the orchestrator from disk
I/O and external downloads, so they run fast and deterministically.

12. Known Limitations
Kaggle data has no published_at, neighborhood, year_built, floor,
or total_floors. These columns exist in the schema but are entirely
null until Divar data is added.
The dataset is small (3235 rows after cleaning). This is a known
limitation that motivates the future Divar ingestion.
price_per_m2 is derived inside the cleaning step. If we ever change
the formula, the listing_id hash should remain stable because it does
not include derived columns.
The Kaggle download flow requires manual consent on the dataset page.
Automatic API download will keep returning 403 until that consent is
given.
13. Definition of Done
Phase 1 is considered complete when all of the following are true:

 Raw data ingested (with manual fallback).
 Cleaning pipeline works end-to-end.
 Canonical schema defined (pandera + pydantic).
 Validation passes on cleaned data.
 build_dataset.py orchestrates the full pipeline.
 At least 35 tests passing (current: 39).
 Output dtypes normalized for downstream consumers.
 Phase 1 documentation written (this file).

Phase 1 status: COMPLETE.

14. What Comes Next
Phase 2 (ML Pipeline) will consume data/processed/tehran_houses.parquet
and add:

feature engineering layer
train / validation split
baseline model
evaluation metrics
artifact and metric logging
