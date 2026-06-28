# Phase 2 - ML Pipeline

This document describes the ML pipeline of the Tehran House Price project:
how features are built, how the data is split, what models are trained,
how they are evaluated, and how a single command reproduces the entire
training flow.

---

## 1. Goal

Turn `data/processed/tehran_houses.parquet` into a fully self-contained,
serializable model artifact that downstream phases (API serving, deployment)
can use without reimplementing any preprocessing logic.

Phase 2 owns the entire path from cleaned data to a trained, evaluated,
audit-ready model. Phase 3 (API) should treat the model as a black box.

---

## 2. Architecture

The ML layer is split into six independent, runnable modules:
features/
constants.py -> feature names, groups, encoding config
transformers.py -> custom sklearn transformers
build_features.py -> build_feature_pipeline(), target transforms

models/
split.py -> stratified train/val split with metadata
baseline.py -> baseline regressors (mean, district median)
evaluation.py -> regression metrics + per-district breakdown
train.py -> main XGBoost model with log-target wrapper
train_pipeline.py -> orchestrator (baselines + main + comparison)

text


Each module exposes:

- a programmatic API (`run()` / `train()` / etc.)
- a CLI entry point (argparse)
- unit tests where logic lives; integration tests where orchestration lives

This layout makes every step debuggable in isolation and reproducible
end-to-end with a single command.

---

## 3. Target Variable Strategy

### 3.1. Predict `price_per_m2`, not `total_price`

`total_price` ranges from a few hundred million to several billion Toman
depending on size. Its distribution is heavily right-skewed.
`price_per_m2` is much more stable across the dataset.

At inference time, total price is recovered as:
predicted_total_price = predicted_price_per_m2 * area_m2

text


This is a standard real-estate ML pattern. It reduces the model's burden
to learning a quantity that already factors out the area effect.

### 3.2. Log-transform the target

Even `price_per_m2` is right-skewed. The model trains on `log1p(y)` and
predictions are mapped back via `expm1`. This:

- makes the target distribution closer to normal
- stabilizes the gradient boosting objective
- prevents huge listings from dominating the loss

Implementation:

- `transform_target_for_training(y)` -> `log1p(y)`
- `inverse_target_for_prediction(y_log)` -> `expm1(y_log)`
- both wrapped inside `LogTargetRegressor` so consumers of the final
  pipeline never see log space

---

## 4. Feature Engineering

### 4.1. Inputs

From the canonical schema, Phase 2 consumes:

| Column        | Type     | Used as                         |
|---------------|----------|---------------------------------|
| area_m2       | float    | numeric feature                 |
| rooms         | int      | numeric feature                 |
| has_parking   | bool     | boolean feature (cast to int)   |
| has_storage   | bool     | boolean feature (cast to int)   |
| has_elevator  | bool     | boolean feature (cast to int)   |
| district      | str      | categorical feature             |
| price_per_m2  | float    | target                          |

Other columns (`year_built`, `floor`, `total_floors`, `neighborhood`,
`published_at`) exist in the schema but are entirely null for the Kaggle
source, so Phase 2 does not consume them. They are reserved for the Divar
source planned in a later phase.

### 4.2. Derived features

| Feature                    | Source            | Reason                                     |
|----------------------------|-------------------|--------------------------------------------|
| area_per_room              | area_m2 / rooms   | captures room density                      |
| district_target_enc        | mean target per district | injects price signal into the model |
| district_freq_enc          | district frequency | captures market depth per district        |

### 4.3. District encoding decision

`district` has ~192 unique values. Three options were considered:

| Method                   | Pros                                  | Cons                            |
|--------------------------|---------------------------------------|---------------------------------|
| One-hot encoding         | simple, interpretable                 | 192 sparse columns, bad on small data |
| Frequency encoding       | one column, fast                      | no price information            |
| Target mean encoding     | one column, encodes price information | risk of leakage if done wrong   |

The project uses **smoothed target mean encoding** plus
**frequency encoding** as a secondary signal. Smoothing pulls rare
districts toward the global mean to prevent overfitting on tiny groups.

### 4.4. Final feature set
area_m2, rooms,
has_parking, has_storage, has_elevator,
district_target_enc, district_freq_enc,
area_per_room

text


Eight features, all numeric. Small but sufficient for the available data.

### 4.5. Pipeline serializability

The full feature pipeline is an `sklearn.pipeline.Pipeline` and is
combined with the model into a single Pipeline object. The entire thing
is persisted with `joblib`. Phase 3 only needs `pipeline.predict(raw_df)`.

---

## 5. Train / Validation Split

Implemented in `models/split.py`.

### 5.1. Strategy: stratified by district

`train_test_split(..., stratify=district_key, random_state=42)`.

Random splits risk putting an entire district only in train or only in val.
Stratification keeps the per-district distribution similar in both sets,
which makes validation metrics meaningful.

### 5.2. Rare districts

Districts with fewer than `min_samples_per_stratum=5` rows are merged into
a single `__rare__` bucket for stratification only. If that bucket still
ends up too small (`< 2`) it is absorbed into the largest existing
stratum. Strategies in order of preference:

1. keep all districts as their own strata
2. merge rare ones into `__rare__`
3. absorb `__rare__` into the largest stratum
4. fall back to a single `__all__` bucket (pathological case)

This guarantees the split never crashes on long-tail categories, which is
critical when Divar data later adds more rare district names.

### 5.3. Reproducibility

Each split run writes
`artifacts/splits/split_metadata.json` containing:

- seed and val_size
- n_train, n_val
- SHA1 hash of sorted `listing_id`s for both train and val

These hashes make the split auditable: any later run with the same seed
must produce the same hashes. Model metadata stores these hashes too,
which is what lets us claim "this model was trained on exactly this data."

### 5.4. Why no test set yet

The processed dataset has 3235 rows. Carving out a held-out test set
right now would leave too few training rows. Once Divar data is added in
a future phase, an explicit test split will be introduced.

---

## 6. Baselines

Implemented in `models/baseline.py`. Two baselines:

### 6.1. MeanPriceBaseline

Predicts `mean(y_train)` for every row. The simplest possible baseline.
Acts as the absolute floor: any real model must beat this.

### 6.2. DistrictMedianBaseline

Predicts the per-district median of `price_per_m2`. Unseen districts at
inference time fall back to the global median.

Median (not mean) is used because price distributions per district are
skewed and a few luxury listings would otherwise pull the per-district
estimate up.

### 6.3. Results on the val split

| Baseline                 | MAE        | RMSE        | MAPE   | R²      |
|--------------------------|-----------:|------------:|-------:|--------:|
| Mean                     | 22,326,482 | 30,157,125  | 1.3151 | -0.0020 |
| District median          |  9,570,207 | 18,196,916  | 0.3337 |  0.6352 |

The district median baseline alone explains 64% of the variance and gives
us a strong benchmark to beat.

### 6.4. Why have two baselines

A model that beats the global mean but not the district median is not
really using its features; it is just learning the district price level.
Both numbers together define a meaningful performance floor.

---

## 7. Evaluation Layer

Implemented in `models/evaluation.py`.

### 7.1. Metrics

| Metric | Why we include it                                 |
|--------|----------------------------------------------------|
| MAE    | average absolute Toman error, easy to communicate  |
| RMSE   | penalizes large errors more, useful for tails      |
| MedAE  | robust to a few outliers                           |
| MAPE   | percentage error, fair across price ranges         |
| R²     | proportion of variance explained                   |

`MAPE` is the headline metric because in real estate the same absolute
error means very different things on a 1B vs 100M listing.

### 7.2. Per-district breakdown

For each district with at least `min_samples_for_breakdown=3` rows in val:

- MAE per district
- MAPE per district
- count per district
- top-K worst districts by MAPE

This is the most actionable part of the evaluation. It tells us where the
model is weakest, which usually maps directly to either data scarcity or
feature scarcity in that area.

### 7.3. Comparison report

`compare_models(reports)` builds a DataFrame ranked by MAPE ascending.
`train_pipeline.py` saves a JSON version of this comparison at
`artifacts/model_evaluation/train_pipeline_comparison.json`.

---

## 8. Main Model: XGBoost

Implemented in `models/train.py`.

### 8.1. Why XGBoost

- excellent on small/medium tabular data
- handles non-linear interactions out of the box
- stable on Windows
- well-known choice for interviews and review
- good interplay with sklearn Pipeline

LightGBM was considered but XGBoost is more friction-free on Windows and
the dataset is too small for LightGBM's speed advantage to matter.

### 8.2. Hyperparameters

```python
{
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}
These are sensible defaults for a small dataset. No automated tuning was
done in Phase 2: tuning before having more data would mostly chase noise.

8.3. No early stopping
Early stopping based on the validation set would leak information from
val into the model. With a small dataset the alternative (further
splitting train) is also wasteful. The trade-off chosen for Phase 2 is
a fixed n_estimators=400 with mild L1+L2 regularization.

8.4. Final artifact
The persisted artifact is a single joblib file containing the full
Pipeline: feature engineering + LogTargetRegressor + XGBRegressor.
Consumers call .predict(raw_df) and get prices in original scale.

A metadata JSON is saved next to it with:

training timestamp
package version
hyperparameters
split seed, val_size, n_train, n_val
train/val ids hashes (from split metadata)
final validation metrics
worst districts
snapshot of baseline metrics for direct comparison
8.5. Results
Model	MAE	RMSE	MedAE	MAPE	R²
baseline_mean	22,326,482	30,157,125	18,847,576	1.3151	-0.002
baseline_district_median	9,570,207	18,196,916	4,903,730	0.3337	0.635
xgb_price_per_m2	8,486,189	15,468,606	4,208,028	0.3274	0.736
Improvements over the strongest baseline (district median):

MAE: -11.3%
RMSE: -15.0%
MedAE: -14.2%
R²: +15.9 percentage points (0.635 -> 0.736)
MAPE: -1.9% (modest)
8.6. Honest interpretation
XGBoost beats the baseline on every metric. The gain on R² and absolute
errors is meaningful. The gain on MAPE is small.

This is a real, defensible finding: the bottleneck right now is not
model complexity, it is feature richness. The model already extracts
most of what is possible from district + area + a few flags. The next
big jump will come from data, not hyperparameters: adding
year_built, floor, neighborhood, and published_at via Divar.

8.7. Worst districts (XGBoost)
District	n	MAE	MAPE
Pasdaran	12	20,442,886	2.88
Persian Gulf Martyrs Lake	15	8,635,021	1.15
Northern Chitgar	3	6,980,300	0.62
Gheitarieh	26	14,996,143	0.60
Elahieh	3	14,256,498	0.60
These are mostly luxury northern Tehran districts where in-district
price variance is enormous and the available features cannot capture it.
This is exactly where richer Divar features would help most.

9. Train Pipeline Orchestration
Implemented in models/train_pipeline.py.

The orchestrator runs the full Phase 2 flow:

(optional) train baselines via baseline.run()
(optional) train main XGBoost via train.train()
evaluate every persisted model on the canonical val split
write a unified comparison JSON
CLI flags:

--skip-baselines: reuse already-trained baselines
--skip-main: only train baselines, useful for debugging
--val-size, --seed, --model-name: standard knobs
Key behaviors:

raises if both skip flags are set (nothing to do)
comparison is always built from all persisted models, not only the
ones trained in the current run, so historical artifacts stay relevant
10. How to Run
All commands assume thpenv is active and the package is installed
editable (pip install -e .).

10.1. Full pipeline
PowerShell

python -m tehran_house_price.models.train_pipeline
10.2. Individual steps
PowerShell

python -m tehran_house_price.models.split
python -m tehran_house_price.models.baseline
python -m tehran_house_price.models.train
python -m tehran_house_price.models.evaluation
10.3. Useful flags
PowerShell

python -m tehran_house_price.models.train_pipeline --skip-baselines
python -m tehran_house_price.models.train_pipeline --skip-main
python -m tehran_house_price.models.train --seed 7 --model-name xgb_seed_7
10.4. Tests
PowerShell

pytest tests/ -v
11. Tests
Test layout added in Phase 2:

text

tests/
├── integration/
│   └── test_train_pipeline.py      # end-to-end orchestrator
└── unit/
    ├── test_transformers.py         # feature transformers
    ├── test_build_features.py       # feature pipeline assembly
    ├── test_split.py                # stratified split + metadata
    ├── test_baseline.py             # baseline regressors
    ├── test_evaluation.py           # metric formulas + report shape
    └── test_train.py                # log-target wrapper + train() flow
End-to-end status: 105 tests, all passing.

Integration tests use monkeypatch to sandbox every disk write into
tmp_path and to make split.run() consume an in-memory toy DataFrame.
This keeps the orchestrator test fast and deterministic.

12. Important Engineering Decisions
Decision	Why
Predict price_per_m2, not total_price	smoother distribution, easier learning target
Log-transform the target	corrects skew, stabilizes XGBoost objective
Wrap log+inverse inside LogTargetRegressor	consumers never see log space
Smoothed target encoding for district	beats one-hot on 192 categories with 3k rows
Stratified split with rare-bucket absorption	never crashes on long-tail districts
Persist entire Pipeline as one joblib file	deployment loads one object, calls .predict(raw_df)
Save model metadata next to model	reproducibility + audit + baseline comparison snapshot
Separate evaluation module	single canonical metric implementation across baselines/model/monitor
__main__ re-import trick in CLIs	avoids __main__.X pickling pitfall when running python -m ...
No automated hyperparameter tuning yet	dataset too small; tuning would chase noise, not signal
13. Known Limitations
The Kaggle dataset has 3235 rows after cleaning. This is small for any
serious gradient boosting. Larger gains will come from data, not model.
year_built, floor, total_floors, neighborhood, published_at
are all null in Kaggle data. The model cannot use them yet.
A previous evaluation surfaced 5 rows where district == "nan" (the
literal string). Root cause is clean_district casting NaN values to
the string "nan" via astype(str) before the dropna step. This is
a known issue to fix in a follow-up cleaning pass.
XGBoost worst-performing districts (Pasdaran, Persian Gulf Martyrs Lake,
Elahieh, etc.) are luxury northern Tehran areas with very high
in-district price variance. Without richer features, no model can do
much better here.
No held-out test set yet; only train and val. To be revisited once the
dataset grows via Divar.
No automated hyperparameter search. Adding Optuna or similar is
deliberately deferred to a later phase.
14. What Phase 2 Enables
After Phase 2, the project has:

a serializable end-to-end model artifact
a metadata file describing exactly how that artifact was trained
a comparison report against meaningful baselines
a single-command training orchestrator
a stable evaluation layer ready for Phase 5 (MLflow) and Phase 7
(monitoring) to consume without rewriting metric logic
This means Phase 3 (FastAPI) only needs to:

load artifacts/models/xgb_price_per_m2.joblib
validate incoming requests with the Pydantic schema from Phase 1
call model.predict(df) and return the result
No feature engineering, no preprocessing, no metric computation belongs
in Phase 3. That is the entire point of how Phase 2 was designed.

15. Definition of Done
Phase 2 is considered complete when:

 Feature engineering pipeline serializable end-to-end
 Deterministic stratified train/val split with metadata
 Two meaningful baselines persisted with metrics
 Evaluation layer with global and per-district metrics
 Main XGBoost model with log-target wrapper
 Model artifact + metadata persisted together
 Main model strictly beats the strongest baseline
 Train pipeline orchestrator with CLI and integration tests
 At least 50 tests passing (current: 105)
 Phase 2 documentation written (this file)
Phase 2 status: COMPLETE.

16. What Comes Next
Phase 3 (API and Serving) will:

expose a FastAPI service that loads the joblib model on startup
validate incoming requests with the HouseListing Pydantic schema
return predictions in original Toman scale
include health, version, and metrics endpoints
