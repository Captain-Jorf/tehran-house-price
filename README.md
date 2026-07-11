# Tehran House Price Prediction — Production MLOps Pipeline

[![CI Pipeline](https://github.com/Captain-Jorf/tehran-house-price/actions/workflows/ci.yml/badge.svg)](https://github.com/Captain-Jorf/tehran-house-price/actions/workflows/ci.yml)
[![Docker Build](https://github.com/Captain-Jorf/tehran-house-price/actions/workflows/docker.yml/badge.svg)](https://github.com/Captain-Jorf/tehran-house-price/actions/workflows/docker.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://github.com/Captain-Jorf/tehran-house-price/pkgs/container/tehran-house-price)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Live API:** https://tehran-house-price-api.onrender.com/docs

An end-to-end MLOps project that predicts residential property prices in Tehran,
Iran. The project covers the full production lifecycle: data ingestion,
feature engineering, model training with experiment tracking, a containerized
REST API, CI/CD automation, observability, and cloud deployment.

The focus of this project is **production engineering practices**, not model
accuracy. See [Known Limitations](#known-limitations) for an honest discussion
of the model's real-world performance.

---

## Table of Contents

- [Live Demo](#live-demo)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Observability](#observability)
- [Deployment](#deployment)
- [Model Details](#model-details)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Live Demo

The API is deployed on Render's free tier. Try it directly:

**Swagger UI (interactive docs):**
https://tehran-house-price-api.onrender.com/docs

**Single prediction with curl:**

```bash
curl -X POST https://tehran-house-price-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Punak",
    "area_m2": 85,
    "rooms": 2,
    "has_parking": true,
    "has_storage": true,
    "has_elevator": true
  }'
Expected response:

JSON

{
  "predicted_price_per_m2": 42368040.9,
  "predicted_total_price": 3601283476.5,
  "currency": "toman",
  "model_name": "xgb_price_per_m2"
}
Note: The free tier spins down after 15 minutes of inactivity.
The first request after idle may take 30-60 seconds (cold start).

What This Project Demonstrates
This project is designed to showcase production-grade ML engineering across
the full stack:

Capability	Implementation
Reproducible data pipeline	Deterministic ingestion, cleaning, validation with Pandera
Feature engineering	Custom sklearn-compatible transformers
Experiment tracking	MLflow with parent/child runs and model registry
Model serving	FastAPI with dependency injection and lifespan management
Containerization	Multi-stage Docker builds with non-root user
CI/CD	GitHub Actions for linting, testing, and image publishing
Observability	Prometheus metrics, structured JSON logs, health probes
Cloud deployment	Infrastructure-as-Code with render.yaml
Configuration	Twelve-factor style with environment variables
Testing	196 tests, 82% coverage, unit + integration + e2e
Architecture
text

                        ┌─────────────────────┐
                        │   Kaggle Dataset    │
                        └──────────┬──────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │  Data Layer                                       │
        │  ingest → clean → validate → build_dataset       │
        └──────────────────────┬───────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │  ML Pipeline                                      │
        │  features → split → baselines + XGBoost          │
        │                    │                              │
        │                    └─→ MLflow (tracking+registry)│
        └──────────────────────┬───────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │  Serving Layer                                    │
        │  FastAPI + ModelService (in-memory)              │
        │       │                                           │
        │       ├─→ Prometheus /metrics                    │
        │       ├─→ Structured logs (JSON)                 │
        │       ├─→ /health/live, /health/ready            │
        │       └─→ Background prediction logging          │
        └──────────────────────┬───────────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────────┐
        │  Deployment                                       │
        │  Docker → GHCR → Render (auto-deploy on push)    │
        └──────────────────────────────────────────────────┘
Request flow:

text

Client
  │
  ▼
Middleware (request_id + JSON logs + metrics)
  │
  ▼
Endpoint handler
  │
  ├─→ ModelService.predict()
  ├─→ record_prediction() (Prometheus counter)
  └─→ BackgroundTask: log_prediction() (non-blocking)
  │
  ▼
Response (with X-Request-ID header)
Tech Stack
Language & Frameworks

Python 3.10
FastAPI 0.111 + Uvicorn (ASGI server)
Pydantic v2 (validation + settings)
Machine Learning

scikit-learn 1.5 (pipelines and transformers)
XGBoost 2.0 (gradient boosted trees)
Pandas 2.2, NumPy, PyArrow
Data & Validation

Pandera (dataframe schemas)
kagglehub (dataset ingestion)
MLOps

MLflow 2.14 (experiment tracking + model registry)
Joblib (model serialization)
Infrastructure

Docker (multi-stage builds)
Docker Compose (local orchestration)
GitHub Actions (CI/CD)
GitHub Container Registry (image hosting)
Render (production deployment)
Observability

prometheus-client (metrics)
Loguru (structured logging)
PostgreSQL (optional prediction logging)
Prometheus + Grafana (local monitoring stack)
Quality

pytest + pytest-cov
ruff (linting)
black (formatting)
mypy (type checking)
pre-commit (git hooks)
Quick Start
Option 1: Pull the pre-built Docker image
Bash

docker pull ghcr.io/captain-jorf/tehran-house-price:latest

docker run -p 8000:8000 \
  -e ARTIFACT_DOWNLOAD_URL=https://github.com/Captain-Jorf/tehran-house-price/releases/download/v0.9.0-model/xgb_price_per_m2.joblib \
  -e ARTIFACT_METADATA_DOWNLOAD_URL=https://github.com/Captain-Jorf/tehran-house-price/releases/download/v0.9.0-model/xgb_price_per_m2_metadata.json \
  ghcr.io/captain-jorf/tehran-house-price:latest

# Then open http://localhost:8000/docs
Option 2: Run from source
Bash

# Clone
git clone https://github.com/Captain-Jorf/tehran-house-price.git
cd tehran-house-price

# Create virtualenv (Python 3.10 required)
python -m venv thpenv
source thpenv/bin/activate           # macOS / Linux
# .\thpenv\Scripts\Activate.ps1      # Windows PowerShell

# Install
pip install -r requirements-dev.txt
pip install -e .

# Run tests
pytest tests/ -q

# Start API (assumes model artifacts are already present in artifacts/models/)
python -m tehran_house_price.api
Option 3: Full pipeline from scratch
Bash

# 1. Configure Kaggle credentials in .env
cp .env.example .env
# edit KAGGLE_API_TOKEN

# 2. Run the data pipeline
python -m tehran_house_price.data.build_dataset

# 3. Train all models (baselines + XGBoost)
python -m tehran_house_price.models.train_pipeline

# 4. Start the API
python -m tehran_house_price.api
API Reference
Method	Endpoint	Description
GET	/	Root, API metadata
GET	/health	Basic health probe with model-loaded status
GET	/health/live	Liveness probe (always 200 if process is alive)
GET	/health/ready	Readiness probe (200 or 503 based on model + disk)
GET	/version	Application and model metadata
GET	/metrics	Prometheus metrics in text exposition format
GET	/docs	Swagger UI
GET	/redoc	ReDoc UI
POST	/predict	Predict price for a single listing
POST	/predict/batch	Predict prices for multiple listings
Single prediction request schema
JSON

{
  "district": "Punak",
  "area_m2": 85,
  "rooms": 2,
  "has_parking": true,
  "has_storage": true,
  "has_elevator": true
}
Response schema
JSON

{
  "predicted_price_per_m2": 42368040.9,
  "predicted_total_price": 3601283476.5,
  "currency": "toman",
  "model_name": "xgb_price_per_m2"
}
Project Structure
text

tehran-house-price/
├── .github/workflows/          # CI and Docker workflows
├── configs/                    # YAML configs (base, logging)
├── data/                       # gitignored: raw, interim, processed
├── artifacts/                  # gitignored: models, splits, evaluations
├── docs/                       # phase-by-phase documentation
├── grafana/                    # Grafana dashboards + provisioning
├── prometheus/                 # Prometheus scrape config
├── src/tehran_house_price/
│   ├── api/                    # FastAPI app, routes, middleware, bootstrap
│   ├── data/                   # ingest, clean, validate, build_dataset
│   ├── features/               # sklearn-compatible transformers
│   ├── models/                 # split, baselines, train, evaluation, pipeline
│   ├── monitoring/             # prediction logger (async, PostgreSQL)
│   ├── tracking/               # MLflow setup, run logger, model registry
│   └── utils/                  # paths, logger
├── tests/
│   ├── unit/                   # unit tests (mocked dependencies)
│   └── integration/            # E2E and pipeline tests
├── Dockerfile                  # dev build (Iran mirrors)
├── Dockerfile.prod             # production build (standard mirrors)
├── docker-compose.yml          # local dev
├── docker-compose.observability.yml  # api + postgres + prometheus + grafana
├── render.yaml                 # Render deployment config (IaC)
├── pyproject.toml              # package + tool configs
├── requirements.txt            # runtime deps
└── requirements-dev.txt        # dev deps
Development
Setup pre-commit hooks
Bash

pre-commit install
Runs on every commit:

ruff (lint + auto-fix)
ruff-format
black
trailing whitespace, end-of-file, large files, merge conflict checks
Common commands
Bash

# Data pipeline
python -m tehran_house_price.data.build_dataset

# Training (all models)
python -m tehran_house_price.models.train_pipeline

# Training (skip baselines)
python -m tehran_house_price.models.train_pipeline --skip-baselines

# API (dev mode)
python -m tehran_house_price.api

# Linting and formatting
ruff check --fix src tests
black src tests
mypy src

# Local observability stack (api + postgres + prometheus + grafana)
docker-compose -f docker-compose.observability.yml up --build
Testing
The project has 196 tests covering unit, integration, and end-to-end scenarios.

Bash

# All tests
pytest tests/ -q

# Unit tests only
pytest tests/unit/ -q

# With coverage
pytest tests/ --cov=src/tehran_house_price --cov-report=term-missing
Coverage: 82%
Test runtime: ~30 seconds

Tests that require model artifacts (which are gitignored) are automatically
skipped in CI. Locally they run once artifacts exist under artifacts/models/.

Observability
Every observability feature is opt-in via environment variables and
wrapped in defensive try/except so that a monitoring failure never crashes
the API.

Feature	Env Var	Default
Structured JSON request logs	REQUEST_LOGGING_ENABLED	true
Prometheus /metrics endpoint	PROMETHEUS_ENABLED	true
Deep /health/live + /health/ready	DEEP_HEALTHCHECK_ENABLED	true
Async prediction logging to PostgreSQL	PREDICTION_LOGGING_ENABLED	false
Master switch	OBSERVABILITY_ENABLED	true
Local monitoring stack
Bash

docker-compose -f docker-compose.observability.yml up --build
Then:

API: http://localhost:8000
Prometheus: http://localhost:9090
Grafana: http://localhost:3000 (anonymous admin, no login)
PostgreSQL: localhost:5432
Sample metrics
text

http_requests_total{method="POST",path="/predict",status_code="200"} 42
http_request_duration_seconds_bucket{le="0.1",path="/predict"} 40
model_predictions_total{endpoint="/predict",model_name="xgb_price_per_m2"} 42
process_resident_memory_bytes 2.35e+08
Deployment
The API is deployed to Render using Infrastructure-as-Code via
render.yaml. Any push to main triggers auto-deploy.

Deployment strategy:

Docker image is built from Dockerfile.prod (multi-stage, non-root user, ~450 MB)
Container starts and runs ensure_model_artifacts() on startup
If model files are missing, they are downloaded atomically from
GitHub Releases (v0.9.0-model tag)
ModelService.load() loads the model into memory
Uvicorn starts serving on port 8000
Render's health check hits /health every 30 seconds
Why download the model at startup instead of baking it into the image?

Keeps the image small and generic
Model can be swapped without rebuilding the image
Code lifecycle is separated from model lifecycle
Model files can be hosted anywhere (GitHub Releases, S3, HF Hub, MLflow Registry)
Model Details
Best model: xgb_price_per_m2 (XGBoost regressor)

Metric	Value
Target	price_per_m2 (then multiplied by area)
Features	area, rooms, parking, storage, elevator, district
Training data	3,235 cleaned Tehran listings from Kaggle (2020-2021)
Test R²	0.74
Test MAE	~8.5 M Toman/m²
Test RMSE	~15 M Toman/m²
Baselines for comparison:

baseline_mean: global mean of prices
baseline_district_median: median price by district
Baselines are trained and evaluated in the same pipeline. XGBoost beats
both on all metrics, which is the sanity check we expect.

Known Limitations
Being upfront about the model's real-world performance is more valuable
than pretending it is perfect.

1. Data Drift: The Training Data is from 2020
The Kaggle dataset used for training reflects the Tehran housing market
around 2020-2021. Since then, prices have risen approximately 6-7x due
to inflation and currency devaluation.

Concrete example:
For a 100 m² apartment in Punak, the model predicts approximately
42 M Toman/m², while the current market price (as of 2026) is
approximately 250-300 M Toman/m².

This is a textbook example of concept drift: the relationship between
features and target has fundamentally changed over time, even though the
features themselves remain the same.

2. Why This Is Not Fixed with a Multiplier
Applying a naive inflation factor would hide the real problem. It would
also:

Miss district-specific inflation rates (luxury areas inflated faster)
Fail to capture non-linear market dynamics
Give a false sense of accuracy
3. The Proper Fix: A Retraining Pipeline
The infrastructure to solve this problem is already in place in this project:

Component	Status	Purpose
MLflow Registry	✅ Implemented	Version and stage model artifacts
Startup model download	✅ Implemented	Swap models without rebuilding Docker
Prediction logging (PostgreSQL)	✅ Implemented	Capture production inputs for retrain
CI/CD pipeline	✅ Implemented	Automated testing on every change
Divar scraper skeleton	✅ Implemented	Foundation for fresh data collection
Scheduled retraining (Airflow / Prefect)	⏳ Future work	Weekly/monthly retrain on fresh data
Drift detection (Evidently AI)	⏳ Future work	Alert when input distribution shifts
4. Missing Features
The model uses only 6 features. Real-world price also depends on:

Year built and renovation status
Floor number
Building orientation
Distance to metro/BRT
Interior condition
Neighborhood safety scores
Adding these features would likely push R² above 0.85 on fresh data.

5. What This Project Actually Proves
The value of this project is not model accuracy — it is the
production infrastructure around the model. Any of the above
limitations can be addressed by re-running the existing pipeline
with better data. The MLOps stack around it is what takes months
to build correctly.
