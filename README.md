# Tehran House Price Prediction

An end-to-end MLOps project that predicts house prices in Tehran using a
combination of Kaggle data and listings scraped from Divar.

Status: work in progress. Currently in Phase 1 (data layer).

## Why this project

I wanted to build something closer to a real production ML system instead
of another notebook with a model and a confusion matrix. The plan is to
cover the full lifecycle: ingestion, training, serving, monitoring, and
deployment.

## Tech stack

- Python 3.10
- pandas, scikit-learn, XGBoost / LightGBM
- FastAPI for serving
- MLflow for experiment tracking
- Docker for containerization
- GitHub Actions for CI/CD
- PostgreSQL for prediction logs
- Hugging Face Spaces or Render for deployment

## Project structure

See `docs/` for details. Short version:

- `src/tehran_house_price/` - main package
- `configs/` - YAML configs
- `data/` - raw, interim and processed datasets (gitignored)
- `artifacts/` - pipeline outputs (gitignored)
- `tests/` - unit and integration tests
- `notebooks/` - exploration notebooks

## Setup

Requires Python 3.10.

```bash
python -m venv thpenv
# Linux / Mac
source thpenv/bin/activate
# Windows
.\thpenv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
pip install -e .
