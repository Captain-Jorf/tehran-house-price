# Phase 5: Experiment Tracking with MLflow

## What this phase adds

Before phase 5, training produced files on disk:

- `artifacts/models/*.joblib`
- `artifacts/model_evaluation/*.json`

To compare two training runs you had to manually open JSON files
and diff numbers by eye. That doesn't scale past a handful of runs.

Phase 5 adds MLflow as an observability layer on top of the existing
training pipeline:

- Every training run is now logged to MLflow with params, metrics,
  artifacts, and the fitted model
- A web UI shows all runs in a table, with sortable metrics
- Models can be registered and promoted through stages (Staging -> Production)
- All of this is **purely additive** - the original `artifacts/` flow
  still works exactly the same

## Architecture
train_pipeline.py
|
+-- baseline.py / train.py / evaluation.py (unchanged)
|
+-- tracking/ (new in phase 5)
mlflow_setup.py -> setup + run context manager
run_logger.py -> safe log_params / log_metrics / log_artifact
registry.py -> register + promote model versions

text


The trainers themselves know nothing about MLflow. Only the orchestrator
(`train_pipeline.py`) calls into `tracking/`. This means:

- Trainers stay framework-agnostic and easy to test
- MLflow can be disabled without changing any code
- If we switch to a different tracking tool later, only one layer changes

## Run structure in MLflow

Each invocation of `train_pipeline.py` creates one parent run with
three nested child runs:
train_pipeline (parent)
|-- baseline_mean (child)
|-- baseline_district_median (child)
+-- xgb_price_per_m2 (child)

text


The parent run holds pipeline-level metrics (number of models compared,
whether main was trained) and the comparison artifact. Each child run
holds its own params, metrics, evaluation JSON, and serialized model.

## Tracking toggle

The whole tracking layer is gated by an env var:

```powershell
$env:MLFLOW_TRACKING_ENABLED="false"
python -m tehran_house_price.models.train_pipeline
Remove-Item Env:\MLFLOW_TRACKING_ENABLED
When disabled, every log_* call is a no-op and get_run_context
yields None. The pipeline still produces all the same files on disk.

This matters for:

CI environments where you don't want hundreds of leftover runs
Quick iteration on training code without polluting the registry
Production environments where you'd rather send metrics elsewhere
Tracking URI
By default, runs go to a local file store at <project>/mlruns/.
This directory is gitignored.

To point at a remote MLflow server later, set:

PowerShell

$env:MLFLOW_TRACKING_URI="http://your-server:5000"
No code changes needed.

What gets logged
Per child run:

Params:
model_name, algorithm, model_role
target_col, target_transform
split.seed, split.val_size, split.n_train, split.n_val
split.train_ids_hash, split.val_ids_hash (main model only)
hp.* for every XGBoost hyperparameter (main model only)
Metrics:
mae, rmse, mape (all models)
medae, r2 (main model)
worst_district_N_mae, worst_district_N_mape for N=1..5 (main model)
Artifacts:
model/<name>.joblib and model/<name>_metadata.json
evaluation/<name>_evaluation.json
sklearn_model/ (main model only, via mlflow.sklearn.log_model)
Tags:
git_commit, package_version, env, phase (default tags)
model_role = "baseline" or "main"
Per parent run:

Metrics:
pipeline.n_models_compared
pipeline.n_baselines
pipeline.main_trained
Artifacts:
comparison/train_pipeline_comparison.json
Tags:
pipeline = "train_pipeline"
skip_baselines, skip_main, pipeline_seed, pipeline_val_size
Model registry
After the main XGBoost model is logged, the pipeline automatically:

Registers it under the name tehran_house_price_xgb
Promotes the new version to the Staging stage
Archives the previous Staging version
This means every training run creates a new model version, and the
latest one is always findable via:

Python

from tehran_house_price.tracking import get_latest_version
version = get_latest_version(stage="Staging")
Why stages instead of aliases
MLflow 2.9 deprecated stages in favor of aliases. We're using stages
anyway because:

They still work and will be supported for years
Most tutorials, blog posts, and interview answers still reference stages
Aliases are more powerful but overkill for a single-model project
If we move to a multi-model setup or want canary deployments, switching
to aliases is a one-line change in registry.py:

Python

# stages
client.transition_model_version_stage(name, version, stage="Staging")

# aliases
client.set_registered_model_alias(name, alias="staging", version=version)
How to run
Train + log + register
PowerShell

python -m tehran_house_price.models.train_pipeline
View runs in the UI
In a separate terminal:

PowerShell

mlflow ui --port 5000
Then open http://localhost:5000.

Compare runs
In the experiment view:

Check the boxes next to the runs you want
Click Compare
Use the Parallel Coordinates plot to see how params affect metrics
Disable tracking for one run
PowerShell

$env:MLFLOW_TRACKING_ENABLED="false"
python -m tehran_house_price.models.train_pipeline
Remove-Item Env:\MLFLOW_TRACKING_ENABLED
Known gotchas
MLflow logs the model with a missing package warning
When mlflow.sklearn.log_model runs, you'll see:

text

WARNING mlflow.utils.requirements_utils: The following packages
were not found in the public PyPI package index ...: {'tehran-house-price'}
This is because our package isn't on PyPI. It's harmless - the model
is still saved correctly. MLflow just can't auto-generate a perfect
requirements.txt for it.

Nested runs need nested=True
When opening a child run inside a parent run, MLflow needs an explicit
nested=True or it raises:

text

Exception: Run with UUID xxx is already active.
Our get_run_context exposes a nested parameter for this.

Pre-commit can rewrite files after commit attempt
If end-of-file-fixer or ruff modifies a file during commit,
the commit fails. Just git add . and git commit again.

mlruns directory grows unbounded
Every training run adds ~50MB to mlruns/. Periodically clean it:

PowerShell

Remove-Item -Recurse -Force mlruns
New-Item -ItemType Directory -Path mlruns | Out-Null
The registry data is also in there, so this wipes everything.

Files changed in this phase
requirements.txt - added mlflow==2.14.1, pinned pyarrow==15.0.2
.gitignore - added mlruns/
src/tehran_house_price/tracking/__init__.py - new package
src/tehran_house_price/tracking/mlflow_setup.py - setup + run context
src/tehran_house_price/tracking/run_logger.py - safe log helpers
src/tehran_house_price/tracking/registry.py - model registration
src/tehran_house_price/models/train_pipeline.py - integration
tests/unit/test_mlflow_setup.py - 19 tests
tests/unit/test_run_logger.py - 11 tests
Next phase
Phase 6: CI/CD with GitHub Actions. Will run tests, build the Docker
image, and (eventually) push to a registry on every merge to main.
