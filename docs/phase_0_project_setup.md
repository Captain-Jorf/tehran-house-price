# Phase 0 - Project Setup

This document describes the initial setup of the Tehran House Price project:
project structure, tooling, dependency management, and the engineering
decisions that shape every later phase.

Phase 0 has no data logic and no model logic. It only sets the foundation.
But every later phase depends on these decisions, so they are documented
explicitly.

---

## 1. Goal

Establish a clean, production-style Python project that:

- is easy to install and run on any machine
- enforces consistent code quality
- separates secrets from code
- makes future phases (data layer, ML pipeline, API, deployment) easy to add
  without restructuring

---

## 2. Project Structure

The project uses the **src-layout** convention:
tehran-house-price/
├── src/
│ └── tehran_house_price/ # the actual Python package
│ ├── data/
│ ├── features/
│ ├── models/
│ ├── api/
│ └── utils/
├── tests/
│ ├── unit/
│ └── integration/
├── configs/
├── data/ # gitignored
├── artifacts/
├── logs/
├── notebooks/
├── scripts/
└── docs/

text


### 2.1. Why src-layout

A flat layout (where `tehran_house_price/` sits at the repo root) often
"works" by accident because Python silently picks up modules from the
current directory. The src-layout forces you to install the package
properly (`pip install -e .`), which means:

- imports behave the same on every machine and CI
- accidental import shadowing is impossible
- the package can be published or containerized without surprises

### 2.2. Why separate `tests/unit` and `tests/integration`

- unit tests: fast, isolated, no I/O, no network
- integration tests: orchestration, full pipelines, may touch disk

Splitting them allows targeted test runs and clearer responsibility.

---

## 3. Dependency Management

Two files, two purposes:

| File                  | Purpose                              |
|-----------------------|--------------------------------------|
| requirements.txt      | runtime dependencies (pinned)        |
| requirements-dev.txt  | development tools (test, lint, etc.) |

`requirements-dev.txt` starts with `-r requirements.txt`, so installing dev
deps automatically pulls in runtime deps.

### 3.1. Why pinned versions

Pinned versions (e.g. `pandas==2.2.2`) guarantee reproducibility. A loose
version (`pandas>=2.0`) can silently break the project months later when a
new minor release changes behavior.

### 3.2. Notable choices

- `pydantic-settings` for environment configuration
- `pandera` for DataFrame validation
- `loguru` for logging (simpler than stdlib `logging` for project-level use)
- `argparse` for CLIs (NOT typer, due to a version conflict with click in
  this environment)

---

## 4. Configuration Strategy

Two configuration sources, used for different purposes:

### 4.1. `configs/base.yaml`

Non-secret, structural configuration:

- data paths
- Kaggle dataset reference
- artifact directories
- output filenames

Loaded via `settings.get_config()` and cached.

### 4.2. `.env`

Secrets and environment-dependent overrides:

- `KAGGLE_API_TOKEN`
- `APP_ENV`
- `LOG_LEVEL`

Loaded via `pydantic-settings` (`AppSettings`).

### 4.3. Why this split

YAML is great for structured config that lives in git. Secrets must never
live in git. Pydantic-settings is the standard, type-safe way to read env
vars in Python. Keeping the two sources separate makes secret handling
explicit and review-friendly.

### 4.4. `.env.example`

A committed file with empty placeholders. New developers copy it to `.env`
and fill in their own values. The real `.env` is gitignored.

---

## 5. Logging

Implemented in `src/tehran_house_price/utils/logger.py`.

- Loguru-based.
- YAML configuration in `configs/logging.yaml`.
- Log file path resolved absolutely so it works regardless of current
  working directory (important for PyCharm test runners).
- Rotating log file under `logs/app.log`.

### 5.1. Why log paths are resolved absolutely

When pytest is launched from PyCharm, the working directory may differ
from the repo root. A relative log path would create log files in random
locations. Absolute resolution removes that class of bug.

---

## 6. Path Helpers

Implemented in `src/tehran_house_price/utils/paths.py`.

A small module that exposes:

- `project_root()`
- `data_dir()`, `raw_dir()`, `interim_dir()`, `processed_dir()`
- `artifacts_dir()`, `logs_dir()`, `configs_dir()`
- `ensure_dir(path)`

### 6.1. Why this exists

Hard-coded relative paths break the moment someone runs the code from a
different directory. Centralizing path resolution means the rest of the
codebase never builds paths by hand.

---

## 7. Code Quality Tooling

### 7.1. Formatters and linters

- `ruff` — fast linter, also handles import sorting
- `black` — opinionated formatter
- `mypy` — static type checking (only for `src/`)

Configured in `pyproject.toml`.

### 7.2. Pre-commit hooks

Configured in `.pre-commit-config.yaml`:

- ruff (lint + autofix)
- ruff-format
- black
- trailing whitespace fixer
- end-of-file fixer
- yaml check
- large file check
- merge conflict check

### 7.3. Pre-commit workflow on Windows PowerShell

When hooks modify files (e.g. reformat), the first commit fails. The
correct workflow is:

```powershell
git add .
git commit -m "message"
# if hooks modified files:
git add .
git commit -m "message"
This is normal pre-commit behavior, not a bug.

8. Testing Setup
pytest as the test runner
pytest-cov for coverage reporting
Tests live in tests/, mirroring the package structure
Each test directory has its own __init__.py to avoid pytest's
"import file mismatch" error when test files share names across
unit and integration directories
8.1. Why __init__.py in test directories
Without them, pytest's rootdir/conftest discovery can collide when two
test files share the same basename (e.g. test_build_dataset.py in both
unit/ and integration/). Adding __init__.py files makes each
directory a proper package and resolves the collision permanently.

9. Editable Install
The package is installed in editable mode:

PowerShell

pip install -e .
This means from tehran_house_price... works everywhere — in scripts,
notebooks, tests, and CLI entrypoints — without PYTHONPATH hacks.

10. .gitignore
Comprehensive, but a few project-specific entries are critical:

data/ (raw and derived data are never committed)
artifacts/ content (but .gitkeep files are kept)
logs/
.env
thpenv/ (virtual environment)
standard Python ignores (__pycache__/, *.pyc, .pytest_cache/)
.gitkeep files preserve empty directories that the project expects to
exist (e.g. data/raw/divar/.gitkeep).

11. Operating Environment
Phase 0 was set up with these assumptions:

OS: Windows
Shell: PowerShell (not bash)
IDE: PyCharm
Python: 3.10
Virtual environment: thpenv/ at project root, gitignored
The Makefile in the repo is convenience for Linux/Mac users only. On
Windows, commands are run directly in PowerShell. This is documented in
the README so future contributors are not confused.

12. Files Created in Phase 0
Configuration and tooling:

pyproject.toml
requirements.txt
requirements-dev.txt
.pre-commit-config.yaml
.gitignore
.env.example
Makefile
README.md
Package skeleton:

src/tehran_house_price/__init__.py (with __version__)
src/tehran_house_price/settings.py
src/tehran_house_price/utils/paths.py
src/tehran_house_price/utils/logger.py
empty package directories for data/, features/, models/, api/
Configs:

configs/base.yaml
configs/logging.yaml
13. Definition of Done
Phase 0 was considered complete when:

 Project structure created (src-layout)
 Virtual environment working
 Editable install working
 All tooling installed and configured (ruff, black, mypy, pytest)
 Pre-commit hooks installed and passing
 Settings module loading both YAML and env variables
 Logger working with file output
 Path helpers in place
 Initial smoke tests passing
 Repository initialized and pushed to GitHub
Phase 0 status: COMPLETE.

14. What Phase 0 Enabled
Phase 1 (data layer) was able to start immediately, without spending a
single minute on:

import path issues
inconsistent code formatting
secret leakage risk
log file location bugs
"works on my machine" problems
