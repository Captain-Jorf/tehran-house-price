# Phase 6: CI/CD with GitHub Actions

## Overview

This phase adds automated Continuous Integration and Continuous Delivery
to the project using GitHub Actions. Every push to `main` and every pull
request triggers automated quality gates and builds, ensuring that
broken code never lands in the main branch.

The pipeline is split into two independent workflows:

1. **CI Pipeline** — lint, format check, type check, and full test suite
2. **Docker Build and Push** — build the production image and publish it
   to GitHub Container Registry (GHCR)

Both workflows must pass before a release tag is cut.

## Architecture
text

             push to main / PR
                    |
      +-------------+-------------+
      |                           |
      v                           v
.github/workflows/ .github/workflows/
ci.yml docker.yml
| |
+------+------+ +-----+-----+
| | | | |
ruff black pytest buildx push
mypy + coverage multi-stage to
GHCR

text


The two workflows run in parallel. This is intentional: a broken test
should not block a Docker image build for investigation, and a Docker
issue should not hide a test failure.

## Workflow 1: CI Pipeline

File: `.github/workflows/ci.yml`

**Triggers**

- Push to `main`
- Pull requests targeting `main`

**Steps**

1. Checkout code
2. Set up Python 3.10 with pip cache
3. Install `requirements.txt` and `requirements-dev.txt`
4. Install the package in editable mode (`pip install -e .`)
5. Run `black --check src tests`
6. Run `ruff check src tests`
7. Run `mypy src` (soft-fail, non-blocking)
8. Run `pytest tests/ --cov=src/tehran_house_price`
9. Upload coverage report as an artifact

**Typical duration:** ~2 minutes with pip cache warm.

**Current metrics**

- 190 tests total (up from 186 after Phase 6.3)
- 176 passing, 10 skipped, 0 failing
- Coverage: 82%

The 10 skipped tests all require the real trained model artifact, which
is not committed to git. They run locally when artifacts are present.

## Workflow 2: Docker Build and Push

File: `.github/workflows/docker.yml`

**Triggers**

- Push to `main`
- Push of any tag matching `v*`
- Manual dispatch via GitHub UI

**Steps**

1. Checkout code
2. Set up Docker Buildx (enables multi-platform builds and GHA caching)
3. Log in to GHCR using the auto-provided `GITHUB_TOKEN`
4. Extract Docker metadata (tags and labels) from git ref
5. Build using `Dockerfile.prod` with GitHub Actions cache backend
6. Push all resulting tags to `ghcr.io/captain-jorf/tehran-house-price`

**Image tags generated**

- `latest` — always points to the latest push on `main`
- `main` — same as latest but branch-scoped
- `sha-<7chars>` — commit-pinned tag for reproducibility
- `v0.6.0-phase6` — semver tag when a git tag is pushed

**Typical duration**

- Cold build: ~5 minutes
- Warm build (cache hit): ~2 minutes

**Registry URL**
ghcr.io/captain-jorf/tehran-house-price:latest

text


## Key Design Decisions

### 1. Model artifacts are NOT baked into the Docker image

Trained model files (`artifacts/models/*.joblib`) are gitignored and
never copied into the image. Instead, models are provided at runtime via
one of:

- Volume mount (local dev via `docker-compose.yml`)
- Cloud storage (planned for Phase 8)
- MLflow Registry (planned for Phase 8)

**Why:**

- Image stays small and fast to pull
- Models can be swapped without rebuilding the image
- Same image runs in staging and production with different models
- Model updates do not require a code release

This decision was forced by an early CI failure: `Dockerfile.prod`
initially tried to `COPY artifacts/models/`, which does not exist on
the CI runner because it is gitignored. The fix was philosophical, not
just tactical: **container image = immutable code, model = mutable
runtime dependency**.

### 2. GHCR instead of Docker Hub

**Why GHCR:**

- Uses `GITHUB_TOKEN` automatically — no extra secret to manage
- No pull rate limits like Docker Hub free tier
- Tightly integrated with GitHub Packages UI
- Image visibility follows repo visibility

**Trade-off:** Users need a GitHub account to pull from private repos.
For a public portfolio project, this is a non-issue.

### 3. `mypy` runs with `|| true` (soft-fail)

The line in `ci.yml`:

```yaml
- name: Type check with mypy
  run: mypy src || true
Why soft-fail: Type coverage is being added incrementally. Making
mypy blocking on day one would force a big-bang type refactor. Instead,
we surface type warnings without blocking merges, and improve gradually.

When to make it strict: When all public functions have type hints
and mypy src returns zero errors, remove the || true.

4. Two Dockerfiles: dev and prod
Dockerfile — uses Iran mirrors for apt and pip, for local dev inside
Iran
Dockerfile.prod — uses standard mirrors, for CI and cloud deploy
The CI workflow explicitly uses Dockerfile.prod. The docker-compose.yml
uses Dockerfile for local dev.

Why not one Dockerfile with build args: Simpler to reason about.
Each file is self-contained and clearly named. Build args add cognitive
load for a two-environment case.

5. Pre-commit versions pinned to match requirements-dev.txt
The .pre-commit-config.yaml uses:

ruff-pre-commit rev: v0.5.0 matches ruff==0.5.0 in requirements
black rev: 24.4.2 matches black==24.4.2 in requirements
pre-commit-hooks rev: v4.6.0
Why: Prevents the classic "passes locally, fails in CI" bug caused
by pre-commit hooks and CI using different tool versions. This is the
single source of truth pattern applied to tool versions.

Local Development Parity
To reproduce the CI pipeline locally before pushing:

PowerShell

# lint (matches CI step)
black --check src tests
ruff check src tests

# type check (matches CI step, without soft-fail)
mypy src

# tests with coverage (matches CI step)
pytest tests/ --cov=src/tehran_house_price --cov-report=term-missing
Or run all pre-commit hooks explicitly:

PowerShell

pre-commit run --all-files
If pre-commit passes locally and CI still fails, check for tool version
drift in .pre-commit-config.yaml vs requirements-dev.txt.

Troubleshooting
Docker workflow fails with "no such file or directory" on COPY
Cause: Dockerfile.prod is trying to COPY a path that is gitignored
and therefore not present on the CI runner.

Fix: Never COPY gitignored paths into an image. If the file is needed
at runtime, mount it as a volume or fetch it at startup.

CI passes locally but fails on GitHub
Cause: Usually a version mismatch between local tools and CI-installed
tools.

Fix: Ensure .pre-commit-config.yaml and requirements-dev.txt list
the same pinned versions.

pre-commit install --install-hooks times out
Cause: pre-commit downloads hook environments from GitHub and PyPI.
Inside Iran, direct connections to files.pythonhosted.org and
github.com frequently time out without a VPN.

Fix: Enable VPN before running pre-commit clean or first-time
pre-commit install --install-hooks. Once envs are cached, no further
downloads are needed unless .pre-commit-config.yaml changes.

GHCR image is not visible after push
Cause: By default, packages pushed to GHCR are private, even from
public repos.

Fix: Go to https://github.com/USERNAME?tab=packages, select the package,
open Package settings, and change visibility to Public.

Docker workflow silently skips push
Cause: Missing permissions: packages: write in the workflow file.

Fix: Ensure the docker.yml workflow declares:

YAML

permissions:
  contents: read
  packages: write
Future Improvements
Ideas deferred to later phases or future work:

Add a release.yml workflow that runs only on tag pushes and creates
a GitHub Release with auto-generated changelog
Multi-platform builds (linux/amd64 + linux/arm64) for M1/M2 Mac
users and ARM cloud instances
Add pytest --benchmark in a nightly workflow to catch performance
regressions
Add a security.yml workflow using pip-audit and trivy for
vulnerability scanning
Make mypy strict once type coverage is complete
Add automated model performance regression check (compare new model
metrics against the deployed baseline)
