# Phase 4: Docker Containerization

## Overview

Phase 4 wraps the FastAPI service into a production-ready Docker image so it
can be shipped to any container runtime (Render, Hugging Face Spaces,
Kubernetes, local Docker). The image bakes the trained model artifact inside
so the container is self-contained and reproducible.

## Goals

- Container-native deployment of the prediction API
- Reproducible image: same artifact, same dependencies, same runtime
- Security: non-root user, minimal base image
- Observability: built-in healthcheck, structured logs
- Small image size: target under 500MB (achieved: 492MB)

## Files added

```text
.dockerignore         excludes thpenv, data, logs, .git, etc. from build context
Dockerfile            local-dev image (uses Iran mirrors)
Dockerfile.prod       production image (uses standard pypi.org and deb.debian.org)
docker-compose.yml    one-command local orchestration
.gitattributes        forces LF line endings for files copied into Linux containers
Image design
Multi-stage build
text

Stage 1: builder
  - python:3.10.14-slim-bookworm
  - install build-essential, gcc, g++
  - pip install all dependencies into /root/.local
  - pip install -e . (the package itself)

Stage 2: runtime
  - python:3.10.14-slim-bookworm (fresh layer, no compilers)
  - install only libgomp1 (required by xgboost OpenMP)
  - copy /root/.local from builder into /home/app/.local
  - create non-root user "app"
  - copy src/, configs/, artifacts/models/, pyproject.toml
  - HEALTHCHECK against /health
  - CMD python -m tehran_house_price.api
The runtime stage never contains compilers or caches, which keeps the final
image small and reduces attack surface.

Why this design
Decision	Reason
Multi-stage build	Keep gcc/g++ out of the runtime image (smaller, safer)
pip install --user	Easy to copy installed packages between stages
Non-root user app	If the container is compromised, attacker is not root
--chown=app:app on COPY	Avoid the "root-owned files" footgun
API_HOST=0.0.0.0 env	Container must bind to all interfaces, not 127.0.0.1
EXPOSE 8000	Documentation for the port (does not actually open it)
HEALTHCHECK via stdlib	No need to install curl, smaller image
start-period=15s	First model load takes a few seconds
Bake artifact into image	Reproducible: image == code + deps + exact model
Two Dockerfiles, one project
This project ships two Dockerfiles because of network realities:

text

Dockerfile          uses Iran mirrors (mirror.iranserver.com,
                    mirror-pypi.runflare.com) so local dev inside
                    Iran can build without VPN issues

Dockerfile.prod     uses standard registries (pypi.org, deb.debian.org)
                    for deployment on cloud platforms outside Iran
                    (Render, Hugging Face Spaces, GitHub Actions)
Both produce the same runtime behavior. The only difference is the package
source URLs. Tests pass identically in both images.

Usage
Build (local dev, inside Iran)
PowerShell

docker build -t tehran-house-price:latest .
Build (production, outside Iran)
PowerShell

docker build -f Dockerfile.prod -t tehran-house-price:prod .
Run a single container
PowerShell

docker run -d --name thp-api -p 8000:8000 tehran-house-price:latest

# Logs
docker logs -f thp-api

# Stop
docker stop thp-api
docker rm thp-api
Run with docker compose
PowerShell

docker compose up -d
docker compose logs -f api
docker compose down
Verify health
PowerShell

Invoke-RestMethod -Uri "http://localhost:8000/health"
# status model_loaded
# ------ ------------
# ok             True

docker inspect --format='{{.State.Health.Status}}' thp-api
# healthy
Verify non-root
PowerShell

docker exec thp-api whoami
# app
Predict from PowerShell
PowerShell

$body = @{
    district     = "Pasdaran"
    area_m2      = 120.0
    rooms        = 2
    has_parking  = $true
    has_storage  = $true
    has_elevator = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" `
                  -Method Post -Body $body `
                  -ContentType "application/json"
Predict with curl.exe
PowerShell

curl.exe -X POST http://localhost:8000/predict `
         -H "Content-Type: application/json" `
         -d '{\"district\":\"Pasdaran\",\"area_m2\":120.0,\"rooms\":2,\"has_parking\":true,\"has_storage\":true,\"has_elevator\":true}'
Note: PowerShell aliases curl to Invoke-WebRequest, so use curl.exe
to get the real curl behavior.

Environment variables
The container reads the following at startup:

text

API_HOST          default 0.0.0.0   bind interface
API_PORT          default 8000      port to listen on
API_LOG_LEVEL     default info      uvicorn log level
API_WORKERS       default 1         number of uvicorn workers
APP_ENV           default prod      application environment label
LOG_LEVEL         default INFO      application logger level
These can be overridden via -e KEY=VALUE on docker run or in the
environment: section of docker-compose.yml.

Verification checklist
text

[x] docker build succeeds without errors
[x] Image size under 500MB (actual: 492MB)
[x] Container starts and loads model
[x] /health returns 200 with model_loaded=true
[x] /version returns model metadata
[x] /predict returns valid predictions
[x] /predict/batch handles multiple listings
[x] Healthcheck passes (status: healthy)
[x] Non-root user (whoami: app)
[x] docker compose up / down works cleanly
[x] All 160 unit + integration tests still pass locally
Iran-specific notes
Inside Iran, default apt-get update against deb.debian.org and
pip install against pypi.org time out or get filtered. The dev
Dockerfile rewrites the apt sources to mirror.iranserver.com and
configures pip to use mirror-pypi.runflare.com. It also drops
bookworm-updates because the Iran mirror only keeps the main and
security repos up to date.

For CI/CD and cloud deployment, use Dockerfile.prod which has none of
these workarounds and is the canonical production image.

Known gotchas
text

1. CRLF vs LF
   Files copied into the Linux container must have LF endings.
   .gitattributes enforces this for new commits. Existing files were
   converted via a PowerShell one-liner during Phase 4.1.

2. Docker Hub rate limit
   Anonymous pulls are throttled to 100 per 6h. If "403 Forbidden"
   appears during build, either wait, run "docker login", or remove
   the "# syntax=docker/dockerfile:1.7" directive (which pulls an
   extra image).

3. BuildKit network stack
   On Windows with VPN, "docker run" sometimes goes through the VPN
   while "docker build" does not. If apt-get update fails inside build
   but not inside a manual run, this is the cause.

4. Model artifact must exist before build
   The COPY step pulls artifacts/models/xgb_price_per_m2.joblib into
   the image. Phase 2 must be complete before Phase 4 build.

5. Healthcheck start_period
   Set to 15s because xgboost model load takes ~2s on first request
   plus uvicorn startup. Too small a start_period causes false
   "unhealthy" reports.

6. PowerShell curl alias
   "curl" in PowerShell is an alias for Invoke-WebRequest, which has
   different syntax than real curl. Use "curl.exe" to invoke the real
   binary, or use Invoke-RestMethod for a more PowerShell-native flow.
What this phase did not do
text

- No image registry push (planned for Phase 6 with GitHub Actions)
- No cloud deployment (planned for Phase 8)
- No horizontal scaling, load balancer, or rate limiting
- No GPU support (XGBoost CPU is fast enough for the dataset size)
- No volume mounts for logs (container is stateless)
