# Phase 8 — Deployment and Final Documentation

Phase 8 takes the project from a working local codebase to a **live,
publicly accessible production service**. It also rewrites the README as
the primary portfolio artifact.

## Goals

- Deploy the API to a public cloud platform
- Handle model artifact delivery without baking models into Docker images
- Use Infrastructure-as-Code for reproducible deployment
- Rewrite the README with live demo, architecture, and honest limitations
- Tag the project as `v1.0.0`

## Deployment Target Decision

Three candidates were considered:

| Platform | Pros | Cons | Decision |
|----------|------|------|----------|
| Hugging Face Spaces | Free, ML-friendly | Better for demos than REST APIs | Runner-up |
| Fly.io | Powerful free tier, global edge | More complex config | Skipped |
| **Render** | Docker-native, IaC support, simple UI, health checks | Cold starts on free tier | **Chosen** |

Render was selected because:
- Direct Docker deployment from GitHub
- Native `render.yaml` support (Infrastructure-as-Code)
- Built-in health check integration
- Free tier sufficient for a portfolio project
- Cloudflare CDN in front of every service (free DDoS protection)

## Model Delivery Strategy

Three approaches were considered:

**Option A: Bake model into Docker image**
Simple but couples code and model lifecycles. Image size grows.

**Option B: Download at startup from public URL** ✅
Chosen approach. The container downloads the model from GitHub Releases on
first startup. Files are cached on the container's disk for subsequent
restarts (until Render spins down).

**Option C: MLflow remote registry**
Production-grade but overkill for this project.

## Implementation

### 1. `src/tehran_house_price/api/bootstrap.py`

A new module responsible for ensuring model files exist on disk before
`ModelService.load()` runs.

Key properties:
- **Idempotent**: no-op if files already exist
- **Opt-in**: no-op if download URLs are unset
- **Atomic**: downloads to `.part` file and renames on success
- **Defensive**: raises `ArtifactDownloadError` on failure (Render will
  restart the container)

### 2. `render.yaml`

Infrastructure-as-Code configuration for Render:

```yaml
services:
  - type: web
    name: tehran-house-price-api
    runtime: docker
    dockerfilePath: ./Dockerfile.prod
    plan: free
    region: oregon
    branch: main
    autoDeploy: true
    healthCheckPath: /health
    envVars:
      - key: ARTIFACT_DOWNLOAD_URL
        sync: false     # Set in Render dashboard, not in code
      - key: ARTIFACT_METADATA_DOWNLOAD_URL
        sync: false
Notable design decisions:

sync: false for URLs even though they are public — treats them as
configuration rather than code
autoDeploy: true — any push to main triggers a redeploy
healthCheckPath: /health — Render restarts the container if the
endpoint stops returning 200
3. Startup Flow
text

Container starts
  │
  ▼
lifespan() begins
  │
  ├─→ ensure_model_artifacts()
  │     │
  │     ├─→ If files exist: skip
  │     ├─→ If URLs missing: log warning, skip
  │     └─→ Otherwise: download atomically
  │
  ├─→ ModelService.load()
  │     │
  │     └─→ joblib.load(artifact_path)
  │
  └─→ Uvicorn starts serving requests
4. Model Hosting
Model files are hosted as GitHub Release assets under tag v0.9.0-model:

xgb_price_per_m2.joblib (1.5 MB)
xgb_price_per_m2_metadata.json (2 KB)
Public URLs are set as environment variables in the Render dashboard.

README Rewrite
The README was fully rewritten with:

Live demo URL and interactive curl examples
CI/CD, Docker, Python, and License badges
ASCII architecture diagram
Tech stack breakdown
Three quick-start options (Docker pull, source, full pipeline)
Complete API reference
Observability documentation
Known Limitations section — an honest discussion of concept drift
and how the existing MLOps infrastructure enables future fixes
The Known Limitations section is deliberately prominent because
demonstrating awareness of model shortcomings is more valuable in
interviews than pretending the model is perfect.

Testing the Deployment
After the first deploy, the following were verified:

Test	Method	Result
Cold start bootstrap	Watch Render logs during startup	✅ Model downloaded and loaded
/health	curl	✅ model_loaded: true
/version	curl	✅ Correct version and artifact path
/predict	curl with sample payload	✅ Consistent with local predictions
/predict/batch	Batch of 3 listings	✅ All predictions returned
/metrics	Browser	✅ Prometheus format, counters incrementing
/docs	Browser	✅ Swagger UI fully functional
Request tracing	Check x-request-id header	✅ UUID present, matches log entries
Structured logging	Render Logs tab	✅ JSON logs with request_id
Lessons Learned
Pydantic protected namespace: field names starting with model_
conflict with Pydantic's reserved namespace. Renamed to
artifact_download_url.
SIM117 rule: nested with statements are flagged by ruff. Python
3.10 parenthesized context manager syntax is the fix.
PowerShell curl != curl: curl in PowerShell is an alias for
Invoke-WebRequest with different syntax. Documentation examples must
use standard bash curl for portability.
Render free tier cold starts: 30-60 seconds for first request after
15 minutes of inactivity. Acceptable for a portfolio project.
Cloudflare in front of Render: every Render service is
automatically behind Cloudflare CDN. Response headers reveal this
(cf-ray, cf-cache-status).
Final State
Project version: 1.0.0
Live URL: https://tehran-house-price-api.onrender.com/docs
Docker image: ghcr.io/captain-jorf/tehran-house-price:latest
All CI pipelines green
196 tests passing (177 pass, 10 skip, 0 fail)
Coverage: 82%
Next Phase (Post-1.0)
Not part of this project's scope but sensible next steps:

Airflow or Prefect DAG for weekly retraining
Evidently AI for drift detection
Divar scraper filled out (currently a stub)
Grafana dashboard JSON provisioning
Model A/B testing via multiple registered stages
