# Phase 7: Monitoring & Observability

## Overview

Phase 7 adds production-grade observability to the Tehran House Price API.
The goal is to answer operational questions without looking at source code:

- Is the API alive and ready to serve traffic?
- What are the p50/p95/p99 latencies?
- What is the error rate per endpoint?
- How many predictions are being served per minute?
- Which inputs caused failures?
- Are prediction inputs drifting from training data?

## Architecture

```text
Client
  -> FastAPI App
  -> RequestObservabilityMiddleware
       - generates / propagates request_id
       - logs structured JSON per request
       - records Prometheus metrics
  -> Endpoint Handler
       - calls ModelService.predict()
       - calls record_prediction() for Prometheus counter
       - schedules log_prediction() as BackgroundTask
  -> Response (with X-Request-ID header)

Background (non-blocking):
  -> log_prediction() writes to PostgreSQL

Prometheus scrapes /metrics every 5s
Grafana reads from Prometheus
Sub-phases Completed
Phase 7.0: Scaffold
Created the package and file structure for monitoring:

text

src/tehran_house_price/api/middleware.py
src/tehran_house_price/api/metrics.py
src/tehran_house_price/api/health.py
src/tehran_house_price/monitoring/__init__.py
src/tehran_house_price/monitoring/prediction_logger.py
src/tehran_house_price/monitoring/schemas.py
prometheus/prometheus.yml
docker-compose.observability.yml
Phase 7.1: Request Tracing and Structured Logging
Every HTTP request is assigned a unique request_id (UUID4).
If the caller sends an X-Request-ID header, that value is used instead
(useful for distributed tracing).

The middleware emits one structured JSON log line per request:

JSON

{
  "event": "http_request_completed",
  "request_id": "c9c90029-5029-4847-b0e4-ec8a28b0dedf",
  "method": "POST",
  "path": "/predict",
  "duration_ms": 13.945,
  "status_code": 200,
  "client_ip": "127.0.0.1"
}
The request_id is also injected into the response header X-Request-ID
so clients can correlate requests with logs.

Key design decisions:

Logging failures never crash the API (wrapped in try/except)
Controlled via REQUEST_LOGGING_ENABLED env var
request_id is stored in Python ContextVar for thread safety
Phase 7.2: Prometheus Metrics
A /metrics endpoint exposes Prometheus text format metrics.

Metrics exposed:

Metric	Type	Labels	Description
http_requests_total	Counter	method, path, status_code	Total HTTP requests
http_request_duration_seconds	Histogram	method, path	Request latency
http_requests_in_progress	Gauge	method, path	In-flight requests
http_request_exceptions_total	Counter	method, path, exception_type	Unhandled exceptions
model_predictions_total	Counter	endpoint, model_name	Predictions served
All metric updates are wrapped in try/except.
A failing metric update never affects the API response.

Controlled via PROMETHEUS_ENABLED env var.

Phase 7.3: Deep Health Checks
Two new endpoints alongside the existing /health:

GET /health/live
Liveness probe. Returns 200 as long as the process is running.
Used by Kubernetes/Docker to know whether to restart the container.

JSON

{"status": "alive"}
GET /health/ready
Readiness probe. Returns 200 only when all checks pass.
Used to know whether the API can serve traffic.

Checks performed:

Model loaded in memory
Model artifact file exists on disk
Free disk space above minimum threshold
JSON

{
  "status": "ready",
  "checks": {
    "model_loaded": true,
    "artifact_exists": true,
    "artifact_path": "/app/artifacts/models/xgb_price_per_m2.joblib",
    "disk_ok": true,
    "disk_free_bytes": 50000000000,
    "disk_min_required_bytes": 100000000
  }
}
If any check fails, returns 503 with "status": "not_ready".

Controlled via DEEP_HEALTHCHECK_ENABLED env var.

Phase 7.4: Prediction Logging to PostgreSQL
Every prediction request is logged asynchronously to PostgreSQL.

The log entry contains:

request_id: correlation ID from middleware
timestamp: UTC datetime
endpoint: which endpoint was called
model_name: which model was used
input_data: full request payload as JSON
output_data: full response payload as JSON
The logging is implemented as a FastAPI BackgroundTask:

The API returns the response immediately
The database write happens after the response is sent
If the database is unavailable, the API continues working normally
All database errors are caught and logged as warnings
Controlled via PREDICTION_LOGGING_ENABLED env var (default: false).
Requires PREDICTION_LOG_DB_URL to be set.

Phase 7.5: Local Observability Stack
docker-compose.observability.yml brings up the full monitoring stack:

text

api        -> FastAPI application (port 8000)
postgres   -> PostgreSQL 15 for prediction logs (port 5432)
prometheus -> Prometheus 2.45 for metrics (port 9090)
grafana    -> Grafana 10.0 for dashboards (port 3000)
To start the full stack locally:

Bash

docker-compose -f docker-compose.observability.yml up --build
Access points:

API: http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana: http://localhost:3000 (anonymous admin access)
Grafana data source: add Prometheus at http://prometheus:9090

Environment Variables
Variable	Default	Description
OBSERVABILITY_ENABLED	true	Master switch for all observability
REQUEST_LOGGING_ENABLED	true	Structured JSON request logs
PROMETHEUS_ENABLED	true	Prometheus metrics and /metrics endpoint
DEEP_HEALTHCHECK_ENABLED	true	/health/live and /health/ready endpoints
HEALTH_MIN_DISK_FREE_BYTES	100000000	Minimum free disk for readiness (100 MB)
PREDICTION_LOGGING_ENABLED	false	Log predictions to PostgreSQL
PREDICTION_LOG_DB_URL	None	PostgreSQL connection URL
REQUEST_ID_HEADER_NAME	X-Request-ID	Header name for request tracing
API Endpoints Added
Method	Path	Description
GET	/health	Basic liveness + model loaded status (existing)
GET	/health/live	Kubernetes liveness probe
GET	/health/ready	Kubernetes readiness probe
GET	/metrics	Prometheus metrics
Testing
New tests added in this phase:

text

tests/unit/test_api_middleware.py     (4 tests)
tests/unit/test_api_metrics.py        (2 tests)
tests/unit/test_api_health.py         (3 tests)
tests/unit/test_prediction_logger.py  (1 test)
tests/integration/test_observability_api.py (1 test)
All observability tests use dependency injection and monkeypatching.
No real database or Prometheus server is required to run the tests.

Design Principles
Observability must be opt-in via env vars: every feature can be
disabled without code changes.

Never let observability crash the API: all metric updates and log
writes are wrapped in try/except. The API always returns a response.

Non-blocking prediction logging: database writes happen in
BackgroundTasks after the response is sent to the client.

Structured logs are machine-readable: JSON format is compatible
with Docker log collectors, ELK stack, Grafana Loki, and CloudWatch.

request_id enables cross-referencing: the same ID appears in
HTTP response headers, structured logs, and prediction database records.
