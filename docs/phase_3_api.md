# Phase 3: API and Serving

This phase wraps the trained model in a production-ready HTTP API
built on FastAPI. It covers schema validation, model loading,
endpoint design, error handling, and a CLI entry point.

## Goals

- Serve predictions over HTTP with a stable, versioned contract
- Load the model once at startup and reuse it across requests
- Provide structured, machine-parseable error responses
- Stay testable without requiring the real model artifact

## Architecture
HTTP request
-> FastAPI router
-> Pydantic validation (api/schemas.py)
-> Endpoint handler (api/app.py)
-> ModelService.predict (api/model_loader.py)
-> Pipeline.predict (sklearn artifact)
HTTP response (Pydantic-serialized)

text


The API layer is intentionally thin: validation lives in the schemas,
inference logic lives in the model service, and endpoints only
translate between them.

## Package layout
src/tehran_house_price/api/
├── init.py re-exports for convenient imports
├── main.py enables python -m tehran_house_price.api
├── app.py FastAPI app factory, lifespan, routes
├── dependencies.py reusable Depends providers
├── errors.py global exception handlers
├── model_loader.py ModelService singleton (load, predict)
├── run.py CLI entry point that wraps uvicorn
└── schemas.py request and response Pydantic models

text


## Endpoints

| Method | Path             | Purpose                                  |
|--------|------------------|------------------------------------------|
| GET    | /health          | Liveness probe and model-loaded flag     |
| GET    | /version         | App version and model metadata           |
| POST   | /predict         | Single house prediction                  |
| POST   | /predict/batch   | Batch prediction, max 1000 listings      |

## Request schema

`POST /predict` accepts:

```json
{
  "district": "Pasdaran",
  "area_m2": 120.0,
  "rooms": 2,
  "has_parking": true,
  "has_storage": true,
  "has_elevator": true
}
All fields are required. The schema rejects:

Empty or placeholder districts ("nan", "null", "none")
Negative or absurd area values
Non-boolean values for boolean flags (no "true" strings)
Unknown fields (extra="forbid")
Response schema
JSON

{
  "predicted_price_per_m2": 95000000.0,
  "predicted_total_price": 11400000000.0,
  "currency": "toman",
  "model_name": "xgb_price_per_m2"
}
Prices are in Toman. predicted_total_price is predicted_price_per_m2 * area_m2.

Error responses
All errors follow a single shape:

JSON

{
  "error": "validation_error",
  "message": "Request payload failed validation",
  "details": [
    { "field": "area_m2", "message": "Input should be greater than 0", "type": "greater_than" }
  ],
  "status_code": 422
}
Mapping:

Source	error	status
Pydantic validation failure	validation_error	422
Model not loaded	http_error	503
ValueError from model layer	bad_request	400
Unknown route	http_error	404
Unexpected internal exception	internal_server_error	500
Internal exceptions are logged with full traceback but never leak
their messages to clients.

Model loading
The model is loaded once at startup via the FastAPI lifespan
handler. Subsequent requests reuse the in-memory instance:

text

startup -> ModelService.load(artifacts/models/xgb_price_per_m2.joblib)
request -> ModelService.predict(rows)
shutdown -> ModelService.reset()
If the artifact is missing, the app still starts but /health
reports model_loaded: false and prediction endpoints return 503.

Dependency injection
Two providers are layered:

get_model_service returns the singleton regardless of state.
Used by /health and /version which must respond even when
the model is not loaded.
get_loaded_model_service depends on the above and raises 503
if the model is not loaded. Used by prediction endpoints.
In tests, overriding the lower-level provider with a stub also
covers the higher-level one, because the latter resolves through
the former.

Running the server
For local development:

PowerShell

python -m tehran_house_price.api --reload
With explicit options:

PowerShell

python -m tehran_house_price.api --host 0.0.0.0 --port 8080 --workers 4
All flags can also be provided as environment variables: API_HOST,
API_PORT, API_RELOAD, API_WORKERS, API_LOG_LEVEL. This is
convenient for container deployments.

The interactive OpenAPI docs are served at /docs.

Testing strategy
tests/unit/test_api_schemas.py: schema validation
tests/unit/test_model_loader.py: loader behavior with fake artifacts
tests/unit/test_api_endpoints.py: routes with stubbed model service
tests/unit/test_api_errors.py: structured error responses
tests/integration/test_api_e2e.py: end-to-end with the real model
(skipped automatically when the artifact is missing)
Stubs use app.dependency_overrides to bypass the lifespan handler,
so the unit suite never touches disk and runs in well under a second.

Known limitations
No authentication or rate limiting; both are deferred to a reverse
proxy or a later phase
No CORS configuration; will be added when a frontend is introduced
The single-process default of uvicorn.run is fine for development
and small deployments; production should use multiple workers
The python-multipart deprecation warning is harmless and will
disappear when FastAPI upgrades its import path
