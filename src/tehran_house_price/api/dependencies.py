"""FastAPI dependency providers.

This module exposes small dependency functions that endpoints use to
access shared services. Keeping them in a separate module makes them
easy to override in tests via app.dependency_overrides.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status

from tehran_house_price.api.model_loader import ModelService, get_model_service


def get_loaded_model_service(
    service: ModelService = Depends(get_model_service),
) -> ModelService:
    """Return the model service, raising HTTP 503 if the model is not loaded.

    This is the standard dependency used by prediction endpoints. Endpoints
    that need to report unloaded state explicitly (like /health) should
    depend on get_model_service directly instead.
    """
    if not service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    return service
