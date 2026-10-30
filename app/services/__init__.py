"""Service layer decoupling API route controllers from model inference."""

from .prediction_service import PredictionService, get_prediction_service

__all__ = ["PredictionService", "get_prediction_service"]
