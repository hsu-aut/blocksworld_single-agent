"""
Models Package - LLM model management and utilities.

This package provides everything needed for working with LLM models:
- Model configurations and pricing (config.py)
- Model creation and API key management (factory.py) 
- Token usage tracking and cost calculation (tracking.py)

The package maintains a clean separation of concerns while providing
a convenient unified import interface.
"""

from .config import AVAILABLE_MODELS, MODELS_BY_PROVIDER, MODELS_BY_PROVIDER_AND_SERIES
from .factory import create_model, get_api_key_for_provider
from .tracking import TokenTrackingCallback

__all__ = [
    # Model configurations
    "AVAILABLE_MODELS",
    "MODELS_BY_PROVIDER", 
    "MODELS_BY_PROVIDER_AND_SERIES",
    
    # Model creation
    "create_model",
    "get_api_key_for_provider",
    
    # Usage tracking
    "TokenTrackingCallback"
]