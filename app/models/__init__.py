"""Models module for SAGE backend.
Handles crop registry, checkpoint downloading, adapter loading, and model lifecycle management.
"""

from .crop_registry import CropRegistry, CropConfig, crop_registry
from .adapter_loader import AdapterLoader, CheckpointMetadata
from .model_manager import ModelManager, get_model_manager

__all__ = [
    "CropRegistry",
    "CropConfig",
    "crop_registry",
    "AdapterLoader",
    "CheckpointMetadata",
    "ModelManager",
    "get_model_manager",
]
