import os
import yaml
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("sage.crop_registry")


class CropConfig(BaseModel):
    """Configuration metadata for a crop model checkpoint."""

    crop_id: str
    display_name: str
    hf_repo: str
    checkpoint_subpath: str
    model_type: str = "peft_lora"
    adapter_name: str
    prompt_template: str
    known_classes: List[str] = Field(default_factory=list)
    loading_status: str = "unloaded"  # "unloaded", "loading", "active", "cached", "error"
    error_message: Optional[str] = None


class CropRegistry:
    """Central registry and manager for supported crop configurations."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or settings.config_path
        self.crops: Dict[str, CropConfig] = {}
        self.default_hf_repo: str = settings.adapter_repo
        self.base_model: str = settings.base_model
        self.load_registry()

    def load_registry(self) -> None:
        """Load and parse crop registry from YAML file."""
        if not os.path.exists(self.config_path):
            logger.error("Configuration file not found at: %s", self.config_path)
            raise FileNotFoundError(f"Crop configuration file not found at: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            self.default_hf_repo = data.get("default_hf_repo", settings.adapter_repo)
            self.base_model = data.get("base_model", settings.base_model)
            crops_data = data.get("crops", {})

            self.crops.clear()
            for crop_id, crop_info in crops_data.items():
                normalized_id = crop_id.lower().strip()
                self.crops[normalized_id] = CropConfig(
                    crop_id=normalized_id,
                    display_name=crop_info.get("display_name", normalized_id.capitalize()),
                    hf_repo=crop_info.get("hf_repo", self.default_hf_repo),
                    checkpoint_subpath=crop_info.get("checkpoint_subpath", ""),
                    model_type=crop_info.get("model_type", "peft_lora"),
                    adapter_name=crop_info.get("adapter_name", f"{normalized_id}_lora"),
                    prompt_template=crop_info.get(
                        "prompt_template",
                        f"Analyze this image of a {normalized_id} plant and provide diagnosis details.",
                    ),
                    known_classes=crop_info.get("known_classes", []),
                    loading_status="unloaded",
                )

            logger.info("Successfully loaded %d crops from %s", len(self.crops), self.config_path)
        except Exception as e:
            logger.error("Failed to parse crop configuration: %s", str(e))
            raise ValueError(f"Invalid crop configuration in {self.config_path}: {e}")

    def get_crop(self, crop_id: str) -> Optional[CropConfig]:
        """Retrieve crop config by case-insensitive identifier."""
        return self.crops.get(crop_id.lower().strip())

    def validate_crop(self, crop_id: str) -> CropConfig:
        """Validate crop identifier; raises ValueError if unsupported."""
        crop_id_norm = (crop_id or "").lower().strip()
        if crop_id_norm not in self.crops:
            supported = ", ".join(self.crops.keys())
            raise ValueError(f"Crop '{crop_id}' is not supported. Supported crops are: {supported}")
        return self.crops[crop_id_norm]

    def list_crop_ids(self) -> List[str]:
        """Return list of all supported crop IDs."""
        return sorted(list(self.crops.keys()))

    def get_all_crops(self) -> Dict[str, CropConfig]:
        """Return full map of registered crops."""
        return self.crops

    def update_loading_status(self, crop_id: str, status: str, error: Optional[str] = None) -> None:
        """Update loading status for a given crop."""
        crop = self.get_crop(crop_id)
        if crop:
            crop.loading_status = status
            crop.error_message = error


# Global singleton crop registry instance
crop_registry = CropRegistry()

