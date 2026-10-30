import os
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from app.core.config import settings
from app.core.logging import get_logger
from app.models.crop_registry import CropConfig

logger = get_logger("sage.adapter_loader")


class CheckpointMetadata(BaseModel):
    """Metadata extracted by inspecting the checkpoint files."""

    crop_id: str
    local_path: str
    is_lora: bool
    has_safetensors: bool
    peft_type: Optional[str] = None
    base_model_reference: Optional[str] = None
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    target_modules: Optional[List[str]] = None
    detected_files: List[str] = []


class AdapterLoader:
    """Manages downloading, caching, and structural inspection of Hugging Face checkpoints."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "models_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_auth_token(self) -> Optional[str]:
        """Retrieve sanitized Hugging Face token if available."""
        token = settings.hf_token
        if token and token.strip():
            return token.strip()
        return None

    def download_checkpoint(self, crop: CropConfig) -> str:
        """Download or retrieve cached checkpoint subdirectory from Hugging Face Hub.
        
        Returns the local directory path containing the checkpoint files.
        """
        token = self._get_auth_token()
        repo_id = crop.hf_repo
        subpath = crop.checkpoint_subpath.strip("/")

        logger.info("Resolving checkpoint for crop '%s' from %s:%s", crop.crop_id, repo_id, subpath)

        try:
            # We download the specific subfolder for this crop checkpoint
            local_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"{subpath}/*"],
                token=token,
                local_dir=os.path.join(self.cache_dir, repo_id.replace("/", "_")),
            )

            target_path = os.path.join(local_dir, subpath)
            if not os.path.exists(target_path):
                # Check if it was extracted directly or with subpath
                raise FileNotFoundError(f"Checkpoint directory {target_path} does not exist after download.")

            logger.info("Checkpoint ready at: %s", target_path)
            return target_path

        except RepositoryNotFoundError as e:
            logger.error("Hugging Face repository '%s' not found or access denied.", repo_id)
            raise ValueError(
                f"Repository '{repo_id}' not found. Please verify repo name and ensure HF_TOKEN is valid for private access."
            ) from e
        except HfHubHTTPError as e:
            logger.error("Hugging Face Hub authentication or connection error: %s", str(e))
            raise ValueError(f"Hugging Face Hub HTTP error: {e}") from e
        except Exception as e:
            logger.error("Failed to download checkpoint for crop '%s': %s", crop.crop_id, str(e))
            raise

    def inspect_checkpoint(self, crop_id: str, checkpoint_dir: str) -> CheckpointMetadata:
        """Inspect checkpoint files and parse adapter_config.json to confirm architecture."""
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        files = os.listdir(checkpoint_dir)
        has_adapter_config = "adapter_config.json" in files
        has_safetensors = any(f.endswith(".safetensors") for f in files)

        metadata = CheckpointMetadata(
            crop_id=crop_id,
            local_path=checkpoint_dir,
            is_lora=has_adapter_config,
            has_safetensors=has_safetensors,
            detected_files=files,
        )

        if has_adapter_config:
            config_file = os.path.join(checkpoint_dir, "adapter_config.json")
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg: Dict[str, Any] = json.load(f)
                
                metadata.peft_type = cfg.get("peft_type", "LORA")
                metadata.base_model_reference = cfg.get("base_model_name_or_path")
                metadata.lora_rank = cfg.get("r")
                metadata.lora_alpha = cfg.get("lora_alpha")
                
                targets = cfg.get("target_modules")
                if isinstance(targets, list):
                    metadata.target_modules = targets
                elif isinstance(targets, str):
                    metadata.target_modules = [targets]

                logger.info(
                    "Checkpoint inspection passed: LoRA r=%s, alpha=%s, base=%s",
                    metadata.lora_rank,
                    metadata.lora_alpha,
                    metadata.base_model_reference,
                )
            except Exception as e:
                logger.warning("Could not parse adapter_config.json in %s: %s", checkpoint_dir, str(e))

        return metadata

