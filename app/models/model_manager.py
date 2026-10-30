import os
import gc
import threading
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.models.crop_registry import crop_registry, CropConfig
from app.models.adapter_loader import AdapterLoader, CheckpointMetadata

logger = get_logger("sage.model_manager")


class GPUStatus(BaseModel):
    available: bool
    device_name: Optional[str] = None
    total_memory_gb: Optional[float] = None
    allocated_memory_gb: Optional[float] = None
    reserved_memory_gb: Optional[float] = None


class ModelStatus(BaseModel):
    base_model: str
    active_crop: Optional[str] = None
    active_adapter: Optional[str] = None
    adapter_mode: str
    base_model_loaded: bool
    loaded_adapters: List[str] = []
    gpu_status: GPUStatus
    mock_mode: bool


class ModelManager:
    """Manages the lifecycle, loading, and switching of base models and PEFT adapters."""

    def __init__(self):
        self._lock = threading.Lock()
        self.loader = AdapterLoader()
        self.base_model_name = settings.base_model
        self.adapter_mode = settings.adapter_mode.lower().strip()
        self.mock_mode = settings.mock_inference

        # Runtime instances
        self.base_model = None
        self.model = None  # Model with adapter(s) attached
        self.processor = None
        self.active_crop: Optional[str] = None
        self.active_adapter_name: Optional[str] = None
        self.loaded_adapters: Dict[str, str] = {}  # crop_id -> local_checkpoint_path
        self._is_base_loaded: bool = False

    def get_gpu_status(self) -> GPUStatus:
        """Query GPU presence and memory consumption."""
        try:
            import torch
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                total = round(props.total_memory / (1024**3), 2)
                allocated = round(torch.cuda.memory_allocated(idx) / (1024**3), 2)
                reserved = round(torch.cuda.memory_reserved(idx) / (1024**3), 2)
                return GPUStatus(
                    available=True,
                    device_name=props.name,
                    total_memory_gb=total,
                    allocated_memory_gb=allocated,
                    reserved_memory_gb=reserved,
                )
            return GPUStatus(available=False, device_name=None)
        except Exception as e:
            logger.debug("GPU status query exception: %s", e)
            return GPUStatus(available=False, device_name=None)

    def get_model_status(self) -> ModelStatus:
        """Return current status of the model, active adapter, and GPU utilization."""
        return ModelStatus(
            base_model=self.base_model_name,
            active_crop=self.active_crop,
            active_adapter=self.active_adapter_name,
            adapter_mode=self.adapter_mode,
            base_model_loaded=self._is_base_loaded,
            loaded_adapters=list(self.loaded_adapters.keys()),
            gpu_status=self.get_gpu_status(),
            mock_mode=self.mock_mode,
        )

    def load_base_model(self) -> None:
        """Download and initialize base Vision-Language Model and Processor once.
        Thread-safe singleton loader.
        """
        with self._lock:
            if self._is_base_loaded:
                logger.info("Base model '%s' is already loaded.", self.base_model_name)
                return

            if self.mock_mode:
                logger.info("MOCK_INFERENCE active: Simulating base model initialization.")
                self._is_base_loaded = True
                return

            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            device = settings.get_resolved_device()
            dtype = settings.get_resolved_dtype() or torch.float16
            token = settings.hf_token

            logger.info(
                "Loading base model '%s' on device '%s' with dtype '%s'...",
                self.base_model_name,
                device,
                dtype,
            )

            try:
                # 1. Load Processor
                self.processor = AutoProcessor.from_pretrained(
                    self.base_model_name,
                    token=token,
                    trust_remote_code=False,
                )

                # 2. Load Base Model in eval mode
                if device == "cuda":
                    self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.base_model_name,
                        torch_dtype=dtype,
                        device_map="auto",
                        token=token,
                        trust_remote_code=False,
                    )
                else:
                    self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.base_model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        token=token,
                        trust_remote_code=False,
                    )
                    self.base_model.to("cpu")

                self.base_model.eval()
                self.model = self.base_model
                self._is_base_loaded = True
                logger.info("Base model '%s' successfully loaded into memory.", self.base_model_name)

            except torch.cuda.OutOfMemoryError as e:
                logger.error("CUDA Out of Memory loading base model: %s", str(e))
                self._free_gpu_memory()
                raise RuntimeError(f"GPU out of memory loading base model {self.base_model_name}") from e
            except Exception as e:
                logger.error("Failed to load base model '%s': %s", self.base_model_name, str(e))
                raise

    def load_adapter(self, crop: CropConfig) -> None:
        """Download and attach crop-specific PEFT LoRA adapter to base model."""
        if self.mock_mode:
            logger.info("MOCK_INFERENCE: Mocking adapter attach for '%s'.", crop.crop_id)
            self.active_crop = crop.crop_id
            self.active_adapter_name = crop.adapter_name
            self.loaded_adapters[crop.crop_id] = crop.checkpoint_subpath
            crop_registry.update_loading_status(crop.crop_id, "active")
            return

        if not self._is_base_loaded or self.base_model is None:
            self.load_base_model()

        from peft import PeftModel

        checkpoint_dir = self.loader.download_checkpoint(crop)
        metadata = self.loader.inspect_checkpoint(crop.crop_id, checkpoint_dir)

        if not metadata.is_lora:
            raise ValueError(
                f"Checkpoint for '{crop.crop_id}' at '{checkpoint_dir}' does not contain 'adapter_config.json'."
            )

        adapter_name = crop.adapter_name

        try:
            if self.adapter_mode == "multi":
                # PEFT Multi-adapter mode: load adapter into existing PeftModel or wrap base_model
                if not isinstance(self.model, PeftModel):
                    logger.info("Initializing multi-adapter PeftModel with '%s'...", adapter_name)
                    self.model = PeftModel.from_pretrained(
                        self.base_model,
                        checkpoint_dir,
                        adapter_name=adapter_name,
                        is_trainable=False,
                    )
                else:
                    if adapter_name not in self.model.peft_config:
                        logger.info("Loading additional adapter '%s' into PeftModel...", adapter_name)
                        self.model.load_adapter(checkpoint_dir, adapter_name=adapter_name)
                    self.model.set_adapter(adapter_name)

            else:
                # Mode A: Single adapter mode (default)
                # Ensure clean detach of prior adapters to conserve VRAM
                logger.info("Attaching single adapter '%s' for crop '%s'...", adapter_name, crop.crop_id)
                self.unload_adapter()
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    checkpoint_dir,
                    adapter_name=adapter_name,
                    is_trainable=False,
                )

            self.model.eval()
            self.active_crop = crop.crop_id
            self.active_adapter_name = adapter_name
            self.loaded_adapters[crop.crop_id] = checkpoint_dir
            crop_registry.update_loading_status(crop.crop_id, "active")
            logger.info("Adapter '%s' active for crop '%s'.", adapter_name, crop.crop_id)

        except Exception as e:
            crop_registry.update_loading_status(crop.crop_id, "error", str(e))
            logger.error("Failed to load adapter for crop '%s': %s", crop.crop_id, str(e))
            raise

    def switch_adapter(self, crop: CropConfig) -> None:
        """Switch active adapter to the requested crop. Thread-safe."""
        with self._lock:
            if self.active_crop == crop.crop_id and self.model is not None:
                logger.debug("Crop '%s' adapter is already active.", crop.crop_id)
                return

            logger.info("Switching adapter from '%s' to '%s'...", self.active_crop, crop.crop_id)

            if self.active_crop and self.active_crop != crop.crop_id:
                crop_registry.update_loading_status(
                    self.active_crop,
                    "cached" if self.adapter_mode == "multi" else "unloaded",
                )

            crop_registry.update_loading_status(crop.crop_id, "loading")

            try:
                if self.adapter_mode == "multi" and crop.crop_id in self.loaded_adapters and self.model is not None:
                    # Quick switch without disk reloading
                    import peft
                    if isinstance(self.model, peft.PeftModel) and crop.adapter_name in self.model.peft_config:
                        self.model.set_adapter(crop.adapter_name)
                        self.active_crop = crop.crop_id
                        self.active_adapter_name = crop.adapter_name
                        crop_registry.update_loading_status(crop.crop_id, "active")
                        logger.info("Switched to preloaded adapter '%s'.", crop.adapter_name)
                        return

                # Otherwise download / load adapter
                self.load_adapter(crop)

            except Exception as e:
                crop_registry.update_loading_status(crop.crop_id, "error", str(e))
                raise

    def unload_adapter(self) -> None:
        """Unload active PEFT adapter and restore base model."""
        if self.mock_mode:
            self.active_crop = None
            self.active_adapter_name = None
            return

        try:
            if self.model is not None and self.base_model is not None and self.model != self.base_model:
                del self.model
                self.model = self.base_model

            self.active_crop = None
            self.active_adapter_name = None
            self.loaded_adapters.clear()
            self._free_gpu_memory()
            logger.debug("Active adapter unloaded and GPU cache freed.")
        except Exception as e:
            logger.warning("Error during adapter unload: %s", str(e))

    def _free_gpu_memory(self) -> None:
        """Garbage collect and release unused GPU memory cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def get_active_crop(self) -> Optional[str]:
        """Return currently active crop identifier."""
        return self.active_crop


# Global singleton ModelManager instance
_model_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Retrieve global ModelManager singleton."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance
