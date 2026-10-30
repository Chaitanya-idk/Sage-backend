import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Hugging Face Authentication
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

    # API Security
    sage_api_key: Optional[str] = Field(default=None, alias="SAGE_API_KEY")

    # Model & Checkpoint Architecture
    base_model: str = Field(default="Qwen/Qwen2.5-VL-3B-Instruct", alias="BASE_MODEL")
    adapter_repo: str = Field(default="IDK-Chaitanya/SAGE5Class-cp", alias="ADAPTER_REPO")
    config_path: str = Field(default="configs/crops.yaml", alias="CONFIG_PATH")

    # Adapter Lifecycle Mode: "single" or "multi"
    adapter_mode: str = Field(default="single", alias="ADAPTER_MODE")

    # Hardware & Precision
    device: str = Field(default="cuda", alias="DEVICE")
    model_dtype: str = Field(default="auto", alias="MODEL_DTYPE")
    mock_inference: bool = Field(default=False, alias="MOCK_INFERENCE")

    # Generation Parameters
    max_new_tokens: int = Field(default=256, alias="MAX_NEW_TOKENS")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    top_p: float = Field(default=0.9, alias="TOP_P")
    repetition_penalty: float = Field(default=1.05, alias="REPETITION_PENALTY")

    # Image Upload Validation
    max_image_size_mb: int = Field(default=10, alias="MAX_IMAGE_SIZE_MB")
    allowed_image_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"],
        alias="ALLOWED_IMAGE_TYPES",
    )

    # Server & Networking
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    allowed_origins: str = Field(default="*", alias="ALLOWED_ORIGINS")

    def get_cors_origins(self) -> List[str]:
        """Parse allowed origins into a list for CORS middleware."""
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    def is_api_key_enabled(self) -> bool:
        """Check if API Key authentication is enforced."""
        return bool(self.sage_api_key and self.sage_api_key.strip())

    def get_resolved_device(self) -> str:
        """Resolve actual execution device considering CUDA availability."""
        if self.mock_inference:
            return "mock"
        try:
            import torch
            if self.device.lower().startswith("cuda") and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"

    def get_resolved_dtype(self):
        """Resolve PyTorch dtype based on hardware support."""
        if self.mock_inference:
            return None
        try:
            import torch
            dtype_str = self.model_dtype.lower().strip()
            if dtype_str == "bfloat16" or (dtype_str == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
                return torch.bfloat16
            elif dtype_str == "float16" or (dtype_str == "auto" and torch.cuda.is_available()):
                return torch.float16
            else:
                return torch.float32
        except (ImportError, Exception):
            return None


# Global singleton instance
settings = Settings()

