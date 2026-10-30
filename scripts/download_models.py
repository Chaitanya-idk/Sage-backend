import os
import sys
from huggingface_hub import snapshot_download

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.models.crop_registry import crop_registry
from app.core.logging import get_logger

logger = get_logger("sage.download")


def download_all(cache_dir: str = "models_cache"):
    """Pre-download base Vision-Language model and all crop LoRA adapters."""
    token = settings.hf_token
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 70)
    print("SAGE Multi-Crop Pre-download Utility")
    print(f"Target Cache Directory: {os.path.abspath(cache_dir)}")
    print("=" * 70)

    # 1. Download Base Model
    base_model = settings.base_model
    print(f"\n[1/2] Downloading Base Model: {base_model}...")
    try:
        base_path = snapshot_download(
            repo_id=base_model,
            token=token,
            local_dir=os.path.join(cache_dir, base_model.replace("/", "_")),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"Base model downloaded successfully to: {base_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download base model: {e}")
        return False

    # 2. Download Crop Checkpoints
    crops = crop_registry.get_all_crops()
    print(f"\n[2/2] Downloading {len(crops)} Crop Checkpoints from {settings.adapter_repo}...")

    repo_id = settings.adapter_repo
    local_repo_dir = os.path.join(cache_dir, repo_id.replace("/", "_"))

    for crop_id, crop_cfg in crops.items():
        subpath = crop_cfg.checkpoint_subpath.strip("/")
        print(f"\nDownloading checkpoint for crop: {crop_cfg.display_name} ({crop_id})...")
        print(f"Subpath: {subpath}")
        try:
            snapshot_download(
                repo_id=repo_id,
                token=token,
                allow_patterns=[f"{subpath}/*"],
                local_dir=local_repo_dir,
            )
            print(f"Successfully downloaded {crop_id} checkpoint.")
        except Exception as e:
            print(f"[ERROR] Failed to download checkpoint for {crop_id}: {e}")
            return False

    print("\n" + "=" * 70)
    print("ALL MODELS AND CHECKPOINTS PRE-DOWNLOADED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = download_all()
    sys.exit(0 if success else 1)

