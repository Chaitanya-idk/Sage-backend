import os
import sys
import json
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.models.crop_registry import crop_registry
from app.core.logging import get_logger

logger = get_logger("sage.verify")


def verify_checkpoints():
    """Verify Hugging Face repository accessibility, checkpoint paths, and adapter files."""
    print("=" * 70)
    print("SAGE Multi-Crop Checkpoint Verification Utility")
    print("=" * 70)

    token = settings.hf_token
    if token:
        masked_token = token[:6] + "..." + token[-4:] if len(token) > 10 else "***"
        print(f"HF_TOKEN detected: {masked_token}")
    else:
        print("HF_TOKEN: Not provided (public repository access only).")

    api = HfApi(token=token)

    repo_id = settings.adapter_repo
    print(f"\nChecking Hugging Face repository: {repo_id}")

    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Repository accessible! Last modified: {repo_info.last_modified}")
        print(f"Private: {repo_info.private}")
    except RepositoryNotFoundError:
        print(f"[ERROR] Repository '{repo_id}' not found. Please verify HF repo ID and token permissions.")
        return False
    except HfHubHTTPError as e:
        print(f"[ERROR] Authentication / HTTP error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Connection error: {e}")
        return False

    # Get repository files tree
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"[ERROR] Could not list repository files: {e}")
        return False

    crops = crop_registry.get_all_crops()
    all_passed = True

    print("\nVerifying crop checkpoints:")
    print("-" * 70)

    for crop_id, crop_cfg in crops.items():
        subpath = crop_cfg.checkpoint_subpath.strip("/")
        print(f"\n[Crop: {crop_cfg.display_name.upper()} ({crop_id})]")
        print(f"  Target subpath: {subpath}")

        # Check for files in this subpath
        matching_files = [f for f in repo_files if f.startswith(subpath)]

        if not matching_files:
            print(f"  [FAIL] No files found for subpath: {subpath}")
            all_passed = False
            continue

        basenames = [f.split("/")[-1] for f in matching_files]
        print(f"  Files detected ({len(basenames)}): {', '.join(basenames)}")

        # Validate adapter_config.json
        has_adapter_config = "adapter_config.json" in basenames
        has_safetensors = any(f.endswith(".safetensors") for f in basenames)

        if has_adapter_config:
            print("  [OK] adapter_config.json present.")
        else:
            print("  [WARN] adapter_config.json is missing.")
            all_passed = False

        if has_safetensors:
            print("  [OK] SafeTensors weights present.")
        else:
            print("  [WARN] No .safetensors file found.")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKPOINTS VERIFIED SUCCESSFULLY! READY FOR INFERENCE.")
    else:
        print("VERIFICATION FOUND WARNINGS OR MISSING FILES.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = verify_checkpoints()
    sys.exit(0 if success else 1)

