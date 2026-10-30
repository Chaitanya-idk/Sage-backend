import os
import sys
import unittest

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.crop_registry import CropRegistry, CropConfig


class TestCropRegistry(unittest.TestCase):
    """Test Crop Registry mapping, validation, and loading status tracking."""

    def setUp(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "crops.yaml")
        self.registry = CropRegistry(config_path=config_path)

    def test_registered_crops_count(self):
        """Ensure all 5 required crops are registered."""
        crops = self.registry.list_crop_ids()
        self.assertEqual(len(crops), 5)
        self.assertIn("corn", crops)
        self.assertIn("rice", crops)
        self.assertIn("soybean", crops)
        self.assertIn("tomato", crops)
        self.assertIn("wheat", crops)

    def test_exact_checkpoint_subpaths(self):
        """Verify exact checkpoint subpath mappings requested in specification."""
        expected_mappings = {
            "corn": "corn_5ep/checkpoints/checkpoint-step-1000",
            "rice": "rice/checkpoints/checkpoint-step-600",
            "soybean": "soybean_5ep/checkpoints/checkpoint-step-1000",
            "tomato": "tomato/checkpoints/checkpoint-step-600",
            "wheat": "wheat/checkpoints/checkpoint-step-500",
        }

        for crop_id, expected_subpath in expected_mappings.items():
            crop = self.registry.get_crop(crop_id)
            self.assertIsNotNone(crop, f"Crop '{crop_id}' should exist in registry.")
            self.assertEqual(
                crop.checkpoint_subpath,
                expected_subpath,
                f"Mismatch for {crop_id}: expected {expected_subpath}, got {crop.checkpoint_subpath}",
            )

    def test_crop_validation_valid(self):
        """Test valid crop validation case insensitivity."""
        c = self.registry.validate_crop("TOMATO")
        self.assertEqual(c.crop_id, "tomato")
        self.assertEqual(c.display_name, "Tomato")

    def test_crop_validation_invalid(self):
        """Test invalid crop raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.registry.validate_crop("avocado")
        self.assertIn("not supported", str(ctx.exception))

    def test_loading_status_updates(self):
        """Test loading status lifecycle transitions."""
        crop_id = "wheat"
        self.registry.update_loading_status(crop_id, "loading")
        self.assertEqual(self.registry.get_crop(crop_id).loading_status, "loading")

        self.registry.update_loading_status(crop_id, "active")
        self.assertEqual(self.registry.get_crop(crop_id).loading_status, "active")


if __name__ == "__main__":
    unittest.main()
