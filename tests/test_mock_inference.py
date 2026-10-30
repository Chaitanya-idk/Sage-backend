import os
import sys
import io
import unittest
from PIL import Image

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.models.model_manager import get_model_manager
from app.inference.predictor import Predictor
from app.inference.image_processor import ImageProcessor
from app.services.prediction_service import PredictionService


class TestMockInference(unittest.TestCase):
    """End-to-end mock inference testing without requiring heavy ML libraries or GPU."""

    def setUp(self):
        # Force mock inference mode
        settings.mock_inference = True
        self.manager = get_model_manager()
        self.manager.mock_mode = True
        self.predictor = Predictor(model_manager=self.manager)
        self.service = PredictionService(predictor=self.predictor)

        # Create dummy RGB image in memory
        img = Image.new("RGB", (256, 256), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        self.sample_image_bytes = buf.getvalue()

    def test_image_processor(self):
        """Test image preprocessing normalization."""
        pil_img, meta = ImageProcessor.load_and_preprocess(self.sample_image_bytes)
        self.assertEqual(meta.channels, 3)
        self.assertEqual(meta.width, 256)
        self.assertEqual(meta.height, 256)
        self.assertEqual(meta.original_format, "JPEG")

    def test_single_mock_prediction(self):
        """Test single crop prediction pipeline."""
        result = self.service.predict_single(
            crop="tomato",
            image_bytes=self.sample_image_bytes,
            prompt="Identify disease and explain symptoms.",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.crop, "tomato")
        self.assertEqual(result.base_model, settings.base_model)
        self.assertEqual(result.input.prompt, "Identify disease and explain symptoms.")
        self.assertIsNotNone(result.prediction.response)
        self.assertGreater(result.inference_time_seconds, 0)

    def test_batch_mock_prediction(self):
        """Test sequential batch prediction across multiple sample images."""
        batch = [
            ("leaf_01.jpg", self.sample_image_bytes),
            ("leaf_02.jpg", self.sample_image_bytes),
        ]
        summary = self.service.predict_batch(
            crop="rice",
            images=batch,
            prompt="Detect any leaf blast or blight.",
        )
        self.assertEqual(summary.total_images, 2)
        self.assertEqual(summary.successful_count, 2)
        self.assertEqual(summary.failed_count, 0)
        self.assertEqual(summary.crop, "rice")
        self.assertEqual(len(summary.results), 2)
        self.assertTrue(summary.results[0].success)
        self.assertEqual(summary.results[0].input.get("prompt"), "Detect any leaf blast or blight.")

    def test_invalid_crop_rejection(self):
        """Ensure unmapped crop yields descriptive ValueError."""
        with self.assertRaises(ValueError):
            self.service.predict_single(
                crop="unsupported_crop_name",
                image_bytes=self.sample_image_bytes,
            )


if __name__ == "__main__":
    unittest.main()
