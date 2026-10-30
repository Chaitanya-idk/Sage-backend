import os
import sys
import io
import unittest
from PIL import Image

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.models.crop_registry import crop_registry
from app.models.model_manager import get_model_manager
from app.inference.predictor import Predictor
from app.inference.image_processor import ImageProcessor
from app.services.prediction_service import PredictionService
from app.api.schemas import PredictionResponse, PredictionInputInfo, PredictionOutputInfo


class TestMultimodalInferencePipeline(unittest.TestCase):
    """Unit tests for Multimodal Image + Text Inference Pipeline.
    
    Verifies all 10 required test criteria:
    1. Missing crop.
    2. Invalid crop.
    3. Missing image.
    4. Missing prompt (default prompt selected).
    5. Custom prompt acceptance.
    6. Correct crop-to-adapter mapping.
    7. Image preprocessing.
    8. Multimodal message construction.
    9. Mock inference response.
    10. API response schema.
    """

    def setUp(self):
        settings.mock_inference = True
        self.manager = get_model_manager()
        self.manager.mock_mode = True
        self.predictor = Predictor(model_manager=self.manager)
        self.service = PredictionService(predictor=self.predictor)

        # Generate sample in-memory JPEG image
        img = Image.new("RGB", (320, 240), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        self.sample_image_bytes = buf.getvalue()

    # 1. Missing crop
    def test_01_missing_crop(self):
        """Test that empty or None crop raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.service.predict_single(
                crop="",
                image_bytes=self.sample_image_bytes,
            )
        self.assertIn("not supported", str(ctx.exception))

    # 2. Invalid crop
    def test_02_invalid_crop(self):
        """Test that unmapped crop identifier raises descriptive ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.service.predict_single(
                crop="banana",
                image_bytes=self.sample_image_bytes,
            )
        self.assertIn("Crop 'banana' is not supported", str(ctx.exception))

    # 3. Missing image
    def test_03_missing_image(self):
        """Test that empty image bytes raise ValueError during preprocessing."""
        with self.assertRaises(ValueError) as ctx:
            self.service.predict_single(
                crop="tomato",
                image_bytes=b"",
            )
        self.assertIn("Failed to process image", str(ctx.exception))

    # 4. Missing prompt (default prompt should be selected)
    def test_04_missing_prompt_selects_default(self):
        """Test that when prompt is omitted or None, the crop's default prompt is used."""
        result = self.service.predict_single(
            crop="tomato",
            image_bytes=self.sample_image_bytes,
            prompt=None,
        )
        expected_default = crop_registry.get_default_prompt("tomato")
        self.assertEqual(result.input.prompt, expected_default)
        self.assertIn("tomato leaf image", result.input.prompt.lower())

    # 5. Custom prompt acceptance
    def test_05_custom_prompt_acceptance(self):
        """Test that custom prompt is preserved and passed to the input info."""
        custom_query = "Assess the visible severity of Early Blight in this leaf."
        result = self.service.predict_single(
            crop="tomato",
            image_bytes=self.sample_image_bytes,
            prompt=custom_query,
        )
        self.assertEqual(result.input.prompt, custom_query)

    # 6. Correct crop-to-adapter mapping
    def test_06_crop_to_adapter_mapping(self):
        """Verify all 5 crops correctly map to their target checkpoint subpaths."""
        mappings = {
            "corn": "corn_5ep/checkpoints/checkpoint-step-1000",
            "rice": "rice/checkpoints/checkpoint-step-600",
            "soybean": "soybean_5ep/checkpoints/checkpoint-step-1000",
            "tomato": "tomato/checkpoints/checkpoint-step-600",
            "wheat": "wheat/checkpoints/checkpoint-step-500",
        }
        for crop_id, expected_subpath in mappings.items():
            crop_cfg = crop_registry.validate_crop(crop_id)
            self.assertEqual(crop_cfg.checkpoint_subpath, expected_subpath)
            self.assertEqual(crop_cfg.adapter_name, f"{crop_id}_lora")

    # 7. Image preprocessing
    def test_07_image_preprocessing(self):
        """Test PIL image normalization to RGB and EXIF correction."""
        pil_img, meta = ImageProcessor.load_and_preprocess(self.sample_image_bytes)
        self.assertEqual(meta.channels, 3)
        self.assertEqual(meta.width, 320)
        self.assertEqual(meta.height, 240)
        self.assertEqual(pil_img.mode, "RGB")

    # 8. Multimodal message construction
    def test_08_multimodal_message_construction(self):
        """Test Qwen2.5-VL conversation message structure containing both image and prompt."""
        pil_img, _ = ImageProcessor.load_and_preprocess(self.sample_image_bytes)
        prompt_text = "Identify any visible pathogens."
        messages = self.predictor.build_prompt_messages(prompt=prompt_text, image=pil_img)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["image"], pil_img)
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(content[1]["text"], prompt_text)

    # 9. Mock inference response
    def test_09_mock_inference_response(self):
        """Test that mock inference generates realistic multi-field output without GPU."""
        result = self.service.predict_single(
            crop="rice",
            image_bytes=self.sample_image_bytes,
            prompt="Explain the visible blast lesions.",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.crop, "rice")
        self.assertEqual(result.base_model, settings.base_model)
        self.assertIn("rice", result.checkpoint.lower())
        self.assertIsNotNone(result.prediction.response)
        self.assertGreater(result.inference_time_seconds, 0)

    # 10. API response schema
    def test_10_api_response_schema(self):
        """Validate response matches the structured SAGE API contract."""
        result = self.service.predict_single(
            crop="wheat",
            image_bytes=self.sample_image_bytes,
        )
        # Validate serialization into PredictionResponse Pydantic schema
        response_model = PredictionResponse(**result.model_dump())
        self.assertTrue(response_model.success)
        self.assertEqual(response_model.crop, "wheat")
        self.assertEqual(response_model.base_model, settings.base_model)
        self.assertEqual(response_model.active_adapter, "wheat")
        self.assertIn("prompt", response_model.input.model_dump())
        self.assertIn("response", response_model.prediction.model_dump())
        self.assertIsNotNone(response_model.raw_output)


if __name__ == "__main__":
    unittest.main()
