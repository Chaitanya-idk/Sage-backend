import os
import sys
import unittest

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.schemas import (
    HealthResponse,
    CropListResponse,
    ModelStatusResponse,
    PredictionResponse,
    PredictionDiagnosis,
)
from app.inference.response_parser import ResponseParser


class TestSchemasAndParser(unittest.TestCase):
    """Test schema serialization and response parsing logic."""

    def test_health_response_schema(self):
        """Test HealthResponse serialization."""
        data = {
            "status": "healthy",
            "service": "SAGE API",
            "device": "cuda",
            "base_model_loaded": True,
            "active_crop": "tomato",
        }
        res = HealthResponse(**data)
        self.assertEqual(res.status, "healthy")
        self.assertEqual(res.device, "cuda")
        self.assertTrue(res.base_model_loaded)

    def test_response_parser_canonical_output(self):
        """Test parsing of model output adhering to SAGE taxonomy format."""
        raw_text = (
            "Crop: Tomato\n"
            "Diagnosis: Early Blight\n"
            "Disease type: Fungal\n"
            "Plant organ: Foliar\n"
            "Visual symptoms: Concentric dark brown rings with target-like appearance.\n"
            "Pathogen: Alternaria solani"
        )
        known = ["Early_Blight", "Late_Blight", "Healthy"]
        parsed = ResponseParser.parse(raw_text, crop_name="Tomato", known_classes=known)

        self.assertEqual(parsed.class_name, "Early Blight")
        self.assertEqual(parsed.crop, "Tomato")
        self.assertEqual(parsed.disease_type, "Fungal")
        self.assertEqual(parsed.plant_organ, "Foliar")
        self.assertEqual(parsed.pathogen, "Alternaria solani")
        self.assertEqual(parsed.matched_taxonomy_class, "Early_Blight")
        self.assertTrue(parsed.is_canonical_match)

    def test_response_parser_unstructured_fallback(self):
        """Test parser behavior on plain text output without explicit keys."""
        raw_text = "Powdery Mildew visible on the leaf surface."
        parsed = ResponseParser.parse(raw_text, crop_name="Wheat")
        self.assertEqual(parsed.class_name, "Powdery Mildew visible on the leaf surface.")
        self.assertEqual(parsed.crop, "Wheat")
        self.assertFalse(parsed.is_canonical_match)

    def test_prediction_response_schema(self):
        """Test full PredictionResponse structure."""
        diag = PredictionDiagnosis(
            class_name="Early Blight",
            crop="tomato",
            disease_type="Fungal",
            matched_taxonomy_class="Early_Blight",
            is_canonical_match=True,
        )
        resp = PredictionResponse(
            success=True,
            crop="tomato",
            model="Qwen2.5-VL-3B-Instruct",
            checkpoint="tomato/checkpoints/checkpoint-step-600",
            prediction=diag,
            raw_output="Crop: Tomato\nDiagnosis: Early Blight",
            inference_time_seconds=1.23,
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.crop, "tomato")
        self.assertEqual(resp.prediction.class_name, "Early Blight")


if __name__ == "__main__":
    unittest.main()

