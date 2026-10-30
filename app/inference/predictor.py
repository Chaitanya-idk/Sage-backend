import time
import threading
from typing import Optional, Dict, Any
from pydantic import BaseModel
from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger
from app.models.crop_registry import crop_registry, CropConfig
from app.models.model_manager import get_model_manager, ModelManager
from app.inference.image_processor import ImageProcessor, ProcessedImage
from app.inference.response_parser import ResponseParser, ParsedPrediction

logger = get_logger("sage.predictor")


class PredictionResult(BaseModel):
    """Complete prediction output contract."""

    success: bool
    crop: str
    model: str
    checkpoint: str
    prediction: ParsedPrediction
    raw_output: str
    inference_time_seconds: float
    image_metadata: Dict[str, Any]


class Predictor:
    """Predictor orchestrating preprocessing, prompt building, Qwen2.5-VL inference, and parsing."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.manager = model_manager or get_model_manager()
        self._inference_lock = threading.Lock()

    def build_prompt_messages(self, crop_config: CropConfig, image: Image.Image):
        """Construct Qwen2.5-VL conversation format message list."""
        user_prompt = crop_config.prompt_template
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

    def predict(
        self,
        crop_id: str,
        image_bytes: bytes,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> PredictionResult:
        """Execute end-to-end inference pipeline for an input image and crop."""
        start_time = time.perf_counter()

        # 1. Validate Crop
        crop_config = crop_registry.validate_crop(crop_id)

        # 2. Validate and Preprocess Image
        rgb_image, img_meta = ImageProcessor.load_and_preprocess(image_bytes)

        # Handle Mock Mode for CPU / Local CI testing
        if self.manager.mock_mode:
            logger.info("MOCK_INFERENCE: Executing mock prediction for crop '%s'.", crop_id)
            mock_class = crop_config.known_classes[0] if crop_config.known_classes else "Sample_Condition"
            mock_output = (
                f"Crop: {crop_config.display_name}\n"
                f"Diagnosis: {mock_class}\n"
                f"Disease type: Fungal\n"
                f"Plant organ: Foliar\n"
                f"Visual symptoms: Irregular lesions with chlorotic halos visible across leaf surface.\n"
                f"Pathogen: Exemplar pathogen sp."
            )
            parsed = ResponseParser.parse(mock_output, crop_config.display_name, crop_config.known_classes)
            elapsed = round(time.perf_counter() - start_time, 4)

            return PredictionResult(
                success=True,
                crop=crop_config.crop_id,
                model=self.manager.base_model_name,
                checkpoint=crop_config.checkpoint_subpath,
                prediction=parsed,
                raw_output=mock_output,
                inference_time_seconds=elapsed,
                image_metadata=img_meta.model_dump(),
            )

        # 3. Synchronize adapter and inference
        with self._inference_lock:
            # Switch or attach crop adapter
            self.manager.switch_adapter(crop_config)

            model = self.manager.model
            processor = self.manager.processor

            if model is None or processor is None:
                raise RuntimeError("Model or processor is not initialized. Please ensure base model is loaded.")

            # 4. Format Prompt and Messages
            messages = self.build_prompt_messages(crop_config, rgb_image)

            # 5. Process Inputs via AutoProcessor
            import torch

            # Qwen2.5-VL uses apply_chat_template
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision and text
            # qwen_vl_utils can be used if present, otherwise standard AutoProcessor
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except ImportError:
                # Direct processor fallback
                inputs = processor(
                    text=[text_prompt],
                    images=[rgb_image],
                    return_tensors="pt",
                )

            # Move inputs to target device
            target_device = next(model.parameters()).device
            inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            # 6. Generation Parameters
            gen_max_tokens = max_new_tokens or settings.max_new_tokens
            gen_temp = temperature if temperature is not None else settings.temperature
            gen_top_p = top_p if top_p is not None else settings.top_p
            gen_rep_penalty = repetition_penalty if repetition_penalty is not None else settings.repetition_penalty

            do_sample = gen_temp > 0.0

            # 7. Generate Tokens
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_max_tokens,
                    do_sample=do_sample,
                    temperature=gen_temp if do_sample else None,
                    top_p=gen_top_p if do_sample else None,
                    repetition_penalty=gen_rep_penalty,
                )

            # 8. Trim input tokens and decode
            input_ids = inputs["input_ids"]
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            decoded_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

            # 9. Parse Output
            parsed = ResponseParser.parse(
                decoded_text, crop_config.display_name, crop_config.known_classes
            )

            elapsed = round(time.perf_counter() - start_time, 4)

            return PredictionResult(
                success=True,
                crop=crop_config.crop_id,
                model=self.manager.base_model_name,
                checkpoint=crop_config.checkpoint_subpath,
                prediction=parsed,
                raw_output=decoded_text,
                inference_time_seconds=elapsed,
                image_metadata=img_meta.model_dump(),
            )


# Global singleton predictor
_predictor_instance: Optional[Predictor] = None


def get_predictor() -> Predictor:
    """Retrieve global Predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance

