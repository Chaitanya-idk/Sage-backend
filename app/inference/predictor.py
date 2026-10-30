import time
import threading
import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel
from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger
from app.models.crop_registry import crop_registry, CropConfig
from app.models.model_manager import get_model_manager, ModelManager
from app.inference.image_processor import ImageProcessor, ProcessedImage
from app.inference.response_parser import ResponseParser, ParsedPrediction
from app.api.schemas import PredictionInputInfo, PredictionOutputInfo

logger = get_logger("sage.predictor")


class PredictionResult(BaseModel):
    """Complete prediction output contract adhering to SAGE multimodal schema."""

    success: bool
    crop: str
    base_model: str
    active_adapter: str
    checkpoint: str
    input: PredictionInputInfo
    prediction: PredictionOutputInfo
    raw_output: str
    inference_time_seconds: float
    image_metadata: Dict[str, Any]


class Predictor:
    """Predictor orchestrating multimodal preprocessing, prompt construction,
    Qwen2.5-VL generation, and structured response parsing.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.manager = model_manager or get_model_manager()
        self._thread_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def build_prompt_messages(self, prompt: str, image: Image.Image):
        """Construct Qwen2.5-VL multimodal conversation message structure."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def resolve_prompt(self, crop_config: CropConfig, user_prompt: Optional[str]) -> str:
        """Resolve effective prompt: returns custom user prompt if provided,
        otherwise falls back to the configured crop-specific default prompt.
        """
        if user_prompt and user_prompt.strip():
            return user_prompt.strip()
        return crop_config.default_prompt

    def predict(
        self,
        crop_id: str,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> PredictionResult:
        """Execute synchronous end-to-end multimodal inference pipeline."""
        start_time = time.perf_counter()

        # 1. Validate Crop
        crop_config = crop_registry.validate_crop(crop_id)

        # 2. Resolve Effective Prompt (Default vs. Custom)
        effective_prompt = self.resolve_prompt(crop_config, prompt)

        # 3. Validate and Preprocess Image
        rgb_image, img_meta = ImageProcessor.load_and_preprocess(image_bytes)

        # 4. Handle Mock Mode (for Local CI / Test execution without GPU)
        if self.manager.mock_mode:
            logger.info("MOCK_INFERENCE: Executing mock multimodal prediction for crop '%s'.", crop_id)
            mock_class = crop_config.known_classes[0] if crop_config.known_classes else "Sample_Condition"
            mock_output = (
                f"Crop: {crop_config.display_name}\n"
                f"Diagnosis: {mock_class}\n"
                f"Disease type: Fungal\n"
                f"Plant organ: Foliar\n"
                f"Visual symptoms: Observed irregular necrotic lesions consistent with {mock_class} as queried in prompt.\n"
                f"Pathogen: Causative pathogen sp."
            )
            parsed = ResponseParser.parse(mock_output, crop_config.display_name, crop_config.known_classes)
            elapsed = round(time.perf_counter() - start_time, 4)

            return PredictionResult(
                success=True,
                crop=crop_config.crop_id,
                base_model=self.manager.base_model_name,
                active_adapter=crop_config.crop_id,
                checkpoint=crop_config.checkpoint_subpath,
                input=PredictionInputInfo(prompt=effective_prompt),
                prediction=PredictionOutputInfo(
                    response=mock_output,
                    class_name=parsed.class_name,
                    crop=parsed.crop,
                    disease_type=parsed.disease_type,
                    plant_organ=parsed.plant_organ,
                    visual_symptoms=parsed.visual_symptoms,
                    pathogen=parsed.pathogen,
                    matched_taxonomy_class=parsed.matched_taxonomy_class,
                    is_canonical_match=parsed.is_canonical_match,
                    attributes=parsed.attributes,
                ),
                raw_output=mock_output,
                inference_time_seconds=elapsed,
                image_metadata=img_meta.model_dump(),
            )

        # 5. Thread-safe adapter switching & inference execution
        with self._thread_lock:
            # Switch or activate crop-specific PEFT LoRA adapter
            self.manager.switch_adapter(crop_config)

            model = self.manager.model
            processor = self.manager.processor

            if model is None or processor is None:
                raise RuntimeError("Model or processor is not initialized. Please ensure base model is loaded.")

            # Construct multimodal conversation message structure
            messages = self.build_prompt_messages(effective_prompt, rgb_image)

            # Process vision and text inputs
            import torch

            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

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
                # Direct AutoProcessor fallback
                inputs = processor(
                    text=[text_prompt],
                    images=[rgb_image],
                    return_tensors="pt",
                )

            # Move tensors to target device
            target_device = next(model.parameters()).device
            inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            gen_max_tokens = max_new_tokens or settings.max_new_tokens
            gen_temp = temperature if temperature is not None else settings.temperature
            gen_top_p = top_p if top_p is not None else settings.top_p
            gen_rep_penalty = repetition_penalty if repetition_penalty is not None else settings.repetition_penalty
            do_sample = gen_temp > 0.0

            # Generate tokens (inference only)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_max_tokens,
                    do_sample=do_sample,
                    temperature=gen_temp if do_sample else None,
                    top_p=gen_top_p if do_sample else None,
                    repetition_penalty=gen_rep_penalty,
                )

            # Slice only the newly generated tokens
            input_ids = inputs["input_ids"]
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            decoded_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

            parsed = ResponseParser.parse(
                decoded_text, crop_config.display_name, crop_config.known_classes
            )

            elapsed = round(time.perf_counter() - start_time, 4)

            return PredictionResult(
                success=True,
                crop=crop_config.crop_id,
                base_model=self.manager.base_model_name,
                active_adapter=crop_config.crop_id,
                checkpoint=crop_config.checkpoint_subpath,
                input=PredictionInputInfo(prompt=effective_prompt),
                prediction=PredictionOutputInfo(
                    response=decoded_text,
                    class_name=parsed.class_name,
                    crop=parsed.crop,
                    disease_type=parsed.disease_type,
                    plant_organ=parsed.plant_organ,
                    visual_symptoms=parsed.visual_symptoms,
                    pathogen=parsed.pathogen,
                    matched_taxonomy_class=parsed.matched_taxonomy_class,
                    is_canonical_match=parsed.is_canonical_match,
                    attributes=parsed.attributes,
                ),
                raw_output=decoded_text,
                inference_time_seconds=elapsed,
                image_metadata=img_meta.model_dump(),
            )

    async def predict_async(
        self,
        crop_id: str,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> PredictionResult:
        """Execute multimodal prediction with async concurrency lock."""
        async with self._async_lock:
            return await asyncio.to_thread(
                self.predict,
                crop_id=crop_id,
                image_bytes=image_bytes,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )


# Global singleton predictor
_predictor_instance: Optional[Predictor] = None


def get_predictor() -> Predictor:
    """Retrieve global Predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance
