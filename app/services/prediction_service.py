from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel

from app.core.logging import get_logger
from app.models.crop_registry import crop_registry, CropConfig
from app.models.model_manager import get_model_manager, ModelManager
from app.inference.predictor import get_predictor, Predictor, PredictionResult

logger = get_logger("sage.prediction_service")


class BatchItemResult(BaseModel):
    """Result for an individual image within a batch request."""

    filename: str
    success: bool
    prediction: Optional[Dict[str, Any]] = None
    raw_output: Optional[str] = None
    inference_time_seconds: Optional[float] = None
    error: Optional[str] = None


class BatchPredictionSummary(BaseModel):
    """Aggregated batch prediction response."""

    total_images: int
    successful_count: int
    failed_count: int
    crop: str
    total_time_seconds: float
    results: List[BatchItemResult]


class PredictionService:
    """Service layer coordinating crop registry, model manager, and predictor execution."""

    def __init__(self, predictor: Optional[Predictor] = None):
        self.predictor = predictor or get_predictor()
        self.model_manager = get_model_manager()

    def get_crops_overview(self) -> Dict[str, Any]:
        """Retrieve list of crops and their metadata."""
        crops = crop_registry.get_all_crops()
        crop_list = []
        for cid, cfg in crops.items():
            crop_list.append({
                "crop_id": cfg.crop_id,
                "display_name": cfg.display_name,
                "hf_repo": cfg.hf_repo,
                "checkpoint_subpath": cfg.checkpoint_subpath,
                "model_type": cfg.model_type,
                "adapter_name": cfg.adapter_name,
                "loading_status": cfg.loading_status,
                "classes_count": len(cfg.known_classes),
                "known_classes": cfg.known_classes,
            })
        return {
            "total_crops": len(crop_list),
            "available_crops": [c["crop_id"] for c in crop_list],
            "crop_details": crop_list,
        }

    def get_status_overview(self) -> Dict[str, Any]:
        """Retrieve operational and hardware status."""
        status = self.model_manager.get_model_status()
        return status.model_dump()

    def predict_single(
        self,
        crop: str,
        image_bytes: bytes,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> PredictionResult:
        """Run single image diagnosis."""
        return self.predictor.predict(
            crop_id=crop,
            image_bytes=image_bytes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    def predict_batch(
        self,
        crop: str,
        images: List[Tuple[str, bytes]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> BatchPredictionSummary:
        """Run sequential batch diagnosis across multiple images for one crop.
        Sequential execution is strictly used to prevent GPU memory spikes (OOM).
        """
        import time

        batch_start = time.perf_counter()
        results: List[BatchItemResult] = []
        success_count = 0
        failure_count = 0

        # Validate crop once up front
        crop_registry.validate_crop(crop)

        for filename, img_bytes in images:
            try:
                res = self.predictor.predict(
                    crop_id=crop,
                    image_bytes=img_bytes,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                success_count += 1
                results.append(
                    BatchItemResult(
                        filename=filename,
                        success=True,
                        prediction=res.prediction.model_dump(),
                        raw_output=res.raw_output,
                        inference_time_seconds=res.inference_time_seconds,
                    )
                )
            except Exception as e:
                logger.error("Error processing image '%s' in batch: %s", filename, str(e))
                failure_count += 1
                results.append(
                    BatchItemResult(
                        filename=filename,
                        success=False,
                        error=str(e),
                    )
                )

        total_time = round(time.perf_counter() - batch_start, 4)

        return BatchPredictionSummary(
            total_images=len(images),
            successful_count=success_count,
            failed_count=failure_count,
            crop=crop.lower().strip(),
            total_time_seconds=total_time,
            results=results,
        )


# Global singleton service
_service_instance: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Retrieve global PredictionService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance

