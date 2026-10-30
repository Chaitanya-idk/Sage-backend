from typing import List, Optional
from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    Form,
    Path,
    Query,
    HTTPException,
    status,
)

from app.core.security import verify_api_key, validate_image_file
from app.core.logging import get_logger
from app.api.schemas import (
    CropListResponse,
    ModelStatusResponse,
    PredictionResponse,
    BatchPredictionResponse,
    ErrorResponse,
)
from app.services.prediction_service import get_prediction_service, PredictionService

logger = get_logger("sage.routes")

router = APIRouter(prefix="/api/v1", tags=["Inference & Diagnostics"])


@router.get(
    "/crops",
    response_model=CropListResponse,
    summary="List Supported Crops",
    description="Retrieve all supported crops, Hugging Face checkpoint subpaths, default prompts, and current loading state.",
)
async def list_available_crops(
    service: PredictionService = Depends(get_prediction_service),
    _auth: bool = Depends(verify_api_key),
) -> CropListResponse:
    """Retrieve available crops and their checkpoint mappings."""
    overview = service.get_crops_overview()
    return CropListResponse(**overview)


@router.get(
    "/model/status",
    response_model=ModelStatusResponse,
    summary="Get Model & Hardware Status",
    description="Query active base model, loaded adapters, and GPU memory utilization.",
)
async def get_model_status(
    service: PredictionService = Depends(get_prediction_service),
    _auth: bool = Depends(verify_api_key),
) -> ModelStatusResponse:
    """Retrieve detailed model runtime status."""
    status_data = service.get_status_overview()
    return ModelStatusResponse(**status_data)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation or crop error"},
        413: {"model": ErrorResponse, "description": "Image payload too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media format"},
        422: {"model": ErrorResponse, "description": "Corrupted or unreadable image"},
    },
    summary="Multimodal Plant Disease Diagnosis (Primary Endpoint)",
    description=(
        "Execute multimodal visual AI diagnosis. Accepts a target crop name (corn, rice, soybean, tomato, wheat), "
        "a leaf/plant image, and an optional custom prompt or instruction. "
        "If prompt is omitted, a fine-tuned default crop disease identification prompt is automatically applied."
    ),
)
async def predict_multimodal(
    crop: str = Form(..., description="Target crop identifier (corn, rice, soybean, tomato, wheat)"),
    image: UploadFile = File(..., description="Plant or leaf image file (JPEG, PNG, WebP)"),
    prompt: Optional[str] = Form(
        default=None,
        description="Optional prompt/instruction for Qwen2.5-VL. If omitted, uses default crop prompt.",
    ),
    max_new_tokens: Optional[int] = Form(default=None, description="Max new tokens to generate (default: 256)"),
    temperature: Optional[float] = Form(default=None, description="Generation temperature (default: 0.0 deterministic)"),
    service: PredictionService = Depends(get_prediction_service),
    _auth: bool = Depends(verify_api_key),
) -> PredictionResponse:
    """Primary multimodal inference endpoint accepting crop, image, and optional prompt."""
    image_bytes = await validate_image_file(image)

    try:
        result = await service.predict_single_async(
            crop=crop,
            image_bytes=image_bytes,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return PredictionResponse(**result.model_dump())
    except ValueError as e:
        logger.warning("Prediction bad request: %s", str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Internal inference error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@router.post(
    "/predict/{crop}",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation or crop error"},
        413: {"model": ErrorResponse, "description": "Image payload too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media format"},
        422: {"model": ErrorResponse, "description": "Corrupted or unreadable image"},
    },
    summary="Diagnose Plant Disease by Crop Path Parameter",
    description="Convenience route accepting the crop name in the URL path, with optional prompt as form parameter.",
)
async def predict_by_crop_path(
    crop: str = Path(..., description="Target crop identifier (corn, rice, soybean, tomato, wheat)"),
    image: UploadFile = File(..., description="Crop leaf or plant image file"),
    prompt: Optional[str] = Form(
        default=None,
        description="Optional custom prompt or instruction. If omitted, uses default crop prompt.",
    ),
    max_new_tokens: Optional[int] = Form(default=None, description="Max tokens to generate"),
    temperature: Optional[float] = Form(default=None, description="Generation temperature"),
    service: PredictionService = Depends(get_prediction_service),
    _auth: bool = Depends(verify_api_key),
) -> PredictionResponse:
    """Multimodal disease diagnosis with crop specified in URL path."""
    image_bytes = await validate_image_file(image)

    try:
        result = await service.predict_single_async(
            crop=crop,
            image_bytes=image_bytes,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return PredictionResponse(**result.model_dump())
    except ValueError as e:
        logger.warning("Prediction bad request: %s", str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Internal inference error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Sequential Batch Plant Disease Diagnosis",
    description="Submit multiple images for a single crop and optional prompt. Images are processed sequentially to prevent GPU OOM.",
)
async def predict_batch(
    crop: str = Form(..., description="Target crop identifier for this batch"),
    images: List[UploadFile] = File(..., description="List of crop images to diagnose"),
    prompt: Optional[str] = Form(
        default=None,
        description="Optional custom prompt applied across batch images. If omitted, uses default crop prompt.",
    ),
    max_new_tokens: Optional[int] = Form(default=None, description="Max tokens to generate"),
    temperature: Optional[float] = Form(default=None, description="Generation temperature"),
    service: PredictionService = Depends(get_prediction_service),
    _auth: bool = Depends(verify_api_key),
) -> BatchPredictionResponse:
    """Execute sequential batch diagnosis across multiple images for one crop."""
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image file must be uploaded for batch processing.",
        )

    # Validate and buffer each image
    buffered_images = []
    for img_file in images:
        raw_bytes = await validate_image_file(img_file)
        buffered_images.append((img_file.filename or "uploaded_image.jpg", raw_bytes))

    try:
        summary = service.predict_batch(
            crop=crop,
            images=buffered_images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return BatchPredictionResponse(**summary.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Internal batch inference error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch execution failed: {str(e)}",
        )
