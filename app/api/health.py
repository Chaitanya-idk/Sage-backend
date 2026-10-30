from fastapi import APIRouter
from app.api.schemas import HealthResponse
from app.models.model_manager import get_model_manager
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Check operational status, compute device, and base model loading state.",
)
async def get_health() -> HealthResponse:
    """Service health check endpoint."""
    manager = get_model_manager()
    status = manager.get_model_status()
    device = "mock" if status.mock_mode else ("cuda" if status.gpu_status.available else "cpu")

    return HealthResponse(
        status="healthy",
        service="SAGE API",
        device=device,
        base_model_loaded=status.base_model_loaded,
        active_crop=status.active_crop,
    )

