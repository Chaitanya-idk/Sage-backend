import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.models.model_manager import get_model_manager
from app.models.crop_registry import crop_registry
from app.api.health import router as health_router
from app.api.routes import router as api_router

# Initialize structured logging
setup_logging(settings.log_level)
logger = get_logger("sage.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown procedures."""
    logger.info("Initializing SAGE Vision AI Inference Backend...")
    logger.info("Configuration: Base Model=%s, Repo=%s, Mode=%s", settings.base_model, settings.adapter_repo, settings.adapter_mode)

    if settings.is_api_key_enabled():
        logger.info("Security: API Key authentication is ENABLED.")
    else:
        logger.warning("Security Warning: SAGE_API_KEY is not set. API endpoints are public!")

    # Pre-initialize ModelManager
    manager = get_model_manager()
    try:
        if not settings.mock_inference:
            logger.info("Lifespan startup: Loading base model into memory...")
            manager.load_base_model()
            # Optionally preload default crop adapter if available
            default_crops = crop_registry.list_crop_ids()
            if default_crops:
                first_crop = crop_registry.get_crop(default_crops[0])
                if first_crop:
                    logger.info("Preloading initial crop adapter for '%s'...", first_crop.crop_id)
                    try:
                        manager.load_adapter(first_crop)
                    except Exception as e:
                        logger.warning("Could not preload initial adapter '%s': %s", first_crop.crop_id, e)
        else:
            logger.info("MOCK_INFERENCE=true: Bypassing heavy weights download for local testing.")
    except Exception as e:
        logger.error("Startup error during model initialization: %s", str(e), exc_info=True)

    yield

    logger.info("Shutting down SAGE backend. Cleaning up model resources...")
    try:
        manager.unload_adapter()
    except Exception as e:
        logger.warning("Shutdown cleanup warning: %s", str(e))
    logger.info("Shutdown completed.")


def create_application() -> FastAPI:
    """FastAPI Application Factory."""
    app = FastAPI(
        title="SAGE – Multi-Crop Vision AI Inference API",
        description=(
            "Autonomous Visual AI Agent for Precision Agriculture powered by fine-tuned "
            "Qwen2.5-VL-3B-Instruct PEFT/LoRA multi-crop disease diagnostics."
        ),
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Request Timing & Logging Middleware
    @app.middleware("http")
    async def log_and_time_request(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(process_time)
        logger.info(
            "%s %s - %d (%s ms)",
            request.method,
            request.url.path,
            response.status_code,
            process_time,
        )
        return response

    # 3. Global Exception Handlers (Safe, no leaked tokens or internal traces)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Request validation failed", "errors": exc.errors()},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("Value error encountered: %s", str(exc))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "error_type": "ValueError"},
        )

    @app.exception_handler(Exception)
    async def global_generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception processing %s: %s", request.url.path, str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred while processing the request.",
                "error_type": exc.__class__.__name__,
            },
        )

    # 4. Include Routers
    app.include_router(health_router)
    app.include_router(api_router)

    return app


# Export application instance
app = create_application()

