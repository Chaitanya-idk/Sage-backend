from fastapi import Security, HTTPException, status, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import io
from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("sage.security")

# HTTP Bearer scheme with auto_error=False to allow custom error handling & optional state
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)) -> bool:
    """Verify API Key if configured in environment.
    
    If SAGE_API_KEY is configured:
        Requires 'Authorization: Bearer <SAGE_API_KEY>'.
    If not configured:
        Allows all requests for development while logging a warning.
    """
    if not settings.is_api_key_enabled():
        return True

    expected_key = settings.sage_api_key.strip()

    if not credentials or not credentials.credentials:
        logger.warning("Unauthorized access attempt: Missing Bearer token in request header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required. Provide 'Authorization: Bearer <YOUR_API_KEY>' in header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_token = credentials.credentials.strip()
    if provided_token != expected_key:
        logger.warning("Unauthorized access attempt: Invalid Bearer token provided.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key provided.",
        )

    return True


async def validate_image_file(file: UploadFile) -> bytes:
    """Validate uploaded image file:
    1. Checks content type against allowed MIME types.
    2. Validates file size against MAX_IMAGE_SIZE_MB.
    3. Validates file content using PIL to ensure it is a non-corrupted readable image.
    
    Returns raw image bytes upon successful validation.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in request.",
        )

    # 1. Content-Type Header check
    content_type = (file.content_type or "").lower()
    if content_type not in [t.lower() for t in settings.allowed_image_types]:
        logger.warning("Rejected file upload with invalid content type: %s", content_type)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: '{content_type}'. Allowed types: {', '.join(settings.allowed_image_types)}",
        )

    # Read content
    try:
        content = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file stream: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file data.",
        )
    finally:
        await file.seek(0)

    # 2. File size validation
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )
    if len(content) > max_bytes:
        size_mb = round(len(content) / (1024 * 1024), 2)
        logger.warning("Image upload exceeded size limit (%s MB > %s MB)", size_mb, settings.max_image_size_mb)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded image is {size_mb} MB, exceeding the maximum permitted size of {settings.max_image_size_mb} MB.",
        )

    # 3. Content and corruption verification via PIL
    try:
        image_stream = io.BytesIO(content)
        with Image.open(image_stream) as img:
            img.verify()  # Fast structural verification without full decode
    except Exception as e:
        logger.warning("Corrupted or invalid image uploaded: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is corrupted or is not a valid recognized image format.",
        )

    return content
