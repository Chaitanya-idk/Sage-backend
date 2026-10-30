import io
from typing import Tuple
from PIL import Image, ImageOps
from pydantic import BaseModel

from app.core.logging import get_logger

logger = get_logger("sage.image_processor")


class ProcessedImage(BaseModel):
    """Container for processed image and metadata."""

    width: int
    height: int
    original_format: str
    channels: int = 3

    class Config:
        arbitrary_types_allowed = True


class ImageProcessor:
    """Handles image validation, color space normalization, and EXIF orientation correction.
    
    Adheres to Qwen2.5-VL specifications:
    Avoids arbitrary manual downscaling so Qwen's vision processor can handle native resolutions
    and dynamic visual token allocation.
    """

    @classmethod
    def load_and_preprocess(cls, image_bytes: bytes) -> Tuple[Image.Image, ProcessedImage]:
        """Loads raw bytes, validates image format, applies EXIF orientation, and converts to RGB.
        
        Returns:
            Tuple[PIL.Image.Image, ProcessedImage]
        """
        try:
            raw_image = Image.open(io.BytesIO(image_bytes))
            orig_format = raw_image.format or "JPEG"

            # 1. EXIF orientation correction (handles mobile camera orientations)
            oriented_image = ImageOps.exif_transpose(raw_image)

            # 2. Color space normalization to RGB (drops alpha channels, handles CMYK/Luminance)
            if oriented_image.mode != "RGB":
                rgb_image = oriented_image.convert("RGB")
            else:
                rgb_image = oriented_image

            width, height = rgb_image.size

            metadata = ProcessedImage(
                width=width,
                height=height,
                original_format=orig_format,
                channels=3,
            )

            return rgb_image, metadata

        except Exception as e:
            logger.error("Image preprocessing error: %s", str(e))
            raise ValueError(f"Failed to process image: {e}") from e
