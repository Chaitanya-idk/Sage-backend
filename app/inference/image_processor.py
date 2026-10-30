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
    """Handles image normalization, RGB conversion, and EXIF orientation correction."""

    # Qwen2.5-VL handles dynamic resolutions up to 28x28 patches natively,
    # but capping extremely large images (> 4096px) prevents OOM and processing hangs.
    MAX_DIMENSION: int = 2048

    @classmethod
    def load_and_preprocess(cls, image_bytes: bytes) -> Tuple[Image.Image, ProcessedImage]:
        """Loads raw bytes, validates, applies EXIF orientation, and converts to RGB.
        
        Returns:
            Tuple[PIL.Image.Image, ProcessedImage]
        """
        try:
            raw_image = Image.open(io.BytesIO(image_bytes))
            orig_format = raw_image.format or "JPEG"

            # 1. EXIF orientation correction
            # Uses PIL.ImageOps.exif_transpose to automatically orient images taken on mobile devices
            oriented_image = ImageOps.exif_transpose(raw_image)

            # 2. Color space normalization to RGB (drop alpha channels, handle CMYK/Grayscale)
            if oriented_image.mode != "RGB":
                rgb_image = oriented_image.convert("RGB")
            else:
                rgb_image = oriented_image

            orig_w, orig_h = rgb_image.size

            # 3. Guard against overly large dimension extremes
            if orig_w > cls.MAX_DIMENSION or orig_h > cls.MAX_DIMENSION:
                scale = min(cls.MAX_DIMENSION / orig_w, cls.MAX_DIMENSION / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                logger.info("Resizing oversized image from (%d, %d) to (%d, %d)", orig_w, orig_h, new_w, new_h)
                rgb_image = rgb_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                w, h = new_w, new_h
            else:
                w, h = orig_w, orig_h

            metadata = ProcessedImage(
                width=w,
                height=h,
                original_format=orig_format,
                channels=3,
            )

            return rgb_image, metadata

        except Exception as e:
            logger.error("Image preprocessing error: %s", str(e))
            raise ValueError(f"Failed to process image: {e}") from e

