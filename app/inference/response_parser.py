import re
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("sage.response_parser")


class ParsedPrediction(BaseModel):
    """Structured result parsed directly from the model's generated response."""

    class_name: str = Field(description="Primary diagnosed disease name or condition")
    crop: Optional[str] = None
    disease_type: Optional[str] = None
    plant_organ: Optional[str] = None
    visual_symptoms: Optional[str] = None
    pathogen: Optional[str] = None
    matched_taxonomy_class: Optional[str] = None
    is_canonical_match: bool = False
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ResponseParser:
    """Parses Qwen2.5-VL generative outputs according to SAGE training schema."""

    # Standard field patterns emitted by fine-tuned SAGE model
    PATTERNS = {
        "crop": re.compile(r"^\s*Crop\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        "diagnosis": re.compile(r"^\s*Diagnosis\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        "disease_type": re.compile(r"^\s*Disease\s+type\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        "plant_organ": re.compile(r"^\s*Plant\s+organ\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        "visual_symptoms": re.compile(r"^\s*Visual\s+symptoms\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        "pathogen": re.compile(r"^\s*Pathogen\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    }

    @classmethod
    def parse(cls, raw_text: str, crop_name: str, known_classes: Optional[List[str]] = None) -> ParsedPrediction:
        """Parse raw model output text into structured SAGE prediction fields."""
        if not raw_text or not raw_text.strip():
            return ParsedPrediction(
                class_name="Unknown",
                crop=crop_name,
                is_canonical_match=False,
            )

        extracted: Dict[str, str] = {}
        for key, pattern in cls.PATTERNS.items():
            match = pattern.search(raw_text)
            if match:
                extracted[key] = match.group(1).strip()

        # Determine class_name
        class_name = extracted.get("diagnosis")
        if not class_name:
            # Fallback: check if the first non-empty line contains a diagnosis or class
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            if lines:
                class_name = lines[0]
            else:
                class_name = "Unknown"

        # Clean any trailing punctuation or markdown artifacts
        class_name = class_name.strip(" *#-_`\"'")

        # Taxonomy matching against known classes
        matched_class = None
        is_canonical = False

        if known_classes:
            normalized_predicted = re.sub(r"[^a-zA-Z0-9]", "", class_name).lower()
            for kc in known_classes:
                normalized_kc = re.sub(r"[^a-zA-Z0-9]", "", kc).lower()
                if normalized_predicted == normalized_kc or normalized_predicted in normalized_kc or normalized_kc in normalized_predicted:
                    matched_class = kc
                    is_canonical = True
                    break

        return ParsedPrediction(
            class_name=class_name,
            crop=extracted.get("crop", crop_name),
            disease_type=extracted.get("disease_type"),
            plant_organ=extracted.get("plant_organ"),
            visual_symptoms=extracted.get("visual_symptoms"),
            pathogen=extracted.get("pathogen"),
            matched_taxonomy_class=matched_class,
            is_canonical_match=is_canonical,
            attributes=extracted,
        )
