"""
ingestion/ocr_options.py — explicit OCR configuration object.

Replaces module-level mutable globals (_ocr_backend_override, _ocr_disabled_override,
_split_spreads_override) with a clean parameter object (Ch 3, F1, G31).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OcrOptions:
    """Immutable OCR configuration for an indexing run.

    Pass this through the extraction pipeline instead of relying on hidden globals.
    """
    disabled: bool = False
    force: bool = False
    backend: Optional[str] = None       # 'tesseract', 'easyocr', or None (use settings)
    language: str = "eng"
    negative: bool = False
    split_spreads: bool = False
