"""
ingestion/registry.py — central registry of extractors.

Clean Code:
- Single place for adding new formats (easy to extend)
- Separates registration from implementation
- No dependency on old rag god module
"""

import os

from .ocr_options import OcrOptions as _OcrOptions

# Import implementations from modular subpackages (direct, no old god module)
from .documents import (
    extract_pdf, extract_pdf_force_ocr,
    extract_epub, extract_mobi,
    extract_text,
    extract_image, extract_multipage_tiff, extract_djvu,
)
from .audio import extract_audio
from .media import extract_media_text  # may be used for some media
# Note: audio extract_audio handles transcription dispatch for media too

EXTRACTORS = {
    '.pdf': extract_pdf,
    '.epub': extract_epub,
    '.mobi': extract_mobi,
    '.txt': extract_text, '.md': extract_text, '.rst': extract_text,
    # audio / video - use audio extractor (handles via transcribe)
    '.mp3': extract_audio, '.wav': extract_audio, '.m4a': extract_audio,
    '.flac': extract_audio, '.ogg': extract_audio, '.opus': extract_audio,
    '.aac': extract_audio, '.mp4': extract_audio, '.m4v': extract_audio,
    '.mov': extract_audio, '.webm': extract_audio, '.mkv': extract_audio,
    '.avi': extract_audio,
    # images (OCR)
    '.png': extract_image, '.jpg': extract_image, '.jpeg': extract_image,
    '.bmp': extract_image, '.webp': extract_image,
    '.tiff': extract_multipage_tiff, '.tif': extract_multipage_tiff,
    '.djvu': extract_djvu, '.djv': extract_djvu,
    '.pbm': extract_image, '.pgm': extract_image, '.ppm': extract_image,
    '.pnm': extract_image,
}

EXTRACTORS_OCR = {**EXTRACTORS, '.pdf': extract_pdf_force_ocr}

VIDEO_EXTENSIONS = {'.mp4', '.m4v', '.mov', '.webm', '.mkv', '.avi'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.aac'}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def get_extractor(ext: str, force_ocr: bool = False):
    table = EXTRACTORS_OCR if force_ocr else EXTRACTORS
    return table.get(ext.lower())


def extract_file(path: str, force_ocr: bool = False, ocr_options: _OcrOptions = None):
    """Dispatcher. Returns (text, is_ocr).

    Parameters
    ----------
    path : str
        File path to extract text from.
    force_ocr : bool
        Force OCR on PDFs even if a text layer exists.
    ocr_options : OcrOptions, optional
        Explicit OCR configuration for an indexing run. Passed down the
        extraction pipeline so that functions can consult them without
        relying on module-level globals.
    """
    ext = os.path.splitext(path)[1].lower()
    func = get_extractor(ext, force_ocr)
    if not func:
        return None, False
    try:
        result = func(path, ocr_options=ocr_options)
        # normalize: some return (text, is_ocr), some just text for audio
        if isinstance(result, tuple):
            return result
        return result, False
    except Exception as e:
        import sys
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            print(f"ERROR: Extraction failed because a dependency is missing: {e}.\n"
                  f"Please install it in your active Python environment.", file=sys.stderr)
        else:
            import traceback
            print(f"ERROR: Extraction failed for {path}:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return None, False
