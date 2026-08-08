"""
ingestion package — all file/media/code extraction + chunking.

SRP: separate files for concerns (documents, ocr, audio, subtitles, media, code).
"""

from . import (
    registry,
    chunking,
    documents,
    ocr,
    audio,
    subtitles,
    media,
    code,
)
from .registry import EXTRACTORS, EXTRACTORS_OCR, extract_file, get_extractor
from .chunking import chunk_text, chunk_code
from .documents import (
    extract_pdf, extract_pdf_force_ocr, extract_epub, extract_mobi,
    extract_text, extract_image, extract_multipage_tiff, extract_djvu,
)
from .ocr import ocr_image, extract_pdf_ocr, extract_images_from_pdf, extract_images_from_epub
from .audio import extract_audio, transcribe_audio, _get_faster_whisper_model, _get_openai_whisper_model
from .subtitles import load_subtitle_file, parse_subtitle_text, search_subdl_subtitles, gather_video_caption_candidates, extract_embedded_subtitle_candidate
from .media import probe_media_metadata, extract_media_text, decorate_media_text, gather_video_caption_candidates
from .code import chunk_code, _detect_language, _get_import_block

