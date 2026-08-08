"""
ingestion/ocr.py — OCR related (tesseract, easyocr, image processing).
OCR functions (complete modular implementation).
"""

import os
import sys
import re
import importlib.util
import shutil
import subprocess
import tempfile
import logging
from PIL import Image, ImageOps
import io
import base64

import numpy as np

import config.settings as cfg  # prefer direct

from .fitz_utils import import_fitz
from .ocr_options import OcrOptions

# --- OCR availability and state (moved full) ---
_ocr_lang = cfg.get('ocr_lang')

_WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def _resolve_tesseract_cmd():
    cmd = shutil.which('tesseract')
    if cmd:
        return cmd
    for path in _WINDOWS_TESSERACT_PATHS:
        if os.path.exists(path):
            return path
    return None


def _check_tesseract():
    return (
        _resolve_tesseract_cmd() is not None
        and importlib.util.find_spec('pytesseract') is not None
    )

def _check_easyocr():
    return importlib.util.find_spec('easyocr') is not None

HAS_TESSERACT = _check_tesseract()
HAS_EASYOCR = _check_easyocr()

_easyocr_reader = None
_easyocr_langs = None
_warned_messages = set()


def _warn_once(message: str):
    if message in _warned_messages:
        return
    _warned_messages.add(message)
    print(message, file=sys.stderr)


def _suppress_torch_cpu_warnings():
    import warnings
    warnings.filterwarnings(
        'ignore',
        message='.*pin_memory.*no accelerator is found.*',
        category=UserWarning,
    )


def _easyocr_gpu_available() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True
        xpu = getattr(torch, 'xpu', None)
        if xpu is not None and xpu.is_available():
            return True
    except Exception:
        pass
    return False


def _release_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _get_easyocr_reader(langs=None):
    global _easyocr_reader, _easyocr_langs
    if langs is None:
        langs = ['en']
    if _easyocr_reader is not None and _easyocr_langs == langs:
        return _easyocr_reader
    # env prep
    try:
        _prepare_env_for_device = lambda d: None  # defined elsewhere or noop here
    except:
        pass
    _suppress_torch_cpu_warnings()
    import easyocr
    use_gpu = _easyocr_gpu_available()
    _easyocr_reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
    _easyocr_langs = langs
    return _easyocr_reader

def _unload_easyocr():
    global _easyocr_reader, _easyocr_langs
    if _easyocr_reader is not None:
        try:
            for component in ('detector', 'recognizer'):
                net = getattr(_easyocr_reader, component, None)
                if net is not None:
                    net.cpu()
        except Exception:
            pass
        del _easyocr_reader
        _easyocr_reader = None
        _easyocr_langs = None
        _release_gpu_memory()

_TESS_TO_EASYOCR = {
    'eng': 'en', 'fra': 'fr', 'deu': 'de', 'spa': 'es', 'ita': 'it',
    'por': 'pt', 'nld': 'nl', 'pol': 'pl', 'rus': 'ru', 'ukr': 'uk',
    'ara': 'ar', 'hin': 'hi', 'ben': 'bn', 'jpn': 'ja', 'kor': 'ko',
    'chi_sim': 'ch_sim', 'chi_tra': 'ch_tra', 'tha': 'th', 'vie': 'vi',
    'tur': 'tr', 'ces': 'cs', 'ron': 'ro', 'hun': 'hu', 'fin': 'fi',
    'swe': 'sv', 'nor': 'no', 'dan': 'da', 'heb': 'he', 'ind': 'id',
    'msa': 'ms', 'tgl': 'tl', 'swa': 'sw', 'lat': 'la',
}

def _tess_lang_to_easyocr(tess_lang):
    langs = []
    for code in tess_lang.split('+'):
        code = code.strip()
        mapped = _TESS_TO_EASYOCR.get(code, code)
        if mapped not in langs:
            langs.append(mapped)
    return langs

_ocr_backend_override = None
_ocr_disabled_override = None
_split_spreads_override = None

def get_ocr_backend(ocr_options=None, emit_warnings=True):
    """Get the active OCR backend, respecting overrides and availability."""
    # Check explicit OcrOptions parameter first
    if ocr_options is not None and ocr_options.backend is not None:
        backend = ocr_options.backend
    # Then check old global override (backward compat)
    elif _ocr_backend_override is not None:
        backend = _ocr_backend_override
    else:
        backend = cfg.get('ocr_backend')
    if backend == 'easyocr' and not HAS_EASYOCR:
        if HAS_TESSERACT:
            if emit_warnings:
                _warn_once("WARNING: EasyOCR not available, falling back to Tesseract")
            return 'tesseract'
        return None
    if backend == 'easyocr' and HAS_EASYOCR and not _easyocr_gpu_available() and HAS_TESSERACT:
        if emit_warnings:
            _warn_once("WARNING: EasyOCR GPU not available, falling back to Tesseract")
        return 'tesseract'
    if backend == 'tesseract' and not HAS_TESSERACT:
        if HAS_EASYOCR:
            if emit_warnings:
                _warn_once("WARNING: Tesseract not available, falling back to EasyOCR")
            return 'easyocr'
        return None
    return backend


def ocr_backend_available(ocr_options=None) -> bool:
    return get_ocr_backend(ocr_options, emit_warnings=False) is not None


def ocr_unavailable_message() -> str:
    return (
        "No OCR backend is available.\n\n"
        "Install one of these in the active Python environment:\n"
        "- Tesseract: install the Tesseract system binary and run "
        "python -m pip install pytesseract\n"
        "- EasyOCR: python -m pip install easyocr\n\n"
        "Bypass: disable OCR or use files with an embedded text layer."
    )

def ocr_disabled(ocr_options=None) -> bool:
    """Check if OCR is globally disabled via settings or runtime override."""
    if ocr_options is not None:
        return ocr_options.disabled
    if _ocr_disabled_override is not None:
        return _ocr_disabled_override
    return cfg.get('disable_ocr')

def split_spreads_enabled(ocr_options=None) -> bool:
    """Check if spread splitting is enabled via settings or CLI override."""
    if ocr_options is not None:
        return ocr_options.split_spreads
    if _split_spreads_override is not None:
        return _split_spreads_override
    return cfg.get('split_spreads')

def set_ocr_lang(lang: str):
    global _ocr_lang
    import subprocess
    tesseract_cmd = _resolve_tesseract_cmd()
    if tesseract_cmd is None:
        print("WARNING: Tesseract executable not found", file=sys.stderr)
        return False
    result = subprocess.run([tesseract_cmd, '--list-langs'], capture_output=True, text=True)
    installed = set(result.stdout.strip().split('\n')[1:])
    requested = set(lang.split('+'))
    missing = requested - installed
    if missing:
        print(f"WARNING: Tesseract language(s) not installed: {', '.join(sorted(missing))}", file=sys.stderr)
        return False
    _ocr_lang = lang
    print(f"OCR language set to: {lang}", file=sys.stderr)
    return True

def reset_ocr_lang():
    global _ocr_lang
    _ocr_lang = cfg.get('ocr_lang')

def _find_gutter(img, margin_frac=0.15):
    gray = np.array(img.convert('L'))
    h, w = gray.shape
    left = int(w * (0.5 - margin_frac))
    right = int(w * (0.5 + margin_frac))
    strip = gray[:, left:right]
    col_sums = strip.sum(axis=0)
    gutter_offset = int(np.argmax(col_sums))
    return left + gutter_offset

def _split_spread(img):
    w, h = img.size
    if w <= h * 1.3:
        return [img]
    gutter_x = _find_gutter(img)
    left = img.crop((0, 0, gutter_x, h))
    right = img.crop((gutter_x, 0, w, h))
    return [left, right]

def _ocr_single(img, ocr_options=None) -> str:
    # Check active OcrOptions for negative flag, fall back to settings
    if ocr_options is not None:
        negative = ocr_options.negative
    else:
        negative = cfg.get('ocr_negative')
    if negative:
        img = ImageOps.invert(img.convert('RGB'))
    backend = get_ocr_backend(ocr_options)
    if backend is None:
        _warn_once("WARNING: No OCR backend available (install tesseract or easyocr)")
        return ""
    if backend == 'easyocr':
        _suppress_torch_cpu_warnings()
        lang = (ocr_options.language if ocr_options is not None else None) or _ocr_lang or 'eng'
        easyocr_langs = _tess_lang_to_easyocr(lang)
        reader = _get_easyocr_reader(easyocr_langs)
        img_array = np.array(img.convert('RGB'))
        results = reader.readtext(img_array)
        return '\n'.join(text for _, text, _ in results)
    else:
        lang = (ocr_options.language if ocr_options is not None else None) or _ocr_lang or 'eng'
        import pytesseract
        tesseract_cmd = _resolve_tesseract_cmd()
        if tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return pytesseract.image_to_string(img, lang=lang)

def ocr_image(img, ocr_options=None) -> str:
    if split_spreads_enabled(ocr_options):
        pages = _split_spread(img)
        if len(pages) == 2:
            return _ocr_single(pages[0], ocr_options) + "\n" + _ocr_single(pages[1], ocr_options)
    return _ocr_single(img, ocr_options)


def extract_pdf_ocr(path: str, ocr_options=None):
    if ocr_disabled(ocr_options):
        print(f"    OCR disabled, skipping", file=sys.stderr)
        return (None, False)
    if not ocr_backend_available(ocr_options):
        _warn_once("WARNING: No OCR backend available (install tesseract or easyocr)")
        return (None, True)
    fitz = import_fitz()
    from PIL import Image as PILImage
    parts = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=cfg.get('render_dpi'))
            img = PILImage.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_image(img, ocr_options)
            if text.strip():
                parts.append(text)
            if (i + 1) % 10 == 0:
                print(f"    OCR page {i+1}/{len(doc)}...", file=sys.stderr)
    text = "\n".join(parts)
    return (text.strip() and sanitize_text(text) or None, True) if text.strip() else (None, True)

def extract_images_from_pdf(path, min_size=None, max_dim=None):
    fitz = import_fitz()
    from PIL import Image as PILImage
    if min_size is None:
        min_size = cfg.get('min_image_size')
    if max_dim is None:
        max_dim = cfg.get('max_image_dim')
    images = []
    seen_xrefs = set()
    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    pil_img = PILImage.open(io.BytesIO(base_image["image"]))
                    w, h = pil_img.size
                except Exception:
                    continue
                if w < min_size or h < min_size:
                    continue
                pil_img.thumbnail((max_dim, max_dim), PILImage.LANCZOS)
                w, h = pil_img.size
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                images.append({
                    'page': page_num,
                    'data': buf.getvalue(),
                    'width': w,
                    'height': h,
                    'xref': xref,
                })
    return images

def extract_images_from_epub(path, min_size=None, max_dim=None):
    return []  # extend if needed for image-backed stores

def sanitize_text(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", '', s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s


_IMAGE_EXTRACTORS = {
    '.pdf': extract_images_from_pdf,
    '.epub': extract_images_from_epub,
}


def associate_images_with_chunks(images, chunks, full_text):
    """Map each image to its nearest chunk by page-based text position.
    Chunks must have 'text' key. Images must have 'page' key.
    Returns images with 'nearest_chunk' index added."""
    if not images or not chunks:
        return images

    chunk_starts = []
    pos = 0
    for i, c in enumerate(chunks):
        snippet = c['text'][:100]
        start = full_text.find(snippet, pos)
        if start == -1:
            start = pos
        chunk_starts.append(start)
        pos = max(pos, start + 1)

    text_len = len(full_text)
    if not images:
         return images
    max_page = max(img['page'] for img in images)

    for img in images:
        target_offset = int((img['page'] / max(max_page, 1)) * text_len)
        best_chunk = 0
        best_dist = abs(chunk_starts[0] - target_offset) if chunk_starts else 0
        for i, cs in enumerate(chunk_starts):
            dist = abs(cs - target_offset)
            if dist < best_dist:
                best_dist = dist
                best_chunk = i
        img['nearest_chunk'] = best_chunk

    return images
