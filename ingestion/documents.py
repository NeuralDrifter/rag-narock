"""
ingestion/documents.py — PDF, EPUB, MOBI, image, DjVu, text extractors.
Document extractors (complete modular implementation).
"""

import os
import re
import sys
import shutil
import tempfile
import logging
import base64
import io

from config import settings as cfg
from .fitz_utils import import_fitz
from .ocr import (
    ocr_image, extract_pdf_ocr, extract_images_from_pdf,
    extract_images_from_epub, ocr_disabled, split_spreads_enabled,
    sanitize_text, ocr_backend_available,
)

def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator='\n')
    except Exception:
        return re.sub(r'<[^>]+>', ' ', html)

def _fetch_url_playwright(url: str):
    try:
        import importlib.util
        if importlib.util.find_spec('playwright') is None:
            return None
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            text = page.inner_text('body')
            browser.close()
            return text
    except Exception:
        return None


def extract_pdf(path: str, ocr_options=None):
    """Returns (text, is_ocr). Falls back to OCR if no text layer."""
    fitz = import_fitz()
    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            t = page.get_text("text")
            if t.strip():
                parts.append(t)
    text = "\n".join(parts)
    if text.strip():
        return (sanitize_text(text), False)
    if ocr_disabled(ocr_options):
        print(f"    No text layer, OCR disabled — skipping", file=sys.stderr)
        return (None, False)
    print(f"    No text layer, trying OCR...", file=sys.stderr)
    return extract_pdf_ocr(path, ocr_options)

def extract_pdf_force_ocr(path: str, ocr_options=None):
    """Always OCR, ignoring any embedded text layer. Returns (text, True)."""
    return extract_pdf_ocr(path, ocr_options)

def extract_image(path: str, ocr_options=None):
    """OCR a single image file. Returns (text, True) or (None, True)."""
    if ocr_disabled(ocr_options):
        return (None, False)
    if not ocr_backend_available(ocr_options):
        print("    No OCR backend available, skipping image OCR", file=sys.stderr)
        return (None, True)
    from PIL import Image
    img = Image.open(path)
    text = ocr_image(img, ocr_options)
    return (sanitize_text(text), True) if text.strip() else (None, True)

def extract_multipage_tiff(path: str, ocr_options=None):
    """OCR a multi-page TIFF by iterating all frames. Returns (text, True)."""
    if ocr_disabled(ocr_options):
        return (None, False)
    if not ocr_backend_available(ocr_options):
        print("    No OCR backend available, skipping TIFF OCR", file=sys.stderr)
        return (None, True)
    from PIL import Image
    parts = []
    img = Image.open(path)
    n_frames = getattr(img, 'n_frames', 1)
    for i in range(n_frames):
        img.seek(i)
        text = ocr_image(img.convert('RGB'), ocr_options)
        if text.strip():
            parts.append(text)
        if (i + 1) % 10 == 0:
            print(f"    OCR TIFF frame {i+1}/{n_frames}...", file=sys.stderr)
    text = "\n".join(parts)
    return (sanitize_text(text), True) if text.strip() else (None, True)

def extract_djvu(path: str, ocr_options=None):
    """Extract text from DjVu using djvutxt, falling back to OCR via ddjvu.
    Returns (text, is_ocr)."""
    import subprocess
    from PIL import Image

    try:
        result = subprocess.run(['djvutxt', path], capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return (sanitize_text(result.stdout), False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if ocr_disabled(ocr_options):
        print(f"    No text layer in DjVu, OCR disabled — skipping", file=sys.stderr)
        return (None, False)
    if not ocr_backend_available(ocr_options):
        print("    No OCR backend available, skipping DjVu OCR", file=sys.stderr)
        return (None, True)
    print(f"    No text layer in DjVu, OCR via ddjvu...", file=sys.stderr)
    tmp_dir = tempfile.mkdtemp(prefix='rag_djvu_')
    try:
        result = subprocess.run(['djvused', path, '-e', 'n'], capture_output=True, text=True, timeout=30)
        n_pages = int(result.stdout.strip()) if result.returncode == 0 else 0
        if n_pages == 0:
            return (None, True)

        parts = []
        for page_num in range(1, n_pages + 1):
            tiff_path = os.path.join(tmp_dir, f'page_{page_num}.tiff')
            subprocess.run(
                ['ddjvu', '-format=tiff', f'-page={page_num}', '-quality=300', path, tiff_path],
                capture_output=True, timeout=60
            )
            if os.path.exists(tiff_path):
                img = Image.open(tiff_path)
                text = ocr_image(img.convert('RGB'), ocr_options)
                if text.strip():
                    parts.append(text)
            if page_num % 10 == 0:
                print(f"    OCR DjVu page {page_num}/{n_pages}...", file=sys.stderr)

        text = "\n".join(parts)
        return (sanitize_text(text), True) if text.strip() else (None, True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def extract_epub_ocr(book, ocr_options=None):
    """OCR an image-only EPUB."""
    if ocr_disabled(ocr_options):
        return (None, False)
    if not ocr_backend_available(ocr_options):
        print("    No OCR backend available, skipping EPUB image OCR", file=sys.stderr)
        return (None, True)
    import ebooklib
    from PIL import Image
    parts = []

    image_items = {}
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        image_items[item.get_name()] = item

    sorted_names = sorted(image_items.keys())
    total = len(sorted_names)

    for i, name in enumerate(sorted_names):
        item = image_items[name]
        content = item.get_content()

        if name.lower().endswith('.svg'):
            svg_text = content.decode('utf-8', errors='replace')
            b64_match = re.search(r'xlink:href="data:image/[^;]+;base64,([^"]+)"', svg_text)
            if b64_match:
                try:
                    img_data = base64.b64decode(b64_match.group(1))
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    text = ocr_image(img, ocr_options)
                    if text.strip():
                        parts.append(text)
                except Exception:
                    pass
        else:
            try:
                img = Image.open(io.BytesIO(content)).convert('RGB')
                min_sz = cfg.get('min_image_size')
                if img.width >= min_sz and img.height >= min_sz:
                    text = ocr_image(img, ocr_options)
                    if text.strip():
                        parts.append(text)
            except Exception:
                pass

        if (i + 1) % 10 == 0:
            print(f"    OCR EPUB image {i+1}/{total}...", file=sys.stderr)

    text = "\n".join(parts)
    return (sanitize_text(text), True) if text.strip() else (None, True)

def extract_epub(path: str, ocr_options=None):
    """Returns (text, is_ocr). Falls back to OCR if no text layer."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    logging.getLogger('ebooklib').setLevel(logging.CRITICAL)
    logging.getLogger('ebooklib.epub').setLevel(logging.CRITICAL)
    book = epub.read_epub(path, options={"ignore_ncx": True})
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        name = (item.get_name() or '').lower()
        if name.endswith('.css'):
            continue
        html = item.get_body_content()
        soup = BeautifulSoup(html, 'html.parser')
        t = soup.get_text(separator='\n')
        if t.strip():
            parts.append(sanitize_text(t))
    if parts:
        return ("\n\n".join(parts), False)
    if ocr_disabled(ocr_options):
        print(f"    No text layer in EPUB, OCR disabled — skipping", file=sys.stderr)
        return (None, False)
    print(f"    No text layer in EPUB, trying OCR on images...", file=sys.stderr)
    return extract_epub_ocr(book, ocr_options)

def extract_mobi(path: str, ocr_options=None):
    """Returns (text, False) — never OCR'd."""
    import mobi
    from bs4 import BeautifulSoup
    temp_dir = None
    try:
        temp_dir, _ = mobi.extract(path)
        parts = []
        for root, _, files in os.walk(temp_dir):
            for f in sorted(files):
                if f.lower().endswith(('.html', '.htm', '.txt')):
                    with open(os.path.join(root, f), 'r', encoding='utf-8', errors='replace') as fh:
                        soup = BeautifulSoup(fh.read(), 'html.parser')
                        t = soup.get_text(separator='\n')
                        if t.strip():
                            parts.append(sanitize_text(t))
        return ("\n\n".join(parts), False) if parts else (None, False)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def extract_text(path: str, ocr_options=None):
    """Returns (text, False) — never OCR'd."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    return (sanitize_text(text), False) if text.strip() else (None, False)

def extract_audio(path: str, ocr_options=None):
    """Delegate to audio handler. Returns (text, is_ocr=False)."""
    from .audio import transcribe_audio
    return transcribe_audio(path)

def extract_url(url: str):
    """Fetch a single URL and extract readable text."""
    import urllib.request, urllib.error
    text = None
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            ct = resp.headers.get('Content-Type', '')
            if 'text/html' in ct or 'application/xhtml' in ct:
                raw = resp.read()
                encoding = resp.headers.get_content_charset() or 'utf-8'
                html = raw.decode(encoding, errors='replace')
                text = _html_to_text(html)
            elif 'text/' in ct or 'application/json' in ct:
                raw = resp.read()
                encoding = resp.headers.get_content_charset() or 'utf-8'
                text = raw.decode(encoding, errors='replace')
            else:
                print(f"  WARN: unsupported content type: {ct}", file=sys.stderr)
                return (None, False)
    except Exception as e:
        print(f"  WARN: simple fetch failed: {e}", file=sys.stderr)

    if text and len(text.strip()) > 200:
        return (sanitize_text(text), False)

    pw_text = _fetch_url_playwright(url)
    if pw_text and len(pw_text.strip()) > 200:
        return (sanitize_text(pw_text), False)

    if text and text.strip():
        return (sanitize_text(text), False)
    if pw_text and pw_text.strip():
        return (sanitize_text(pw_text), False)
    return (None, False)

def _probe_needs_ocr(path: str) -> bool:
    """Quick check if a file will likely need OCR."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.txt', '.md', '.rst', '.mobi'):
        return False
    if ext in {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.aac', '.mp4', '.m4v', '.mov', '.webm', '.mkv', '.avi'}:
        return False
    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.pbm', '.pgm', '.ppm', '.pnm'):
        return True
    if ext in ('.tiff', '.tif'):
        return True
    if ext == '.pdf':
        try:
            fitz = import_fitz()
            with fitz.open(path) as doc:
                for i, page in enumerate(doc):
                    if i >= 3:
                        break
                    if page.get_text("text").strip():
                        return False
            return True
        except Exception:
            return True
    if ext == '.epub':
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            book = epub.read_epub(path, options={'ignore_ncx': True})
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                if soup.get_text(strip=True):
                    return False
            return True
        except Exception:
            return True
    if ext in ('.djvu', '.djv'):
        try:
            import subprocess
            r = subprocess.run(['djvutxt', path], capture_output=True, text=True, timeout=5)
            return not r.stdout.strip()
        except Exception:
            return True
    return False


def detect_document_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.aac'}
    VIDEO_EXTENSIONS = {'.mp4', '.m4v', '.mov', '.webm', '.mkv', '.avi'}
    if ext in AUDIO_EXTENSIONS:
        return 'audio'
    if ext in VIDEO_EXTENSIONS:
        return 'video'
    return 'book'
