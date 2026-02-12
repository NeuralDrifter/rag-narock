#!/usr/bin/env python3
"""
RAG-Narock — local RAG system for Claude Code.
- CPU-only embedding model (no GPU interference)
- Model loaded on-demand, unloaded after each operation
- Point at a folder of books -> index -> query via CLI

Usage:
    python3 rag.py index /path/to/books --name mybooks
    python3 rag.py query "search terms" --name mybooks --top-k 5
    python3 rag.py list
    python3 rag.py settings
"""

import os, sys, json, gc, argparse, re, unicodedata, hashlib, signal
from pathlib import Path
from typing import List, Optional
import rag_settings
import rag_backends

# Force CPU-only BEFORE any ML imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ONEAPI_DEVICE_SELECTOR"] = "opencl:cpu"
os.environ["SYCL_DEVICE_FILTER"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Offline-first: never contact HF Hub unless model isn't cached
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RAG_HOME = os.path.expanduser("~/.local/share/rag")

# ── File hashing for duplicate detection ──

def file_hash(path: str) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load_index_hashes(index_dir: str) -> dict:
    """Load stored hashes from an index. Returns {hash: filename}."""
    backend = rag_backends.get_backend(index_dir, rag_backends.detect_backend(index_dir))
    return backend.get_hashes()

def save_index_hashes(index_dir: str, hashes: dict):
    """Save hashes to an index."""
    backend = rag_backends.get_backend(index_dir, rag_backends.detect_backend(index_dir))
    backend.save_hashes(hashes)

# ── Index management helpers (module-level) ──

def get_indexes():
    """Return list of existing index names."""
    if not os.path.exists(RAG_HOME):
        return []
    return sorted(n for n in os.listdir(RAG_HOME)
                  if os.path.exists(os.path.join(RAG_HOME, n, "meta.json")))

def get_index_info(name):
    """Return meta.json dict for an index, or None."""
    mp = os.path.join(RAG_HOME, name, "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f)
    return None

def get_index_sources(name):
    """Return dict of {source_name: chunk_count} for an index."""
    index_dir = os.path.join(RAG_HOME, name)
    backend_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, backend_type)
    chunks = backend.get_chunks()
    sources = {}
    for c in chunks:
        src = c['source']
        sources[src] = sources.get(src, 0) + 1
    return sources

def is_index_locked(name):
    """Check if index has a .locked file."""
    return os.path.exists(os.path.join(RAG_HOME, name, ".locked"))

def remove_source_from_index(index_name, source_name):
    """Remove all chunks for a source, rebuild index, update hashes+meta.
    Returns {'removed_chunks', 'remaining_chunks', 'remaining_files'}."""
    index_dir = os.path.join(RAG_HOME, index_name)
    backend_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, backend_type)

    if not backend.exists():
        raise FileNotFoundError(f"Index '{index_name}' not found or incomplete")

    result = backend.remove_source(source_name)

    # Update meta
    mp = os.path.join(index_dir, "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
        meta['n_chunks'] = result['remaining_chunks']
        meta['n_files'] = result['remaining_files']
        with open(mp, 'w') as f:
            json.dump(meta, f, indent=2)

    return result

def delete_index(index_name, force=False):
    """Delete entire index directory. Raises ValueError if locked and not force."""
    import shutil
    index_dir = os.path.join(RAG_HOME, index_name)
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Index '{index_name}' not found")
    if is_index_locked(index_name) and not force:
        raise ValueError(f"Index '{index_name}' is LOCKED. Unlock it first or use force=True.")
    shutil.rmtree(index_dir)

# ── Document extraction (ideas from bulk_convert.py) ──

def sanitize_text(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", '', s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s

_ocr_lang = rag_settings.get('ocr_lang')  # module-level OCR language, set via --ocr-lang or GUI

def set_ocr_lang(lang: str):
    """Set the Tesseract language for all OCR operations.
    Examples: 'eng', 'hin', 'chi_sim', 'jpn', 'eng+hin', 'chi_sim+eng'.
    Install packs: sudo apt install tesseract-ocr-hin tesseract-ocr-chi-sim etc."""
    global _ocr_lang
    import subprocess
    # Validate that requested languages are installed
    result = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True)
    installed = set(result.stdout.strip().split('\n')[1:])  # skip header line
    requested = set(lang.split('+'))
    missing = requested - installed
    if missing:
        print(f"WARNING: Tesseract language(s) not installed: {', '.join(sorted(missing))}", file=sys.stderr)
        pkgs = ' '.join('tesseract-ocr-' + l.replace('_', '-') for l in sorted(missing))
        print(f"  Install with: sudo apt install {pkgs}", file=sys.stderr)
        print(f"  Available: {', '.join(sorted(installed))}", file=sys.stderr)
        return False
    _ocr_lang = lang
    print(f"OCR language set to: {lang}", file=sys.stderr)
    return True

def reset_ocr_lang():
    """Reset OCR language to English (default)."""
    global _ocr_lang
    _ocr_lang = rag_settings.get('ocr_lang')

def ocr_image(img) -> str:
    """OCR a PIL Image using Tesseract. Uses module-level _ocr_lang setting.
    If ocr_negative is enabled in settings, inverts the image first."""
    import pytesseract
    from PIL import ImageOps
    if rag_settings.get('ocr_negative'):
        img = ImageOps.invert(img.convert('RGB'))
    return pytesseract.image_to_string(img, lang=_ocr_lang)

def extract_pdf_ocr(path: str):
    """OCR a PDF by rendering each page to an image. Returns (text, True) or (None, True)."""
    import fitz
    from PIL import Image
    import io
    parts = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=rag_settings.get('render_dpi'))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_image(img)
            if text.strip():
                parts.append(text)
            if (i + 1) % 10 == 0:
                print(f"    OCR page {i+1}/{len(doc)}...", file=sys.stderr)
    text = "\n".join(parts)
    return (sanitize_text(text), True) if text.strip() else (None, True)

def extract_pdf(path: str):
    """Returns (text, is_ocr). Falls back to OCR if no text layer."""
    import fitz
    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            t = page.get_text("text")
            if t.strip():
                parts.append(t)
    text = "\n".join(parts)
    if text.strip():
        return (sanitize_text(text), False)
    # No embedded text — automatic OCR fallback
    print(f"    No text layer, trying OCR...", file=sys.stderr)
    return extract_pdf_ocr(path)

def extract_pdf_force_ocr(path: str):
    """Always OCR, ignoring any embedded text layer. Returns (text, True)."""
    return extract_pdf_ocr(path)

def extract_image(path: str):
    """OCR a single image file. Returns (text, True) or (None, True)."""
    from PIL import Image
    img = Image.open(path)
    text = ocr_image(img)
    return (sanitize_text(text), True) if text.strip() else (None, True)

def extract_multipage_tiff(path: str):
    """OCR a multi-page TIFF by iterating all frames. Returns (text, True)."""
    from PIL import Image
    parts = []
    img = Image.open(path)
    n_frames = getattr(img, 'n_frames', 1)
    for i in range(n_frames):
        img.seek(i)
        text = ocr_image(img.convert('RGB'))
        if text.strip():
            parts.append(text)
        if (i + 1) % 10 == 0:
            print(f"    OCR TIFF frame {i+1}/{n_frames}...", file=sys.stderr)
    text = "\n".join(parts)
    return (sanitize_text(text), True) if text.strip() else (None, True)

def extract_djvu(path: str):
    """Extract text from DjVu using djvutxt, falling back to OCR via ddjvu.
    Returns (text, is_ocr)."""
    import subprocess, tempfile, shutil
    from PIL import Image

    # Try embedded text layer first (fast)
    try:
        result = subprocess.run(['djvutxt', path], capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return (sanitize_text(result.stdout), False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # No text layer — OCR each page via ddjvu → TIFF → Tesseract
    print(f"    No text layer in DjVu, OCR via ddjvu...", file=sys.stderr)
    tmp_dir = tempfile.mkdtemp(prefix='rag_djvu_')
    try:
        # Get page count
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
                text = ocr_image(img.convert('RGB'))
                if text.strip():
                    parts.append(text)
            if page_num % 10 == 0:
                print(f"    OCR DjVu page {page_num}/{n_pages}...", file=sys.stderr)

        text = "\n".join(parts)
        return (sanitize_text(text), True) if text.strip() else (None, True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def extract_epub_ocr(book):
    """OCR an image-only EPUB. Handles SVGs with base64 rasters + direct raster images."""
    import ebooklib
    from PIL import Image
    import io, base64
    parts = []

    # Collect all image items by name for quick lookup
    image_items = {}
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        image_items[item.get_name()] = item

    # Sort by name for page order
    sorted_names = sorted(image_items.keys())
    total = len(sorted_names)

    for i, name in enumerate(sorted_names):
        item = image_items[name]
        content = item.get_content()

        if name.lower().endswith('.svg'):
            # SVG wrapping a base64-encoded raster (common in scanned EPUBs)
            svg_text = content.decode('utf-8', errors='replace')
            b64_match = re.search(r'xlink:href="data:image/[^;]+;base64,([^"]+)"', svg_text)
            if b64_match:
                try:
                    img_data = base64.b64decode(b64_match.group(1))
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    text = ocr_image(img)
                    if text.strip():
                        parts.append(text)
                except Exception:
                    pass
        else:
            # Direct raster image (png, jpg, etc.)
            try:
                img = Image.open(io.BytesIO(content)).convert('RGB')
                # Skip tiny images (icons, decorations)
                min_sz = rag_settings.get('min_image_size')
                if img.width >= min_sz and img.height >= min_sz:
                    text = ocr_image(img)
                    if text.strip():
                        parts.append(text)
            except Exception:
                pass

        if (i + 1) % 10 == 0:
            print(f"    OCR EPUB image {i+1}/{total}...", file=sys.stderr)

    text = "\n".join(parts)
    return (sanitize_text(text), True) if text.strip() else (None, True)

def extract_epub(path: str):
    """Returns (text, is_ocr). Falls back to OCR if no text layer."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    import logging
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
    # No text — try OCR on embedded images
    print(f"    No text layer in EPUB, trying OCR on images...", file=sys.stderr)
    return extract_epub_ocr(book)

def extract_mobi(path: str):
    """Returns (text, False) — never OCR'd."""
    import mobi, shutil, tempfile
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

def extract_text(path: str):
    """Returns (text, False) — never OCR'd."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    return (sanitize_text(text), False) if text.strip() else (None, False)

EXTRACTORS = {
    '.pdf': extract_pdf, '.epub': extract_epub, '.mobi': extract_mobi,
    '.txt': extract_text, '.md': extract_text, '.rst': extract_text,
    # Images (OCR)
    '.png': extract_image, '.jpg': extract_image, '.jpeg': extract_image,
    '.bmp': extract_image, '.webp': extract_image,
    # Multi-page TIFF (iterates all frames)
    '.tiff': extract_multipage_tiff, '.tif': extract_multipage_tiff,
    # DjVu (text layer → OCR fallback)
    '.djvu': extract_djvu, '.djv': extract_djvu,
    # PNM family (OCR)
    '.pbm': extract_image, '.pgm': extract_image, '.ppm': extract_image,
    '.pnm': extract_image,
}

# When --ocr is passed, override PDF extractor to force OCR
EXTRACTORS_OCR = {**EXTRACTORS, '.pdf': extract_pdf_force_ocr}

def extract_file(path: str, force_ocr: bool = False):
    """Returns (text, is_ocr) tuple, or (None, False) on failure."""
    ext = os.path.splitext(path)[1].lower()
    table = EXTRACTORS_OCR if force_ocr else EXTRACTORS
    fn = table.get(ext)
    if fn is None:
        return (None, False)
    try:
        return fn(path)
    except Exception as e:
        print(f"  WARN: {os.path.basename(path)}: {e}", file=sys.stderr)
        return (None, False)

# ── Chunking ──

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    if chunk_size is None:
        chunk_size = rag_settings.get('chunk_size')
    if overlap is None:
        overlap = rag_settings.get('overlap')
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(current.strip())
            # keep overlap from end of previous chunk
            current = current[-overlap:] + "\n\n" + para if len(current) > overlap else para
        else:
            current = (current + "\n\n" + para) if current else para

    if current.strip():
        chunks.append(current.strip())

    # force-split any oversized chunks
    final = []
    for c in chunks:
        if len(c) > chunk_size * 2:
            for i in range(0, len(c), chunk_size - overlap):
                sub = c[i:i + chunk_size]
                if sub.strip():
                    final.append(sub.strip())
        else:
            final.append(c)

    min_len = rag_settings.get('min_chunk_length')
    return [c for c in final if len(c) > min_len]

# ── Embedding (lazy load, CPU-only) ──

_model = None

_model_name = None  # tracks which model is currently loaded

def _get_model(model_name=None):
    global _model, _model_name
    if model_name is None:
        model_name = rag_settings.get('embedding_model')

    # If a different model is requested, unload current one
    if _model is not None and _model_name != model_name:
        print(f"Switching model: '{_model_name}' -> '{model_name}'", file=sys.stderr)
        del _model
        _model = None
        _model_name = None
        gc.collect()

    if _model is None:
        import warnings
        warnings.filterwarnings('ignore', message='.*UNEXPECTED.*')
        import logging
        logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
        # Suppress all progress bars and load reports
        os.environ["TQDM_DISABLE"] = "1"
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(8, os.cpu_count() or 4))

        # Try loading from local cache first (no network)
        try:
            print(f"Loading embedding model '{model_name}' (CPU-only, local cache)...", file=sys.stderr)
            # Suppress the BertModel LOAD REPORT (writes to C-level fd, not Python streams)
            _saved_stdout = os.dup(1)
            _saved_stderr = os.dup(2)
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull_fd, 1)
            os.dup2(_devnull_fd, 2)
            try:
                _model = SentenceTransformer(model_name, device='cpu')
            finally:
                os.dup2(_saved_stdout, 1)
                os.dup2(_saved_stderr, 2)
                os.close(_saved_stdout)
                os.close(_saved_stderr)
                os.close(_devnull_fd)
            _model_name = model_name
            print("Model ready (offline).", file=sys.stderr)
        except Exception:
            # Model not cached — allow one-time download
            print(f"Model '{model_name}' not cached locally. Downloading...", file=sys.stderr)
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("TQDM_DISABLE", None)  # show download progress
            _model = SentenceTransformer(model_name, device='cpu')
            _model_name = model_name
            os.environ["TQDM_DISABLE"] = "1"
            # Re-enable offline mode for future loads
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            print("Model downloaded and cached. Future loads will be offline.", file=sys.stderr)
    return _model

def _unload_model():
    global _model
    # API backends have no local model to unload
    if rag_settings.get('embedding_backend') in ('ollama', 'lmstudio'):
        return
    if _model is not None:
        del _model
        _model = None
        gc.collect()

def embed_texts_api(texts: List[str], backend: str, override_model=None, override_url=None):
    """Embed texts via Ollama or LM Studio OpenAI-compatible API.
    Uses urllib.request (stdlib) — no new dependencies."""
    import numpy as np
    import urllib.request

    api_model = override_model or rag_settings.get('api_model')
    if override_url:
        base_url = override_url.rstrip('/')
    elif backend == 'ollama':
        base_url = rag_settings.get('ollama_url').rstrip('/')
    else:
        base_url = rag_settings.get('lmstudio_url').rstrip('/')
    url = f"{base_url}/v1/embeddings"

    all_embs = []
    batch_size = 32
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        if total > batch_size:
            print(f"  API embed batch {start // batch_size + 1}/{(total + batch_size - 1) // batch_size} "
                  f"({len(batch)} texts)...", file=sys.stderr)

        payload = json.dumps({"model": api_model, "input": batch}).encode('utf-8')
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach {backend} at {base_url}: {e}\n"
                f"Is {backend} running? Start it and ensure model '{api_model}' is available."
            ) from e

        # Sort by index to preserve order (API may return out of order)
        sorted_data = sorted(result['data'], key=lambda d: d['index'])
        for item in sorted_data:
            all_embs.append(item['embedding'])

    embs = np.array(all_embs, dtype=np.float32)
    # L2-normalize (API may not normalize)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs

def embed_texts(texts: List[str], batch_size: int = 64,
               override_backend=None, override_model=None, override_url=None):
    import numpy as np
    backend = override_backend or rag_settings.get('embedding_backend')
    if backend in ('ollama', 'lmstudio'):
        return embed_texts_api(texts, backend, override_model=override_model, override_url=override_url)
    # Local CPU path
    model = _get_model(model_name=override_model)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 100,
                        normalize_embeddings=True)
    return np.array(embs, dtype=np.float32)

# ── Commands ──

# ── Per-index model auto-resolution ──

def resolve_embedding_for_index(index_name):
    """Resolve the correct embedding model+backend for an index.
    Returns {'backend': str, 'model': str, 'warning': str|None, 'api_url': str|None,
             'storage_backend': str}."""
    index_dir = os.path.join(RAG_HOME, index_name)
    meta = rag_backends.get_index_meta_with_defaults(index_dir)

    emb_backend = meta.get('embedding_backend', 'local')
    emb_model = meta.get('embedding_model', 'all-MiniLM-L6-v2')
    storage = meta.get('storage_backend', 'faiss')
    api_url = None
    warning = None

    # Check storage backend availability
    if storage == 'sqlite-vec':
        try:
            import sqlite_vec
        except ImportError:
            return {
                'backend': emb_backend, 'model': emb_model, 'api_url': None,
                'storage_backend': storage,
                'warning': (
                    f"ERROR: Index '{index_name}' uses SQLite-vec for storage, but sqlite-vec is not\n"
                    f"installed in your Python environment.\n\n"
                    f"To fix this:\n"
                    f"  1. Install it:  pip install sqlite-vec\n"
                    f"  2. Verify:      python -c \"import sqlite_vec; print('OK')\"\n"
                    f"  3. Or re-index using FAISS (the default): rag.py settings -> RAG tab -> Storage Backend -> FAISS"
                ),
            }

    # Check embedding model availability
    if emb_backend == 'local':
        # Local model — try to load (will auto-download if needed)
        # We just validate it's loadable; actual loading happens in embed_texts
        pass  # _get_model() handles download + caching transparently

    elif emb_backend == 'ollama':
        import urllib.request
        api_url = meta.get('ollama_url', rag_settings.get('ollama_url'))
        base_url = api_url.rstrip('/')
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                model_names = [m.get('name', '').split(':')[0] for m in data.get('models', [])]
                if emb_model not in model_names:
                    # Try auto-pulling
                    try:
                        pull_data = json.dumps({"name": emb_model}).encode('utf-8')
                        pull_req = urllib.request.Request(
                            f"{base_url}/api/pull", data=pull_data,
                            headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(pull_req, timeout=300):
                            pass
                        warning = f"Auto-pulled Ollama model '{emb_model}'"
                    except Exception:
                        warning = (
                            f"WARNING: Ollama model '{emb_model}' not found. "
                            f"Pull it with: ollama pull {emb_model}"
                        )
        except Exception:
            return {
                'backend': emb_backend, 'model': emb_model, 'api_url': api_url,
                'storage_backend': storage,
                'warning': (
                    f"ERROR: Index '{index_name}' requires Ollama model '{emb_model}' but Ollama\n"
                    f"is not reachable at {base_url}.\n\n"
                    f"To fix this:\n"
                    f"  1. Start Ollama:  ollama serve\n"
                    f"  2. Pull the model: ollama pull {emb_model}\n"
                    f"  3. Verify it's running: curl {base_url}/api/tags\n"
                    f"  4. If Ollama is on a different URL, update it in: rag.py settings -> Models tab"
                ),
            }

    elif emb_backend == 'lmstudio':
        import urllib.request
        api_url = meta.get('lmstudio_url', rag_settings.get('lmstudio_url'))
        base_url = api_url.rstrip('/')
        try:
            req = urllib.request.Request(f"{base_url}/v1/models", method='GET')
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            return {
                'backend': emb_backend, 'model': emb_model, 'api_url': api_url,
                'storage_backend': storage,
                'warning': (
                    f"ERROR: Index '{index_name}' requires LM Studio model '{emb_model}' but\n"
                    f"LM Studio is not reachable at {base_url}.\n\n"
                    f"To fix this:\n"
                    f"  1. Open LM Studio and start the local server\n"
                    f"  2. Load the model '{emb_model}' in the Embeddings tab\n"
                    f"  3. Verify it's running: curl {base_url}/v1/models\n"
                    f"  4. If LM Studio is on a different URL, update it in: rag.py settings -> Models tab"
                ),
            }

    return {
        'backend': emb_backend, 'model': emb_model, 'api_url': api_url,
        'storage_backend': storage, 'warning': warning,
    }


def cmd_index(args):
    import numpy as np

    source_dir = os.path.abspath(args.path)
    if not os.path.isdir(source_dir):
        print(f"Not a directory: {source_dir}")
        return 1

    index_dir = os.path.join(RAG_HOME, args.name)

    # protect locked indexes from accidental overwrite (append is always OK)
    lock_file = os.path.join(index_dir, ".locked")
    if os.path.exists(lock_file) and not args.append:
        if not args.force:
            print(f"Index '{args.name}' is LOCKED. Use --force to overwrite, or --append to add.")
            return 1
        else:
            print(f"WARNING: Overwriting locked index '{args.name}'")

    os.makedirs(index_dir, exist_ok=True)

    # find files (EXTRACTORS already includes image formats)
    files = []
    for root, dirs, fnames in os.walk(source_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in EXTRACTORS:
                files.append(os.path.join(root, f))

    if not files:
        print(f"No supported files in {source_dir}")
        return 1

    # load existing hashes for duplicate detection
    existing_hashes = load_index_hashes(index_dir) if args.append else {}
    existing_sources = set()
    if args.append and os.path.exists(os.path.join(index_dir, "chunks.json")):
        with open(os.path.join(index_dir, "chunks.json"), 'r') as f:
            existing_sources = set(c['source'] for c in json.load(f))

    # deduplicate files by hash and name
    deduped_files = []
    dup_files = []
    new_hashes = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        fhash = file_hash(fpath)

        # check hash match (same content, possibly different filename)
        if fhash in existing_hashes:
            dup_files.append((fname, f"duplicate of '{existing_hashes[fhash]}' (same content)"))
            continue
        # check name match
        relname = os.path.relpath(fpath, source_dir)
        if relname in existing_sources:
            dup_files.append((fname, "already in index (same name)"))
            continue
        # check hash collision within this batch
        if fhash in new_hashes:
            dup_files.append((fname, f"duplicate of '{new_hashes[fhash]}' in this batch"))
            continue

        deduped_files.append(fpath)
        new_hashes[fhash] = fname

    if dup_files:
        print(f"Skipping {len(dup_files)} duplicate(s):")
        for name, reason in dup_files:
            print(f"  - {name}: {reason}")

    if not deduped_files:
        print("All files already in index. Nothing to add.")
        return 0

    # Resolve settings fallback for None args
    if args.chunk_size is None:
        args.chunk_size = rag_settings.get('chunk_size')
    if args.overlap is None:
        args.overlap = rag_settings.get('overlap')

    force_ocr = args.ocr if args.ocr is not None else rag_settings.get('force_ocr')
    ocr_lang = args.ocr_lang if args.ocr_lang is not None else rag_settings.get('ocr_lang')
    ocr_neg = args.ocr_negative if args.ocr_negative is not None else rag_settings.get('ocr_negative')
    if ocr_lang != rag_settings.get('ocr_lang'):
        if not set_ocr_lang(ocr_lang):
            return 1
    # Temporarily apply negative setting for this run
    if ocr_neg != rag_settings.get('ocr_negative'):
        cfg = rag_settings.load()
        cfg['ocr_negative'] = ocr_neg
        rag_settings.save(cfg)
    opts = []
    if force_ocr:
        opts.append("force-OCR")
    if ocr_neg:
        opts.append("negative")
    if opts:
        print(f"OCR mode: {', '.join(opts)}")

    print(f"Indexing {len(deduped_files)} files from {source_dir}")

    # extract + chunk
    all_chunks = []
    skipped_files = []
    file_hashes_map = {}
    for i, fpath in enumerate(deduped_files):
        fname = os.path.basename(fpath)
        print(f"[{i+1}/{len(deduped_files)}] {fname}", end="")
        text, is_ocr = extract_file(fpath, force_ocr=force_ocr)
        if not text:
            skipped_files.append(fname)
            print(" - SKIP (no text)")
            continue
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        ocr_tag = " [OCR]" if is_ocr else ""
        for ci, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'source': os.path.relpath(fpath, source_dir),
                'chunk': ci,
                'of': len(chunks),
                'ocr': is_ocr,
            })
        file_hashes_map[file_hash(fpath)] = fname
        print(f" - {len(chunks)} chunks{ocr_tag}")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} file(s) with no extractable text:")
        for f in skipped_files:
            print(f"  - {f}")

    if not all_chunks:
        print("No content extracted!")
        return 1

    print(f"\nNew: {len(all_chunks)} chunks. Embedding...")

    # embed new chunks
    texts = [c['text'] for c in all_chunks]
    embeddings = embed_texts(texts)
    _unload_model()

    dim = embeddings.shape[1]

    # Resolve storage backend
    storage_type = getattr(args, 'storage_backend', None) or rag_settings.get('storage_backend', 'faiss')
    # If appending, use whatever backend already exists
    if args.append:
        storage_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, storage_type)

    # merge hashes
    all_hashes = {**existing_hashes, **file_hashes_map}

    # append or full write
    existing_chunks = []
    if args.append and backend.exists():
        existing_chunks = backend.get_chunks()
        backend.append(all_chunks, embeddings, file_hashes_map)
        all_chunks = existing_chunks + all_chunks
    else:
        backend.save(all_chunks, embeddings, all_hashes)

    # metadata last — derived from actual saved data
    n_sources = len(set(c['source'] for c in all_chunks))
    emb_backend = rag_settings.get('embedding_backend')
    emb_model = rag_settings.get('embedding_model') if emb_backend == 'local' else rag_settings.get('api_model')
    meta = {
        'source_dir': source_dir if not existing_chunks else 'multiple',
        'chunk_size': args.chunk_size,
        'overlap': args.overlap,
        'n_chunks': len(all_chunks),
        'n_files': n_sources,
        'dim': dim,
        'storage_backend': storage_type,
        'embedding_backend': emb_backend,
        'embedding_model': emb_model,
    }
    with open(os.path.join(index_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    # Auto-lock after successful index
    lock_file = os.path.join(index_dir, ".locked")
    if not os.path.exists(lock_file):
        with open(lock_file, 'w') as f:
            f.write("locked\n")

    # Restore negative setting if we overrode it
    if ocr_neg != rag_settings.get('ocr_negative'):
        cfg = rag_settings.load()
        cfg['ocr_negative'] = not ocr_neg
        rag_settings.save(cfg)

    print(f"\nDone! Index '{args.name}' → {index_dir} [AUTO-LOCKED]")
    print(f"  {len(all_chunks)} chunks from {n_sources} files, {dim}-dim embeddings")
    return 0

def cmd_query(args):
    import numpy as np

    # Resolve settings fallback
    if args.top_k is None:
        args.top_k = rag_settings.get('top_k')

    index_dir = os.path.join(RAG_HOME, args.name)

    # Detect backend and check existence
    backend_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, backend_type)

    if not backend.exists():
        print(f"Index '{args.name}' not found. Run: rag.py index /path --name {args.name}")
        return 1

    chunks = backend.get_chunks()

    # Auto-resolve embedding model for this index
    resolved = resolve_embedding_for_index(args.name)
    if resolved['warning'] and resolved['warning'].startswith('ERROR'):
        print(resolved['warning'], file=sys.stderr)
        return 1
    if resolved['warning']:
        print(resolved['warning'], file=sys.stderr)

    # embed query using the index's model
    q_emb = embed_texts([args.query],
                        override_backend=resolved['backend'],
                        override_model=resolved['model'],
                        override_url=resolved['api_url'])
    _unload_model()

    # search
    search_results = backend.search(q_emb, args.top_k)

    # output
    OCR_NOTE = "[NOTE: This text was produced by OCR and may contain recognition errors — misspellings, garbled characters, or missing words.]\n"

    results = []
    for rank, (score, idx) in enumerate(search_results):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        is_ocr = c.get('ocr', False)
        results.append({
            'rank': rank + 1,
            'score': float(score),
            'source': c['source'],
            'chunk': f"{c['chunk']+1}/{c['of']}",
            'text': c['text'],
            'ocr': is_ocr,
        })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"=== RAG: \"{args.query}\" ({len(results)} results) ===\n")
        for r in results:
            ocr_str = " [OCR]" if r['ocr'] else ""
            print(f"--- [{r['rank']}] {r['source']} (chunk {r['chunk']}, score {r['score']:.3f}){ocr_str} ---")
            if r['ocr']:
                print(OCR_NOTE)
            print(r['text'][:3000])
            print()

    return 0

def cmd_list(args):
    if not os.path.exists(RAG_HOME):
        print("No indexes.")
        return 0
    found = False
    for name in sorted(os.listdir(RAG_HOME)):
        mp = os.path.join(RAG_HOME, name, "meta.json")
        if os.path.exists(mp):
            found = True
            locked = os.path.exists(os.path.join(RAG_HOME, name, ".locked"))
            lock_icon = " [LOCKED]" if locked else ""
            meta = rag_backends.get_index_meta_with_defaults(os.path.join(RAG_HOME, name))
            storage = meta.get('storage_backend', 'faiss')
            emb_backend = meta.get('embedding_backend', 'local')
            emb_model = meta.get('embedding_model', 'all-MiniLM-L6-v2')
            print(f"  {name}: {meta['n_chunks']} chunks from {meta['n_files']} files{lock_icon}")
            print(f"    Source: {meta.get('source_dir', '?')}  |  Storage: {storage}  |  Model: {emb_model} ({emb_backend})")
    if not found:
        print("No indexes.")
    return 0

def cmd_lock(args):
    index_dir = os.path.join(RAG_HOME, args.name)
    if not os.path.exists(os.path.join(index_dir, "meta.json")):
        print(f"Index '{args.name}' not found")
        return 1
    lock_file = os.path.join(index_dir, ".locked")
    with open(lock_file, 'w') as f:
        f.write("locked\n")
    print(f"Index '{args.name}' is now LOCKED. Use 'index --force' to overwrite.")
    return 0

def cmd_unlock(args):
    index_dir = os.path.join(RAG_HOME, args.name)
    lock_file = os.path.join(index_dir, ".locked")
    if os.path.exists(lock_file):
        os.remove(lock_file)
        print(f"Index '{args.name}' is now unlocked.")
    else:
        print(f"Index '{args.name}' was not locked.")
    return 0

def cmd_delete(args):
    import shutil
    index_dir = os.path.join(RAG_HOME, args.name)
    if not os.path.exists(index_dir):
        print(f"Index '{args.name}' not found")
        return 0
    lock_file = os.path.join(index_dir, ".locked")
    if os.path.exists(lock_file) and not args.force:
        print(f"Index '{args.name}' is LOCKED. Use --force to delete.")
        return 1
    shutil.rmtree(index_dir)
    print(f"Deleted index '{args.name}'")
    return 0

def cmd_gui(args):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import threading

    SUPPORTED_EXTS = tuple(EXTRACTORS.keys())
    _shutdown = threading.Event()  # signals all threads to stop

    # ── helpers (use module-level get_indexes / get_index_info) ──

    # ── splash screen ──
    SPLASH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_splash.png')
    splash = None
    if os.path.exists(SPLASH_PATH):
        splash = tk.Tk()
        splash.overrideredirect(True)  # borderless
        try:
            from PIL import Image, ImageTk
            img = Image.open(SPLASH_PATH)
            # Scale to fit nicely — cap at 600px wide
            max_w = 600
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(splash, image=photo, bg='black')
            lbl.image = photo  # prevent GC
            lbl.pack()
            # Center on screen
            splash.update_idletasks()
            sw = splash.winfo_screenwidth()
            sh = splash.winfo_screenheight()
            x = (sw - img.width) // 2
            y = (sh - img.height) // 2
            splash.geometry(f"{img.width}x{img.height}+{x}+{y}")
            splash.update()
            splash.after(2500, splash.destroy)
            splash.mainloop()
        except Exception:
            if splash and splash.winfo_exists():
                splash.destroy()
        splash = None

    # ── window ──
    root = tk.Tk()
    root.title("RAG-Narock")
    root.geometry("700x580")
    root.resizable(True, True)

    # ── Dark Norse theme ──
    BG       = '#0f1626'   # deep navy background
    BG2      = '#1a2332'   # slightly lighter panel bg
    BG3      = '#243044'   # raised elements (entries, listbox)
    FG       = '#d4d4dc'   # main text — pale silver
    FG_DIM   = '#7a8599'   # dimmed text
    ACCENT   = '#e8781e'   # fire orange — buttons, highlights
    ACCENT2  = '#ff9a3c'   # lighter orange — hover
    ICE      = '#5eb8d4'   # ice blue — info, links
    FIRE_RED = '#c0392b'   # deep red — toggle off / errors
    GREEN    = '#27ae60'   # toggle on

    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use('clam')

    # Global defaults
    style.configure('.', background=BG, foreground=FG, fieldbackground=BG3,
                    bordercolor=BG3, darkcolor=BG, lightcolor=BG2,
                    troughcolor=BG2, selectbackground=ACCENT, selectforeground='#ffffff',
                    font=('sans-serif', 10))

    # Frames
    style.configure('TFrame', background=BG)
    style.configure('TLabelframe', background=BG, foreground=ACCENT,
                    bordercolor='#2a3a52')
    style.configure('TLabelframe.Label', background=BG, foreground=ACCENT,
                    font=('sans-serif', 10, 'bold'))

    # Labels
    style.configure('TLabel', background=BG, foreground=FG)
    style.configure('Info.TLabel', foreground=ICE)
    style.configure('Count.TLabel', foreground=FG_DIM)
    style.configure('Title.TLabel', foreground=ACCENT, font=('sans-serif', 11, 'bold'))

    # Buttons — fire orange
    style.configure('TButton', background=ACCENT, foreground='#ffffff',
                    bordercolor=ACCENT, font=('sans-serif', 9, 'bold'), padding=(8, 4))
    style.map('TButton',
              background=[('active', ACCENT2), ('disabled', BG3)],
              foreground=[('disabled', FG_DIM)])

    # Primary action button — slightly different
    style.configure('Primary.TButton', background='#c0581e', foreground='#ffffff',
                    font=('sans-serif', 10, 'bold'), padding=(12, 5))
    style.map('Primary.TButton',
              background=[('active', ACCENT), ('disabled', BG3)])

    # Entry / Combobox
    style.configure('TEntry', fieldbackground=BG3, foreground=FG,
                    insertcolor=FG, bordercolor='#2a3a52')
    style.configure('TCombobox', fieldbackground=BG3, foreground=FG,
                    arrowcolor=ACCENT, bordercolor='#2a3a52')
    style.map('TCombobox', fieldbackground=[('readonly', BG3)],
              selectbackground=[('readonly', ACCENT)])
    # Combobox dropdown colors
    root.option_add('*TCombobox*Listbox.background', BG3)
    root.option_add('*TCombobox*Listbox.foreground', FG)
    root.option_add('*TCombobox*Listbox.selectBackground', ACCENT)
    root.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')

    # Checkbutton
    style.configure('TCheckbutton', background=BG, foreground=FG,
                    indicatorcolor=BG3)
    style.map('TCheckbutton',
              indicatorcolor=[('selected', ACCENT), ('!selected', BG3)],
              background=[('active', BG2)])

    # Progressbar — fire orange trough
    style.configure('TProgressbar', background=ACCENT, troughcolor=BG3,
                    bordercolor=BG3, darkcolor=ACCENT, lightcolor=ACCENT2)

    # Scrollbar
    style.configure('TScrollbar', background=BG2, troughcolor=BG,
                    arrowcolor=FG_DIM, bordercolor=BG)
    style.map('TScrollbar', background=[('active', BG3)])

    # ── busy state — disables all action buttons during background work ──
    _action_buttons = []  # populated after button creation

    def set_busy(busy, status_msg=None):
        """Enable/disable all action buttons. Non-blocking."""
        state = 'disabled' if busy else 'normal'
        for btn in _action_buttons:
            btn.config(state=state)
        if status_msg:
            log(status_msg)

    # ── index selector frame ──
    idx_frame = ttk.LabelFrame(root, text="Target Index", padding=8)
    idx_frame.pack(fill='x', padx=10, pady=(10, 5))

    ttk.Label(idx_frame, text="Index:").pack(side='left')
    idx_var = tk.StringVar()
    idx_combo = ttk.Combobox(idx_frame, textvariable=idx_var, width=20)
    indexes = get_indexes()
    idx_combo['values'] = indexes + ['(new index...)']
    if 'tech_books' in indexes:
        idx_var.set('tech_books')
    elif indexes:
        idx_var.set(indexes[0])
    else:
        idx_var.set('(new index...)')
    idx_combo.pack(side='left', padx=(5, 10))

    new_name_var = tk.StringVar()
    new_name_entry = ttk.Entry(idx_frame, textvariable=new_name_var, width=20)
    new_name_label = ttk.Label(idx_frame, text="Name:")

    def on_index_change(*_):
        if idx_var.get() == '(new index...)':
            new_name_label.pack(side='left')
            new_name_entry.pack(side='left', padx=(5, 0))
        else:
            new_name_label.pack_forget()
            new_name_entry.pack_forget()
    idx_combo.bind('<<ComboboxSelected>>', on_index_change)
    on_index_change()

    info_label = ttk.Label(idx_frame, text="", style='Info.TLabel')
    info_label.pack(side='right')

    def refresh_info(*_):
        name = idx_var.get()
        if name == '(new index...)':
            info_label.config(text="Will create new index")
        else:
            info = get_index_info(name)
            locked = os.path.exists(os.path.join(RAG_HOME, name, ".locked"))
            if info:
                lock_str = " [LOCKED]" if locked else ""
                info_label.config(text=f"{info['n_chunks']} chunks, {info['n_files']} files{lock_str}")
            else:
                info_label.config(text="")

    # ── file queue frame ──
    queue_frame = ttk.LabelFrame(root, text="File Queue", padding=8)
    queue_frame.pack(fill='both', expand=True, padx=10, pady=5)

    file_queue = []  # list of absolute paths
    queue_hashes = {}  # hash -> filename, updated incrementally
    queue_paths = set()  # fast path lookup

    # cached index state — loaded async, refreshed after each index operation
    _idx_cache = {'name': None, 'hashes': {}, 'sources': set()}

    def refresh_idx_cache_async(on_done=None):
        """Load index hashes + sources in a background thread."""
        name = idx_var.get()
        if name == '(new index...)':
            _idx_cache.update(name=None, hashes={}, sources=set())
            if on_done:
                on_done()
            return

        def worker():
            idx_dir = os.path.join(RAG_HOME, name)
            hashes = load_index_hashes(idx_dir)
            sources = set()
            cp = os.path.join(idx_dir, "chunks.json")
            if os.path.exists(cp):
                with open(cp, 'r') as f:
                    sources = set(c['source'] for c in json.load(f))
            # Apply on main thread
            def apply():
                _idx_cache['name'] = name
                _idx_cache['hashes'] = hashes
                _idx_cache['sources'] = sources
                if on_done:
                    on_done()
            root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    # load cache on startup (async)
    refresh_idx_cache_async()

    def on_combo_select(e=None):
        on_index_change()
        refresh_info()
        refresh_idx_cache_async()
    idx_combo.bind('<<ComboboxSelected>>', on_combo_select)
    refresh_info()

    queue_list = tk.Listbox(queue_frame, selectmode='extended', font=('monospace', 9),
                            bg=BG3, fg=FG, selectbackground=ACCENT, selectforeground='#ffffff',
                            highlightthickness=0, borderwidth=1, relief='flat')
    queue_scroll = ttk.Scrollbar(queue_frame, orient='vertical', command=queue_list.yview)
    queue_list.configure(yscrollcommand=queue_scroll.set)
    queue_list.pack(side='left', fill='both', expand=True)
    queue_scroll.pack(side='right', fill='y')

    def refresh_queue():
        queue_list.delete(0, 'end')
        for f in file_queue:
            ext = os.path.splitext(f)[1].lower()
            size_mb = os.path.getsize(f) / (1024 * 1024)
            queue_list.insert('end', f"  {os.path.basename(f)}  ({ext}, {size_mb:.1f} MB)")

    def _apply_candidates(candidates, label):
        """Apply hashed candidates to queue on main thread (fast dict lookups only)."""
        count = 0
        dups = []
        for fpath, fh in candidates:
            fname = os.path.basename(fpath)
            if fpath in queue_paths:
                dups.append((fname, "already in queue"))
            elif fh in queue_hashes:
                dups.append((fname, f"same content as '{queue_hashes[fh]}' in queue"))
            elif fh in _idx_cache['hashes']:
                dups.append((fname, f"already indexed as '{_idx_cache['hashes'][fh]}'"))
            elif fname in _idx_cache['sources']:
                dups.append((fname, "filename already in index"))
            else:
                file_queue.append(fpath)
                queue_hashes[fh] = fname
                queue_paths.add(fpath)
                count += 1
        refresh_queue()
        set_busy(False)
        progress.config(value=0, mode='determinate')
        msg = f"Added {count} files{label}"
        if dups:
            msg += f"\nSkipped {len(dups)} duplicate(s):"
            for name, reason in dups:
                msg += f"\n  - {name}: {reason}"
        log(msg)
        if dups:
            messagebox.showinfo("Duplicates skipped",
                f"{count} file(s) added, {len(dups)} duplicate(s) skipped.\n\n" +
                "\n".join(f"- {n}: {r}" for n, r in dups))

    # ── buttons frame ──
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill='x', padx=10, pady=5)

    def add_folder():
        folder = filedialog.askdirectory(title="Select folder of books")
        if not folder:
            return
        set_busy(True, f"Scanning {os.path.basename(folder)}...")
        progress.config(mode='indeterminate')
        progress.start(15)

        def worker():
            candidates = []
            for root_dir, dirs, fnames in os.walk(folder):
                if _shutdown.is_set():
                    return
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for f in sorted(fnames):
                    fpath = os.path.join(root_dir, f)
                    if os.path.splitext(f)[1].lower() not in SUPPORTED_EXTS:
                        continue
                    fh = file_hash(fpath)
                    candidates.append((fpath, fh))
            root.after(0, lambda: (progress.stop(), _apply_candidates(candidates, f" from {os.path.basename(folder)}")))

        threading.Thread(target=worker, daemon=True).start()

    def add_files():
        filetypes = [
            ("Supported files", "*.pdf *.epub *.mobi *.djvu *.djv *.txt *.md *.rst *.png *.jpg *.jpeg *.tiff *.tif *.bmp *.webp *.pbm *.pgm *.ppm *.pnm"),
            ("PDF", "*.pdf"), ("EPUB", "*.epub"), ("MOBI", "*.mobi"),
            ("DjVu", "*.djvu *.djv"),
            ("Images (OCR)", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.webp *.pbm *.pgm *.ppm *.pnm"),
            ("Text", "*.txt *.md *.rst"), ("All files", "*.*"),
        ]
        paths = filedialog.askopenfilenames(title="Select book files", filetypes=filetypes)
        if not paths:
            return
        set_busy(True, f"Hashing {len(paths)} file(s)...")
        progress.config(mode='indeterminate')
        progress.start(15)

        def worker():
            candidates = []
            for p in paths:
                if _shutdown.is_set():
                    return
                if os.path.splitext(p)[1].lower() not in SUPPORTED_EXTS:
                    continue
                fh = file_hash(p)
                candidates.append((p, fh))
            root.after(0, lambda: (progress.stop(), _apply_candidates(candidates, "")))

        threading.Thread(target=worker, daemon=True).start()

    def remove_selected():
        sel = sorted(queue_list.curselection(), reverse=True)
        for i in sel:
            removed = file_queue.pop(i)
            queue_paths.discard(removed)
            to_del = [h for h, n in queue_hashes.items() if n == os.path.basename(removed)]
            for h in to_del:
                del queue_hashes[h]
        refresh_queue()

    def clear_queue():
        file_queue.clear()
        queue_hashes.clear()
        queue_paths.clear()
        refresh_queue()

    btn_add_folder = ttk.Button(btn_frame, text="Add Folder", command=add_folder)
    btn_add_folder.pack(side='left', padx=2)
    btn_add_files = ttk.Button(btn_frame, text="Add Files", command=add_files)
    btn_add_files.pack(side='left', padx=2)
    btn_remove = ttk.Button(btn_frame, text="Remove Selected", command=remove_selected)
    btn_remove.pack(side='left', padx=2)
    btn_clear = ttk.Button(btn_frame, text="Clear", command=clear_queue)
    btn_clear.pack(side='left', padx=2)

    # OCR options
    ocr_var = tk.BooleanVar(value=rag_settings.get('force_ocr'))
    ocr_neg_var = tk.BooleanVar(value=rag_settings.get('ocr_negative'))
    ocr_lang_var = tk.StringVar(value=rag_settings.get('ocr_lang'))
    ttk.Entry(btn_frame, textvariable=ocr_lang_var, width=10).pack(side='right', padx=(0, 5))
    ttk.Label(btn_frame, text="OCR lang:").pack(side='right')
    ttk.Checkbutton(btn_frame, text="Negative", variable=ocr_neg_var).pack(side='right', padx=2)
    ttk.Checkbutton(btn_frame, text="Force OCR", variable=ocr_var).pack(side='right', padx=2)

    # ── progress + log frame ──
    log_frame = ttk.LabelFrame(root, text="Log", padding=8)
    log_frame.pack(fill='x', padx=10, pady=(5, 5))

    progress = ttk.Progressbar(log_frame, mode='determinate')
    progress.pack(fill='x', pady=(0, 5))

    log_text = tk.Text(log_frame, height=6, font=('monospace', 9), state='disabled', wrap='word',
                       bg=BG3, fg=ICE, insertbackground=FG,
                       highlightthickness=0, borderwidth=1, relief='flat')
    log_text.pack(fill='x')

    def log(msg):
        log_text.config(state='normal')
        log_text.insert('end', msg + "\n")
        log_text.see('end')
        log_text.config(state='disabled')

    # ── index button ──
    action_frame = ttk.Frame(root)
    action_frame.pack(fill='x', padx=10, pady=(0, 10))

    indexing = False

    def do_index():
        nonlocal indexing
        if indexing:
            return
        if not file_queue:
            messagebox.showwarning("No files", "Add some files first.")
            return

        # resolve index name
        name = idx_var.get()
        if name == '(new index...)':
            name = new_name_var.get().strip()
            if not name:
                messagebox.showwarning("No name", "Enter a name for the new index.")
                return
            if not re.match(r'^[\w\-]+$', name):
                messagebox.showwarning("Bad name", "Index name can only contain letters, numbers, - and _")
                return

        indexing = True
        set_busy(True)
        progress.config(mode='determinate', value=0)

        def worker():
            try:
                import numpy as np

                index_dir = os.path.join(RAG_HOME, name)
                os.makedirs(index_dir, exist_ok=True)

                total = len(file_queue)
                all_new_chunks = []
                skipped_files = []
                added_files = []
                new_file_hashes = {}
                force_ocr = ocr_var.get()
                neg = ocr_neg_var.get()
                lang = ocr_lang_var.get().strip() or 'eng'
                if lang != 'eng':
                    if not set_ocr_lang(lang):
                        root.after(0, lambda: messagebox.showerror("OCR language error",
                            f"Language '{lang}' not installed.\nInstall with:\nsudo apt install tesseract-ocr-{lang.replace('_', '-')}"))
                        return

                # Temporarily override negative setting for this run
                _saved_neg = rag_settings.get('ocr_negative')
                if neg != _saved_neg:
                    cfg = rag_settings.load()
                    cfg['ocr_negative'] = neg
                    rag_settings.save(cfg)

                opts = []
                if force_ocr:
                    opts.append("force-OCR")
                if neg:
                    opts.append("negative")
                if opts:
                    root.after(0, lambda: log(f"OCR mode: {', '.join(opts)}, lang={lang}"))

                # extract + chunk
                for i, fpath in enumerate(file_queue):
                    if _shutdown.is_set():
                        return
                    fname = os.path.basename(fpath)
                    root.after(0, lambda m=f"[{i+1}/{total}] Extracting: {fname}": log(m))
                    root.after(0, lambda v=(i / total) * 50: progress.config(value=v))

                    text, is_ocr = extract_file(fpath, force_ocr=force_ocr)
                    if not text:
                        skipped_files.append(fname)
                        root.after(0, lambda m=f"  SKIP: {fname} (no text)": log(m))
                        continue
                    chunks = chunk_text(text)
                    ocr_tag = " [OCR]" if is_ocr else ""
                    for ci, chunk in enumerate(chunks):
                        all_new_chunks.append({
                            'text': chunk,
                            'source': os.path.basename(fpath),
                            'chunk': ci,
                            'of': len(chunks),
                            'ocr': is_ocr,
                        })
                    added_files.append(fname)
                    new_file_hashes[file_hash(fpath)] = fname
                    root.after(0, lambda m=f"  -> {len(chunks)} chunks{ocr_tag}": log(m))

                if not all_new_chunks:
                    msg = "No text could be extracted from any file."
                    if skipped_files:
                        msg += f"\n\nSkipped ({len(skipped_files)}):\n" + "\n".join(f"  - {f}" for f in skipped_files)
                    root.after(0, lambda: log(msg))
                    root.after(0, lambda m=msg: messagebox.showwarning("Nothing indexed", m))
                    return

                # embed
                if _shutdown.is_set():
                    return
                root.after(0, lambda: log(f"Embedding {len(all_new_chunks)} chunks..."))
                root.after(0, lambda: progress.config(value=50))
                texts = [c['text'] for c in all_new_chunks]
                new_embs = embed_texts(texts)
                _unload_model()
                dim = new_embs.shape[1]

                root.after(0, lambda: progress.config(value=80))

                # load existing if present (backend-agnostic)
                existing_chunks = []
                existing_sources = set()
                detect_type = rag_backends.detect_backend(index_dir)
                detect_backend = rag_backends.get_backend(index_dir, detect_type)
                if detect_backend.exists():
                    existing_chunks = detect_backend.get_chunks()
                    existing_sources = set(c['source'] for c in existing_chunks)

                    keep_idx = [i for i, c in enumerate(all_new_chunks) if c['source'] not in existing_sources]
                    if not keep_idx:
                        root.after(0, lambda: log("All files already in index. Nothing new to add."))
                        return

                    filtered_chunks = [all_new_chunks[i] for i in keep_idx]
                    filtered_embs = new_embs[keep_idx]

                    skipped = len(all_new_chunks) - len(filtered_chunks)
                    if skipped:
                        root.after(0, lambda m=f"Skipping {skipped} chunks from existing sources": log(m))

                    # Read old embeddings via FAISS or re-embed (for sqlite-vec, rebuild is done in save)
                    if detect_type == 'faiss':
                        import faiss
                        old_index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
                        old_embs = faiss.rev_swig_ptr(old_index.get_xb(), old_index.ntotal * dim).reshape(old_index.ntotal, dim).copy()
                        combined_embs = np.vstack([old_embs, filtered_embs])
                    else:
                        # For sqlite-vec: append mode — just use new embeddings
                        combined_embs = filtered_embs
                    final_chunks = existing_chunks + filtered_chunks
                else:
                    combined_embs = new_embs
                    final_chunks = all_new_chunks

                # save via backend
                storage_type = rag_settings.get('storage_backend', 'faiss')
                if os.path.exists(os.path.join(index_dir, "index.faiss")) or os.path.exists(os.path.join(index_dir, "index.db")):
                    storage_type = rag_backends.detect_backend(index_dir)
                gui_backend = rag_backends.get_backend(index_dir, storage_type)

                if existing_chunks and storage_type == 'sqlite-vec':
                    # For sqlite-vec with existing data, use append
                    gui_backend.append(
                        [c for c in final_chunks if c['source'] not in existing_sources],
                        combined_embs, {})
                else:
                    gui_backend.save(final_chunks, combined_embs, {})

                # save hashes (merge with existing)
                all_hashes = gui_backend.get_hashes()
                all_hashes.update(new_file_hashes)
                gui_backend.save_hashes(all_hashes)

                # metadata
                n_sources = len(set(c['source'] for c in final_chunks))
                emb_backend = rag_settings.get('embedding_backend')
                emb_model = rag_settings.get('embedding_model') if emb_backend == 'local' else rag_settings.get('api_model')
                meta = {
                    'source_dir': 'multiple (gui)',
                    'chunk_size': rag_settings.get('chunk_size'),
                    'overlap': rag_settings.get('overlap'),
                    'n_chunks': len(final_chunks),
                    'n_files': n_sources,
                    'dim': dim,
                    'storage_backend': storage_type,
                    'embedding_backend': emb_backend,
                    'embedding_model': emb_model,
                }
                with open(os.path.join(index_dir, "meta.json"), 'w') as f:
                    json.dump(meta, f, indent=2)

                # Auto-lock
                lock_path = os.path.join(index_dir, ".locked")
                if not os.path.exists(lock_path):
                    with open(lock_path, 'w') as f:
                        f.write("locked\n")

                root.after(0, lambda: progress.config(value=100))
                root.after(0, lambda m=f"Done! '{name}': {len(final_chunks)} chunks from {n_sources} files [AUTO-LOCKED]": log(m))
                root.after(0, lambda: refresh_info())

                new_indexes = get_indexes()
                root.after(0, lambda: idx_combo.config(values=new_indexes + ['(new index...)']))

                summary = f"Added {len(added_files)} file(s) to '{name}'."
                if skipped_files:
                    summary += f"\n\nSkipped {len(skipped_files)} file(s) (no extractable text):\n"
                    summary += "\n".join(f"  - {f}" for f in skipped_files)
                if skipped_files:
                    root.after(0, lambda m=summary: messagebox.showwarning("Done (with warnings)", m))
                else:
                    root.after(0, lambda m=summary: messagebox.showinfo("Done", m))

            except Exception as e:
                root.after(0, lambda m=f"ERROR: {e}": log(m))
            finally:
                nonlocal indexing
                indexing = False
                reset_ocr_lang()
                # Restore negative setting if we overrode it
                if neg != _saved_neg:
                    cfg = rag_settings.load()
                    cfg['ocr_negative'] = _saved_neg
                    rag_settings.save(cfg)
                root.after(0, lambda: set_busy(False))
                root.after(0, lambda: file_queue.clear())
                root.after(0, lambda: queue_hashes.clear())
                root.after(0, lambda: queue_paths.clear())
                root.after(0, refresh_queue)
                # refresh cache async so main thread stays responsive
                root.after(0, lambda: refresh_idx_cache_async())

        threading.Thread(target=worker, daemon=True).start()

    def open_settings():
        from rag_settings import SettingsDialog
        SettingsDialog(root)

    settings_btn = ttk.Button(action_frame, text="Settings", command=open_settings)
    settings_btn.pack(side='left', padx=2)

    def open_editor():
        from rag_editor import EditorDialog
        def on_change():
            new_indexes = get_indexes()
            idx_combo.config(values=new_indexes + ['(new index...)'])
            refresh_info()
            refresh_idx_cache_async()
        EditorDialog(root, on_change=on_change)

    editor_btn = ttk.Button(action_frame, text="Index Editor", command=open_editor)
    editor_btn.pack(side='left', padx=2)

    index_btn = ttk.Button(action_frame, text="Add to Index", command=do_index, style='Primary.TButton')
    index_btn.pack(side='right', padx=2)

    # Register all action buttons for busy-state management
    _action_buttons.extend([btn_add_folder, btn_add_files, btn_remove, btn_clear, index_btn, settings_btn, editor_btn])

    count_label = ttk.Label(action_frame, text="", style='Count.TLabel')
    count_label.pack(side='left')

    def update_count(*_):
        if not _shutdown.is_set():
            count_label.config(text=f"{len(file_queue)} files in queue")
            root.after(500, update_count)
    update_count()

    def on_close():
        """Clean shutdown: unload model, signal threads, destroy window."""
        _shutdown.set()
        _unload_model()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def sigint_handler(sig, frame):
        _shutdown.set()
        _unload_model()
        root.after(0, root.destroy)
    signal.signal(signal.SIGINT, sigint_handler)

    root.mainloop()
    _unload_model()
    return 0

def cmd_settings(args):
    from rag_settings import SettingsTUI
    tui = SettingsTUI()
    tui.run()
    return 0

def cmd_editor(args):
    """Launch index editor (TUI or --gui for dialog)."""
    if getattr(args, 'gui', False):
        import tkinter as tk
        from rag_editor import EditorDialog
        root = tk.Tk()
        root.withdraw()
        EditorDialog(root)
        root.destroy()
    else:
        from rag_editor import EditorTUI
        tui = EditorTUI()
        tui.run()
    return 0

def main():
    p = argparse.ArgumentParser(description="RAG-Narock — local RAG for Claude Code (CPU-only)")
    sub = p.add_subparsers(dest='cmd')

    pi = sub.add_parser('index', help='Index a folder of documents')
    pi.add_argument('path', help='Folder of books/documents')
    pi.add_argument('--name', default='default', help='Index name')
    pi.add_argument('--chunk-size', type=int, default=None, help='Chunk size (default: from settings)')
    pi.add_argument('--overlap', type=int, default=None, help='Overlap (default: from settings)')
    pi.add_argument('--force', action='store_true', help='Overwrite a locked index')
    pi.add_argument('--append', action='store_true', help='Add to existing index without overwriting')
    pi.add_argument('--ocr', action='store_true', default=None, help='Force OCR on all PDFs (ignore text layers)')
    pi.add_argument('--ocr-lang', default=None, help='Tesseract language(s) for OCR (e.g. hin, chi_sim, eng+hin)')
    pi.add_argument('--ocr-negative', action='store_true', default=None, help='Invert image colors before OCR (helps with light-on-dark text)')
    pi.add_argument('--storage-backend', choices=['faiss', 'sqlite-vec'], default=None, help='Storage backend (default: from settings)')

    pq = sub.add_parser('query', help='Query the index')
    pq.add_argument('query', help='Search query text')
    pq.add_argument('--name', default='default', help='Index name')
    pq.add_argument('--top-k', type=int, default=None, help='Top K results (default: from settings)')
    pq.add_argument('--json', action='store_true', help='JSON output (for Claude)')

    sub.add_parser('list', help='List indexes')

    pl = sub.add_parser('lock', help='Lock an index to prevent overwriting')
    pl.add_argument('name', help='Index name to lock')

    pu = sub.add_parser('unlock', help='Unlock an index')
    pu.add_argument('name', help='Index name to unlock')

    pd = sub.add_parser('delete', help='Delete an index')
    pd.add_argument('name', help='Index name to delete')
    pd.add_argument('--force', action='store_true', help='Delete even if locked')

    sub.add_parser('gui', help='Open GUI to add books')
    sub.add_parser('settings', help='Open settings TUI (or --gui for dialog)')

    pe = sub.add_parser('editor', help='Edit/manage indexes (TUI or --gui)')
    pe.add_argument('--gui', action='store_true', help='Open GUI editor instead of TUI')

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return 1

    cmds = {'index': cmd_index, 'query': cmd_query, 'list': cmd_list,
            'lock': cmd_lock, 'unlock': cmd_unlock, 'delete': cmd_delete,
            'gui': cmd_gui, 'settings': cmd_settings, 'editor': cmd_editor}
    return cmds[args.cmd](args)

if __name__ == '__main__':
    def _cleanup_handler(sig, frame):
        _unload_model()
        sys.exit(1)
    signal.signal(signal.SIGINT, _cleanup_handler)
    signal.signal(signal.SIGTERM, _cleanup_handler)
    try:
        sys.exit(main())
    finally:
        _unload_model()
