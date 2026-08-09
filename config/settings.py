"""
config/settings.py — pure configuration (no UI).

This module owns:
- Settings schema (TABS), DEFAULTS, SCHEMA
- load / save / get
- data directory and external index registry helpers
- DANGEROUS_KEYS for embedding model changes
- Optional dependency checks

Extracted during modular refactor (SRP / Ch 10). Original behavior preserved exactly.
UI code (TUI/Dialog) remains in rag_settings.py temporarily until split into tui/ + gui/.
"""

import os
import sys
import json
import importlib.util
import shutil
import subprocess
from pathlib import Path

SETTINGS_DIR = os.path.expanduser("~/.local/share/rag")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")

# ── Schema ──────────────────────────────────────────────────────────────────
# Each item: (key, label, type, options_or_none, default)
# type: 'choice', 'toggle', 'text'

TABS = [
    ("Models", [
        ("embedding_backend", "Backend", "choice", [
            ("local",    "Local (CPU)"),
            ("ollama",   "Ollama (GPU)"),
            ("lmstudio", "LM Studio (GPU)"),
        ], "local"),
        ("gpu_indexing",  "GPU Indexing",  "toggle", None, False),
        ("conda_env",     "Conda Env",    "text",   None, "ai-env"),
        ("embedding_model", "Local Model", "choice", [
            ("all-MiniLM-L6-v2",                        "all-MiniLM-L6-v2 (English, 384d, fast)"),
            ("all-MiniLM-L12-v2",                       "all-MiniLM-L12-v2 (English, 384d, balanced)"),
            ("paraphrase-multilingual-MiniLM-L12-v2",   "multilingual-MiniLM-L12 (50+ langs, 384d)"),
            ("all-mpnet-base-v2",                       "all-mpnet-base-v2 (English, 768d, best quality)"),
        ], "all-MiniLM-L6-v2"),
        ("api_model",    "API Model",    "text", None, "nomic-embed-text"),
        ("ollama_url",   "Ollama URL",   "text", None, "http://localhost:11434"),
        ("lmstudio_url", "LM Studio URL","text", None, "http://localhost:1234"),
    ]),
    ("OCR", [
        ("disable_ocr",    "Disable OCR",      "toggle", None, False),
        ("ocr_backend",    "OCR Backend",      "choice", [
            ("chrome",    "Chrome Screen AI (Fastest)"),
            ("tesseract", "Tesseract (CPU)"),
            ("easyocr",   "EasyOCR (GPU/XPU)"),
        ], "chrome"),
        ("ocr_lang",       "OCR Language",     "text",   None, "eng"),
        ("force_ocr",      "Force OCR",        "toggle", None, False),
        ("ocr_negative",   "OCR Negative",     "toggle", None, False),
        ("split_spreads",  "Split Spreads",    "toggle", None, False),
        ("render_dpi",     "Render DPI",       "choice", [
            (150, "150"), (200, "200"), (300, "300"), (400, "400"),
        ], 300),
        ("min_image_size", "Min Image Size",   "choice", [
            (100, "100 px"), (200, "200 px"), (300, "300 px"), (400, "400 px"),
        ], 200),
    ]),
    ("Audio", [
        ("disable_transcription", "Disable Transcription", "toggle", None, False),
        ("transcription_backend", "Transcription Backend", "choice", [
            ("auto",            "Auto-detect"),
            ("faster-whisper",  "faster-whisper"),
            ("whisper",         "openai-whisper"),
            ("whisper-cli",     "whisper CLI"),
        ], "auto"),
        ("transcription_model", "Whisper Model", "choice", [
            ("tiny",   "tiny"),
            ("base",   "base"),
            ("small",  "small"),
            ("medium", "medium"),
            ("large",  "large"),
        ], "base"),
        ("transcription_language", "Language", "text", None, ""),
        ("transcription_device", "Device", "choice", [
            ("auto", "Auto"),
            ("cpu",  "CPU"),
            ("xpu",  "XPU"),
            ("cuda", "CUDA"),
        ], "auto"),
        ("prefer_subtitles", "Prefer Subtitles", "toggle", None, True),
        ("subtitle_language", "Subtitle Language", "text", None, ""),
        ("external_subtitles", "External Subtitles", "toggle", None, False),
        ("subtitle_provider", "Subtitle Provider", "choice", [
            ("subdl", "SubDL"),
        ], "subdl"),
        ("subdl_api_key", "SubDL API Key", "text", None, ""),
    ]),
    ("RAG", [
        ("data_dir", "Data Directory", "text", None, "~/.local/share/rag"),
        ("storage_backend", "Storage Backend",  "choice", [
            ("faiss",       "FAISS (default)"),
            ("sqlite-vec",  "SQLite-vec"),
            ("sqlite-doc",  "SQLite-doc (document-aware)"),
        ], "faiss"),
        ("chunk_size",      "Chunk Size",       "choice", [
            (500, "500"), (1000, "1000"), (1500, "1500"), (2000, "2000"), (3000, "3000"),
        ], 1500),
        ("overlap",         "Overlap",          "choice", [
            (50, "50"), (100, "100"), (200, "200"), (300, "300"),
        ], 200),
        ("top_k",           "Top K Results",    "choice", [
            (3, "3"), (5, "5"), (10, "10"), (15, "15"), (20, "20"),
        ], 5),
        ("min_chunk_length","Min Chunk Length",  "choice", [
            (20, "20"), (50, "50"), (100, "100"),
        ], 50),
        ("code_chunk_size", "Code Chunk Size",  "choice", [
            (1500, "1500"), (2000, "2000"), (3000, "3000"), (4000, "4000"), (5000, "5000"),
        ], 3000),
        ("code_overlap",    "Code Overlap",     "choice", [
            (100, "100"), (200, "200"), (300, "300"), (500, "500"),
        ], 200),
        ("extract_images",  "Extract Images",   "toggle", None, False),
        ("max_image_dim",   "Max Image Dim",    "choice", [
            (512, "512 px"), (1024, "1024 px"), (2048, "2048 px"),
        ], 1024),
    ]),
]

# Flat defaults for quick access
DEFAULTS = {}
SCHEMA = {}  # key -> (label, type, options, default, tab_name)
for tab_name, items in TABS:
    for key, label, typ, options, default in items:
        DEFAULTS[key] = default
        SCHEMA[key] = (label, typ, options, default, tab_name)


# ── Optional Dependencies ──────────────────────────────────────────────────

def _dep_check_pymupdf():
    return importlib.util.find_spec('fitz') is not None

def _dep_check_ebooklib():
    return importlib.util.find_spec('ebooklib') is not None

def _dep_check_mobi():
    return importlib.util.find_spec('mobi') is not None

def _dep_check_bs4():
    return importlib.util.find_spec('bs4') is not None

def _dep_check_tesseract():
    if shutil.which('tesseract') is not None:
        return True
    return any(os.path.exists(path) for path in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ))

def _dep_check_pytesseract():
    return importlib.util.find_spec('pytesseract') is not None

def _dep_check_pdf2image():
    return importlib.util.find_spec('pdf2image') is not None

def _dep_check_pillow():
    return importlib.util.find_spec('PIL') is not None

def _dep_check_easyocr():
    return importlib.util.find_spec('easyocr') is not None

def _dep_check_ffmpeg():
    return shutil.which('ffmpeg') is not None

def _dep_check_faster_whisper():
    return importlib.util.find_spec('faster_whisper') is not None

def _dep_check_openai_whisper():
    return importlib.util.find_spec('whisper') is not None

def _dep_check_playwright():
    return importlib.util.find_spec('playwright') is not None

def _dep_check_playwright_browsers():
    """Check if Playwright has browsers installed."""
    try:
        r = subprocess.run([sys.executable, '-m', 'playwright', 'install', '--dry-run'],
                           capture_output=True, text=True, timeout=10)
        return 'chromium' not in r.stdout.lower() or r.returncode == 0
    except Exception:
        return False

OPTIONAL_DEPS = [
    ("Documents", [
        ("PyMuPDF",       _dep_check_pymupdf,       "pip install PyMuPDF",         None,                        "PDF text extraction"),
        ("ebooklib",      _dep_check_ebooklib,       "pip install ebooklib",        None,                        "EPUB support"),
        ("mobi",          _dep_check_mobi,           "pip install mobi",            None,                        "MOBI support"),
        ("BeautifulSoup", _dep_check_bs4,            "pip install beautifulsoup4",  None,                        "HTML parsing (URL indexing)"),
        ("Pillow",        _dep_check_pillow,          "pip install Pillow",          None,                        "Image processing"),
    ]),
    ("OCR", [
        ("Tesseract",     _dep_check_tesseract,      None,                          "sudo apt install tesseract-ocr", "OCR engine (system binary)"),
        ("pytesseract",   _dep_check_pytesseract,    "pip install pytesseract",     None,                        "Python Tesseract bindings"),
        ("pdf2image",     _dep_check_pdf2image,      "pip install pdf2image",       None,                        "PDF page rasterization for OCR"),
        ("EasyOCR",       _dep_check_easyocr,        "pip install easyocr",         None,                        "GPU-accelerated OCR"),
    ]),
    ("Transcription", [
        ("FFmpeg",        _dep_check_ffmpeg,          None,                          "sudo apt install ffmpeg",   "Audio/video processing (system binary)"),
        ("faster-whisper", _dep_check_faster_whisper, "pip install faster-whisper",  None,                        "Fast speech-to-text"),
        ("openai-whisper", _dep_check_openai_whisper, "pip install openai-whisper",  None,                        "OpenAI speech-to-text"),
    ]),
    ("Web / URL", [
        ("Playwright",    _dep_check_playwright,      "pip install playwright",      None,                        "JS-rendered page fetching"),
    ]),
]


def check_all_deps():
    """Return dict of {name: bool} for all optional dependencies."""
    results = {}
    for group_label, deps in OPTIONAL_DEPS:
        for name, check_fn, pip_cmd, apt_cmd, desc in deps:
            try:
                results[name] = check_fn()
            except Exception:
                results[name] = False
    return results


# Settings whose change makes existing embeddings incompatible
DANGEROUS_KEYS = {'embedding_backend', 'embedding_model', 'api_model', 'storage_backend'}

_settings_cache = None


# ── Load / Save / Get ───────────────────────────────────────────────────────

def load():
    """Load settings, merging with defaults for any missing keys."""
    cfg = dict(DEFAULTS)
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            for key, val in saved.items():
                if key in SCHEMA:
                    cfg[key] = val
            if 'external_indexes' in saved and isinstance(saved['external_indexes'], list):
                cfg['external_indexes'] = saved['external_indexes']
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save(cfg):
    """Save settings to disk."""
    global _settings_cache
    _settings_cache = dict(cfg)
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)


def get(key):
    """Get a single setting value (loads from disk once, cached)."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = load()
    return _settings_cache.get(key, DEFAULTS.get(key))


def set(key, value):
    """Set a single setting value and save to disk."""
    global _settings_cache
    cfg = load()
    cfg[key] = value
    save(cfg)
    _settings_cache = cfg


def get_data_dir():
    """Return resolved absolute data directory path."""
    raw = get('data_dir')
    return os.path.abspath(os.path.expanduser(raw)) if raw else os.path.expanduser("~/.local/share/rag")


def get_external_indexes():
    """Return list of registered external index absolute paths."""
    cfg = load()
    return cfg.get('external_indexes', [])


def add_external_index(path):
    """Register an external index path in settings. Returns True if added."""
    path = os.path.abspath(os.path.expanduser(path))
    cfg = load()
    externals = cfg.get('external_indexes', [])
    if path in externals:
        return False
    externals.append(path)
    cfg['external_indexes'] = externals
    save(cfg)
    return True


def remove_external_index(path):
    """Unregister an external index path from settings. Returns True if removed."""
    path = os.path.abspath(os.path.expanduser(path))
    cfg = load()
    externals = cfg.get('external_indexes', [])
    if path not in externals:
        return False
    externals.remove(path)
    cfg['external_indexes'] = externals
    save(cfg)
    return True


def verify_python_environment():
    """Verify packages and optional backend availability. Returns warning text or None."""
    import sys
    import importlib.util
    
    required = {
        'numpy': 'numpy',
        'sentence_transformers': 'sentence-transformers',
        'fitz': 'PyMuPDF'
    }
    missing = []
    for module, pkg in required.items():
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(pkg)
        except Exception:
            missing.append(pkg)
            
    messages = []
    if missing:
        messages.append(
            f"WARNING: Your current Python environment ({sys.executable}) is missing required package(s): {', '.join(missing)}.\n"
            f"Please ensure you are using the correct Python environment or install them using:\n"
            f"  pip install {' '.join(missing)}"
        )

    try:
        from storage import get_missing_backend_warnings
        backend_warnings = get_missing_backend_warnings()
    except Exception as e:
        backend_warnings = [f"Storage backend dependency check failed: {e}"]

    if backend_warnings:
        messages.append(
            "WARNING: Some storage backends are not available in this Python environment.\n"
            + "\n".join(f"  - {warning}" for warning in backend_warnings)
        )

    return "\n\n".join(messages) if messages else None
