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

import os, sys, json, gc, logging, argparse, re, unicodedata, hashlib, signal
from pathlib import Path
from typing import List, Optional
import rag_settings
import rag_backends

# Modular ingestion (complete)
try:
    from ingestion.registry import EXTRACTORS, EXTRACTORS_OCR, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS, MEDIA_EXTENSIONS, extract_file
except Exception as e:
    logging.getLogger(__name__).debug("EXTRACTORS import failed: %s", e)
    EXTRACTORS = {}

from core.hashing import file_hash, load_index_hashes, save_index_hashes
from core.integrity import (
    compute_index_integrity,
    save_index_integrity,
    check_index_integrity,
    suppress_index_integrity,
    _cli_integrity_gate,
)
from core.index_manager import (
    resolve_index_dir,
    get_indexes,
    get_index_info,
    get_index_sources,
    is_index_locked,
    remove_source_from_index,
    delete_index,
    _export_filename,
    export_source,
    export_index,
)

# Force CPU-only BEFORE any ML imports — but don't block XPU if EasyOCR
# GPU backend might be needed (env vars can't be un-done after torch imports)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Only force CPU for SYCL/oneAPI if EasyOCR GPU won't be needed
_ocr_may_use_gpu = (
    rag_settings.get('ocr_backend') == 'easyocr'
    and not rag_settings.get('disable_ocr')
)
if not _ocr_may_use_gpu and not rag_settings.get('gpu_indexing'):
    os.environ["ONEAPI_DEVICE_SELECTOR"] = "opencl:cpu"
    os.environ["SYCL_DEVICE_FILTER"] = ""
# Initialize environment checks and fallback paths
import sys
from config.settings import verify_python_environment
from indexing.embedder import setup_hf_home_fallback

warn_msg = verify_python_environment()
if warn_msg:
    print(warn_msg, file=sys.stderr)

setup_hf_home_fallback()

# --- Thin launcher section ---
# All heavy logic delegated to cli/, gui/, ingestion/, core/, etc. (clean code Ch 11)

import cli.main as _cli


def main():
    return _cli.main()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    def _cleanup_handler(sig, frame):
        try:
            # unload if defined
            pass
        except Exception as e:
            logger.debug("Cleanup handler: %s", e)
        sys.exit(1)
    signal.signal(signal.SIGINT, _cleanup_handler)
    signal.signal(signal.SIGTERM, _cleanup_handler)
    try:
        sys.exit(main())
    finally:
        pass
