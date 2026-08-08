#!/usr/bin/env python3
"""
RAG Backends — thin shim (Clean Code).

Logic in storage/
"""
from storage import *
from storage.base import StorageBackend, META_DEFAULTS, get_index_meta_with_defaults, detect_backend
from storage import get_backend


# ── Abstract Base Class ──────────────────────────────────────────────────────

