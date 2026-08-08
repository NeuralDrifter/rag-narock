"""
ingestion/code.py — code specific helpers (chunking primary in chunking.py).
"""
from .chunking import chunk_code, _detect_language, _get_import_block

__all__ = ['chunk_code', '_detect_language', '_get_import_block']
