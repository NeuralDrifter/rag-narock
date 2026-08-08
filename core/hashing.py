"""
core/hashing.py — SHA-256 file hashing and index hash store helpers.

Extracted from rag.py (original god file) during modular refactor.
Preserves exact original behavior.
"""

import hashlib


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
    # Lazy import to avoid circular dependency (core → rag_backends → storage → core)
    from storage import get_backend, detect_backend
    backend = get_backend(index_dir, detect_backend(index_dir))
    return backend.get_hashes()


def save_index_hashes(index_dir: str, hashes: dict):
    """Save hashes to an index."""
    # Lazy import to avoid circular dependency (core → rag_backends → storage → core)
    from storage import get_backend, detect_backend
    backend = get_backend(index_dir, detect_backend(index_dir))
    backend.save_hashes(hashes)
