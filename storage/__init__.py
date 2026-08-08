"""
storage package — pluggable backends for RAG indexes.

Public API (same as old rag_backends):
  from storage import get_backend, detect_backend, get_index_meta_with_defaults

Backends:
- faiss (default)
- sqlite-vec
- sqlite-doc (full document + context)

Extracted from rag_backends.py for modularity (Ch 10/11).
"""

"""
storage/__init__.py — re-exports the storage API.

Full implementations restored from original source of truth.
"""
from .base import (
    StorageBackend,
    META_DEFAULTS,
    get_index_meta_with_defaults,
    detect_backend,
)

from importlib.util import find_spec


BACKEND_DEPENDENCIES = {
    "faiss": {
        "module": "faiss",
        "package": "faiss-cpu",
    },
    "sqlite-vec": {
        "module": "sqlite_vec",
        "package": "sqlite-vec",
    },
    "sqlite-doc": {
        "module": "sqlite_vec",
        "package": "sqlite-vec",
    },
}


class MissingBackendDependency(RuntimeError):
    """Raised when a selected storage backend cannot run in this Python env."""

    def __init__(self, backend_type: str, module: str, package: str):
        self.backend_type = backend_type
        self.module = module
        self.package = package
        super().__init__(
            f"Storage backend '{backend_type}' requires Python package '{package}' "
            f"(module '{module}'). Install it with: python -m pip install {package}. "
            "Or choose another installed storage backend."
        )


def _module_is_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except Exception:
        return False


def get_backend_status():
    """Return install status for storage backends without importing them."""
    status = {}
    for name, dep in BACKEND_DEPENDENCIES.items():
        status[name] = {
            "installed": _module_is_available(dep["module"]),
            "module": dep["module"],
            "package": dep["package"],
        }
    return status


def get_missing_backend_warnings():
    """Human-readable warnings for optional storage backends not installed."""
    warnings = []
    for name, info in get_backend_status().items():
        if not info["installed"]:
            warnings.append(
                f"Storage backend '{name}' is unavailable because package "
                f"'{info['package']}' is not installed. Install with: "
                f"python -m pip install {info['package']}. You can keep using "
                "other installed storage backends."
            )
    return warnings


def _require_backend_dependency(backend_type: str):
    dep = BACKEND_DEPENDENCIES.get(backend_type, BACKEND_DEPENDENCIES["faiss"])
    if not _module_is_available(dep["module"]):
        raise MissingBackendDependency(backend_type, dep["module"], dep["package"])

def get_backend(index_dir: str, backend_type: str = "faiss") -> StorageBackend:
    if backend_type == "sqlite-doc":
        _require_backend_dependency("sqlite-doc")
        from .sqlite_doc_backend import SqliteDocBackend
        return SqliteDocBackend(index_dir)
    if backend_type == "sqlite-vec":
        _require_backend_dependency("sqlite-vec")
        from .sqlite_vec_backend import SqliteVecBackend
        return SqliteVecBackend(index_dir)
    _require_backend_dependency("faiss")
    from .faiss_backend import FaissBackend
    return FaissBackend(index_dir)


def __getattr__(name):
    if name == "FaissBackend":
        _require_backend_dependency("faiss")
        from .faiss_backend import FaissBackend
        return FaissBackend
    if name == "SqliteVecBackend":
        _require_backend_dependency("sqlite-vec")
        from .sqlite_vec_backend import SqliteVecBackend
        return SqliteVecBackend
    if name == "SqliteDocBackend":
        _require_backend_dependency("sqlite-doc")
        from .sqlite_doc_backend import SqliteDocBackend
        return SqliteDocBackend
    raise AttributeError(name)

__all__ = [
    'get_backend',
    'get_backend_status',
    'get_missing_backend_warnings',
    'detect_backend',
    'get_index_meta_with_defaults',
    'StorageBackend',
    'META_DEFAULTS',
    'MissingBackendDependency',
]
