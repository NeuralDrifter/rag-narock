"""
storage/base.py — abstract base + shared helpers.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from core.constants import META_FILENAME, INDEX_DB_FILENAME, COSINE_TO_SIMILARITY_SCALE

logger = logging.getLogger(__name__)

# Fields added to meta.json for new indexes; old indexes get these defaults
META_DEFAULTS = {
    "storage_backend": "faiss",
    "embedding_backend": "local",
    "embedding_model": "all-MiniLM-L6-v2",
}


def get_index_meta_with_defaults(index_dir: str) -> dict:
    """Load meta.json and fill missing fields with backward-compatible defaults."""
    mp = os.path.join(index_dir, META_FILENAME)
    meta = {}
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    for key, default in META_DEFAULTS.items():
        if key not in meta:
            meta[key] = default
    return meta


def detect_backend(index_dir: str) -> str:
    """Detect which storage backend an index uses by checking files on disk."""
    db_path = os.path.join(index_dir, INDEX_DB_FILENAME)
    if os.path.exists(db_path):
        import sqlite3
        try:
            db = sqlite3.connect(db_path)
            tables = {r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            db.close()
            if 'documents' in tables:
                return "sqlite-doc"
        except Exception as e:
            logger.debug("detect_backend failed: %s", e)
        return "sqlite-vec"
    # Default / fallback — FAISS (even if index.faiss doesn't exist yet)
    return "faiss"


class StorageBackend(ABC):
    """Interface for RAG index storage."""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir

    @abstractmethod
    def save(self, chunks: List[dict], embeddings, hashes: dict, **kwargs):
        """Full write — replace all data."""
        ...

    @abstractmethod
    def append(self, new_chunks: List[dict], new_embeddings, new_hashes: dict, **kwargs):
        """Merge new data into existing index."""
        ...

    @abstractmethod
    def search(self, query_embedding, top_k: int) -> List[Tuple[float, int]]:
        """KNN search. Returns [(score, chunk_index), ...]."""
        ...

    @abstractmethod
    def remove_source(self, source_name: str) -> dict:
        """Remove all chunks for a source. Returns {removed, remaining, remaining_files}."""
        ...

    @abstractmethod
    def export_source(self, source_name: str, output_dir: str) -> dict:
        """Export a single source to a file. Returns {files_written, source, chunks_exported}."""
        ...

    @abstractmethod
    def get_chunks(self) -> List[dict]:
        """Return all chunk metadata."""
        ...

    @abstractmethod
    def get_hashes(self) -> dict:
        """Return {hash: filename} dict."""
        ...

    @abstractmethod
    def save_hashes(self, hashes: dict):
        """Write hashes to storage."""
        ...

    @abstractmethod
    def exists(self) -> bool:
        """Check if this index has been created/populated."""
        ...

    @abstractmethod
    def get_dim(self) -> int:
        """Return embedding dimension, or 0 if not yet created."""
        ...

    @abstractmethod
    def get_total(self) -> int:
        """Return total number of indexed vectors."""
        ...

    # ── Shared helpers for sqlite backends ─────────────────────────────────

    def _cosine_to_similarity(self, distance: float) -> float:
        """Convert sqlite-vec cosine distance (0=identical, 2=opposite) to [0,1] similarity."""
        return 1.0 - distance / COSINE_TO_SIMILARITY_SCALE

    def _get_dim_from_vec_table(self, db) -> int:
        """Read embedding dimension from vec_index virtual table. Returns 0 on failure."""
        try:
            row = db.execute("SELECT embedding FROM vec_index LIMIT 1").fetchone()
            if row and row[0]:
                return len(row[0]) // 4  # float32 = 4 bytes
        except Exception as e:
            logger.debug("get_dim_from_vec_table failed: %s", e)
        return 0

    def _get_total_from_chunks(self, db) -> int:
        """Read total chunk count. Returns 0 if table missing."""
        try:
            row = db.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.debug("get_total_from_chunks failed: %s", e)
            return 0

    # polymorphic document retrieval fallbacks
    def get_document(self, source_name: str):
        chunks = self.get_document_chunks(source_name)
        if not chunks:
            return None
        text = "\n\n".join(c.get('text', '') for c in chunks)
        # return compatible dictionary format
        return {
            'source': source_name,
            'full_text': text,
            'doc_type': chunks[0].get('doc_type', 'book') if chunks else 'book',
            'language': chunks[0].get('language', 'eng') if chunks else 'eng',
            'ocr': any(c.get('ocr', False) for c in chunks)
        }

    def get_document_chunks(self, source_name: str):
        chunks = self.get_chunks()
        matching = []
        for c in chunks:
            if c.get('source') == source_name:
                matching.append(c)
        try:
            matching.sort(key=lambda x: x.get('chunk', 0))
        except Exception:
            pass
        return matching

    def list_documents(self):
        return []

    def search_with_context(self, query_embedding, top_k: int,
                            context: int = 0, source_filter: str = "") -> List[dict]:
        """Base implementation of context-aware search.
        Falls back to standard KNN search if the backend does not natively support documents."""
        chunks = self.get_chunks()
        fetch_k = top_k * 10 if source_filter else top_k
        search_results = self.search(query_embedding, fetch_k)

        results = []
        for score, idx in search_results:
            if idx < 0 or idx >= len(chunks):
                continue
            c = chunks[idx]
            source = c.get('source', '')
            if source_filter and source_filter not in source:
                continue
            is_ocr = c.get('ocr', False)
            results.append({
                'id': idx,
                'score': float(score),
                'source': source,
                'chunk': c.get('chunk', 0),
                'of': c.get('of', 0),
                'text': c.get('text', ''),
                'ocr': is_ocr,
                'adjacent': [],
            })
            if len(results) >= top_k:
                break
        return results
