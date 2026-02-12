#!/usr/bin/env python3
"""
RAG-Narock Storage Backends — abstraction layer for index storage.

Supports:
  - FAISS + JSON (default, existing behavior)
  - SQLite-vec (single index.db file, optional)

Factory:
  get_backend(index_dir, backend_type="faiss") -> StorageBackend
  detect_backend(index_dir) -> str   # "faiss" | "sqlite-vec"

Meta helpers:
  get_index_meta_with_defaults(index_dir) -> dict
"""

import os, json
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

RAG_HOME = os.path.expanduser("~/.local/share/rag")

# Fields added to meta.json for new indexes; old indexes get these defaults
META_DEFAULTS = {
    "storage_backend": "faiss",
    "embedding_backend": "local",
    "embedding_model": "all-MiniLM-L6-v2",
}


def get_index_meta_with_defaults(index_dir: str) -> dict:
    """Load meta.json and fill missing fields with backward-compatible defaults."""
    mp = os.path.join(index_dir, "meta.json")
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
    if os.path.exists(os.path.join(index_dir, "index.db")):
        return "sqlite-vec"
    # Default / fallback — FAISS (even if index.faiss doesn't exist yet)
    return "faiss"


def get_backend(index_dir: str, backend_type: str = "faiss") -> "StorageBackend":
    """Factory: return the appropriate StorageBackend instance."""
    if backend_type == "sqlite-vec":
        return SqliteVecBackend(index_dir)
    return FaissBackend(index_dir)


# ── Abstract Base Class ──────────────────────────────────────────────────────

class StorageBackend(ABC):
    """Interface for RAG index storage."""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir

    @abstractmethod
    def save(self, chunks: List[dict], embeddings, hashes: dict):
        """Full write — replace all data."""
        ...

    @abstractmethod
    def append(self, new_chunks: List[dict], new_embeddings, new_hashes: dict):
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


# ── FAISS Backend ────────────────────────────────────────────────────────────

class FaissBackend(StorageBackend):
    """FAISS flat inner-product index + JSON chunk/hash files.
    Same file layout as original rag.py: index.faiss, chunks.json, hashes.json."""

    @property
    def _faiss_path(self):
        return os.path.join(self.index_dir, "index.faiss")

    @property
    def _chunks_path(self):
        return os.path.join(self.index_dir, "chunks.json")

    @property
    def _hashes_path(self):
        return os.path.join(self.index_dir, "hashes.json")

    def save(self, chunks, embeddings, hashes):
        import numpy as np
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        faiss.write_index(index, self._faiss_path)
        with open(self._chunks_path, 'w') as f:
            json.dump(chunks, f)
        self.save_hashes(hashes)

    def append(self, new_chunks, new_embeddings, new_hashes):
        import numpy as np
        import faiss
        dim = new_embeddings.shape[1]

        if os.path.exists(self._faiss_path):
            old_index = faiss.read_index(self._faiss_path)
            old_embs = faiss.rev_swig_ptr(
                old_index.get_xb(), old_index.ntotal * dim
            ).reshape(old_index.ntotal, dim).copy()
            with open(self._chunks_path, 'r') as f:
                old_chunks = json.load(f)
            combined_embs = np.vstack([old_embs, new_embeddings])
            combined_chunks = old_chunks + new_chunks
        else:
            combined_embs = new_embeddings
            combined_chunks = new_chunks

        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(combined_embs, dtype=np.float32))
        faiss.write_index(index, self._faiss_path)
        with open(self._chunks_path, 'w') as f:
            json.dump(combined_chunks, f)

        # Merge hashes
        existing = self.get_hashes()
        existing.update(new_hashes)
        self.save_hashes(existing)

    def search(self, query_embedding, top_k):
        import faiss
        index = faiss.read_index(self._faiss_path)
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((float(score), int(idx)))
        return results

    def remove_source(self, source_name):
        import numpy as np
        import faiss

        if not os.path.exists(self._chunks_path) or not os.path.exists(self._faiss_path):
            raise FileNotFoundError(f"Index incomplete in {self.index_dir}")

        with open(self._chunks_path, 'r') as f:
            chunks = json.load(f)

        old_index = faiss.read_index(self._faiss_path)
        dim = old_index.d

        keep_indices = []
        removed = 0
        for i, c in enumerate(chunks):
            if c['source'] == source_name:
                removed += 1
            else:
                keep_indices.append(i)

        if removed == 0:
            raise ValueError(f"Source '{source_name}' not found")

        all_vecs = faiss.rev_swig_ptr(
            old_index.get_xb(), old_index.ntotal * dim
        ).reshape(old_index.ntotal, dim).copy()

        if keep_indices:
            keep_vecs = all_vecs[keep_indices]
            keep_chunks = [chunks[i] for i in keep_indices]
        else:
            keep_vecs = np.zeros((0, dim), dtype=np.float32)
            keep_chunks = []

        new_index = faiss.IndexFlatIP(dim)
        if len(keep_vecs) > 0:
            new_index.add(keep_vecs)
        faiss.write_index(new_index, self._faiss_path)

        with open(self._chunks_path, 'w') as f:
            json.dump(keep_chunks, f)

        # Update hashes
        hashes = self.get_hashes()
        hashes = {h: fn for h, fn in hashes.items() if fn != source_name}
        self.save_hashes(hashes)

        remaining_files = len(set(c['source'] for c in keep_chunks))
        return {
            'removed_chunks': removed,
            'remaining_chunks': len(keep_chunks),
            'remaining_files': remaining_files,
        }

    def get_chunks(self):
        if not os.path.exists(self._chunks_path):
            return []
        with open(self._chunks_path, 'r') as f:
            return json.load(f)

    def get_hashes(self):
        if os.path.exists(self._hashes_path):
            with open(self._hashes_path) as f:
                return json.load(f)
        return {}

    def save_hashes(self, hashes):
        with open(self._hashes_path, 'w') as f:
            json.dump(hashes, f, indent=2)

    def exists(self):
        return os.path.exists(self._faiss_path)

    def get_dim(self):
        if not self.exists():
            return 0
        import faiss
        index = faiss.read_index(self._faiss_path)
        return index.d

    def get_total(self):
        if not self.exists():
            return 0
        import faiss
        index = faiss.read_index(self._faiss_path)
        return index.ntotal


# ── SQLite-vec Backend ───────────────────────────────────────────────────────

class SqliteVecBackend(StorageBackend):
    """Single index.db file using sqlite-vec for vector search."""

    @property
    def _db_path(self):
        return os.path.join(self.index_dir, "index.db")

    def _connect(self):
        import sqlite3
        import sqlite_vec
        db = sqlite3.connect(self._db_path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return db

    def _ensure_tables(self, db, dim: int):
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT,
                source TEXT,
                chunk_idx INTEGER,
                chunk_of INTEGER,
                ocr INTEGER DEFAULT 0
            )
        """)
        # vec0 virtual table for KNN — cosine distance
        db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_index
            USING vec0(embedding float[{dim}] distance_metric=cosine)
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS hashes (
                hash TEXT PRIMARY KEY,
                filename TEXT
            )
        """)
        db.commit()

    def save(self, chunks, embeddings, hashes):
        import struct
        dim = embeddings.shape[1]
        db = self._connect()
        # Drop and recreate for full write
        db.execute("DROP TABLE IF EXISTS chunks")
        db.execute("DROP TABLE IF EXISTS vec_index")
        db.execute("DROP TABLE IF EXISTS hashes")
        self._ensure_tables(db, dim)

        for i, c in enumerate(chunks):
            db.execute(
                "INSERT INTO chunks (id, text, source, chunk_idx, chunk_of, ocr) VALUES (?, ?, ?, ?, ?, ?)",
                (i, c['text'], c['source'], c['chunk'], c['of'], int(c.get('ocr', False)))
            )
            vec_bytes = embeddings[i].tobytes()
            db.execute("INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)", (i, vec_bytes))

        for h, fn in hashes.items():
            db.execute("INSERT INTO hashes (hash, filename) VALUES (?, ?)", (h, fn))

        db.commit()
        db.close()

    def append(self, new_chunks, new_embeddings, new_hashes):
        import struct
        dim = new_embeddings.shape[1]
        db = self._connect()
        self._ensure_tables(db, dim)

        # Find next id
        row = db.execute("SELECT COALESCE(MAX(id), -1) FROM chunks").fetchone()
        next_id = row[0] + 1

        for i, c in enumerate(new_chunks):
            rid = next_id + i
            db.execute(
                "INSERT INTO chunks (id, text, source, chunk_idx, chunk_of, ocr) VALUES (?, ?, ?, ?, ?, ?)",
                (rid, c['text'], c['source'], c['chunk'], c['of'], int(c.get('ocr', False)))
            )
            vec_bytes = new_embeddings[i].tobytes()
            db.execute("INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)", (rid, vec_bytes))

        for h, fn in new_hashes.items():
            db.execute("INSERT OR IGNORE INTO hashes (hash, filename) VALUES (?, ?)", (h, fn))

        db.commit()
        db.close()

    def search(self, query_embedding, top_k):
        db = self._connect()
        vec_bytes = query_embedding[0].tobytes()
        rows = db.execute(
            "SELECT rowid, distance FROM vec_index WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (vec_bytes, top_k)
        ).fetchall()
        db.close()
        # sqlite-vec cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score (1 - distance/2) for compatibility with FAISS inner product
        results = []
        for rowid, distance in rows:
            score = 1.0 - distance / 2.0
            results.append((score, int(rowid)))
        return results

    def remove_source(self, source_name):
        db = self._connect()

        # Count chunks to remove
        row = db.execute("SELECT COUNT(*) FROM chunks WHERE source = ?", (source_name,)).fetchone()
        removed = row[0]
        if removed == 0:
            db.close()
            raise ValueError(f"Source '{source_name}' not found")

        # Get ids to remove from vec_index
        ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE source = ?", (source_name,)).fetchall()]

        # Remove from chunks table
        db.execute("DELETE FROM chunks WHERE source = ?", (source_name,))

        # Remove from vec_index
        for rid in ids:
            db.execute("DELETE FROM vec_index WHERE rowid = ?", (rid,))

        # Remove matching hashes
        db.execute("DELETE FROM hashes WHERE filename = ?", (source_name,))

        remaining = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        remaining_files = db.execute("SELECT COUNT(DISTINCT source) FROM chunks").fetchone()[0]

        db.commit()
        db.close()

        return {
            'removed_chunks': removed,
            'remaining_chunks': remaining,
            'remaining_files': remaining_files,
        }

    def get_chunks(self):
        if not os.path.exists(self._db_path):
            return []
        db = self._connect()
        rows = db.execute("SELECT text, source, chunk_idx, chunk_of, ocr FROM chunks ORDER BY id").fetchall()
        db.close()
        return [
            {'text': r[0], 'source': r[1], 'chunk': r[2], 'of': r[3], 'ocr': bool(r[4])}
            for r in rows
        ]

    def get_hashes(self):
        if not os.path.exists(self._db_path):
            return {}
        db = self._connect()
        rows = db.execute("SELECT hash, filename FROM hashes").fetchall()
        db.close()
        return {r[0]: r[1] for r in rows}

    def save_hashes(self, hashes):
        db = self._connect()
        # Ensure table exists (might be called before save/append)
        db.execute("""
            CREATE TABLE IF NOT EXISTS hashes (
                hash TEXT PRIMARY KEY,
                filename TEXT
            )
        """)
        db.execute("DELETE FROM hashes")
        for h, fn in hashes.items():
            db.execute("INSERT INTO hashes (hash, filename) VALUES (?, ?)", (h, fn))
        db.commit()
        db.close()

    def exists(self):
        return os.path.exists(self._db_path)

    def get_dim(self):
        if not self.exists():
            return 0
        import sqlite3
        db = self._connect()
        try:
            # Query vec0 shadow table for dimension info
            row = db.execute("SELECT embedding FROM vec_index LIMIT 1").fetchone()
            if row and row[0]:
                import struct
                # Each float32 = 4 bytes
                return len(row[0]) // 4
            return 0
        except Exception:
            return 0
        finally:
            db.close()

    def get_total(self):
        if not self.exists():
            return 0
        db = self._connect()
        row = db.execute("SELECT COUNT(*) FROM chunks").fetchone()
        db.close()
        return row[0] if row else 0
