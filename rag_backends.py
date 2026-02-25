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
    db_path = os.path.join(index_dir, "index.db")
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
        except Exception:
            pass
        return "sqlite-vec"
    # Default / fallback — FAISS (even if index.faiss doesn't exist yet)
    return "faiss"


def get_backend(index_dir: str, backend_type: str = "faiss") -> "StorageBackend":
    """Factory: return the appropriate StorageBackend instance."""
    if backend_type == "sqlite-doc":
        return SqliteDocBackend(index_dir)
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


# ── SQLite-doc Backend ──────────────────────────────────────────────────────

class SqliteDocBackend(StorageBackend):
    """Document-aware SQLite backend. Stores full documents alongside chunks,
    enabling context expansion, document retrieval, and chunk navigation."""

    _schema_checked = None  # tracks which db path has been migrated

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
        # One-time schema migration for old DBs
        if SqliteDocBackend._schema_checked != self._db_path:
            self._migrate_schema(db)
            SqliteDocBackend._schema_checked = self._db_path
        return db

    def _migrate_schema(self, db):
        """Add source column to chunks table if missing (backward compat)."""
        try:
            cols = {r[1] for r in db.execute("PRAGMA table_info(chunks)").fetchall()}
            if cols and 'source' not in cols:
                db.execute("ALTER TABLE chunks ADD COLUMN source TEXT")
                db.execute(
                    "UPDATE chunks SET source = ("
                    "SELECT d.source FROM documents d WHERE d.id = chunks.doc_id"
                    ") WHERE doc_id IS NOT NULL AND source IS NULL"
                )
                db.commit()
        except Exception:
            pass

    def _ensure_tables(self, db, dim: int):
        db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                source TEXT UNIQUE,
                full_text TEXT,
                doc_type TEXT DEFAULT 'book',
                language TEXT,
                ocr INTEGER DEFAULT 0
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER REFERENCES documents(id),
                source TEXT,
                text TEXT,
                chunk_idx INTEGER,
                chunk_of INTEGER
            )
        """)
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

    def save(self, chunks, embeddings, hashes, documents=None):
        """Full write — replace all data.
        documents: list of dicts with keys: source, full_text, doc_type, language, ocr"""
        dim = embeddings.shape[1]
        db = self._connect()
        db.execute("DROP TABLE IF EXISTS chunks")
        db.execute("DROP TABLE IF EXISTS documents")
        db.execute("DROP TABLE IF EXISTS vec_index")
        db.execute("DROP TABLE IF EXISTS hashes")
        self._ensure_tables(db, dim)

        # Build doc_id mapping from documents list
        doc_ids = {}  # source -> doc_id
        if documents:
            for di, doc in enumerate(documents):
                db.execute(
                    "INSERT INTO documents (id, source, full_text, doc_type, language, ocr) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (di, doc['source'], doc.get('full_text', ''),
                     doc.get('doc_type', 'book'), doc.get('language'),
                     int(doc.get('ocr', False)))
                )
                doc_ids[doc['source']] = di

        for i, c in enumerate(chunks):
            doc_id = doc_ids.get(c['source'])
            db.execute(
                "INSERT INTO chunks (id, doc_id, source, text, chunk_idx, chunk_of) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (i, doc_id, c['source'], c['text'], c['chunk'], c['of'])
            )
            db.execute("INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)",
                       (i, embeddings[i].tobytes()))

        for h, fn in hashes.items():
            db.execute("INSERT INTO hashes (hash, filename) VALUES (?, ?)", (h, fn))

        db.commit()
        db.close()

    def append(self, new_chunks, new_embeddings, new_hashes, documents=None):
        """Merge new data into existing index."""
        dim = new_embeddings.shape[1]
        db = self._connect()
        self._ensure_tables(db, dim)

        # Insert new documents
        doc_ids = {}
        if documents:
            for doc in documents:
                # Use INSERT OR IGNORE for idempotency
                db.execute(
                    "INSERT OR IGNORE INTO documents (source, full_text, doc_type, language, ocr) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (doc['source'], doc.get('full_text', ''),
                     doc.get('doc_type', 'book'), doc.get('language'),
                     int(doc.get('ocr', False)))
                )
                row = db.execute("SELECT id FROM documents WHERE source = ?",
                                 (doc['source'],)).fetchone()
                if row:
                    doc_ids[doc['source']] = row[0]

        # For chunks whose source was already in documents, look up existing doc_id
        if not documents:
            for c in new_chunks:
                row = db.execute("SELECT id FROM documents WHERE source = ?",
                                 (c['source'],)).fetchone()
                if row:
                    doc_ids[c['source']] = row[0]

        # Find next chunk id
        row = db.execute("SELECT COALESCE(MAX(id), -1) FROM chunks").fetchone()
        next_id = row[0] + 1

        for i, c in enumerate(new_chunks):
            rid = next_id + i
            doc_id = doc_ids.get(c['source'])
            db.execute(
                "INSERT INTO chunks (id, doc_id, source, text, chunk_idx, chunk_of) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rid, doc_id, c['source'], c['text'], c['chunk'], c['of'])
            )
            db.execute("INSERT INTO vec_index (rowid, embedding) VALUES (?, ?)",
                       (rid, new_embeddings[i].tobytes()))

        for h, fn in new_hashes.items():
            db.execute("INSERT OR IGNORE INTO hashes (hash, filename) VALUES (?, ?)", (h, fn))

        db.commit()
        db.close()

    def search(self, query_embedding, top_k):
        db = self._connect()
        vec_bytes = query_embedding[0].tobytes()
        rows = db.execute(
            "SELECT rowid, distance FROM vec_index WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (vec_bytes, top_k)
        ).fetchall()
        db.close()
        results = []
        for rowid, distance in rows:
            score = 1.0 - distance / 2.0
            results.append((score, int(rowid)))
        return results

    def remove_source(self, source_name):
        db = self._connect()

        # Find chunks by source column OR via documents join
        ids = [r[0] for r in db.execute(
            "SELECT c.id FROM chunks c LEFT JOIN documents d ON c.doc_id = d.id "
            "WHERE c.source = ? OR d.source = ?",
            (source_name, source_name)).fetchall()]

        if not ids:
            db.close()
            raise ValueError(f"Source '{source_name}' not found")

        for rid in ids:
            db.execute("DELETE FROM vec_index WHERE rowid = ?", (rid,))
        db.execute("DELETE FROM chunks WHERE id IN ({})".format(
            ','.join('?' * len(ids))), ids)
        db.execute("DELETE FROM documents WHERE source = ?", (source_name,))
        db.execute("DELETE FROM hashes WHERE filename = ?", (source_name,))

        remaining = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        remaining_files = db.execute(
            "SELECT COUNT(DISTINCT source) FROM chunks WHERE source IS NOT NULL AND source != ''"
        ).fetchone()[0]

        db.commit()
        db.close()
        return {
            'removed_chunks': len(ids),
            'remaining_chunks': remaining,
            'remaining_files': remaining_files,
        }

    def get_chunks(self):
        if not os.path.exists(self._db_path):
            return []
        db = self._connect()
        # Use c.source directly; fall back to d.source for old DBs without source column
        try:
            rows = db.execute(
                "SELECT c.text, COALESCE(c.source, d.source, ''), c.chunk_idx, c.chunk_of, "
                "COALESCE(d.ocr, 0) "
                "FROM chunks c LEFT JOIN documents d ON c.doc_id = d.id "
                "ORDER BY c.id"
            ).fetchall()
        except Exception:
            # Fallback for DBs where chunks has no source column
            rows = db.execute(
                "SELECT c.text, COALESCE(d.source, ''), c.chunk_idx, c.chunk_of, "
                "COALESCE(d.ocr, 0) "
                "FROM chunks c LEFT JOIN documents d ON c.doc_id = d.id "
                "ORDER BY c.id"
            ).fetchall()
        db.close()
        return [
            {'text': r[0], 'source': r[1] or '', 'chunk': r[2], 'of': r[3],
             'ocr': bool(r[4])}
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
        db = self._connect()
        try:
            row = db.execute("SELECT embedding FROM vec_index LIMIT 1").fetchone()
            if row and row[0]:
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

    # ── Document-aware methods (sqlite-doc only) ────────────────────────────

    def get_document(self, source: str) -> Optional[dict]:
        """Return full document text + metadata for a source path."""
        db = self._connect()
        row = db.execute(
            "SELECT id, source, full_text, doc_type, language, ocr FROM documents "
            "WHERE source = ?", (source,)
        ).fetchone()
        db.close()
        if not row:
            return None
        return {
            'id': row[0], 'source': row[1], 'full_text': row[2],
            'doc_type': row[3], 'language': row[4], 'ocr': bool(row[5]),
        }

    def get_document_by_chunk_id(self, chunk_id: int) -> Optional[dict]:
        """Return full document from any chunk hit."""
        db = self._connect()
        row = db.execute(
            "SELECT d.id, d.source, d.full_text, d.doc_type, d.language, d.ocr "
            "FROM documents d JOIN chunks c ON d.id = c.doc_id "
            "WHERE c.id = ?", (chunk_id,)
        ).fetchone()
        db.close()
        if not row:
            return None
        return {
            'id': row[0], 'source': row[1], 'full_text': row[2],
            'doc_type': row[3], 'language': row[4], 'ocr': bool(row[5]),
        }

    def get_adjacent_chunks(self, chunk_id: int, context: int = 1) -> List[dict]:
        """Return N chunks before/after the given chunk from the same document."""
        db = self._connect()
        # Get the chunk's doc_id and chunk_idx
        row = db.execute(
            "SELECT doc_id, chunk_idx FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            db.close()
            return []
        doc_id, chunk_idx = row

        rows = db.execute(
            "SELECT c.id, c.text, c.chunk_idx, c.chunk_of, "
            "COALESCE(c.source, d.source, ''), COALESCE(d.ocr, 0) "
            "FROM chunks c LEFT JOIN documents d ON c.doc_id = d.id "
            "WHERE c.doc_id = ? AND c.chunk_idx BETWEEN ? AND ? "
            "ORDER BY c.chunk_idx",
            (doc_id, chunk_idx - context, chunk_idx + context)
        ).fetchall()
        db.close()
        return [
            {'id': r[0], 'text': r[1], 'chunk': r[2], 'of': r[3],
             'source': r[4] or '', 'ocr': bool(r[5]) if r[5] is not None else False,
             'is_hit': r[2] == chunk_idx}
            for r in rows
        ]

    def get_document_chunks(self, source: str) -> List[dict]:
        """Return all chunks for a source, ordered by chunk_idx."""
        db = self._connect()
        rows = db.execute(
            "SELECT c.id, c.text, c.chunk_idx, c.chunk_of "
            "FROM chunks c LEFT JOIN documents d ON c.doc_id = d.id "
            "WHERE c.source = ? OR d.source = ? ORDER BY c.chunk_idx",
            (source, source)
        ).fetchall()
        db.close()
        return [
            {'id': r[0], 'text': r[1], 'chunk': r[2], 'of': r[3]}
            for r in rows
        ]

    def list_documents(self) -> List[dict]:
        """Return all documents with metadata (no full_text)."""
        if not os.path.exists(self._db_path):
            return []
        db = self._connect()
        rows = db.execute(
            "SELECT d.id, d.source, d.doc_type, d.language, d.ocr, "
            "COUNT(c.id) as chunk_count "
            "FROM documents d LEFT JOIN chunks c ON d.id = c.doc_id "
            "GROUP BY d.id ORDER BY d.source"
        ).fetchall()
        db.close()
        return [
            {'id': r[0], 'source': r[1], 'doc_type': r[2], 'language': r[3],
             'ocr': bool(r[4]), 'chunk_count': r[5]}
            for r in rows
        ]

    def search_with_context(self, query_embedding, top_k: int,
                            context: int = 0, source_filter: str = "") -> List[dict]:
        """Search + auto-expand with adjacent chunks.
        Returns list of dicts with hit info and optional adjacent chunks."""
        db = self._connect()
        vec_bytes = query_embedding[0].tobytes()
        # Over-fetch when filtering by source to ensure we get enough results
        fetch_k = top_k * 10 if source_filter else top_k
        rows = db.execute(
            "SELECT rowid, distance FROM vec_index WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (vec_bytes, fetch_k)
        ).fetchall()

        results = []
        for rowid, distance in rows:
            score = 1.0 - distance / 2.0

            # Get chunk info with source
            chunk_row = db.execute(
                "SELECT c.id, c.text, c.chunk_idx, c.chunk_of, c.doc_id, "
                "COALESCE(c.source, d.source, ''), COALESCE(d.ocr, 0) "
                "FROM chunks c LEFT JOIN documents d ON c.doc_id = d.id "
                "WHERE c.id = ?", (int(rowid),)
            ).fetchone()
            if not chunk_row:
                continue

            source = chunk_row[5] or ''
            # Apply source filter
            if source_filter and source_filter not in source:
                continue

            hit = {
                'id': chunk_row[0],
                'text': chunk_row[1],
                'chunk': chunk_row[2],
                'of': chunk_row[3],
                'source': source,
                'ocr': bool(chunk_row[6]) if chunk_row[6] is not None else False,
                'score': score,
            }

            # Expand with adjacent chunks if requested
            if context > 0 and chunk_row[4] is not None:
                adj_rows = db.execute(
                    "SELECT c.id, c.text, c.chunk_idx, c.chunk_of "
                    "FROM chunks c WHERE c.doc_id = ? "
                    "AND c.chunk_idx BETWEEN ? AND ? "
                    "AND c.id != ? ORDER BY c.chunk_idx",
                    (chunk_row[4], chunk_row[2] - context,
                     chunk_row[2] + context, chunk_row[0])
                ).fetchall()
                hit['adjacent'] = [
                    {'id': r[0], 'text': r[1], 'chunk': r[2], 'of': r[3]}
                    for r in adj_rows
                ]
            else:
                hit['adjacent'] = []

            results.append(hit)
            if len(results) >= top_k:
                break

        db.close()
        return results
