"""
storage/sqlite_vec_backend.py — SQLite-vec backend implementation.
Ported from original reference.
"""

import os
import json
import struct

import sqlite3
import sqlite_vec

from .base import StorageBackend
from core.constants import INDEX_DB_FILENAME


class SqliteVecBackend(StorageBackend):
    """Single index.db file using sqlite-vec for vector search."""

    @property
    def _db_path(self):
        return os.path.join(self.index_dir, INDEX_DB_FILENAME)

    def _connect(self):
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

    def save(self, chunks, embeddings, hashes, **k):
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

    def append(self, new_chunks, new_embeddings, new_hashes, **k):
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
            score = self._cosine_to_similarity(distance)
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

    def export_source(self, source_name, output_dir):
        """Export one source to a .txt file in output_dir."""
        from core.index_manager import _export_filename

        all_chunks = self.get_chunks()
        chunks = [c for c in all_chunks if c.get('source') == source_name]
        if not chunks:
            raise FileNotFoundError(f"Source '{source_name}' not found")

        chunks.sort(key=lambda c: c.get('chunk', 0))
        os.makedirs(output_dir, exist_ok=True)
        filename = _export_filename(source_name)
        out_path = os.path.join(output_dir, filename)

        text = '\n\n'.join(c['text'] for c in chunks)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return {'files_written': [out_path], 'source': source_name, 'chunks_exported': len(chunks)}

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
        db = self._connect()
        try:
            return self._get_dim_from_vec_table(db)
        finally:
            db.close()

    def get_total(self):
        if not self.exists():
            return 0
        db = self._connect()
        try:
            return self._get_total_from_chunks(db)
        finally:
            db.close()
