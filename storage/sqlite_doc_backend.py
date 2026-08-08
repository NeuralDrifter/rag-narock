"""
storage/sqlite_doc_backend.py — SQLite-doc backend implementation.
Document-aware. Stores full documents + chunks + vectors + images.
Ported from original reference.
"""

import os
import json
import logging

import sqlite3
import sqlite_vec

from .base import StorageBackend
from core.constants import INDEX_DB_FILENAME
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SqliteDocBackend(StorageBackend):
    """Document-aware SQLite backend. Stores full documents alongside chunks,
    enabling context expansion, document retrieval, and chunk navigation."""

    _schema_checked = None  # tracks which db path has been migrated

    @property
    def _db_path(self):
        return os.path.join(self.index_dir, INDEX_DB_FILENAME)

    def _connect(self):
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
        """Migrate old schemas: add missing columns and tables."""
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
        except Exception as e:
            logger.debug("Migrate schema (add source column) failed: %s", e)
        try:
            tables = {r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if 'images' not in tables:
                self._create_images_table(db)
        except Exception as e:
            logger.debug("Migrate schema (create images table) failed: %s", e)
        try:
            tables = {r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if 'chunks_fts' not in tables:
                db.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text, chunk_id UNINDEXED)
                """)
                db.execute("INSERT INTO chunks_fts (rowid, text, chunk_id) SELECT id, text, id FROM chunks")
                db.commit()
        except Exception as e:
            logger.debug("Migrate schema (fts5) failed: %s", e)

    def _create_images_table(self, db):
        db.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                doc_id INTEGER REFERENCES documents(id),
                page_number INTEGER,
                nearest_chunk_id INTEGER REFERENCES chunks(id),
                image_data BLOB NOT NULL,
                mime_type TEXT DEFAULT 'image/png',
                width INTEGER,
                height INTEGER,
                original_size INTEGER,
                xref INTEGER
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_images_doc ON images(doc_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_images_chunk ON images(nearest_chunk_id)")
        db.commit()

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
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(text, chunk_id UNINDEXED)
        """)
        self._create_images_table(db)
        db.commit()

    def save(self, chunks, embeddings, hashes, **k):
        """Full write — replace all data.
        Supports documents= kwarg for full docs.
        """
        documents = k.get('documents')
        dim = embeddings.shape[1]
        db = self._connect()
        db.execute("DROP TABLE IF EXISTS chunks")
        db.execute("DROP TABLE IF EXISTS documents")
        db.execute("DROP TABLE IF EXISTS vec_index")
        db.execute("DROP TABLE IF EXISTS hashes")
        db.execute("DROP TABLE IF EXISTS chunks_fts")
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
            db.execute("INSERT INTO chunks_fts (rowid, text, chunk_id) VALUES (?, ?, ?)",
                       (i, c['text'], i))

        for h, fn in hashes.items():
            db.execute("INSERT INTO hashes (hash, filename) VALUES (?, ?)", (h, fn))

        db.commit()
        db.close()

    def append(self, new_chunks, new_embeddings, new_hashes, **k):
        """Merge new data into existing index."""
        documents = k.get('documents')
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
            db.execute("INSERT INTO chunks_fts (rowid, text, chunk_id) VALUES (?, ?, ?)",
                       (rid, c['text'], rid))

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
            score = self._cosine_to_similarity(distance)
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
        db.execute("DELETE FROM chunks_fts WHERE chunk_id IN ({})".format(
            ','.join('?' * len(ids))), ids)
        db.execute("DELETE FROM images WHERE nearest_chunk_id IN ({})".format(
            ','.join('?' * len(ids))), ids)
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

    def export_source(self, source_name, output_dir):
        """Export one source to a .txt file in output_dir."""
        from core.index_manager import _export_filename

        doc = self.get_document(source_name)
        if not doc:
            # Fallback: try from chunks
            all_chunks = self.get_chunks()
            chunks = [c for c in all_chunks if c.get('source') == source_name]
            if not chunks:
                raise FileNotFoundError(f"Source '{source_name}' not found")
            chunks.sort(key=lambda c: c.get('chunk', 0))
            text = '\n\n'.join(c['text'] for c in chunks)
            chunk_count = len(chunks)
        else:
            text = doc['full_text']
            chunk_count = len(self.get_document_chunks(source_name) or [])

        os.makedirs(output_dir, exist_ok=True)
        filename = _export_filename(source_name)
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)

        return {'files_written': [out_path], 'source': source_name, 'chunks_exported': chunk_count}

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
        except Exception as e:
            logger.debug("get_chunks fallback for old DB: %s", e)
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
            "AND c.id != ? ORDER BY c.chunk_idx",
            (doc_id, chunk_idx - context, chunk_idx + context, chunk_id)
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
            score = self._cosine_to_similarity(distance)

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

    def search_fts(self, query_text: str, top_k: int):
        db = self._connect()
        import re
        words = re.findall(r'\w+', query_text.lower())
        if not words:
            return []
        fts_query = ' OR '.join(words)
        try:
            rows = db.execute(
                "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (fts_query, top_k)
            ).fetchall()
        except Exception as e:
            logger.debug("FTS query failed: %s", e)
            rows = []
        db.close()
        return [(-float(row[1]), int(row[0])) for row in rows]

    def get_hits(self, chunk_ids: List[int], context: int = 0, source_filter: str = "") -> List[dict]:
        if not chunk_ids:
            return []
        db = self._connect()
        results = []
        for cid in chunk_ids:
            chunk_row = db.execute(
                "SELECT c.id, c.text, c.chunk_idx, c.chunk_of, c.doc_id, "
                "COALESCE(c.source, d.source, ''), COALESCE(d.ocr, 0) "
                "FROM chunks c LEFT JOIN documents d ON c.doc_id = d.id "
                "WHERE c.id = ?", (cid,)
            ).fetchone()
            if not chunk_row:
                continue

            source = chunk_row[5] or ''
            if source_filter and source_filter not in source:
                continue

            hit = {
                'id': chunk_row[0],
                'text': chunk_row[1],
                'chunk': chunk_row[2],
                'of': chunk_row[3],
                'source': source,
                'ocr': bool(chunk_row[6]) if chunk_row[6] is not None else False,
                'score': 1.0,
            }

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
        db.close()
        return results

    def save_images(self, images, doc_id_map, chunk_id_map):
        """Store extracted images.
        images: list of dicts with keys: source, page, data, width, height, xref, nearest_chunk
        doc_id_map: {source: doc_id}
        chunk_id_map: {(source, chunk_idx): chunk_id}
        """
        if not images:
            return
        db = self._connect()
        for img in images:
            did = doc_id_map.get(img['source'])
            cid = chunk_id_map.get((img['source'], img['nearest_chunk']))
            db.execute(
                "INSERT INTO images (doc_id, page_number, nearest_chunk_id, "
                "image_data, mime_type, width, height, original_size, xref) "
                "VALUES (?, ?, ?, ?, 'image/png', ?, ?, ?, ?)",
                (did, img['page'], cid, img['data'],
                 img['width'], img['height'], img.get('original_size', len(img['data'])),
                 img.get('xref'))
            )
        db.commit()
        db.close()

    def get_images_for_chunks(self, chunk_ids):
        """Return {chunk_id: [{'data': bytes, 'width': int, 'height': int}, ...]}"""
        if not chunk_ids:
            return {}
        db = self._connect()
        placeholders = ','.join('?' * len(chunk_ids))
        rows = db.execute(
            f"SELECT nearest_chunk_id, image_data, width, height "
            f"FROM images WHERE nearest_chunk_id IN ({placeholders}) "
            f"ORDER BY nearest_chunk_id, page_number",
            chunk_ids
        ).fetchall()
        db.close()
        result = {}
        for row in rows:
            chunk_id = row[0]
            if chunk_id not in result:
                result[chunk_id] = []
            result[chunk_id].append({
                'data': row[1], 'width': row[2], 'height': row[3],
            })
        return result

    def get_chunk_id_map(self):
        """Return {(source, chunk_idx): chunk_id} for all chunks."""
        db = self._connect()
        rows = db.execute("SELECT id, source, chunk_idx FROM chunks").fetchall()
        db.close()
        return {(r[1], r[2]): r[0] for r in rows}

    def get_doc_id_map(self):
        """Return {source: doc_id} for all documents."""
        db = self._connect()
        rows = db.execute("SELECT id, source FROM documents").fetchall()
        db.close()
        return {r[1]: r[0] for r in rows}

    def get_image_count(self):
        db = self._connect()
        try:
            row = db.execute("SELECT COUNT(*) FROM images").fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.debug("Image count query failed: %s", e)
            return 0
        finally:
            db.close()
