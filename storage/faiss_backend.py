"""
storage/faiss_backend.py — FAISS backend implementation.
Ported from original reference.
"""

import os
import json

import numpy as np
import faiss

from .base import StorageBackend
from core.constants import INDEX_FAISS_FILENAME, CHUNKS_FILENAME, HASHES_FILENAME


class FaissBackend(StorageBackend):
    """FAISS flat inner-product index + JSON chunk/hash files.
    Same file layout as original: index.faiss, chunks.json, hashes.json."""

    @property
    def _faiss_path(self):
        return os.path.join(self.index_dir, INDEX_FAISS_FILENAME)

    @property
    def _chunks_path(self):
        return os.path.join(self.index_dir, CHUNKS_FILENAME)

    @property
    def _hashes_path(self):
        return os.path.join(self.index_dir, HASHES_FILENAME)

    def save(self, chunks, embeddings, hashes, **k):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        faiss.write_index(index, self._faiss_path)
        with open(self._chunks_path, 'w') as f:
            json.dump(chunks, f)
        self.save_hashes(hashes)

    def append(self, new_chunks, new_embeddings, new_hashes, **k):
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
        if not self.exists():
            return []
        index = faiss.read_index(self._faiss_path)
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((float(score), int(idx)))
        return results

    def remove_source(self, source_name):
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

    def export_source(self, source_name, output_dir):
        """Export one source to a .txt file in output_dir."""
        from core.index_manager import _export_filename

        all_chunks = self.get_chunks()
        if not all_chunks:
            if os.path.exists(self._chunks_path):
                with open(self._chunks_path, 'r', encoding='utf-8') as f:
                    all_chunks = json.load(f)

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
        index = faiss.read_index(self._faiss_path)
        return index.d

    def get_total(self):
        if not self.exists():
            return 0
        index = faiss.read_index(self._faiss_path)
        return index.ntotal
