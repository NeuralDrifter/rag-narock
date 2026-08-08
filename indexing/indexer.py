"""indexing/indexer.py — high level chunk + embed + store pipeline (facade)."""
import numpy as np

from ingestion.chunking import chunk_text, chunk_code
from .embedder import embed_text
import storage as backends_mod
from core.hashing import file_hash
from config import settings as cfg

def index_chunks(chunks, embs, index_dir, storage_type=None, hashes=None, docs=None):
    """Store chunks + embeddings into the chosen backend."""
    if storage_type is None:
        storage_type = cfg.get('storage_backend') or 'faiss'
    backend = backends_mod.get_backend(index_dir, storage_type)
    if docs:
        backend.append(chunks, embs, hashes or {}, documents=docs)
    else:
        backend.append(chunks, embs, hashes or {})
    return 0

