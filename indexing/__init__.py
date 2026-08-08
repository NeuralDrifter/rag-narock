"""
indexing package — embedding, indexing, query logic.

Clean separation from ingestion and UI.
"""
from . import embedder, indexer, querier
from .embedder import embed_text, get_embedder, unload_model, resolve_for_index
from .querier import query_index
from .indexer import index_chunks
