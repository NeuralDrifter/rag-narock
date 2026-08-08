"""indexing/querier.py — query + retrieval facade (uses backends + embedder)."""
import numpy as np
from .embedder import embed_text
import storage as backends_mod
from core.index_manager import resolve_index_dir
from config import settings as cfg


def query_index(query_text, index_name='default', top_k=None, context=0, source_filter=None):
    """Query an index and return ranked results with metadata."""
    if top_k is None:
        top_k = cfg.get('top_k')
    index_dir = resolve_index_dir(index_name)
    backend_type = backends_mod.detect_backend(index_dir)
    backend = backends_mod.get_backend(index_dir, backend_type)
    if not backend.exists():
        return []
    q_embs = embed_text([query_text])
    if not q_embs or len(q_embs[0]) == 0:
        return []
    q_emb = np.array([q_embs[0]], dtype=np.float32)

    # 1. Semantic Search
    vec_results = backend.search(q_emb, top_k * 2)

    # 2. Keyword Search
    if hasattr(backend, 'search_fts'):
        fts_results = backend.search_fts(query_text, top_k * 2)
    else:
        fts_results = []

    # 3. Blending (RRF)
    if hasattr(backend, 'get_hits') and (vec_results or fts_results):
        scores = {}
        for rank, (score, cid) in enumerate(vec_results):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        for rank, (score, cid) in enumerate(fts_results):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_cids = [cid for cid, score in ranked[:top_k * 2]]
        
        # 4. Fetch Details
        results = backend.get_hits(top_cids, context, source_filter or "")
        
        # Preserve RRF rank order
        cid_to_hit = {h['id']: h for h in results}
        final_results = []
        for cid in top_cids:
            if cid in cid_to_hit:
                h = cid_to_hit[cid]
                h['score'] = scores[cid]
                final_results.append(h)
                if len(final_results) >= top_k:
                    break
        return final_results

    # Fallback
    return backend.search_with_context(q_emb, top_k, context=context, source_filter=source_filter or "")
