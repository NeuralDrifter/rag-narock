"""
core/index_manager.py — high-level index lifecycle operations.

Functions moved from the original rag.py god file (SRP extraction).
These are the "nouns" of index management: resolve, list, inspect, mutate, export.

Temporary imports from the flat rag_* modules will be cleaned up in later slices
when config/ and storage/ are fully in place.
"""

import json
import logging
import os
import re
import shutil

import config.settings as settings
import storage as backends  # new modular location (slice 3)

logger = logging.getLogger(__name__)

from .integrity import save_index_integrity
from .hashing import load_index_hashes  # not directly used here but available
from .constants import META_FILENAME, CHUNKS_FILENAME, INDEX_DB_FILENAME, LOCK_FILENAME


# ── Index directory resolution ──

def resolve_index_dir(name):
    """Given an index name, return its absolute directory path.
    Checks data_dir first, then external registry, falls back to data_dir/name for creation."""
    data_dir = settings.get_data_dir()
    local = os.path.join(data_dir, name)
    if os.path.exists(os.path.join(local, META_FILENAME)):
        return local
    # Check external registry
    for ext_path in settings.get_external_indexes():
        if os.path.basename(ext_path) == name and os.path.exists(os.path.join(ext_path, META_FILENAME)):
            return ext_path
    # Fallback: data_dir/name (for creation or not-yet-existing)
    return local


# ── Index management helpers (module-level) ──

def get_indexes():
    """Return list of existing index names (from data_dir + external registry)."""
    names = set()
    data_dir = settings.get_data_dir()
    if os.path.exists(data_dir):
        for n in os.listdir(data_dir):
            if os.path.exists(os.path.join(data_dir, n, META_FILENAME)):
                names.add(n)
    # External indexes (skip if name collision with local — local wins)
    for ext_path in settings.get_external_indexes():
        if os.path.exists(os.path.join(ext_path, META_FILENAME)):
            ext_name = os.path.basename(ext_path)
            if ext_name not in names:
                names.add(ext_name)
    return sorted(names)


def get_index_info(name):
    """Return meta.json dict for an index, or None."""
    mp = os.path.join(resolve_index_dir(name), META_FILENAME)
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f)
    return None


def get_index_sources(name):
    """Return dict of {source_name: chunk_count} for an index.
    Robust: tries backend, falls back to chunks.json for persisted data.
    """
    index_dir = resolve_index_dir(name)
    backend_type = backends.detect_backend(index_dir)
    backend = backends.get_backend(index_dir, backend_type)
    chunks = []
    try:
        chunks = backend.get_chunks() or []
    except Exception as e:
        logger.debug("get_chunks failed: %s", e)
    if not chunks:
        # fallback to persisted chunks.json (common after refactor stubs)
        cj = os.path.join(index_dir, CHUNKS_FILENAME)
        if os.path.exists(cj):
            try:
                with open(cj, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
            except Exception as e:
                logger.debug("chunks.json load failed: %s", e)
    sources = {}
    for c in chunks:
        src = c.get('source') or c.get('source_name')
        if src:
            sources[src] = sources.get(src, 0) + 1
    if not sources:
        # direct sqlite: prefer 'chunks' table for accurate *chunk* counts (documents table stores 1 row per source)
        dbp = os.path.join(index_dir, INDEX_DB_FILENAME)
        if os.path.exists(dbp):
            try:
                import sqlite3
                conn = sqlite3.connect(dbp)
                tables = [r[0] for r in conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]
                if 'chunks' in tables:
                    for s, c in conn.execute('SELECT source, COUNT(*) FROM chunks GROUP BY source').fetchall():
                        sources[s] = c
                elif 'documents' in tables:
                    # fallback, but this will be ~1 per source (full docs)
                    for s, c in conn.execute('SELECT source, COUNT(*) FROM documents GROUP BY source').fetchall():
                        sources[s] = c
                conn.close()
            except Exception as e:
                logger.debug("direct sqlite sources query failed: %s", e)
    return sources


def is_index_locked(name):
    """Check if index has a .locked file."""
    return os.path.exists(os.path.join(resolve_index_dir(name), LOCK_FILENAME))


def remove_source_from_index(index_name, source_name):
    """Remove all chunks for a source, rebuild index, update hashes+meta.
    Returns {'removed_chunks', 'remaining_chunks', 'remaining_files'}."""
    index_dir = resolve_index_dir(index_name)
    backend_type = backends.detect_backend(index_dir)
    backend = backends.get_backend(index_dir, backend_type)

    if not backend.exists():
        raise FileNotFoundError(f"Index '{index_name}' not found or incomplete")

    result = backend.remove_source(source_name)

    # Update meta
    mp = os.path.join(index_dir, META_FILENAME)
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
        meta['n_chunks'] = result['remaining_chunks']
        meta['n_files'] = result['remaining_files']
        with open(mp, 'w') as f:
            json.dump(meta, f, indent=2)

    save_index_integrity(index_dir)
    return result


def update_source_in_index(index_name, source_name, new_text):
    """Remove a source, chunk the new text, re-embed, and insert it back to the index.
    Returns {'removed_chunks', 'new_chunks', 'remaining_chunks', 'remaining_files'}."""
    index_dir = resolve_index_dir(index_name)
    backend_type = backends.detect_backend(index_dir)
    backend = backends.get_backend(index_dir, backend_type)

    if not backend.exists():
        raise FileNotFoundError(f"Index '{index_name}' not found or incomplete")

    # 1. Fetch chunk settings from meta.json
    meta = backends.get_index_meta_with_defaults(index_dir)
    chunk_size = meta.get('chunk_size')
    overlap = meta.get('overlap')

    # 2. Remove old source data from backend
    try:
        remove_result = remove_source_from_index(index_name, source_name)
        removed_count = remove_result['removed_chunks']
    except ValueError:
        # Source didn't exist or was already removed
        removed_count = 0

    # 3. Generate new chunks
    from ingestion.chunking import chunk_text
    chunks = chunk_text(new_text, chunk_size, overlap)
    
    new_chunks = []
    for ci, chunk in enumerate(chunks):
        new_chunks.append({
            'text': chunk,
            'source': source_name,
            'chunk': ci,
            'of': len(chunks),
            'ocr': False,
        })

    # 4. Embed the new chunks using the index's configured model/backend
    from indexing.embedder import resolve_embedding_for_index, embed_texts, _resolve_embedding_device, _unload_embedding_model

    emb_info = resolve_embedding_for_index(index_name)
    if emb_info.get('warning') and "ERROR" in emb_info['warning']:
        raise ValueError(emb_info['warning'])

    device = _resolve_embedding_device()
    texts = [c['text'] for c in new_chunks]
    embeddings = embed_texts(
        texts,
        override_backend=emb_info['backend'],
        override_model=emb_info['model'],
        override_url=emb_info['api_url'],
        device=device
    )
    _unload_embedding_model()

    # 5. Insert updated chunks into backend
    import hashlib
    new_hash = hashlib.sha256(new_text.encode('utf-8')).hexdigest()
    new_hashes = {new_hash: source_name}

    docs_arg = None
    if backend_type == 'sqlite-doc':
        docs_arg = [{
            'source': source_name,
            'full_text': new_text,
            'doc_type': 'book',
            'language': None,
            'ocr': False,
        }]

    if backend_type == 'sqlite-doc':
        backend.append(new_chunks, embeddings, new_hashes, documents=docs_arg)
    else:
        backend.append(new_chunks, embeddings, new_hashes)

    # 6. Re-calculate totals and update meta.json
    updated_chunks = backend.get_chunks()
    unique_sources = len(set(c.get('source') for c in updated_chunks))
    total_chunks = len(updated_chunks)

    meta['n_chunks'] = total_chunks
    meta['n_files'] = unique_sources

    mp = os.path.join(index_dir, META_FILENAME)
    with open(mp, 'w') as f:
        json.dump(meta, f, indent=2)

    save_index_integrity(index_dir)

    return {
        'removed_chunks': removed_count,
        'new_chunks': len(new_chunks),
        'remaining_chunks': total_chunks,
        'remaining_files': unique_sources,
    }


def delete_index(index_name, force=False):
    """Delete entire index directory. Raises ValueError if locked and not force."""
    index_dir = resolve_index_dir(index_name)
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Index '{index_name}' not found")
    if is_index_locked(index_name) and not force:
        raise ValueError(f"Index '{index_name}' is LOCKED. Unlock it first or use force=True.")
    shutil.rmtree(index_dir)
    # If it was an external index, unregister it
    settings.remove_external_index(index_dir)


# ── Export ───────────────────────────────────────────────────────────────────

def _export_filename(source_name):
    """Sanitize source name into a safe filename."""
    name = source_name.replace('/', '__').replace('\\', '__')
    name = name.replace(' ', '_')
    name = re.sub(r'[<>:"|?*\x00-\x1f]', '', name)
    name = name.strip('. ')
    if not name:
        name = 'unnamed'
    # Always use .txt — export produces plain text regardless of original format
    root, ext = os.path.splitext(name)
    if root:
        name = root + '.txt'
    else:
        name += '.txt'
    return name


def export_source(index_name, source_name, output_dir):
    """Export one source from an index to a file on disk. Thin dispatcher (G30 fixed)."""
    index_dir = resolve_index_dir(index_name)
    backend_type = backends.detect_backend(index_dir)
    backend = backends.get_backend(index_dir, backend_type)
    return backend.export_source(source_name, output_dir)


def export_index(index_name, output_dir, source_filter=None):
    """Export multiple sources from an index to files on disk.

    Returns dict with sources_exported, files_written, skipped.
    """
    index_dir = resolve_index_dir(index_name)
    backend_type = backends.detect_backend(index_dir)
    backend = backends.get_backend(index_dir, backend_type)

    if backend_type == 'sqlite-doc':
        docs = backend.list_documents()
        source_names = [d['source'] for d in docs]
    else:
        sources = get_index_sources(index_name)
        source_names = list(sources.keys())

    if source_filter:
        source_names = [s for s in source_names if source_filter in s]

    files_written = []
    skipped = []
    for src in source_names:
        try:
            result = backend.export_source(src, output_dir)
            files_written.extend(result['files_written'])
        except Exception as e:
            skipped.append({'source': src, 'error': str(e)})

    return {'sources_exported': len(files_written), 'files_written': files_written, 'skipped': skipped}
