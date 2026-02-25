#!/usr/bin/env python3
"""
RAG-Narock MCP Server — local-only, stdio transport.
Exposes the RAG index as tools for any MCP-compatible LLM CLI.

Register with Claude Code:
    claude mcp add rag -s user --transport stdio -- /home/mike/miniforge3/envs/ai-env/bin/python3 /home/mike/tools/rag_mcp.py
"""

import os, sys, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_settings
import rag_backends
import rag

# Force CPU-only, offline, no GPU interference — same as rag.py
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ONEAPI_DEVICE_SELECTOR"] = "opencl:cpu"
os.environ["SYCL_DEVICE_FILTER"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="RAG-Narock",
    instructions="Local RAG system. Workflow: 1) rag_list to see indexes, 2) rag_sources to find document names, 3) rag_query with source_filter to search within a specific document.",
)


def _list_index_names():
    """Fresh scan of available indexes every time."""
    return rag.get_indexes()


def _get_index_info(name):
    """Load metadata for an index (with backward-compatible defaults)."""
    index_dir = rag.resolve_index_dir(name)
    return rag_backends.get_index_meta_with_defaults(index_dir)


def _mcp_integrity_gate(index_name):
    """Check index integrity for MCP. Returns None if OK, or error string.
    Hard refuses — never serves potentially tampered or unverified data."""
    index_dir = rag.resolve_index_dir(index_name)
    result = rag.check_index_integrity(index_dir)
    if result['ok']:
        return None
    if result.get('untracked'):
        return (f"UNVERIFIED INDEX: '{index_name}' has no integrity record and cannot be served. "
                f"IMPORTANT: Tell the user this index needs to be verified before it can be used. "
                f"They should open the RAG editor (rag.py editor) and press 'h' to hash it, "
                f"or run: rag.py integrity --rehash --name {index_name}")
    details = "; ".join(result['details'])
    return (f"INTEGRITY ERROR: Index '{index_name}' failed integrity check ({details}). "
            f"Refusing to serve data — it may have been tampered with. "
            f"IMPORTANT: Warn the user about this integrity failure. "
            f"They should open the RAG editor (rag.py editor) to inspect and fix it, "
            f"or run: rag.py verify --name {index_name}")


def _available_indexes_str():
    """Human-readable summary of all indexes, freshly scanned."""
    names = _list_index_names()
    if not names:
        return "No indexes available."
    lines = []
    for name in names:
        info = _get_index_info(name)
        locked = rag.is_index_locked(name)
        lock_str = " [LOCKED]" if locked else ""
        # Integrity check
        index_dir = rag.resolve_index_dir(name)
        integrity = rag.check_index_integrity(index_dir)
        if not integrity['ok']:
            warn_str = " [UNVERIFIED]" if integrity.get('untracked') else " [INTEGRITY WARNING]"
        else:
            warn_str = ""
        if info and 'n_chunks' in info:
            storage = info.get('storage_backend', 'faiss')
            emb_model = info.get('embedding_model', '?')
            index_type = info.get('index_type', 'book')
            n_files = info.get('n_files', 0)
            type_str = f" ({index_type})" if index_type != 'book' else ""
            large_hint = ' — use source_filter or rag_sources(filter="...") to narrow searches' if n_files > 100 else ""
            lines.append(f"  {name}: {info['n_chunks']} chunks from {n_files} files{lock_str}{warn_str}{type_str} [{storage}, {emb_model}]{large_hint}")
            if storage == 'sqlite-doc':
                lines.append(f"    -> Supports: context expansion, document retrieval, chunk navigation")
        else:
            lines.append(f"  {name}: (metadata missing){lock_str}{warn_str}")
    return "\n".join(lines)


@mcp.tool()
def rag_list() -> str:
    """List available RAG indexes with their stats. Call this FIRST to discover which indexes exist before querying."""
    result = _available_indexes_str()
    if result == "No indexes available.":
        return result
    return f"Available RAG indexes:\n{result}\n\nUse rag_query with the index_name parameter to search a specific index."


_RAG_WARNING = "WARNING: Only use text returned above. Do NOT fabricate, hallucinate, or guess content that is not in these results. If the answer is not here, say so and suggest refining the query."


def _build_result_header(index_name: str, query: str, n_results: int,
                         source_filter: str, auto_filtered: bool,
                         seen_sources: set) -> str:
    """Build a result header with filter status and source summary."""
    header = f"=== RAG '{index_name}': \"{query}\" ({n_results} results"
    if source_filter:
        how = "auto-detected" if auto_filtered else "filtered"
        header += f", {how}: {source_filter}"
    header += ") ==="
    # Multi-source, no filter: show SOURCES summary + TIP
    if not source_filter and len(seen_sources) > 1:
        source_list = ", ".join(sorted(seen_sources))
        header += f"\nSOURCES: {source_list}"
        first_source = sorted(seen_sources)[0]
        header += f'\nTIP: call rag_query again with source_filter="{first_source}" to focus on one document'
    header += f"\n{_RAG_WARNING}"
    return header


def _auto_detect_source_filter(query: str, backend, backend_type: str) -> str:
    """If the query mentions a source name, return it as a filter. Min 8 char match, case-insensitive."""
    if backend_type == 'sqlite-doc':
        docs = backend.list_documents()
        source_names = [d['source'] for d in docs]
    else:
        hashes = backend.get_hashes()
        source_names = list(hashes.values()) if hashes else []

    query_lower = query.lower()
    best_match = ""
    best_len = 0
    for name in source_names:
        # Strip extension for matching
        stem = name.rsplit('.', 1)[0] if '.' in name else name
        stem_lower = stem.lower()
        if len(stem_lower) >= 8 and stem_lower in query_lower:
            if len(stem_lower) > best_len:
                best_match = name
                best_len = len(stem_lower)
        # Also try words from the stem (e.g. "Red Badge of Courage" from "The Red Badge of Courage")
        # Match substrings of the stem that appear in the query
        words = stem_lower.split()
        if len(words) >= 3:
            # Try progressively shorter subsequences from the stem
            for start in range(len(words)):
                for end in range(len(words), start, -1):
                    sub = " ".join(words[start:end])
                    if len(sub) >= 8 and sub in query_lower:
                        if len(sub) > best_len:
                            best_match = name
                            best_len = len(sub)
    return best_match


@mcp.tool()
def rag_query(query: str, index_name: str = "", top_k: int = 0,
              context: int = 0, source_filter: str = "") -> str:
    """Search indexed books for relevant passages. Use source_filter to limit results to one document (e.g. source_filter="Red Badge"). Returns top-K chunks with source, chunk number, and relevance score. If index_name is empty or invalid, returns a list of available indexes so you can pick one."""
    import numpy as np

    # If no index specified or invalid, show what's available
    available = _list_index_names()
    if not index_name or index_name not in available:
        hint = f"Index '{index_name}' not found. " if index_name else "No index specified. "
        return f"{hint}Available indexes:\n{_available_indexes_str()}\n\nPass index_name to search a specific index."

    index_dir = rag.resolve_index_dir(index_name)
    err = _mcp_integrity_gate(index_name)
    if err:
        return err

    # Detect and get backend
    backend_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, backend_type)

    if not backend.exists():
        return f"Index '{index_name}' has no data. Available:\n{_available_indexes_str()}"

    # Auto-detect source filter from query text if not explicitly provided
    auto_filtered = False
    if not source_filter:
        detected = _auto_detect_source_filter(query, backend, backend_type)
        if detected:
            source_filter = detected
            auto_filtered = True

    # Resolve top_k from settings if not specified
    if top_k <= 0:
        top_k = rag_settings.get('top_k')

    # Resolve per-index embedding model
    meta = rag_backends.get_index_meta_with_defaults(index_dir)
    emb_backend = meta.get('embedding_backend', 'local')
    emb_model = meta.get('embedding_model', 'all-MiniLM-L6-v2')
    api_url = None
    if emb_backend == 'ollama':
        api_url = meta.get('ollama_url', rag_settings.get('ollama_url'))
    elif emb_backend == 'lmstudio':
        api_url = meta.get('lmstudio_url', rag_settings.get('lmstudio_url'))

    # Embed query using the index's model
    q_emb = _embed([query], override_backend=emb_backend,
                    override_model=emb_model, override_url=api_url)

    OCR_NOTE = "[NOTE: This text was produced by OCR and may contain recognition errors — misspellings, garbled characters, or missing words.]"

    # Use context-aware search for sqlite-doc backends
    if backend_type == 'sqlite-doc' and (context > 0 or source_filter):
        ctx_results = backend.search_with_context(q_emb, top_k,
                                                   context=context,
                                                   source_filter=source_filter)
        results = []
        seen_sources = set()
        for rank, hit in enumerate(ctx_results):
            seen_sources.add(hit['source'])
            is_ocr = hit.get('ocr', False)
            ocr_str = " [OCR]" if is_ocr else ""
            ocr_block = f"\n{OCR_NOTE}\n" if is_ocr else ""

            text_parts = []
            for adj in hit.get('adjacent', []):
                if adj['chunk'] < hit['chunk']:
                    text_parts.append(f"[context: chunk {adj['chunk']+1}]\n{adj['text'][:1500]}")
            text_parts.append(hit['text'][:3000])
            for adj in hit.get('adjacent', []):
                if adj['chunk'] > hit['chunk']:
                    text_parts.append(f"[context: chunk {adj['chunk']+1}]\n{adj['text'][:1500]}")

            combined_text = "\n\n".join(text_parts)
            results.append(
                f"[{rank+1}] {hit['source']} | chunk {hit['chunk']+1}/{hit['of']} | "
                f"score {hit['score']:.3f}{ocr_str}{ocr_block}\n{combined_text}"
            )

        if not results:
            if source_filter:
                return (f"No results found in '{index_name}' for: {query} (filtered to: {source_filter})\n"
                        f"TIP: Try rag_read_range(index_name=\"{index_name}\", source=\"{source_filter}\") to read from the start.\n"
                        f"{_RAG_WARNING}")
            return f"No results found in '{index_name}' for: {query}\n{_RAG_WARNING}"
        header = _build_result_header(index_name, query, len(results), source_filter, auto_filtered, seen_sources)
        return header + "\n\n" + "\n\n---\n\n".join(results)

    # Standard search path
    chunks = backend.get_chunks()
    fetch_k = top_k * 10 if source_filter else top_k
    search_results = backend.search(q_emb, fetch_k)

    results = []
    seen_sources = set()
    for rank, (score, idx) in enumerate(search_results):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        if source_filter and source_filter not in c.get('source', ''):
            continue
        seen_sources.add(c.get('source', ''))
        is_ocr = c.get('ocr', False)
        ocr_str = " [OCR]" if is_ocr else ""
        ocr_block = f"\n{OCR_NOTE}\n" if is_ocr else ""
        results.append(
            f"[{len(results)+1}] {c['source']} | chunk {c['chunk']+1}/{c['of']} | score {score:.3f}{ocr_str}{ocr_block}\n{c['text'][:3000]}"
        )
        if len(results) >= top_k:
            break

    if not results:
        if source_filter:
            return (f"No results found in '{index_name}' for: {query} (filtered to: {source_filter})\n"
                    f"TIP: Try rag_read_range(index_name=\"{index_name}\", source=\"{source_filter}\") to read from the start.\n"
                    f"{_RAG_WARNING}")
        return f"No results found in '{index_name}' for: {query}\n{_RAG_WARNING}"

    header = _build_result_header(index_name, query, len(results), source_filter, auto_filtered, seen_sources)
    return header + "\n\n" + "\n\n---\n\n".join(results)


@mcp.tool()
def rag_sources(index_name: str = "", filter: str = "") -> str:
    """List document names in a RAG index. Use BEFORE rag_query to find exact source names for source_filter. Pass filter to search by name substring (case-insensitive). If index_name is empty or invalid, shows available indexes."""
    available = _list_index_names()
    if not index_name or index_name not in available:
        hint = f"Index '{index_name}' not found. " if index_name else "No index specified. "
        return f"{hint}Available indexes:\n{_available_indexes_str()}"

    index_dir = rag.resolve_index_dir(index_name)
    err = _mcp_integrity_gate(index_name)
    if err:
        return err
    backend_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, backend_type)
    hashes = backend.get_hashes()

    if not hashes:
        return f"Index '{index_name}' has no hash data."
    names = sorted(hashes.values())

    if filter:
        filter_lower = filter.lower()
        matched = [n for n in names if filter_lower in n.lower()]
        if not matched:
            return f"No sources matching '{filter}' in '{index_name}'. Use rag_sources(index_name=\"{index_name}\") to see all."
        return f"{len(matched)} sources matching '{filter}' in '{index_name}':\n" + "\n".join(f"  - {n}" for n in matched)

    # Unfiltered: truncate at 50
    if len(names) > 50:
        listing = "\n".join(f"  - {n}" for n in names[:50])
        return (f"{len(names)} files in '{index_name}' (showing first 50):\n{listing}\n\n"
                f"TIP: Use filter parameter to narrow results, e.g. rag_sources(index_name=\"{index_name}\", filter=\"keyword\")")
    return f"{len(names)} files in '{index_name}':\n" + "\n".join(f"  - {n}" for n in names)


@mcp.tool()
def rag_read(index_name: str, source: str) -> str:
    """Return full document text from a sqlite-doc index. Use rag_sources to find valid source paths."""
    available = _list_index_names()
    if not index_name or index_name not in available:
        hint = f"Index '{index_name}' not found. " if index_name else "No index specified. "
        return f"{hint}Available indexes:\n{_available_indexes_str()}"

    index_dir = rag.resolve_index_dir(index_name)
    err = _mcp_integrity_gate(index_name)
    if err:
        return err
    backend_type = rag_backends.detect_backend(index_dir)

    if backend_type != 'sqlite-doc':
        return (f"Index '{index_name}' uses {backend_type} backend. "
                f"Document retrieval requires sqlite-doc. "
                f"Re-index with: rag.py code /path --name {index_name}")

    backend = rag_backends.get_backend(index_dir, backend_type)
    doc = backend.get_document(source)

    if not doc:
        # Try partial match
        docs = backend.list_documents()
        matches = [d for d in docs if source in d['source']]
        if matches:
            suggestion = "\n".join(f"  - {d['source']} ({d['chunk_count']} chunks)" for d in matches[:10])
            return f"Source '{source}' not found. Similar:\n{suggestion}"
        return f"Source '{source}' not found in '{index_name}'."

    lang_str = f" [{doc['language']}]" if doc['language'] else ""
    return f"=== {doc['source']}{lang_str} ===\n\n{doc['full_text']}"


@mcp.tool()
def rag_chunk(index_name: str, source: str, chunk_number: int) -> str:
    """Return a specific chunk by number (0-indexed) from a sqlite-doc index."""
    available = _list_index_names()
    if not index_name or index_name not in available:
        hint = f"Index '{index_name}' not found. " if index_name else "No index specified. "
        return f"{hint}Available indexes:\n{_available_indexes_str()}"

    index_dir = rag.resolve_index_dir(index_name)
    err = _mcp_integrity_gate(index_name)
    if err:
        return err
    backend_type = rag_backends.detect_backend(index_dir)

    if backend_type != 'sqlite-doc':
        return (f"Index '{index_name}' uses {backend_type} backend. "
                f"Chunk navigation requires sqlite-doc. "
                f"Re-index with: rag.py code /path --name {index_name}")

    backend = rag_backends.get_backend(index_dir, backend_type)
    chunks = backend.get_document_chunks(source)

    if not chunks:
        return f"Source '{source}' not found in '{index_name}'."

    if chunk_number < 0 or chunk_number >= len(chunks):
        return f"Chunk {chunk_number} out of range. '{source}' has {len(chunks)} chunks (0-{len(chunks)-1})."

    c = chunks[chunk_number]
    return f"=== {source} (chunk {c['chunk']+1}/{c['of']}) ===\n\n{c['text']}"


@mcp.tool()
def rag_context(index_name: str, source: str, chunk_number: int, window: int = 1) -> str:
    """Return a chunk with N neighboring chunks from the same document. Requires sqlite-doc backend."""
    available = _list_index_names()
    if not index_name or index_name not in available:
        hint = f"Index '{index_name}' not found. " if index_name else "No index specified. "
        return f"{hint}Available indexes:\n{_available_indexes_str()}"

    index_dir = rag.resolve_index_dir(index_name)
    err = _mcp_integrity_gate(index_name)
    if err:
        return err
    backend_type = rag_backends.detect_backend(index_dir)

    if backend_type != 'sqlite-doc':
        return (f"Index '{index_name}' uses {backend_type} backend. "
                f"Context expansion requires sqlite-doc. "
                f"Re-index with: rag.py code /path --name {index_name}")

    backend = rag_backends.get_backend(index_dir, backend_type)
    chunks = backend.get_document_chunks(source)

    if not chunks:
        return f"Source '{source}' not found in '{index_name}'."

    if chunk_number < 0 or chunk_number >= len(chunks):
        return f"Chunk {chunk_number} out of range. '{source}' has {len(chunks)} chunks (0-{len(chunks)-1})."

    # Get the window range
    start = max(0, chunk_number - window)
    end = min(len(chunks), chunk_number + window + 1)

    parts = []
    for c in chunks[start:end]:
        marker = " <<< HIT" if c['chunk'] == chunk_number else ""
        parts.append(f"--- chunk {c['chunk']+1}/{c['of']}{marker} ---\n{c['text']}")

    return f"=== {source} (chunks {start+1}-{end}/{len(chunks)}) ===\n\n" + "\n\n".join(parts)


@mcp.tool()
def rag_read_range(index_name: str, source: str, start: int = 0, count: int = 10) -> str:
    """Read a sequential range of chunks from a document, joined as continuous text. start is 0-indexed, count is how many chunks (max 30). Returns text with a note about remaining chunks."""
    available = _list_index_names()
    if not index_name or index_name not in available:
        hint = f"Index '{index_name}' not found. " if index_name else "No index specified. "
        return f"{hint}Available indexes:\n{_available_indexes_str()}"

    index_dir = rag.resolve_index_dir(index_name)
    err = _mcp_integrity_gate(index_name)
    if err:
        return err
    backend_type = rag_backends.detect_backend(index_dir)

    if backend_type != 'sqlite-doc':
        return (f"Index '{index_name}' uses {backend_type} backend. "
                f"Sequential reading requires sqlite-doc. "
                f"Re-index with: rag.py code /path --name {index_name}")

    backend = rag_backends.get_backend(index_dir, backend_type)
    chunks = backend.get_document_chunks(source)

    if not chunks:
        # Try partial match
        docs = backend.list_documents()
        matches = [d for d in docs if source.lower() in d['source'].lower()]
        if matches:
            suggestion = "\n".join(f"  - {d['source']} ({d['chunk_count']} chunks)" for d in matches[:10])
            return f"Source '{source}' not found. Similar:\n{suggestion}"
        return f"Source '{source}' not found in '{index_name}'."

    count = min(count, 30)  # cap at 30 chunks per call
    if start < 0:
        start = 0
    if start >= len(chunks):
        return f"Start chunk {start} out of range. '{source}' has {len(chunks)} chunks (0-{len(chunks)-1})."

    end = min(start + count, len(chunks))
    selected = chunks[start:end]
    text = "\n\n".join(c['text'] for c in selected)

    remaining = len(chunks) - end
    status = f"{remaining} chunks remaining — call again with start={end} to continue." if remaining > 0 else "End of document."

    return f"=== {source} (chunks {start+1}-{end} of {len(chunks)}) ===\n\n{text}\n\n[{status}]"


# ── Internal helpers (embedding model, lazy loaded) ──

_model = None
_model_name = None  # tracks which model is currently loaded


def _embed_api(texts, override_model=None, override_url=None):
    """Embed texts via Ollama or LM Studio OpenAI-compatible API."""
    import numpy as np
    import urllib.request

    backend = rag_settings.get('embedding_backend')
    api_model = override_model or rag_settings.get('api_model')
    if override_url:
        base_url = override_url.rstrip('/')
    elif backend == 'ollama':
        base_url = rag_settings.get('ollama_url').rstrip('/')
    else:
        base_url = rag_settings.get('lmstudio_url').rstrip('/')
    url = f"{base_url}/v1/embeddings"

    all_embs = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = json.dumps({"model": api_model, "input": batch}).encode('utf-8')
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach {backend} at {base_url}: {e}\n"
                f"Is {backend} running? Ensure model '{api_model}' is available."
            ) from e

        sorted_data = sorted(result['data'], key=lambda d: d['index'])
        for item in sorted_data:
            all_embs.append(item['embedding'])

    embs = np.array(all_embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs


def _embed(texts, override_backend=None, override_model=None, override_url=None):
    """Embed texts. Dispatches to API or local model based on settings/overrides."""
    global _model, _model_name
    import numpy as np

    backend = override_backend or rag_settings.get('embedding_backend')
    if backend in ('ollama', 'lmstudio'):
        return _embed_api(texts, override_model=override_model, override_url=override_url)

    model_name = override_model or rag_settings.get('embedding_model')

    # If a different model is requested, unload current one
    if _model is not None and _model_name != model_name:
        del _model
        _model = None
        _model_name = None
        gc.collect()

    # Local CPU path — load on first call, stays resident
    if _model is None:
        import warnings
        warnings.filterwarnings('ignore', message='.*UNEXPECTED.*')
        import logging
        logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(8, os.cpu_count() or 4))

        # Suppress LOAD REPORT noise at fd level
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            _model = SentenceTransformer(model_name, device='cpu')
            _model_name = model_name
        except Exception:
            # Model not cached — try download
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _model = SentenceTransformer(model_name, device='cpu')
            _model_name = model_name
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            saved_out = os.dup(1)  # re-capture for finally
            saved_err = os.dup(2)
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            os.close(devnull)

    embs = _model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embs, dtype=np.float32)


if __name__ == "__main__":
    mcp.run(transport="stdio")
