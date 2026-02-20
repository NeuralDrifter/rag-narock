#!/usr/bin/env python3
# Copyright (c) 2026 Michael Burgus (https://github.com/NeuralDrifter)
# Licensed under the MIT License. See LICENSE file for details.
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
    instructions="RAG-Narock — local RAG system for searching indexed technical books. All processing runs on-device, nothing leaves the machine. IMPORTANT: Always call rag_list first to discover available indexes before querying.",
)


def _list_index_names():
    """Fresh scan of available indexes every time."""
    return rag.get_indexes()


def _get_index_info(name):
    """Load metadata for an index (with backward-compatible defaults)."""
    index_dir = rag.resolve_index_dir(name)
    return rag_backends.get_index_meta_with_defaults(index_dir)


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
        if info and 'n_chunks' in info:
            storage = info.get('storage_backend', 'faiss')
            emb_model = info.get('embedding_model', '?')
            lines.append(f"  {name}: {info['n_chunks']} chunks from {info['n_files']} files{lock_str} [{storage}, {emb_model}]")
        else:
            lines.append(f"  {name}: (metadata missing){lock_str}")
    return "\n".join(lines)


@mcp.tool()
def rag_list() -> str:
    """List all available RAG indexes with their stats. Call this FIRST to discover which indexes exist before querying. Indexes can be added at any time via the GUI or CLI, so always check for the latest list."""
    result = _available_indexes_str()
    if result == "No indexes available.":
        return result
    return f"Available RAG indexes:\n{result}\n\nUse rag_query with the index_name parameter to search a specific index."


@mcp.tool()
def rag_query(query: str, index_name: str = "", top_k: int = 0) -> str:
    """Search indexed books for relevant passages. Returns top-K matching text chunks with source attribution and relevance scores. Use natural language queries for best results. If index_name is empty or invalid, returns a list of available indexes so you can pick one."""
    import numpy as np

    # If no index specified or invalid, show what's available
    available = _list_index_names()
    if not index_name or index_name not in available:
        hint = f"Index '{index_name}' not found. " if index_name else "No index specified. "
        return f"{hint}Available indexes:\n{_available_indexes_str()}\n\nPass index_name to search a specific index."

    index_dir = rag.resolve_index_dir(index_name)

    # Detect and get backend
    backend_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, backend_type)

    if not backend.exists():
        return f"Index '{index_name}' has no data. Available:\n{_available_indexes_str()}"

    # Resolve top_k from settings if not specified
    if top_k <= 0:
        top_k = rag_settings.get('top_k')

    # Load chunks
    chunks = backend.get_chunks()

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

    # Search
    search_results = backend.search(q_emb, top_k)

    OCR_NOTE = "[NOTE: This text was produced by OCR and may contain recognition errors — misspellings, garbled characters, or missing words.]"

    # Format results
    results = []
    for rank, (score, idx) in enumerate(search_results):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        is_ocr = c.get('ocr', False)
        ocr_str = " [OCR]" if is_ocr else ""
        ocr_block = f"\n{OCR_NOTE}\n" if is_ocr else ""
        results.append(
            f"[{rank+1}] {c['source']} (chunk {c['chunk']+1}/{c['of']}, score {score:.3f}){ocr_str}{ocr_block}\n{c['text'][:3000]}"
        )

    if not results:
        return f"No results found in '{index_name}' for: {query}"

    return f"=== RAG '{index_name}': \"{query}\" ({len(results)} results) ===\n\n" + "\n\n---\n\n".join(results)


@mcp.tool()
def rag_sources(index_name: str = "") -> str:
    """List all book/document titles in a RAG index. If index_name is empty or invalid, shows available indexes."""
    available = _list_index_names()
    if not index_name or index_name not in available:
        hint = f"Index '{index_name}' not found. " if index_name else "No index specified. "
        return f"{hint}Available indexes:\n{_available_indexes_str()}"

    index_dir = rag.resolve_index_dir(index_name)
    backend_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, backend_type)
    hashes = backend.get_hashes()

    if not hashes:
        return f"Index '{index_name}' has no hash data."
    names = sorted(hashes.values())
    return f"{len(names)} files in '{index_name}':\n" + "\n".join(f"  - {n}" for n in names)


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
