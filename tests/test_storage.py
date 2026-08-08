"""
Characterization tests for storage backends — FAISS and SQLite implementations.

Tests the StorageBackend contract: save, append, search, remove_source,
get_chunks, get_hashes, exists, get_dim, get_total.
"""

import os
import json
import tempfile
import shutil

import numpy as np
import pytest

from storage.base import detect_backend, StorageBackend, META_DEFAULTS, get_index_meta_with_defaults
from storage.faiss_backend import FaissBackend
from storage import (
    MissingBackendDependency,
    get_backend,
    get_backend_status,
    get_missing_backend_warnings,
)

# Try importing sqlite backends (may not be available in all envs)
try:
    from storage.sqlite_vec_backend import SqliteVecBackend
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

try:
    from storage.sqlite_doc_backend import SqliteDocBackend
    HAS_SQLITE_DOC = True
except ImportError:
    HAS_SQLITE_DOC = False


# ── Test fixtures ─────────────────────────────────────────────────────────────

def _make_chunks(n=3):
    """Create n dummy chunks."""
    return [
        {'text': f'This is chunk {i}', 'source': f'source_{i % 2}.txt',
         'chunk': i, 'of': n, 'ocr': False}
        for i in range(n)
    ]


def _make_embeddings(dim=4, n=3):
    """Create n dummy embeddings."""
    rng = np.random.RandomState(42)
    return rng.rand(n, dim).astype(np.float32)


def _make_hashes():
    """Create dummy hashes dict."""
    return {
        'abc123hash': 'source_0.txt',
        'def456hash': 'source_1.txt',
    }


@pytest.fixture(params=['faiss'])
def backend(request, tmp_path):
    """Create a fresh backend instance for each test."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    if request.param == 'faiss':
        yield FaissBackend(str(index_dir))
    elif request.param == 'sqlite-vec' and HAS_SQLITE_VEC:
        yield SqliteVecBackend(str(index_dir))
    elif request.param == 'sqlite-doc' and HAS_SQLITE_DOC:
        yield SqliteDocBackend(str(index_dir))
    # Cleanup
    shutil.rmtree(str(index_dir), ignore_errors=True)


# ── detect_backend ────────────────────────────────────────────────────────────

def test_detect_backend_defaults_to_faiss(tmp_path):
    """detect_backend() returns 'faiss' for empty or non-existent directories."""
    assert detect_backend(str(tmp_path)) == "faiss"


def test_detect_backend_faiss_after_save(tmp_path):
    """detect_backend() returns 'faiss' even after FAISS data written."""
    backend = FaissBackend(str(tmp_path))
    backend.save(_make_chunks(), _make_embeddings(), _make_hashes())
    assert detect_backend(str(tmp_path)) == "faiss"


# ── FAISS backend save / exists / get_dim / get_total ─────────────────────────

def test_faiss_not_exists_initially(tmp_path):
    """FaissBackend.exists() is False before data is saved."""
    backend = FaissBackend(str(tmp_path))
    assert backend.exists() is False


def test_faiss_save_creates_index(tmp_path):
    """FaissBackend.save() creates the index and marks it as existing."""
    backend = FaissBackend(str(tmp_path))
    backend.save(_make_chunks(), _make_embeddings(), _make_hashes())
    assert backend.exists() is True
    assert os.path.exists(os.path.join(str(tmp_path), "index.faiss"))
    assert os.path.exists(os.path.join(str(tmp_path), "chunks.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "hashes.json"))


def test_faiss_get_dim_after_save(tmp_path):
    """FaissBackend.get_dim() returns the embedding dimension."""
    backend = FaissBackend(str(tmp_path))
    embs = _make_embeddings(dim=8)
    backend.save(_make_chunks(), embs, _make_hashes())
    assert backend.get_dim() == 8


def test_faiss_get_total_after_save(tmp_path):
    """FaissBackend.get_total() returns the number of indexed vectors."""
    backend = FaissBackend(str(tmp_path))
    chunks = _make_chunks(5)
    embs = _make_embeddings(n=5)
    backend.save(chunks, embs, _make_hashes())
    assert backend.get_total() == 5


def test_faiss_get_dim_before_save(tmp_path):
    """FaissBackend.get_dim() returns 0 before any data is saved."""
    backend = FaissBackend(str(tmp_path))
    assert backend.get_dim() == 0


def test_faiss_get_total_before_save(tmp_path):
    """FaissBackend.get_total() returns 0 before any data is saved."""
    backend = FaissBackend(str(tmp_path))
    assert backend.get_total() == 0


# ── FAISS get_chunks / get_hashes ─────────────────────────────────────────────

def test_faiss_get_chunks_empty_when_no_file(tmp_path):
    """FaissBackend.get_chunks() returns [] when no chunks.json exists."""
    backend = FaissBackend(str(tmp_path))
    assert backend.get_chunks() == []


def test_faiss_get_chunks_roundtrip(tmp_path):
    """FaissBackend.get_chunks() returns saved chunks."""
    backend = FaissBackend(str(tmp_path))
    chunks = _make_chunks(3)
    backend.save(chunks, _make_embeddings(n=3), _make_hashes())
    loaded = backend.get_chunks()
    assert len(loaded) == 3
    assert loaded[0]['text'] == chunks[0]['text']


def test_faiss_get_hashes_empty_when_no_file(tmp_path):
    """FaissBackend.get_hashes() returns {} when no hashes.json exists."""
    backend = FaissBackend(str(tmp_path))
    assert backend.get_hashes() == {}


def test_faiss_get_hashes_roundtrip(tmp_path):
    """FaissBackend.get_hashes() returns saved hashes."""
    backend = FaissBackend(str(tmp_path))
    hashes = {'hash1': 'file1.txt', 'hash2': 'file2.txt'}
    backend.save(_make_chunks(), _make_embeddings(), hashes)
    loaded = backend.get_hashes()
    assert loaded == hashes


def test_faiss_save_hashes_updates(tmp_path):
    """FaissBackend.save_hashes() replaces all hashes."""
    backend = FaissBackend(str(tmp_path))
    backend.save(_make_chunks(), _make_embeddings(), {'old': 'file.txt'})
    backend.save_hashes({'new': 'other.txt'})
    assert backend.get_hashes() == {'new': 'other.txt'}


# ── FAISS append ──────────────────────────────────────────────────────────────

def test_faiss_append_merges_new_data(tmp_path):
    """FaissBackend.append() adds new chunks to existing index."""
    backend = FaissBackend(str(tmp_path))
    # Initial save
    backend.save(
        [{'text': 'chunk 0', 'source': 'a.txt', 'chunk': 0, 'of': 1, 'ocr': False}],
        _make_embeddings(n=1),
        {'h0': 'a.txt'}
    )
    # Append
    backend.append(
        [{'text': 'chunk 1', 'source': 'b.txt', 'chunk': 0, 'of': 1, 'ocr': False}],
        _make_embeddings(n=1),
        {'h1': 'b.txt'}
    )
    assert backend.get_total() == 2
    chunks = backend.get_chunks()
    assert len(chunks) == 2
    hashes = backend.get_hashes()
    assert 'h0' in hashes
    assert 'h1' in hashes


# ── FAISS search ──────────────────────────────────────────────────────────────

def test_faiss_search_empty_index(tmp_path):
    """FaissBackend.search() returns [] when index is empty."""
    backend = FaissBackend(str(tmp_path))
    query = np.random.rand(1, 4).astype(np.float32)
    assert backend.search(query, 5) == []


def test_faiss_search_returns_results(tmp_path):
    """FaissBackend.search() returns (score, index) tuples."""
    backend = FaissBackend(str(tmp_path))
    embs = _make_embeddings(n=5)
    backend.save(_make_chunks(5), embs, _make_hashes())
    # Search with the first embedding — should find itself
    query = embs[0:1]
    results = backend.search(query, 3)
    assert len(results) >= 1
    assert len(results[0]) == 2  # (score, idx)
    assert isinstance(results[0][0], float)


def test_faiss_search_respects_top_k(tmp_path):
    """FaissBackend.search() returns at most top_k results."""
    backend = FaissBackend(str(tmp_path))
    embs = _make_embeddings(n=10)
    backend.save(_make_chunks(10), embs, _make_hashes())
    results = backend.search(embs[0:1], 3)
    assert len(results) == 3


# ── FAISS remove_source ───────────────────────────────────────────────────────

def test_faiss_remove_source_removes_chunks(tmp_path):
    """FaissBackend.remove_source() removes all chunks for a source."""
    backend = FaissBackend(str(tmp_path))
    chunks = [
        {'text': 'a1', 'source': 'a.txt', 'chunk': 0, 'of': 2, 'ocr': False},
        {'text': 'a2', 'source': 'a.txt', 'chunk': 1, 'of': 2, 'ocr': False},
        {'text': 'b1', 'source': 'b.txt', 'chunk': 0, 'of': 1, 'ocr': False},
    ]
    embs = _make_embeddings(n=3)
    backend.save(chunks, embs, {'ha': 'a.txt', 'hb': 'b.txt'})
    result = backend.remove_source('a.txt')
    assert result['removed_chunks'] == 2
    assert result['remaining_chunks'] == 1
    assert result['remaining_files'] == 1
    remaining = backend.get_chunks()
    assert all(c['source'] == 'b.txt' for c in remaining)


def test_faiss_remove_nonexistent_source(tmp_path):
    """FaissBackend.remove_source() raises ValueError for unknown source."""
    backend = FaissBackend(str(tmp_path))
    backend.save(_make_chunks(), _make_embeddings(), _make_hashes())
    with pytest.raises(ValueError, match="not found"):
        backend.remove_source("nonexistent.txt")


def test_faiss_remove_all_sources(tmp_path):
    """FaissBackend.remove_source() works when removing the last source."""
    backend = FaissBackend(str(tmp_path))
    chunks = [{'text': 'only', 'source': 'only.txt', 'chunk': 0, 'of': 1, 'ocr': False}]
    backend.save(chunks, _make_embeddings(n=1), {'h': 'only.txt'})
    result = backend.remove_source('only.txt')
    assert result['removed_chunks'] == 1
    assert result['remaining_chunks'] == 0
    assert result['remaining_files'] == 0


# ── META_DEFAULTS / get_index_meta_with_defaults ──────────────────────────────

def test_meta_defaults_has_required_keys():
    """META_DEFAULTS contains storage_backend, embedding_backend, embedding_model."""
    assert 'storage_backend' in META_DEFAULTS
    assert 'embedding_backend' in META_DEFAULTS
    assert 'embedding_model' in META_DEFAULTS


def test_get_index_meta_with_defaults_empty_dir(tmp_path):
    """get_index_meta_with_defaults() returns only defaults for empty dir."""
    meta = get_index_meta_with_defaults(str(tmp_path))
    assert meta['storage_backend'] == META_DEFAULTS['storage_backend']


def test_get_index_meta_with_defaults_merges(tmp_path):
    """get_index_meta_with_defaults() fills missing keys from existing meta."""
    meta_path = os.path.join(str(tmp_path), "meta.json")
    with open(meta_path, 'w') as f:
        json.dump({"n_chunks": 42}, f)
    meta = get_index_meta_with_defaults(str(tmp_path))
    assert meta['n_chunks'] == 42
    assert meta['storage_backend'] == META_DEFAULTS['storage_backend']


# ── get_backend factory ───────────────────────────────────────────────────────

def test_get_backend_returns_faiss_by_default(tmp_path):
    """get_backend() returns FaissBackend for 'faiss' type."""
    backend = get_backend(str(tmp_path), 'faiss')
    assert isinstance(backend, FaissBackend)


def test_get_backend_returns_faiss_for_unknown_type(tmp_path):
    """get_backend() defaults to FaissBackend for unknown types."""
    backend = get_backend(str(tmp_path), 'unknown-type')
    assert isinstance(backend, FaissBackend)


def test_backend_status_reports_supported_backends():
    """Dependency status includes every supported storage backend."""
    status = get_backend_status()
    assert set(status) == {'faiss', 'sqlite-vec', 'sqlite-doc'}
    assert status['faiss']['package'] == 'faiss-cpu'
    assert status['sqlite-vec']['package'] == 'sqlite-vec'
    assert status['sqlite-doc']['package'] == 'sqlite-vec'


def test_storage_wildcard_exports_do_not_force_backend_imports():
    """Backend classes stay direct-importable but are not eager wildcard exports."""
    import storage

    assert 'FaissBackend' not in storage.__all__
    assert 'SqliteVecBackend' not in storage.__all__
    assert 'SqliteDocBackend' not in storage.__all__


def test_missing_backend_dependency_is_actionable(tmp_path, monkeypatch):
    """The selected backend fails with install guidance instead of import traceback."""
    import storage

    monkeypatch.setattr(storage, "_module_is_available", lambda module: False)

    with pytest.raises(MissingBackendDependency, match="python -m pip install faiss-cpu"):
        get_backend(str(tmp_path), 'faiss')

    warnings = get_missing_backend_warnings()
    assert any("faiss-cpu" in warning for warning in warnings)
