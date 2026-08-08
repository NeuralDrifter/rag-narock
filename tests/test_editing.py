"""
Unit and integration tests for document editing and re-indexing in RAG-Narock.
"""

import os
import json
import shutil
import numpy as np
import pytest
import hashlib

import config.settings as settings
from storage import get_backend, detect_backend
from core.index_manager import (
    resolve_index_dir,
    get_index_sources,
    update_source_in_index,
)
from core.integrity import check_index_integrity


# Parametrize test across all three storage backends
@pytest.mark.parametrize('storage_backend', ['faiss', 'sqlite-vec', 'sqlite-doc'])
def test_update_source_in_index(storage_backend, tmp_path, monkeypatch):
    # 1. Redirect settings to tmp_path
    monkeypatch.setattr(settings, 'get_data_dir', lambda: str(tmp_path))
    
    # 2. Mock embedding functions to avoid loading heavy sentence-transformer models
    def mock_embed_texts(texts, **kwargs):
        # Return dummy 384-dim normalized vectors
        embs = np.ones((len(texts), 384), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    monkeypatch.setattr('indexing.embedder.embed_texts', mock_embed_texts)
    monkeypatch.setattr('indexing.embedder._resolve_embedding_device', lambda *a, **k: 'cpu')
    monkeypatch.setattr('indexing.embedder._unload_embedding_model', lambda *a, **k: None)

    index_name = "edit_test_index"
    index_dir = os.path.join(str(tmp_path), index_name)
    os.makedirs(index_dir, exist_ok=True)

    # 3. Create meta.json
    meta = {
        'storage_backend': storage_backend,
        'embedding_backend': 'local',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': 100,
        'overlap': 10,
        'dim': 384,
        'n_chunks': 1,
        'n_files': 1,
    }
    with open(os.path.join(index_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    # 4. Save initial document to backend
    backend = get_backend(index_dir, storage_backend)
    initial_text = "This is the original text. It contains some keywords like antispam and banana."
    
    initial_chunks = [{
        'text': initial_text,
        'source': 'doc1.txt',
        'chunk': 0,
        'of': 1,
        'ocr': False
    }]
    initial_embs = mock_embed_texts([initial_text])
    initial_hash = hashlib.sha256(initial_text.encode('utf-8')).hexdigest()
    initial_hashes = {initial_hash: 'doc1.txt'}

    if storage_backend == 'sqlite-doc':
        backend.save(initial_chunks, initial_embs, initial_hashes, documents=[{
            'source': 'doc1.txt',
            'full_text': initial_text,
            'doc_type': 'book',
            'language': None,
            'ocr': False
        }])
    else:
        backend.save(initial_chunks, initial_embs, initial_hashes)

    # Ensure index exists and is valid
    assert backend.exists() is True
    assert backend.get_total() == 1
    assert backend.get_chunks()[0]['text'] == initial_text

    # 5. Call update_source_in_index
    new_text = "This is the newly updated edited text. The keywords are gone, now we talk about antigravity."
    result = update_source_in_index(index_name, 'doc1.txt', new_text)

    # Check update returns
    assert result['removed_chunks'] == 1
    assert result['new_chunks'] > 0
    assert result['remaining_files'] == 1

    # Re-fetch backend instance and verify chunks updated
    backend = get_backend(index_dir, storage_backend)
    chunks = backend.get_chunks()
    assert len(chunks) == result['new_chunks']
    assert all(c['source'] == 'doc1.txt' for c in chunks)
    assert any("antigravity" in c['text'] for c in chunks)
    assert not any("banana" in c['text'] for c in chunks)

    # Check updated hashes
    hashes = backend.get_hashes()
    new_hash = hashlib.sha256(new_text.encode('utf-8')).hexdigest()
    assert new_hash in hashes
    assert hashes[new_hash] == 'doc1.txt'
    assert initial_hash not in hashes

    # Verify search finds the edited document
    query_emb = mock_embed_texts(["antigravity"])
    search_results = backend.search(query_emb, top_k=1)
    assert len(search_results) > 0
    score, idx = search_results[0]
    assert idx < len(chunks)
    assert "antigravity" in chunks[idx]['text']

    # Verify integrity record exists and matches
    assert check_index_integrity(index_dir)['ok'] is True
