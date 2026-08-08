"""
Characterization tests for core/integrity.py — index integrity verification.
"""

import os
import json
import tempfile
import pytest

from core.integrity import (
    compute_index_integrity,
    save_index_integrity,
    check_index_integrity,
    suppress_index_integrity,
    _should_hash_file,
)
from core.constants import INTEGRITY_FILENAME, LOCK_FILENAME


# ── _should_hash_file ─────────────────────────────────────────────────────────

def test_should_hash_normal_files():
    """_should_hash_file() returns True for normal index files."""
    assert _should_hash_file("chunks.json") is True
    assert _should_hash_file("index.faiss") is True
    assert _should_hash_file("meta.json") is True


def test_should_skip_integrity_file():
    """_should_hash_file() returns False for .integrity file."""
    assert _should_hash_file(INTEGRITY_FILENAME) is False


def test_should_skip_lock_file():
    """_should_hash_file() returns False for .locked file."""
    assert _should_hash_file(LOCK_FILENAME) is False


def test_should_skip_wal_shm_journal():
    """_should_hash_file() returns False for SQLite WAL/SHM/journal files."""
    assert _should_hash_file("index.db-wal") is False
    assert _should_hash_file("index.db-shm") is False
    assert _should_hash_file("index.db-journal") is False


# ── compute / save / check cycle ──────────────────────────────────────────────

def test_compute_integrity_structure(tmp_path):
    """compute_index_integrity() returns proper structure."""
    # Create some test files
    (tmp_path / "test.txt").write_text("hello world")
    (tmp_path / "chunks.json").write_text('{"key": "value"}')
    (tmp_path / ".integrity").write_text("{}")  # should be skipped
    (tmp_path / ".locked").write_text("locked")   # should be skipped

    result = compute_index_integrity(str(tmp_path))
    assert result['version'] == 1
    assert result['suppressed'] is False
    assert 'files' in result
    # Only normal files should be hashed
    assert 'test.txt' in result['files']
    assert 'chunks.json' in result['files']
    assert INTEGRITY_FILENAME not in result['files']
    assert LOCK_FILENAME not in result['files']


def test_compute_integrity_includes_sha256(tmp_path):
    """compute_index_integrity() includes sha256, size, and mtime for each file."""
    (tmp_path / "data.txt").write_text("test content for hashing")
    result = compute_index_integrity(str(tmp_path))
    info = result['files']['data.txt']
    assert 'sha256' in info
    assert 'size' in info
    assert 'mtime' in info
    assert info['size'] == len("test content for hashing")
    assert len(info['sha256']) == 64  # SHA-256 hex digest


def test_save_and_check_cycle(tmp_path):
    """save_index_integrity() → check_index_integrity() roundtrip passes."""
    (tmp_path / "data.txt").write_text("hello")
    (tmp_path / "meta.json").write_text('{"n_chunks": 1}')

    save_index_integrity(str(tmp_path))
    result = check_index_integrity(str(tmp_path))
    assert result['ok'] is True
    assert result['suppressed'] is False
    assert result['untracked'] is False
    assert result['details'] == []


def test_check_detects_missing_file(tmp_path):
    """check_index_integrity() reports MISSING files."""
    (tmp_path / "data.txt").write_text("hello")
    save_index_integrity(str(tmp_path))
    # Delete the file
    os.remove(tmp_path / "data.txt")

    result = check_index_integrity(str(tmp_path))
    assert result['ok'] is False
    assert any("MISSING" in d for d in result['details'])


def test_check_detects_modified_file(tmp_path):
    """check_index_integrity() detects file modifications (fast mode)."""
    (tmp_path / "data.txt").write_text("original content")
    save_index_integrity(str(tmp_path))
    # Modify the file
    (tmp_path / "data.txt").write_text("modified content for testing")

    result = check_index_integrity(str(tmp_path), fast=True)
    assert result['ok'] is False
    assert any("size/mtime changed" in d for d in result['details'])


def test_check_untracked_index(tmp_path):
    """check_index_integrity() returns untracked=True if meta.json exists but no .integrity."""
    (tmp_path / "meta.json").write_text('{"n_chunks": 10}')
    result = check_index_integrity(str(tmp_path))
    assert result['ok'] is False
    assert result['untracked'] is True


def test_check_empty_directory(tmp_path):
    """check_index_integrity() returns ok=True for empty directories (no meta.json)."""
    result = check_index_integrity(str(tmp_path))
    assert result['ok'] is True
    assert result['untracked'] is False


# ── suppress ──────────────────────────────────────────────────────────────────

def test_suppress_sets_flag(tmp_path):
    """suppress_index_integrity() sets suppressed=True."""
    (tmp_path / "data.txt").write_text("hello")
    save_index_integrity(str(tmp_path))

    suppress_index_integrity(str(tmp_path))
    result = check_index_integrity(str(tmp_path))
    assert result['ok'] is True
    assert result['suppressed'] is True


def test_suppress_no_file_is_noop(tmp_path):
    """suppress_index_integrity() is a no-op when .integrity doesn't exist."""
    # Should not raise
    suppress_index_integrity(str(tmp_path))
