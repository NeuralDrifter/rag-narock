"""
Characterization tests for core/hashing.py — file hashing.
"""

import os
import tempfile
import pytest

from core.hashing import file_hash


def _write_temp(content: str) -> str:
    """Write content to a temp file, close it, return path. Caller must delete."""
    fd, path = tempfile.mkstemp(suffix='.txt', text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def test_file_hash_is_deterministic():
    """file_hash() returns the same hash for the same content."""
    path = _write_temp("hello world")
    try:
        h1 = file_hash(path)
        h2 = file_hash(path)
        assert h1 == h2
    finally:
        os.unlink(path)


def test_file_hash_changes_with_content():
    """file_hash() returns different hashes for different content."""
    path1 = _write_temp("hello world")
    path2 = _write_temp("different content")
    try:
        h1 = file_hash(path1)
        h2 = file_hash(path2)
        assert h1 != h2
    finally:
        os.unlink(path1)
        os.unlink(path2)


def test_file_hash_is_sha256_hex():
    """file_hash() returns a 64-character hex string (SHA-256)."""
    path = _write_temp("test")
    try:
        result = file_hash(path)
    finally:
        os.unlink(path)
    assert len(result) == 64
    assert all(c in '0123456789abcdef' for c in result)


def test_file_hash_empty_file():
    """file_hash() works on empty files."""
    path = _write_temp("")
    try:
        result = file_hash(path)
    finally:
        os.unlink(path)
    assert len(result) == 64
    # SHA-256 of empty input
    assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_file_hash_large_file():
    """file_hash() handles files larger than the read buffer (64KB)."""
    path = _write_temp("x" * 200000)  # ~200KB
    try:
        result = file_hash(path)
    finally:
        os.unlink(path)
    assert len(result) == 64
