"""
core/integrity.py — index integrity verification (SHA + stat + suppress).

Extracted from rag.py during Clean Code modular refactor (Ch 10 SRP, Ch 17 G30).
Exact original semantics preserved.
"""

import json
import os
import sys

from .constants import INTEGRITY_FILENAME, LOCK_FILENAME
from .hashing import file_hash

# Also delegates to load/save_index_hashes in same package for some paths, but
# current callers go through the functions that already existed in rag.py.

_INTEGRITY_SKIP = {LOCK_FILENAME, INTEGRITY_FILENAME, '-wal', '-shm', '-journal'}


def _should_hash_file(filename):
    """Return True if this file should be included in integrity hashing."""
    if filename in _INTEGRITY_SKIP:
        return False
    for suffix in ('-wal', '-shm', '-journal'):
        if filename.endswith(suffix):
            return False
    return True


def compute_index_integrity(index_dir):
    """Hash all data files in index_dir. Returns dict with per-file sha256/size/mtime."""
    files = {}
    for fname in sorted(os.listdir(index_dir)):
        if not _should_hash_file(fname):
            continue
        fpath = os.path.join(index_dir, fname)
        if not os.path.isfile(fpath):
            continue
        st = os.stat(fpath)
        files[fname] = {
            'sha256': file_hash(fpath),
            'size': st.st_size,
            'mtime': st.st_mtime,
        }
    return {'version': 1, 'files': files, 'suppressed': False}


def save_index_integrity(index_dir):
    """Compute and write .integrity file with suppressed=false."""
    integrity = compute_index_integrity(index_dir)
    with open(os.path.join(index_dir, INTEGRITY_FILENAME), 'w') as f:
        json.dump(integrity, f, indent=2)


def check_index_integrity(index_dir, fast=True):
    """Check index integrity. Returns dict with 'ok', 'suppressed', 'untracked', 'details'.
    fast=True uses stat check (size+mtime), fast=False does full SHA-256.
    Missing .integrity = untracked (never verified), treated as not ok."""
    ipath = os.path.join(index_dir, INTEGRITY_FILENAME)
    if not os.path.exists(ipath):
        if os.path.exists(os.path.join(index_dir, 'meta.json')):
            return {'ok': False, 'suppressed': False, 'untracked': True, 'details': ['no .integrity file — index has never been verified']}
        return {'ok': True, 'suppressed': False, 'untracked': False, 'details': []}

    with open(ipath) as f:
        stored = json.load(f)

    if stored.get('suppressed', False):
        return {'ok': True, 'suppressed': True, 'untracked': False, 'details': []}

    details = []
    for fname, info in stored.get('files', {}).items():
        fpath = os.path.join(index_dir, fname)
        if not os.path.exists(fpath):
            details.append(f"{fname}: MISSING")
            continue
        st = os.stat(fpath)
        if fast:
            if st.st_size != info['size'] or abs(st.st_mtime - info['mtime']) > 0.01:
                details.append(f"{fname}: size/mtime changed")
        else:
            actual_hash = file_hash(fpath)
            if actual_hash != info['sha256']:
                details.append(f"{fname}: SHA-256 mismatch")

    return {'ok': len(details) == 0, 'suppressed': False, 'untracked': False, 'details': details}


def suppress_index_integrity(index_dir):
    """Set suppressed=true in .integrity file."""
    ipath = os.path.join(index_dir, INTEGRITY_FILENAME)
    if not os.path.exists(ipath):
        return
    with open(ipath) as f:
        data = json.load(f)
    data['suppressed'] = True
    with open(ipath, 'w') as f:
        json.dump(data, f, indent=2)


def _cli_integrity_gate(name, index_dir):
    """CLI interactive warning handler. Returns True if safe to proceed."""
    result = check_index_integrity(index_dir)
    if result['ok']:
        return True

    if result.get('untracked'):
        print(f"\n*** UNVERIFIED INDEX '{name}' ***", file=sys.stderr)
        print(f"  This index was created before integrity tracking was enabled.", file=sys.stderr)
        print(f"  Open the editor (rag.py editor) and press 'h' to hash it,", file=sys.stderr)
        print(f"  or run: rag.py integrity --rehash --name {name}", file=sys.stderr)
    else:
        print(f"\n*** INTEGRITY WARNING for '{name}' ***", file=sys.stderr)
        for d in result['details']:
            print(f"  - {d}", file=sys.stderr)
        print(f"\nIndex data may have been modified outside RAG-Narock.", file=sys.stderr)

    if not sys.stdin.isatty():
        print("Aborting. Fix via editor (rag.py editor) or CLI (rag.py integrity --rehash).", file=sys.stderr)
        return False

    while True:
        try:
            choice = input("\nProceed? [y/N/suppress/delete] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if choice in ('', 'n', 'no'):
            return False
        if choice in ('y', 'yes'):
            return True
        if choice == 'suppress':
            suppress_index_integrity(index_dir)
            print("Warning suppressed. Will re-verify on next index write.", file=sys.stderr)
            return True
        if choice == 'delete':
            print(f"To delete this index, run: rag.py delete {name}", file=sys.stderr)
            return False
        print("Invalid choice. Enter y, n, suppress, or delete.", file=sys.stderr)
