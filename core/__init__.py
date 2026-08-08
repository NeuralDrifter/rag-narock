"""
core package — pure(ish) domain logic for RAG-Narock.

Contains:
- constants (canonical filenames and magic values)
- hashing
- integrity verification
- index management (resolve, list, sources, lock, delete, export, etc.)

These modules have no UI or CLI dependencies. They may delegate to config/ and storage/ after full migration.
"""
from .constants import *  # noqa: F401, F403 — re-export for convenience
from .hashing import file_hash, load_index_hashes, save_index_hashes  # noqa: F401
from .integrity import (  # noqa: F401
    compute_index_integrity,
    save_index_integrity,
    check_index_integrity,
    suppress_index_integrity,
    _cli_integrity_gate,
)
from .index_manager import (  # noqa: F401
    resolve_index_dir,
    get_indexes,
    get_index_info,
    get_index_sources,
    is_index_locked,
    remove_source_from_index,
    update_source_in_index,
    delete_index,
    _export_filename,
    export_source,
    export_index,
)
