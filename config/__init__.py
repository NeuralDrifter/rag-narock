"""
config package — pure configuration for RAG-Narock.

Holds:
- settings schema, defaults, load/save/get
- data dir resolution
- external index registry
- dangerous keys list
- optional dependency checks

No UI code lives here. TUI/GUI consume this.
"""
from .settings import (  # noqa: F401
    SETTINGS_DIR, SETTINGS_PATH,
    TABS, DEFAULTS, SCHEMA,
    OPTIONAL_DEPS, check_all_deps,
    DANGEROUS_KEYS,
    load, save, get,
    get_data_dir, get_external_indexes,
    add_external_index, remove_external_index,
)
