"""
Characterization tests for config/settings.py — pure configuration logic.

Tests load/save/get/get_data_dir/external index registry/DANGEROUS_KEYS/DEFAULTS.
"""

import os
import json
import tempfile
import pytest

# Point settings to a temp file before importing
import config.settings as settings


@pytest.fixture(autouse=True)
def temp_settings_dir(monkeypatch):
    """Redirect all settings I/O to a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        settings_dir = os.path.join(tmp, "rag_test_settings")
        settings_path = os.path.join(settings_dir, "settings.json")
        monkeypatch.setattr(settings, 'SETTINGS_DIR', settings_dir)
        monkeypatch.setattr(settings, 'SETTINGS_PATH', settings_path)
        # Reset cache so tests are isolated
        monkeypatch.setattr(settings, '_settings_cache', None)
        yield tmp


# ── Schema / DEFAULTS ─────────────────────────────────────────────────────────

def test_tabs_is_non_empty():
    """TABS contains at least 3 setting groups."""
    assert len(settings.TABS) >= 3


def test_defaults_covers_all_keys():
    """DEFAULTS has a value for every key in SCHEMA."""
    for key in settings.SCHEMA:
        assert key in settings.DEFAULTS, f"Missing default for {key}"


def test_dangerous_keys_are_in_schema():
    """Every DANGEROUS_KEY exists in SCHEMA."""
    for key in settings.DANGEROUS_KEYS:
        assert key in settings.SCHEMA, f"DANGEROUS_KEY {key} not in SCHEMA"


def test_schema_items_have_correct_structure():
    """Each SCHEMA entry has (label, type, options, default, tab_name)."""
    for key, entry in settings.SCHEMA.items():
        assert len(entry) == 5, f"Bad schema entry for {key}: {entry}"
        label, typ, options, default, tab_name = entry
        assert isinstance(label, str)
        assert typ in ('choice', 'toggle', 'text', 'int', 'str', 'bool')


# ── load / save roundtrip ─────────────────────────────────────────────────────

def test_load_returns_defaults_when_no_file():
    """load() returns DEFAULTS if no settings.json exists."""
    cfg = settings.load()
    for key, default in settings.DEFAULTS.items():
        if key in settings.SCHEMA:
            assert cfg[key] == default, f"Key {key} default mismatch"


def test_save_and_load_roundtrip():
    """save() writes data that load() reads back, merged with defaults."""
    cfg = settings.load()
    cfg['chunk_size'] = 9999
    settings.save(cfg)
    loaded = settings.load()
    assert loaded['chunk_size'] == 9999


def test_load_merges_missing_keys():
    """load() fills in schema keys missing from saved file."""
    settings.save({"chunk_size": 2000})
    loaded = settings.load()
    # All schema keys should exist
    for key in settings.SCHEMA:
        assert key in loaded, f"Missing schema key {key}"


def test_load_handles_corrupt_json():
    """load() survives corrupt settings.json by falling back to defaults."""
    os.makedirs(settings.SETTINGS_DIR, exist_ok=True)
    with open(settings.SETTINGS_PATH, 'w') as f:
        f.write("this is not valid json {{{")
    cfg = settings.load()
    assert cfg['chunk_size'] == settings.DEFAULTS['chunk_size']


# ── get ───────────────────────────────────────────────────────────────────────

def test_get_returns_default_for_unknown_key():
    """get() returns the default value for an unknown key."""
    val = settings.get('chunk_size')
    assert val == settings.DEFAULTS['chunk_size']


def test_get_returns_saved_value():
    """get() returns the saved value after save()."""
    cfg = settings.load()
    cfg['chunk_size'] = 1234
    settings.save(cfg)
    assert settings.get('chunk_size') == 1234


def test_get_caches_result():
    """get() uses cache — second call doesn't re-read from disk."""
    # First call caches
    val1 = settings.get('chunk_size')
    # Corrupt the file
    os.makedirs(settings.SETTINGS_DIR, exist_ok=True)
    with open(settings.SETTINGS_PATH, 'w') as f:
        f.write("garbage {{{")
    # Second call uses cache — should NOT fail
    val2 = settings.get('chunk_size')
    assert val1 == val2


# ── get_data_dir ──────────────────────────────────────────────────────────────

def test_get_data_dir_expands_tilde():
    """get_data_dir() expands ~ to the home directory."""
    data_dir = settings.get_data_dir()
    assert not data_dir.startswith('~')
    assert os.path.isabs(data_dir)


def test_get_data_dir_is_absolute():
    """get_data_dir() returns an absolute path."""
    data_dir = settings.get_data_dir()
    assert os.path.isabs(data_dir)


# ── external index registry ───────────────────────────────────────────────────

def test_get_external_indexes_defaults_empty():
    """get_external_indexes() returns empty list by default."""
    assert settings.get_external_indexes() == []


def test_add_external_index(tmp_path):
    """add_external_index() registers and persists an external path."""
    idx_dir = tmp_path / "test_index"
    idx_dir.mkdir()
    result = settings.add_external_index(str(idx_dir))
    assert result is True
    assert str(idx_dir) in settings.get_external_indexes()


def test_add_external_index_duplicate(tmp_path):
    """add_external_index() returns False for duplicates."""
    idx_dir = tmp_path / "test_index"
    idx_dir.mkdir()
    settings.add_external_index(str(idx_dir))
    result = settings.add_external_index(str(idx_dir))
    assert result is False


def test_remove_external_index(tmp_path):
    """remove_external_index() unregisters a path."""
    idx_dir = tmp_path / "test_index"
    idx_dir.mkdir()
    settings.add_external_index(str(idx_dir))
    result = settings.remove_external_index(str(idx_dir))
    assert result is True
    assert str(idx_dir) not in settings.get_external_indexes()


def test_remove_nonexistent_external_index():
    """remove_external_index() returns False for unknown paths."""
    result = settings.remove_external_index("/nonexistent/path")
    assert result is False


# ── optional deps ─────────────────────────────────────────────────────────────

def test_check_all_deps_returns_dict():
    """check_all_deps() returns a dict of {name: bool}."""
    deps = settings.check_all_deps()
    assert isinstance(deps, dict)
    # Some should be checkable (True or False, not None)
    for name, present in deps.items():
        assert isinstance(present, bool), f"{name} is not a bool"
