"""
Characterization tests for acl/acl.py — access control logic.

These tests verify the pure access control functions:
create_client, revoke_client, rotate_key, check_access, filter_indexes,
list_clients, set_enabled, is_enabled, mask_key, clean_stale_indexes.
"""

import os
import json
import tempfile
import pytest

import acl.acl as acl


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_acl_dir(monkeypatch):
    """Point ACL_DIR to a temporary directory for isolated tests."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(acl, 'ACL_DIR', tmp)
        monkeypatch.setattr(acl, 'ACL_PATH', os.path.join(tmp, 'access_control.json'))
        yield tmp


@pytest.fixture
def empty_acl(temp_acl_dir):
    """Ensure ACL file starts empty for each test."""
    acl.save({"enabled": False, "clients": {}})


# ── load / save ───────────────────────────────────────────────────────────────

def test_load_returns_defaults_when_no_file(temp_acl_dir):
    """load() returns defaults when no ACL file exists."""
    result = acl.load()
    assert result == {"enabled": False, "clients": {}}


def test_save_and_load_roundtrip(temp_acl_dir):
    """save() writes data that load() can read back."""
    data = {"enabled": True, "clients": {"key1": {"name": "test"}}}
    acl.save(data)
    result = acl.load()
    assert result["enabled"] is True
    assert "key1" in result["clients"]


def test_load_merges_missing_keys(temp_acl_dir):
    """load() fills in missing keys with defaults."""
    acl.save({"enabled": True})  # no 'clients' key
    result = acl.load()
    assert result["clients"] == {}


# ── generate_key ──────────────────────────────────────────────────────────────

def test_generate_key_format():
    """generate_key() produces keys with 'rag-' prefix and 16 hex chars."""
    key = acl.generate_key()
    assert key.startswith("rag-")
    assert len(key) == 4 + 16  # rag- + 16 hex chars


def test_generate_key_is_unique():
    """generate_key() produces unique keys."""
    keys = {acl.generate_key() for _ in range(100)}
    assert len(keys) == 100


# ── create_client / list_clients ──────────────────────────────────────────────

def test_create_client_adds_entry(temp_acl_dir):
    """create_client() adds a client to the ACL."""
    key = acl.create_client("TestBot")
    clients = dict(acl.list_clients())
    assert key in clients
    assert clients[key]["name"] == "TestBot"


def test_create_client_defaults_to_all_privileges(temp_acl_dir):
    """create_client() without privileges gives ALL_PRIVILEGES."""
    key = acl.create_client("Bot")
    _, client = acl.list_clients()[0]
    assert set(client["privileges"]) == set(acl.ALL_PRIVILEGES)


def test_create_client_with_specific_privileges(temp_acl_dir):
    """create_client() with explicit privileges uses them."""
    key = acl.create_client("Reader", privileges=["list", "query"])
    _, client = acl.list_clients()[0]
    assert set(client["privileges"]) == {"list", "query"}


def test_create_client_rejects_invalid_privileges(temp_acl_dir):
    """create_client() raises ValueError for invalid privilege names."""
    with pytest.raises(ValueError, match="Invalid privileges"):
        acl.create_client("Bad", privileges=["fly_to_moon"])


# ── revoke_client ─────────────────────────────────────────────────────────────

def test_revoke_client_removes_entry(temp_acl_dir):
    """revoke_client() removes the client."""
    key = acl.create_client("Temp")
    assert acl.revoke_client(key) is True
    assert key not in dict(acl.list_clients())


def test_revoke_nonexistent_returns_false(temp_acl_dir):
    """revoke_client() on unknown key returns False."""
    assert acl.revoke_client("nonexistent") is False


# ── rotate_key ────────────────────────────────────────────────────────────────

def test_rotate_key_changes_key_preserves_data(temp_acl_dir):
    """rotate_key() returns new key, old key stops working."""
    old_key = acl.create_client("RotateMe", privileges=["list"])
    new_key = acl.rotate_key(old_key)
    assert new_key != old_key
    assert new_key.startswith("rag-")
    # Old key no longer works
    assert old_key not in dict(acl.list_clients())
    # New key has same data
    _, client = acl.list_clients()[0]
    assert client["name"] == "RotateMe"
    assert set(client["privileges"]) == {"list"}


def test_rotate_nonexistent_raises_keyerror(temp_acl_dir):
    """rotate_key() on unknown key raises KeyError."""
    with pytest.raises(KeyError):
        acl.rotate_key("nonexistent")


# ── update_client ─────────────────────────────────────────────────────────────

def test_update_client_name(temp_acl_dir):
    """update_client() changes the client name."""
    key = acl.create_client("OldName")
    acl.update_client(key, name="NewName")
    _, client = acl.list_clients()[0]
    assert client["name"] == "NewName"


def test_update_client_privileges(temp_acl_dir):
    """update_client() changes privileges."""
    key = acl.create_client("Bot")
    acl.update_client(key, privileges=["list", "query"])
    _, client = acl.list_clients()[0]
    assert set(client["privileges"]) == {"list", "query"}


def test_update_nonexistent_raises_keyerror(temp_acl_dir):
    """update_client() on unknown key raises KeyError."""
    with pytest.raises(KeyError):
        acl.update_client("nonexistent", name="X")


# ── enabled/disabled ──────────────────────────────────────────────────────────

def test_is_enabled_defaults_false(temp_acl_dir):
    """is_enabled() returns False by default."""
    assert acl.is_enabled() is False


def test_set_enabled_toggle(temp_acl_dir):
    """set_enabled() toggles the enabled flag."""
    acl.set_enabled(True)
    assert acl.is_enabled() is True
    acl.set_enabled(False)
    assert acl.is_enabled() is False


# ── check_access ──────────────────────────────────────────────────────────────

def test_check_access_disabled_allows_all(temp_acl_dir):
    """check_access() allows everything when ACL is disabled."""
    ok, msg = acl.check_access(None, "delete", index_name="any")
    assert ok is True
    assert msg is None


def test_check_access_no_key_when_enabled(temp_acl_dir):
    """check_access() denies when ACL enabled and no key provided."""
    acl.set_enabled(True)
    ok, msg = acl.check_access(None, "list")
    assert ok is False
    assert "No API key" in msg


def test_check_access_invalid_key(temp_acl_dir):
    """check_access() denies invalid keys."""
    acl.set_enabled(True)
    ok, msg = acl.check_access("bad-key", "list")
    assert ok is False
    assert "Invalid API key" in msg


def test_check_access_missing_privilege(temp_acl_dir):
    """check_access() denies when key lacks required privilege."""
    acl.set_enabled(True)
    key = acl.create_client("Reader", privileges=["list"])
    ok, msg = acl.check_access(key, "delete")
    assert ok is False
    assert "PRIVILEGE DENIED" in msg


def test_check_access_has_privilege(temp_acl_dir):
    """check_access() allows when key has the privilege."""
    acl.set_enabled(True)
    key = acl.create_client("Admin")
    ok, _ = acl.check_access(key, "delete")
    assert ok is True


def test_check_access_index_restriction(temp_acl_dir):
    """check_access() denies access to restricted indexes."""
    acl.set_enabled(True)
    key = acl.create_client("Limited", indexes=["index_a"])
    ok, msg = acl.check_access(key, "query", index_name="index_b")
    assert ok is False
    assert "INDEX ACCESS DENIED" in msg


def test_check_access_wildcard_indexes(temp_acl_dir):
    """check_access() allows any index when client has '*' indexes."""
    acl.set_enabled(True)
    key = acl.create_client("Full", indexes="*")
    ok, _ = acl.check_access(key, "query", index_name="any_index")
    assert ok is True


# ── filter_indexes ────────────────────────────────────────────────────────────

def test_filter_indexes_disabled_returns_all(temp_acl_dir):
    """filter_indexes() returns all when ACL is disabled."""
    result = acl.filter_indexes(None, ["a", "b", "c"])
    assert result == ["a", "b", "c"]


def test_filter_indexes_invalid_key_returns_empty(temp_acl_dir):
    """filter_indexes() returns empty list for invalid key."""
    acl.set_enabled(True)
    result = acl.filter_indexes("bad", ["a", "b"])
    assert result == []


def test_filter_indexes_wildcard(temp_acl_dir):
    """filter_indexes() returns all for wildcard access."""
    acl.set_enabled(True)
    key = acl.create_client("Full", indexes="*")
    result = acl.filter_indexes(key, ["a", "b"])
    assert result == ["a", "b"]


def test_filter_indexes_restricted(temp_acl_dir):
    """filter_indexes() only returns allowed indexes."""
    acl.set_enabled(True)
    key = acl.create_client("Limited", indexes=["a", "c"])
    result = acl.filter_indexes(key, ["a", "b", "c"])
    assert result == ["a", "c"]


# ── mask_key ──────────────────────────────────────────────────────────────────

def test_mask_key_short():
    """mask_key() returns short keys unchanged."""
    assert acl.mask_key("abc123") == "abc123"


def test_mask_key_long():
    """mask_key() masks long keys: first 8 + '...' + last 4."""
    masked = acl.mask_key("rag-a1b2c3d4e5f6g7h8")
    assert masked == "rag-a1b2...g7h8"


# ── clean_stale_indexes ───────────────────────────────────────────────────────

def test_clean_stale_indexes_removes_deleted(temp_acl_dir):
    """clean_stale_indexes() removes references to deleted indexes."""
    key = acl.create_client("Bot", indexes=["idx1", "idx2", "idx3"])
    changed = acl.clean_stale_indexes(["idx1", "idx3"])  # idx2 is gone
    assert changed is True
    _, client = acl.list_clients()[0]
    assert client["indexes"] == ["idx1", "idx3"]


def test_clean_stale_indexes_no_change(temp_acl_dir):
    """clean_stale_indexes() returns False when nothing changes."""
    acl.create_client("Bot", indexes=["idx1"])
    changed = acl.clean_stale_indexes(["idx1", "idx2"])
    assert changed is False


def test_clean_stale_indexes_wildcard_unchanged(temp_acl_dir):
    """clean_stale_indexes() leaves '*' wildcard alone."""
    acl.create_client("Bot", indexes="*")
    changed = acl.clean_stale_indexes(["idx1"])
    assert changed is False
