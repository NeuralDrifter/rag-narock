"""
acl/acl.py — access control logic (pure).

Moved from rag_acl.py during modular refactor.
All functions and constants preserved exactly.
"""

import os, json, secrets

ACL_DIR = os.path.expanduser("~/.local/share/rag")
ACL_PATH = os.path.join(ACL_DIR, "access_control.json")

ALL_PRIVILEGES = ["list", "query", "read", "export", "index", "remove", "delete", "lock"]
READ_PRIVILEGES = ["list", "query", "read", "export"]
WRITE_PRIVILEGES = ["index", "remove"]
ADMIN_PRIVILEGES = ["delete", "lock"]


def load():
    """Load ACL config from disk, merging with defaults."""
    acl = {"enabled": False, "clients": {}}
    if os.path.exists(ACL_PATH):
        try:
            with open(ACL_PATH) as f:
                saved = json.load(f)
            if isinstance(saved.get('enabled'), bool):
                acl['enabled'] = saved['enabled']
            if isinstance(saved.get('clients'), dict):
                acl['clients'] = saved['clients']
        except (json.JSONDecodeError, OSError):
            pass
    return acl


def save(acl):
    """Save ACL config to disk."""
    os.makedirs(ACL_DIR, exist_ok=True)
    with open(ACL_PATH, 'w') as f:
        json.dump(acl, f, indent=2)


def generate_key():
    """Generate a new API key: rag-<16 hex chars>."""
    return "rag-" + secrets.token_hex(8)


def create_client(name, indexes="*", privileges=None):
    acl = load()
    key = generate_key()
    if privileges is None:
        privileges = list(ALL_PRIVILEGES)
    invalid = set(privileges) - set(ALL_PRIVILEGES)
    if invalid:
        raise ValueError(f"Invalid privileges: {', '.join(sorted(invalid))}. "
                         f"Valid: {', '.join(ALL_PRIVILEGES)}")
    acl['clients'][key] = {
        'name': name,
        'indexes': indexes,
        'privileges': privileges,
    }
    save(acl)
    return key


def revoke_client(key):
    acl = load()
    if key in acl['clients']:
        del acl['clients'][key]
        save(acl)
        return True
    return False


def rotate_key(old_key):
    acl = load()
    if old_key not in acl['clients']:
        raise KeyError(f"Key not found: {old_key}")
    client = acl['clients'].pop(old_key)
    new_key = generate_key()
    acl['clients'][new_key] = client
    save(acl)
    return new_key


def update_client(key, name=None, indexes=None, privileges=None):
    acl = load()
    if key not in acl['clients']:
        raise KeyError(f"Key not found: {key}")
    client = acl['clients'][key]
    if name is not None:
        client['name'] = name
    if indexes is not None:
        client['indexes'] = indexes
    if privileges is not None:
        invalid = set(privileges) - set(ALL_PRIVILEGES)
        if invalid:
            raise ValueError(f"Invalid privileges: {', '.join(sorted(invalid))}. "
                             f"Valid: {', '.join(ALL_PRIVILEGES)}")
        client['privileges'] = privileges
    save(acl)


def list_clients():
    acl = load()
    return list(acl['clients'].items())


def set_enabled(enabled):
    acl = load()
    acl['enabled'] = bool(enabled)
    save(acl)


def is_enabled():
    return load()['enabled']


def check_access(api_key, operation, index_name=None):
    acl = load()
    if not acl['enabled']:
        return (True, None)
    if not api_key:
        return (False, "ACCESS DENIED: No API key provided. "
                "Set RAG_API_KEY environment variable in your MCP client registration.")
    client = acl['clients'].get(api_key)
    if client is None:
        return (False, "ACCESS DENIED: Invalid API key.")
    if operation not in client.get('privileges', []):
        return (False, f"PRIVILEGE DENIED: Key '{client['name']}' lacks '{operation}' privilege.")
    if index_name is not None:
        allowed_indexes = client.get('indexes', [])
        if allowed_indexes != "*":
            if index_name not in allowed_indexes:
                return (False, f"INDEX ACCESS DENIED: Key '{client['name']}' "
                        f"cannot access index '{index_name}'.")
    return (True, None)


def filter_indexes(api_key, index_names):
    acl = load()
    if not acl['enabled']:
        return index_names
    client = acl['clients'].get(api_key)
    if client is None:
        return []
    allowed = client.get('indexes', [])
    if allowed == "*":
        return index_names
    return [n for n in index_names if n in allowed]


def clean_stale_indexes(available_indexes):
    acl = load()
    available_set = set(available_indexes)
    changed = False
    for client in acl['clients'].values():
        indexes = client.get('indexes', '*')
        if isinstance(indexes, list):
            new_indexes = [i for i in indexes if i in available_set]
            if len(new_indexes) != len(indexes):
                client['indexes'] = new_indexes
                changed = True
    if changed:
        save(acl)
    return changed


def mask_key(key):
    if len(key) > 14:
        return key[:8] + '...' + key[-4:]
    return key
