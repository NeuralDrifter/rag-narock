"""
gui/dialogs/acl_dialog.py — Modal tkinter dialogs for managing RAG access control.
Moved from rag_settings.py per separation of concerns (SRP / Ch 10 & 11).
"""

import tkinter as tk
from tkinter import ttk
import acl.acl as rag_acl
from gui.theme import show_copyable_info, ask_copyable_yesno, show_copyable_error

class ACLDialog:
    """Modal tkinter dialog for managing RAG access control."""

    BG       = '#0f1626'
    BG2      = '#1a2332'
    BG3      = '#243044'
    FG       = '#d4d4dc'
    FG_DIM   = '#7a8599'
    ACCENT   = '#e8781e'
    ICE      = '#5eb8d4'
    WARN_RED = '#e74c3c'
    GREEN    = '#2ecc71'

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("RAG Access Control")
        self.win.geometry("620x420")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.configure(bg=self.BG)

        # Enable/disable toggle
        top_frame = tk.Frame(self.win, bg=self.BG)
        top_frame.pack(fill='x', padx=15, pady=(15, 5))

        self.enabled_var = tk.BooleanVar(value=rag_acl.is_enabled())
        tk.Label(top_frame, text="Access Control:", bg=self.BG, fg=self.FG,
                 font=('sans-serif', 11, 'bold')).pack(side='left')
        self.toggle_btn = tk.Button(top_frame,
                                     text="ENABLED" if self.enabled_var.get() else "DISABLED",
                                     bg=self.GREEN if self.enabled_var.get() else self.WARN_RED,
                                     fg='white', font=('sans-serif', 9, 'bold'), relief='flat',
                                     padx=10, command=self._toggle_enabled)
        self.toggle_btn.pack(side='left', padx=10)

        # Treeview for clients
        tree_frame = tk.Frame(self.win, bg=self.BG)
        tree_frame.pack(fill='both', expand=True, padx=15, pady=5)

        style = ttk.Style(self.win)
        style.configure('ACL.Treeview', background=self.BG2, foreground=self.FG,
                         fieldbackground=self.BG2, rowheight=28)
        style.configure('ACL.Treeview.Heading', background=self.BG3, foreground=self.ACCENT,
                         font=('sans-serif', 9, 'bold'))

        columns = ('name', 'key', 'indexes', 'privileges')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                  style='ACL.Treeview', selectmode='browse')
        self.tree.heading('name', text='Name')
        self.tree.heading('key', text='Key')
        self.tree.heading('indexes', text='Indexes')
        self.tree.heading('privileges', text='Privileges')
        self.tree.column('name', width=100)
        self.tree.column('key', width=120)
        self.tree.column('indexes', width=120)
        self.tree.column('privileges', width=240)

        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Buttons
        btn_frame = tk.Frame(self.win, bg=self.BG)
        btn_frame.pack(fill='x', padx=15, pady=(5, 5))

        for text, cmd in [("New Key", self._create_key), ("Edit", self._edit_key),
                          ("Rotate", self._rotate_key), ("Copy Key", self._copy_key),
                          ("Revoke", self._revoke_key)]:
            tk.Button(btn_frame, text=text, command=cmd, bg=self.BG3, fg=self.FG,
                       relief='flat', padx=8, pady=3).pack(side='left', padx=3)

        tk.Button(btn_frame, text="Close", command=self.win.destroy, bg=self.BG3, fg=self.FG,
                   relief='flat', padx=8, pady=3).pack(side='right', padx=3)

        # Status bar
        self.status_var = tk.StringVar(value="")
        tk.Label(self.win, textvariable=self.status_var, bg=self.BG, fg=self.ICE,
                 font=('monospace', 8), anchor='w').pack(fill='x', padx=15, pady=(0, 10))

        self._refresh()

        # Center on parent
        self.win.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.win.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.win.winfo_height()) // 2
        self.win.geometry(f"+{max(0,px)}+{max(0,py)}")

        self.win.wait_window()

    def _refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.acl_clients = rag_acl.list_clients()
        for key, client in self.acl_clients:
            masked = rag_acl.mask_key(key)
            indexes = client.get('indexes', '*')
            idx_str = "*" if indexes == "*" else ", ".join(indexes) if indexes else "(none)"
            privs = ", ".join(client.get('privileges', []))
            self.tree.insert('', 'end', iid=key,
                              values=(client.get('name', '?'), masked, idx_str, privs))

    def _toggle_enabled(self):
        new_state = not rag_acl.is_enabled()
        rag_acl.set_enabled(new_state)
        self.enabled_var.set(new_state)
        self.toggle_btn.config(text="ENABLED" if new_state else "DISABLED",
                                bg=self.GREEN if new_state else self.WARN_RED)
        self.status_var.set("ACL " + ("enabled" if new_state else "disabled"))

    def _get_selected_key(self):
        sel = self.tree.selection()
        if not sel:
            self.status_var.set("No client selected")
            return None
        return sel[0]

    def _create_key(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("New API Key", "Client name:", parent=self.win)
        if not name:
            return
        key = rag_acl.create_client(name.strip())
        self._refresh()
        show_copyable_info(self.win, "Key Created",
                             f"Name: {name.strip()}\nKey: {key}\n\n"
                             "Add to MCP client config:\n"
                             f'  "env": {{"RAG_API_KEY": "{key}"}}')
        self.status_var.set(f"Created key for {name.strip()}")

    def _edit_key(self):
        key = self._get_selected_key()
        if not key:
            return
        client = dict(rag_acl.load()['clients'].get(key, {}))
        if not client:
            return
        ACLEditDialog(self.win, key, client)
        self._refresh()

    def _rotate_key(self):
        key = self._get_selected_key()
        if not key:
            return
        client = rag_acl.load()['clients'].get(key, {})
        if not ask_copyable_yesno(self.win, "Rotate Key",
                                    f"Rotate key for '{client.get('name', '?')}'?\n"
                                    "The old key will stop working immediately."):
            return
        new_key = rag_acl.rotate_key(key)
        self._refresh()
        show_copyable_info(self.win, "Key Rotated", f"New key: {new_key}")
        self.status_var.set(f"Key rotated: {new_key}")

    def _copy_key(self):
        key = self._get_selected_key()
        if not key:
            return
        self.win.clipboard_clear()
        self.win.clipboard_append(key)
        self.status_var.set(f"Copied: {key}")

    def _revoke_key(self):
        key = self._get_selected_key()
        if not key:
            return
        client = rag_acl.load()['clients'].get(key, {})
        if not ask_copyable_yesno(self.win, "Revoke Key",
                                    f"Revoke key for '{client.get('name', '?')}'?\n"
                                    "This cannot be undone."):
            return
        rag_acl.revoke_client(key)
        self._refresh()
        self.status_var.set("Key revoked")


class ACLEditDialog:
    """Sub-dialog for editing a client's name, indexes, and privileges."""

    BG  = '#0f1626'
    BG2 = '#1a2332'
    BG3 = '#243044'
    FG  = '#d4d4dc'

    def __init__(self, parent, key, client):
        self.key = key
        self.win = tk.Toplevel(parent)
        self.win.title(f"Edit: {client.get('name', '?')}")
        self.win.geometry("400x350")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.configure(bg=self.BG)

        pad = {'padx': 15, 'pady': 5}

        tk.Label(self.win, text="Name:", bg=self.BG, fg=self.FG).pack(anchor='w', **pad)
        self.name_var = tk.StringVar(value=client.get('name', ''))
        tk.Entry(self.win, textvariable=self.name_var, width=40).pack(anchor='w', padx=15)

        tk.Label(self.win, text="Indexes (* for all, or comma-separated):",
                 bg=self.BG, fg=self.FG).pack(anchor='w', **pad)
        indexes = client.get('indexes', '*')
        idx_str = "*" if indexes == "*" else ", ".join(indexes)
        self.idx_var = tk.StringVar(value=idx_str)
        tk.Entry(self.win, textvariable=self.idx_var, width=40).pack(anchor='w', padx=15)

        tk.Label(self.win, text="Privileges:", bg=self.BG, fg=self.FG).pack(anchor='w', **pad)
        self.priv_vars = {}
        priv_frame = tk.Frame(self.win, bg=self.BG)
        priv_frame.pack(anchor='w', padx=15)
        current_privs = set(client.get('privileges', []))
        for priv in rag_acl.ALL_PRIVILEGES:
            var = tk.BooleanVar(value=priv in current_privs)
            tk.Checkbutton(priv_frame, text=priv, variable=var, bg=self.BG, fg=self.FG,
                            selectcolor=self.BG2, activebackground=self.BG).pack(anchor='w')
            self.priv_vars[priv] = var

        btn_frame = tk.Frame(self.win, bg=self.BG)
        btn_frame.pack(fill='x', padx=15, pady=15)
        tk.Button(btn_frame, text="Cancel", command=self.win.destroy, bg=self.BG3, fg=self.FG,
                   relief='flat', padx=10).pack(side='right', padx=3)
        tk.Button(btn_frame, text="Save", command=self._save, bg=self.BG3, fg=self.FG,
                   relief='flat', padx=10).pack(side='right', padx=3)

        self.win.wait_window()

    def _save(self):
        name = self.name_var.get().strip() or None
        idx_str = self.idx_var.get().strip()
        if idx_str == "*":
            indexes = "*"
        else:
            indexes = [i.strip() for i in idx_str.split(",") if i.strip()]
        privileges = [p for p, v in self.priv_vars.items() if v.get()]
        rag_acl.update_client(self.key, name=name, indexes=indexes, privileges=privileges)
        self.win.destroy()
