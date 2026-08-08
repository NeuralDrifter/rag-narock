"""
gui/dialogs/editor_dialog.py — Full EditorDialog (Tkinter) ported from original.
Provides list of indexes and sources (entries), ability to read/view counts and details,
delete, rename, lock/unlock, hash/integrity, remove source, export specific document/source,
export whole index. Full functionality for GUI systems.
"""

import os
import re

from core.index_manager import (
    get_indexes,
    resolve_index_dir,
    get_index_sources,
    delete_index,
    remove_source_from_index,
    export_source,
    export_index,
    is_index_locked,
)
from core.integrity import check_index_integrity, save_index_integrity
from storage.base import get_index_meta_with_defaults
from gui.theme import show_copyable_warning, show_copyable_error, ask_copyable_yesno
from gui.dialogs.source_viewer_dialog import SourceViewerDialog
from rag_backends import detect_backend, get_backend


def _rag():
    # For compatibility shims if needed, but we use direct
    pass


class EditorDialog:
    """Modal tkinter index editor dialog. Full port."""

    BG       = '#0f1626'
    BG2      = '#1a2332'
    BG3      = '#243044'
    FG       = '#d4d4dc'
    FG_DIM   = '#7a8599'
    ACCENT   = '#e8781e'
    ACCENT2  = '#ff9a3c'
    ICE      = '#5eb8d4'
    FIRE_RED = '#c0392b'
    GREEN    = '#27ae60'

    def __init__(self, parent, on_change=None):
        import tkinter as tk
        from tkinter import ttk

        self.on_change = on_change
        self.parent = parent

        # Initialize all tracking attributes immediately
        self.selected_index_idx = None
        self.selected_source_idx = None
        self.checked_indexes = set()
        self.checked_sources = set()

        # Also pre-init structures populated by _refresh_indexes()
        self._indexes = []
        self._sources = []

        self.win = tk.Toplevel(parent)
        self.win.title("RAG-Narock Index Editor")
        self.win.geometry("700x480")
        self.win.resizable(True, True)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.configure(bg=self.BG)

        # Styles
        style = ttk.Style(self.win)
        style.configure('Editor.TLabelframe', background=self.BG, foreground=self.ACCENT,
                        bordercolor='#2a3a52')
        style.configure('Editor.TLabelframe.Label', background=self.BG, foreground=self.ACCENT,
                        font=('sans-serif', 10, 'bold'))
        style.configure('Editor.TButton', background=self.ACCENT, foreground='#ffffff',
                        bordercolor=self.ACCENT, font=('sans-serif', 9, 'bold'), padding=(8, 4))
        style.map('Editor.TButton',
                  background=[('active', self.ACCENT2), ('disabled', self.BG3)],
                  foreground=[('disabled', self.FG_DIM)])
        style.configure('Danger.TButton', background=self.FIRE_RED, foreground='#ffffff',
                        bordercolor=self.FIRE_RED, font=('sans-serif', 9, 'bold'), padding=(8, 4))
        style.map('Danger.TButton',
                  background=[('active', '#e74c3c'), ('disabled', self.BG3)],
                  foreground=[('disabled', self.FG_DIM)])

        # Main layout
        main_frame = ttk.Frame(self.win)
        main_frame.pack(fill='both', expand=True, padx=10, pady=(10, 5))

        # Left: Indexes
        left_frame = ttk.LabelFrame(main_frame, text="Indexes", style='Editor.TLabelframe', padding=5)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        self.index_list = tk.Listbox(left_frame, font=('monospace', 9),
                                      bg=self.BG3, fg=self.FG, selectbackground=self.ACCENT,
                                      selectforeground='#ffffff', highlightthickness=0,
                                      borderwidth=1, relief='flat', exportselection=False)
        idx_scroll = ttk.Scrollbar(left_frame, orient='vertical', command=self.index_list.yview)
        self.index_list.configure(yscrollcommand=idx_scroll.set)
        self.index_list.pack(side='left', fill='both', expand=True)
        idx_scroll.pack(side='right', fill='y')
        self.index_list.bind('<<ListboxSelect>>', self._on_index_select)

        # Right: Sources
        self.source_label_var = tk.StringVar(value="Sources")
        self.right_frame = ttk.LabelFrame(main_frame, text="Sources",
                                           style='Editor.TLabelframe', padding=5)
        right_frame = self.right_frame
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        self.source_list = tk.Listbox(right_frame, font=('monospace', 9),
                                       bg=self.BG3, fg=self.FG, selectbackground=self.ACCENT,
                                       selectforeground='#ffffff', highlightthickness=0,
                                       borderwidth=1, relief='flat', exportselection=False)
        src_scroll = ttk.Scrollbar(right_frame, orient='vertical', command=self.source_list.yview)
        self.source_list.configure(yscrollcommand=src_scroll.set)
        self.source_list.pack(side='left', fill='both', expand=True)
        src_scroll.pack(side='right', fill='y')

        self.source_list.bind('<<ListboxSelect>>', self._on_source_select)
        self.index_list.bind('<ButtonRelease-1>', self._on_index_checkbox_click)
        self.source_list.bind('<ButtonRelease-1>', self._on_source_checkbox_click)
        self.source_list.bind('<Double-1>', self._read_source)

        # Buttons
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill='x', padx=10, pady=5)

        self.delete_btn = ttk.Button(btn_frame, text="Delete Index", style='Danger.TButton',
                                      command=self._delete_index)
        self.delete_btn.pack(side='left', padx=2)

        self.rename_btn = ttk.Button(btn_frame, text="Rename", style='Editor.TButton',
                                      command=self._rename_index)
        self.rename_btn.pack(side='left', padx=2)

        self.lock_btn = ttk.Button(btn_frame, text="Lock/Unlock", style='Editor.TButton',
                                    command=self._toggle_lock)
        self.lock_btn.pack(side='left', padx=2)

        self.hash_btn = ttk.Button(btn_frame, text="Hash", style='Editor.TButton',
                                    command=self._hash_index)
        self.hash_btn.pack(side='left', padx=2)

        self.remove_btn = ttk.Button(btn_frame, text="Remove Source", style='Danger.TButton',
                                      command=self._remove_source)
        self.remove_btn.pack(side='right', padx=2)

        self.export_all_btn = ttk.Button(btn_frame, text="Export All", style='Editor.TButton',
                                          command=self._export_all)
        self.export_all_btn.pack(side='right', padx=2)

        self.export_btn = ttk.Button(btn_frame, text="Export Source", style='Editor.TButton',
                                      command=self._export_source)
        self.export_btn.pack(side='right', padx=2)

        self.read_btn = ttk.Button(btn_frame, text="Read Source", style='Editor.TButton',
                                    command=self._read_source)
        self.read_btn.pack(side='right', padx=2)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(self.win, textvariable=self.status_var, bg=self.BG, fg=self.ICE,
                                 font=('sans-serif', 9), anchor='w')
        status_label.pack(fill='x', padx=12, pady=(0, 10))

        self._refresh_indexes()

        # Center on parent
        self.win.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.win.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.win.winfo_height()) // 2
        self.win.geometry(f"+{max(0,px)}+{max(0,py)}")

        self.win.wait_window()

    def _refresh_indexes(self):
        self.index_list.delete(0, 'end')
        self._indexes = []
        for name in get_indexes():
            index_dir = resolve_index_dir(name)
            meta = get_index_meta_with_defaults(index_dir)
            locked = is_index_locked(name)
            n_chunks = meta.get('n_chunks', 0)
            n_files = meta.get('n_files', 0)
            storage = meta.get('storage_backend', 'faiss')
            emb_model = meta.get('embedding_model', '?')
            integrity = check_index_integrity(index_dir)
            tampered = not integrity['ok'] and not integrity.get('untracked')
            unverified = integrity.get('untracked', False)
            lock_str = " [LOCKED]" if locked else ""
            if tampered:
                integrity_str = " [TAMPERED]"
            elif unverified:
                integrity_str = " [UNVERIFIED]"
            else:
                integrity_str = ""
            checked = "☑" if name in self.checked_indexes else "☐"
            self.index_list.insert('end', f" {checked}  {name}  ({n_chunks:,}ch, {n_files}f){lock_str}{integrity_str} [{storage}, {emb_model}]")
            self._indexes.append({
                'name': name, 'n_chunks': n_chunks,
                'n_files': n_files, 'locked': locked,
                'tampered': tampered, 'unverified': unverified,
            })
        self.source_list.delete(0, 'end')
        self.right_frame.configure(text="Sources")
        self.selected_source_idx = None
        self.selected_index_idx = None  # user will re-select after full refresh
        # keep checked_* for bulk selection persistence across refreshes

    def _on_index_select(self, event=None):
        sel = self.index_list.curselection()
        if not sel:
            return
        new_idx = sel[0]
        # Only reset source checks when actually switching to a different index
        if self.selected_index_idx != new_idx:
            self.checked_sources = set()
        self.selected_index_idx = new_idx
        self.selected_source_idx = None
        idx = self._indexes[self.selected_index_idx]
        sources = get_index_sources(idx['name'])
        self.source_list.delete(0, 'end')
        self._sources = sorted(sources.items(), key=lambda x: x[0])
        for src_name, count in self._sources:
            checked = "☑" if src_name in self.checked_sources else "☐"
            self.source_list.insert('end', f" {checked}  {src_name}  ({count:,} chunks)")
        self.right_frame.configure(text=f"Sources in: {idx['name']}")
        if idx.get('tampered'):
            self.status_var.set(f"WARNING: '{idx['name']}' integrity mismatch — data may have been modified. Use 'Hash' button to re-verify.")
        elif idx.get('unverified'):
            self.status_var.set(f"UNVERIFIED: '{idx['name']}' has no integrity record. Use 'Hash' button to compute hashes.")

    def _on_source_select(self, event=None):
        sel = self.source_list.curselection()
        if sel:
            self.selected_source_idx = sel[0]
            # Make sure the parent index stays selected (prevents "select index first" surprises)
            if self.selected_index_idx is not None:
                try:
                    self.index_list.selection_clear(0, 'end')
                    self.index_list.selection_set(self.selected_index_idx)
                except:
                    pass

    def _on_index_checkbox_click(self, event):
        idx = self.index_list.nearest(event.y)
        if 0 <= idx < len(self._indexes):
            name = self._indexes[idx]['name']
            if name in self.checked_indexes:
                self.checked_indexes.remove(name)
            else:
                self.checked_indexes.add(name)
            self._refresh_indexes()
            # restore view / highlight
            try:
                self.index_list.selection_set(idx)
                self._on_index_select()
            except:
                pass
        return "break"

    def _on_source_checkbox_click(self, event):
        idx = self.source_list.nearest(event.y)
        if 0 <= idx < len(self._sources):
            src_name = self._sources[idx][0]
            if src_name in self.checked_sources:
                self.checked_sources.remove(src_name)
            else:
                self.checked_sources.add(src_name)
            # Surgically update only this row's text with the new checkbox state.
            # This avoids full delete/insert which can interfere with selection/click processing.
            ch = "☑" if src_name in self.checked_sources else "☐"
            cnt = self._sources[idx][1]
            new_text = f" {ch}  {src_name}  ({cnt:,} chunks)"
            try:
                self.source_list.delete(idx)
                self.source_list.insert(idx, new_text)
                # Keep it highlighted as current
                self.source_list.selection_set(idx)
            except:
                pass
            # Record as current for fallback single-select logic
            self.selected_source_idx = idx
            # Keep parent index selected visually (no full reselect logic)
            if self.selected_index_idx is not None:
                try:
                    self.index_list.selection_set(self.selected_index_idx)
                except:
                    pass
        return "break"

    def _selected_index(self):
        if self.selected_index_idx is not None and 0 <= self.selected_index_idx < len(self._indexes):
            return self._indexes[self.selected_index_idx]
        sel = self.index_list.curselection()
        if sel:
            self.selected_index_idx = sel[0]
            return self._indexes[sel[0]]
        return None

    def _notify_change(self):
        if self.on_change:
            self.on_change()

    def _delete_index(self):
        to_delete = list(self.checked_indexes) if self.checked_indexes else []
        if not to_delete:
            idx = self._selected_index()
            if idx:
                to_delete = [idx['name']]
        if not to_delete:
            return
        for name in to_delete:
            idx_info = next((i for i in self._indexes if i['name'] == name), None)
            if idx_info and idx_info['locked']:
                show_copyable_warning(self.win, "Locked",
                    f"Index '{name}' is LOCKED.\nUnlock it first.")
                continue
            confirm = ask_copyable_yesno(self.win,
                "Delete Index",
                f"PERMANENTLY delete index '{name}'?\n\nCANNOT BE UNDONE.")
            if not confirm:
                continue
            try:
                delete_index(name, force=True)
                self.status_var.set(f"Deleted index '{name}'")
                if name in self.checked_indexes:
                    self.checked_indexes.remove(name)
            except Exception as e:
                show_copyable_error(self.win, "Error", str(e))
        self._refresh_indexes()
        self._notify_change()

    def _rename_index(self):
        from tkinter import simpledialog
        idx = self._selected_index()
        if not idx:
            return
        if idx['locked']:
            show_copyable_warning(self.win, "Locked",
                f"Index '{idx['name']}' is LOCKED.\nUnlock it first.")
            return
        new_name = simpledialog.askstring("Rename Index",
            f"Rename '{idx['name']}' to:",
            initialvalue=idx['name'], parent=self.win)
        if not new_name or new_name.strip() == idx['name']:
            return
        new_name = new_name.strip()
        if not re.match(r'^[A-Za-z0-9_-]+$', new_name):
            show_copyable_error(self.win, "Invalid Name",
                "Use letters, digits, underscores or hyphens only.")
            return
        existing = {ix['name'] for ix in self._indexes}
        if new_name in existing:
            show_copyable_error(self.win, "Name Exists",
                f"Index '{new_name}' already exists.")
            return
        try:
            old_dir = resolve_index_dir(idx['name'])
            new_dir = os.path.join(os.path.dirname(old_dir), new_name)
            os.rename(old_dir, new_dir)
            self.status_var.set(f"Renamed '{idx['name']}' -> '{new_name}'")
            self._refresh_indexes()
            self._notify_change()
        except Exception as e:
            show_copyable_error(self.win, "Error", str(e))

    def _toggle_lock(self):
        to_toggle = list(self.checked_indexes) if self.checked_indexes else []
        if not to_toggle:
            idx = self._selected_index()
            if idx:
                to_toggle = [idx['name']]
        if not to_toggle:
            return
        for name in to_toggle:
            lock_file = os.path.join(resolve_index_dir(name), ".locked")
            idx_info = next((i for i in self._indexes if i['name'] == name), {'locked': False})
            if idx_info.get('locked'):
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                self.status_var.set(f"Unlocked '{name}'")
            else:
                with open(lock_file, 'w') as f:
                    f.write("locked\n")
                self.status_var.set(f"Locked '{name}'")
            if name in self.checked_indexes:
                self.checked_indexes.remove(name)
        self._refresh_indexes()
        self._notify_change()

    def _hash_index(self):
        """Compute and save integrity hashes for the selected/checked indexes."""
        to_hash = list(self.checked_indexes) if self.checked_indexes else []
        if not to_hash:
            idx = self._selected_index()
            if idx:
                to_hash = [idx['name']]
        if not to_hash:
            return
        for name in to_hash:
            try:
                index_dir = resolve_index_dir(name)
                save_index_integrity(index_dir)
                self.status_var.set(f"Integrity hashes computed for '{name}'")
                if name in self.checked_indexes:
                    self.checked_indexes.remove(name)
            except Exception as e:
                show_copyable_error(self.win, "Error", f"For {name}: {e}")
        self._refresh_indexes()
        self._notify_change()

    def _remove_source(self):
        idx = self._selected_index()
        if not idx:
            return
        if idx['locked']:
            show_copyable_warning(self.win, "Locked",
                f"Index '{idx['name']}' is LOCKED.\nUnlock it first.")
            return
        sources_to_remove = list(self.checked_sources) if self.checked_sources else []
        if not sources_to_remove:
            if self.selected_source_idx is None:
                sel = self.source_list.curselection()
                if sel:
                    self.selected_source_idx = sel[0]
            if self.selected_source_idx is not None and self._sources:
                sources_to_remove = [self._sources[self.selected_source_idx][0]]
        if not sources_to_remove:
            return
        for src_name in sources_to_remove[:]:  # copy to modify during loop
            confirm = ask_copyable_yesno(self.win, "Remove Source",
                f"Remove '{src_name}' from '{idx['name']}'?\n\nCANNOT BE UNDONE.")
            if not confirm:
                continue
            try:
                result = remove_source_from_index(idx['name'], src_name)
                self.status_var.set(
                    f"Removed '{src_name}' ({result['removed_chunks']:,} chunks). "
                    f"{result['remaining_chunks']:,} remain.")
                if src_name in self.checked_sources:
                    self.checked_sources.remove(src_name)
            except Exception as e:
                show_copyable_error(self.win, "Error", str(e))
        self._refresh_indexes()
        # re-select the index if possible
        if self.selected_index_idx is not None and self.selected_index_idx < len(self._indexes):
            try:
                self.index_list.selection_set(self.selected_index_idx)
                self._on_index_select()
            except:
                pass
        self._notify_change()

    def _export_source(self):
        """Export the selected/checked source(s) to a directory."""
        from tkinter import filedialog
        idx = self._selected_index()
        if not idx:
            self.status_var.set("Select an index first")
            return

        sources_to_export = list(self.checked_sources) if self.checked_sources else []
        if not sources_to_export:
            if self.selected_source_idx is None:
                src_sel = self.source_list.curselection()
                if src_sel:
                    self.selected_source_idx = src_sel[0]
            if self.selected_source_idx is not None and self._sources:
                sources_to_export = [self._sources[self.selected_source_idx][0]]

        if not sources_to_export:
            self.status_var.set("Select a source to export")
            return

        output_dir = filedialog.askdirectory(
            title=f"Export source(s) from {idx['name']} to...",
            initialdir=os.path.expanduser(f"~/rag-export/{idx['name']}"),
            parent=self.win)
        if not output_dir:
            return

        for src_name in sources_to_export:
            try:
                result = export_source(idx['name'], src_name, output_dir)
                self.status_var.set(
                    f"Exported '{src_name}' ({result.get('chunks_exported', '?')} chunks) -> {output_dir}")
            except Exception as e:
                show_copyable_error(self.win, "Export Error", f"{src_name}: {e}")
            if src_name in self.checked_sources:
                self.checked_sources.remove(src_name)

    def _export_all(self):
        """Export all sources from the selected/checked indexes."""
        from tkinter import filedialog
        to_export = list(self.checked_indexes) if self.checked_indexes else []
        if not to_export:
            idx = self._selected_index()
            if idx:
                to_export = [idx['name']]
        if not to_export:
            self.status_var.set("Select an index first")
            return

        output_dir = filedialog.askdirectory(
            title=f"Export all sources from selected indexes to...",
            initialdir=os.path.expanduser("~/rag-export/"),
            parent=self.win)
        if not output_dir:
            return

        total_exported = 0
        for name in to_export:
            try:
                result = export_index(name, output_dir)
                total_exported += result.get('sources_exported', 0)
                if result.get('skipped'):
                    self.status_var.set(f"Some skipped for {name}")
            except Exception as e:
                show_copyable_error(self.win, "Export Error", f"For {name}: {e}")
        self.status_var.set(f"Exported from {len(to_export)} indexes, {total_exported} sources total -> {output_dir}")

    def _read_source(self, event=None):
        """Load and display full source text in a read-only viewer."""
        if self.selected_index_idx is None:
            self.status_var.set("Select an index first")
            return
        idx = self._indexes[self.selected_index_idx]
        
        sel = self.source_list.curselection()
        if sel:
            selected_idx = sel[0]
        elif self.selected_source_idx is not None:
            selected_idx = self.selected_source_idx
        else:
            self.status_var.set("Select a source to read")
            return
            
        src_name = self._sources[selected_idx][0]
        self.status_var.set(f"Loading '{src_name}'...")
        self.win.update_idletasks()
        
        try:
            index_dir = resolve_index_dir(idx['name'])
            backend_type = detect_backend(index_dir)
            backend = get_backend(index_dir, backend_type)
            doc = backend.get_document(src_name)
            if doc and doc.get('full_text'):
                text = doc['full_text']
            elif hasattr(backend, 'get_document_chunks'):
                chunks = backend.get_document_chunks(src_name) or []
                if chunks:
                    text = '\n\n'.join(
                        c['text'] for c in sorted(chunks, key=lambda c: c.get('chunk', 0))
                    )
                else:
                    text = "No text available for this source."
            else:
                text = "No text available for this source."
                
            self.status_var.set("Ready")
            edit_index_name = idx['name'] if not is_index_locked(idx['name']) else None
            SourceViewerDialog(
                self.win,
                source_name=src_name,
                text=text,
                index_name=edit_index_name,
                on_save=lambda: (self._on_index_select(), self._notify_change())
            )
        except Exception as e:
            self.status_var.set("Load failed")
            show_copyable_error(self.win, "Load Error", f"Cannot load document: {e}")
