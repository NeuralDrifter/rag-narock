"""
gui/widgets/index_selector.py — widget for choosing or creating index.

Small, SRP class per clean code.
"""

import tkinter as tk
from tkinter import ttk


class IndexSelector(ttk.Frame):
    """Index dropdown + new name entry."""

    def __init__(self, parent, get_indexes_fn):
        super().__init__(parent)
        self.get_indexes = get_indexes_fn
        self.var = tk.StringVar(value='default')
        self.new_var = tk.StringVar()

        ttk.Label(self, text="Index:").pack(side='left')
        self.combo = ttk.Combobox(self, textvariable=self.var, width=18)
        self.combo.pack(side='left', padx=4)
        self._refresh()

        self.new_label = ttk.Label(self, text="New:")
        self.new_entry = ttk.Entry(self, textvariable=self.new_var, width=14)
        
        self.combo.bind("<<ComboboxSelected>>", self._on_select)
        self._on_select()

    def _on_select(self, event=None):
        if self.var.get() == '(new index...)':
            self.new_label.pack(side='left', padx=(6, 2))
            self.new_entry.pack(side='left', padx=2)
        else:
            self.new_label.pack_forget()
            self.new_entry.pack_forget()

    def _refresh(self):
        try:
            vals = self.get_indexes() + ['(new index...)']
            self.combo['values'] = vals
        except Exception:
            pass

    def get_name(self):
        if self.var.get() == '(new index...)':
            return self.new_var.get().strip() or 'new'
        return self.var.get()
