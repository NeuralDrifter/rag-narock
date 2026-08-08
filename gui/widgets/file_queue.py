"""
gui/widgets/file_queue.py — dedicated widget for managing the list of files to index.

Clean Code:
- One reason to change (Ch 10 SRP)
- Small methods
- Clear interface
"""

import tkinter as tk
from tkinter import ttk, filedialog
import os


class FileQueueWidget(ttk.Frame):
    """Encapsulates the file queue listbox + add/remove buttons."""

    def __init__(self, parent, supported_exts, log_fn):
        super().__init__(parent)
        self.supported_exts = supported_exts
        self.log = log_fn
        self.files = []  # internal list

        self._build()

    def _build(self):
        self.listbox = tk.Listbox(self, height=8)
        self.listbox.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(self, command=self.listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.listbox.configure(yscrollcommand=scrollbar.set)

        btns = ttk.Frame(self)
        btns.pack(side='left', fill='y', padx=4)

        ttk.Button(btns, text="Add Folder", command=self.add_folder).pack(fill='x')
        ttk.Button(btns, text="Add Files", command=self.add_files).pack(fill='x')
        ttk.Button(btns, text="Remove", command=self.remove_selected).pack(fill='x', pady=4)
        ttk.Button(btns, text="Clear", command=self.clear).pack(fill='x')

    def add_folder(self):
        path = filedialog.askdirectory()
        if not path:
            return
        for root, _, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                if ext in self.supported_exts and full not in self.files:
                    self.files.append(full)
                    self.listbox.insert(tk.END, full)

    def add_files(self):
        paths = tk.filedialog.askopenfilenames() if hasattr(tk, 'filedialog') else []
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert(tk.END, p)

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            del self.files[i]
            self.listbox.delete(i)

    def clear(self):
        self.files.clear()
        self.listbox.delete(0, tk.END)

    def get_files(self):
        return list(self.files)
