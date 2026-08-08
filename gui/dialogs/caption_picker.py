"""
gui/dialogs/caption_picker.py — dialog for choosing subtitle source.

Small class.
"""

import tkinter as tk
from tkinter import ttk


class CaptionPickerDialog:
    def __init__(self, parent, path, candidates):
        self.result = {'kind': 'transcription'}
        self.win = tk.Toplevel(parent)
        self.win.title("Caption Source")
        ttk.Label(self.win, text=f"For {path}").pack()
        self.list = tk.Listbox(self.win)
        self.list.pack()
        for c in candidates:
            self.list.insert(tk.END, c.get('label', str(c)))
        ttk.Button(self.win, text="Use", command=self._choose).pack()
        ttk.Button(self.win, text="Transcription", command=self._trans).pack()
        self.win.wait_window()

    def _choose(self):
        self.result = {'kind': 'embedded'}
        self.win.destroy()

    def _trans(self):
        self.result = {'kind': 'transcription'}
        self.win.destroy()
