"""
gui/widgets/log_panel.py — scrollable log + progress.

Small widget.
"""

import tkinter as tk
from tkinter import ttk


class LogPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.text = tk.Text(self, height=6, wrap='word')
        self.text.pack(fill='both', expand=True)
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.pack(fill='x', pady=2)

    def log(self, msg):
        self.text.insert(tk.END, msg + '\n')
        self.text.see(tk.END)
        self.update_idletasks()

    def set_progress(self, val):
        self.progress.config(value=val)
