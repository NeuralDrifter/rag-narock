"""
gui/widgets/options_bar.py — OCR / GPU / options checkboxes, synced with settings.
"""

import tkinter as tk
from tkinter import ttk
import config.settings as settings


class OptionsBar(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.force_ocr = tk.BooleanVar()
        self.no_ocr = tk.BooleanVar()
        self.gpu = tk.BooleanVar()
        
        # Load initial values
        self.sync_from_settings()

        # Pack checkbuttons and bind commands to save to global settings on toggle
        ttk.Checkbutton(
            self, text="Force OCR", variable=self.force_ocr,
            command=lambda: settings.set('force_ocr', self.force_ocr.get())
        ).pack(side='left', padx=10)
        
        ttk.Checkbutton(
            self, text="No OCR", variable=self.no_ocr,
            command=lambda: settings.set('disable_ocr', self.no_ocr.get())
        ).pack(side='left', padx=10)
        
        ttk.Checkbutton(
            self, text="GPU", variable=self.gpu,
            command=lambda: settings.set('gpu_indexing', self.gpu.get())
        ).pack(side='left', padx=10)

    def sync_from_settings(self):
        self.force_ocr.set(bool(settings.get('force_ocr')))
        self.no_ocr.set(bool(settings.get('disable_ocr')))
        self.gpu.set(bool(settings.get('gpu_indexing')))

    def get_options(self):
        return {
            'force_ocr': self.force_ocr.get(),
            'no_ocr': self.no_ocr.get(),
            'gpu': self.gpu.get()
        }
