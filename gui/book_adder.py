"""
gui/book_adder.py — Modular GUI for adding documents to indexes.

Follows Clean Code:
- Small, focused classes (SRP, Ch 10)
- Composition over huge functions (Ch 3)
- Controller separates UI from domain logic (Ch 11, boundaries)
- No god classes

Currently delegates to functions in the main module for ingestion.
Future: will use ingestion/ and indexing/ packages directly.
"""

import os
import threading
import logging
import tkinter as tk
from tkinter import ttk, filedialog
from gui.theme import show_copyable_info, show_copyable_error, show_copyable_warning

# Theme centralized
from . import theme as gui_theme

# Modular imports (now possible after restoring full backends etc.)
from core.constants import META_FILENAME
from core.hashing import file_hash
from core.index_manager import get_indexes
import config.settings as rag_settings
from storage import get_backend, get_backend_status, get_missing_backend_warnings
from ingestion.chunking import chunk_text
from ingestion.documents import detect_document_type
from ingestion.ocr import get_ocr_backend, ocr_backend_available, ocr_unavailable_message
from ingestion.ocr_options import OcrOptions
from ingestion.registry import extract_file
from indexing.embedder import (
    embed_texts,
    _resolve_embedding_device,
    _resolve_embedding_model_name,
    _unload_embedding_model,
)
import storage as backends  # for detect etc. if needed

logger = logging.getLogger(__name__)

OCR_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.webp',
    '.tiff', '.tif', '.djvu', '.djv', '.pbm',
    '.pgm', '.ppm', '.pnm', '.pdf', '.epub',
}


def _build_gui_ocr_options(opts: dict) -> OcrOptions:
    """Mirror CLI OCR option resolution for GUI checkboxes + settings."""
    no_ocr = opts.get('no_ocr')
    if no_ocr is None:
        no_ocr = rag_settings.get('disable_ocr')
    force_ocr = opts.get('force_ocr')
    if force_ocr is None:
        force_ocr = rag_settings.get('force_ocr')
    if no_ocr:
        force_ocr = False
    return OcrOptions(
        disabled=bool(no_ocr),
        force=bool(force_ocr),
        backend=None,
        language=rag_settings.get('ocr_lang') or 'eng',
        negative=bool(rag_settings.get('ocr_negative')),
        split_spreads=bool(rag_settings.get('split_spreads')),
    )


def _format_ocr_status(ocr_opts: OcrOptions) -> str:
    if ocr_opts.disabled:
        return "OCR disabled"
    parts = []
    if ocr_opts.force:
        parts.append("force-OCR")
    backend = get_ocr_backend(ocr_opts)
    if backend:
        parts.append(backend)
    return ", ".join(parts) if parts else "auto (text layer, OCR fallback)"


class BookAdderController:
    """Handles the non-UI logic for indexing (Ch 11: separate construction from use)."""

    def __init__(self, log_callback, progress_callback, shutdown_event):
        self.log = log_callback
        self.set_progress = progress_callback
        self.shutdown = shutdown_event
        self.indexing = False

    def do_index(self, name, file_queue, ocr_options, gui_vars):
        """Full indexing using restored modular components (ingestion + indexing + storage)."""
        if self.indexing or not file_queue:
            self.log("Nothing to index.")
            return

        self.indexing = True
        self.set_progress(0)

        if name == '(new index...)':
            name = gui_vars.get('new_name', '').strip() or 'new-index'

        index_dir = os.path.join(rag_settings.get_data_dir(), name)
        os.makedirs(index_dir, exist_ok=True)

        try:
            from core.hashing import save_index_hashes
            from core.integrity import save_index_integrity
            from storage.base import get_index_meta_with_defaults
            import json

            ocr_opts = _build_gui_ocr_options(ocr_options)
            self.log(f"OCR: {_format_ocr_status(ocr_opts)}")

            total = len(file_queue)
            all_chunks = []
            all_documents = []
            file_hashes_map = {}

            # 1. Extract + chunk
            for i, fpath in enumerate(file_queue):
                if self.shutdown.is_set():
                    break
                fname = os.path.basename(fpath)
                self.log(f"[{i+1}/{total}] {fname}")
                self.set_progress((i / total) * 50)

                text, is_ocr = extract_file(
                    fpath,
                    force_ocr=ocr_opts.force,
                    ocr_options=ocr_opts,
                )
                if not text:
                    self.log("  (skipped empty)")
                    continue

                try:
                    doc_type = detect_document_type(fpath)
                except Exception:
                    doc_type = 'book'

                h = file_hash(fpath)
                file_hashes_map[h] = fname

                chunks = chunk_text(text)
                for ci, chunk in enumerate(chunks):
                    all_chunks.append({
                        'text': chunk,
                        'source': fname,
                        'chunk': ci,
                        'of': len(chunks),
                        'ocr': is_ocr,
                    })

                all_documents.append({
                    'source': fname,
                    'full_text': text,
                    'doc_type': doc_type,
                    'language': None,
                    'ocr': is_ocr,
                })

                ocr_tag = " [OCR]" if is_ocr else ""
                self.log(f"  extracted, {len(chunks)} chunks{ocr_tag}")

            if not all_chunks:
                self.log("No content to index.")
                return

            self.set_progress(55)
            self.log(f"Total chunks: {len(all_chunks)}")

            # 2. Embed
            use_gpu = ocr_options.get('gpu')
            if use_gpu is None:
                use_gpu = rag_settings.get('gpu_indexing')
            device = _resolve_embedding_device(use_gpu)
            self.log(f"Embedding ({device.upper()})...")
            embeddings = embed_texts([c['text'] for c in all_chunks], device=device)
            _unload_embedding_model()
            self.set_progress(70)

            # 3. Save to backend
            storage_type = gui_vars.get('storage_backend') or rag_settings.get('storage_backend')
            backend = get_backend(index_dir, storage_type)
            docs_arg = all_documents if storage_type == 'sqlite-doc' else None

            if hasattr(backend, 'save') and not backend.exists():
                if docs_arg is not None:
                    backend.save(all_chunks, embeddings, file_hashes_map, documents=docs_arg)
                else:
                    backend.save(all_chunks, embeddings, file_hashes_map)
            else:
                if docs_arg is not None:
                    backend.append(all_chunks, embeddings, file_hashes_map, documents=docs_arg)
                else:
                    backend.append(all_chunks, embeddings, file_hashes_map)

            self.set_progress(85)

            # 4. Meta, hashes, integrity
            emb_backend, emb_model = _resolve_embedding_model_name()
            meta = get_index_meta_with_defaults(index_dir)
            meta.update({
                'n_chunks': len(all_chunks),
                'n_files': len(file_hashes_map),
                'storage_backend': storage_type,
                'embedding_backend': emb_backend,
                'embedding_model': emb_model,
                'chunk_size': rag_settings.get('chunk_size'),
                'overlap': rag_settings.get('overlap'),
            })
            with open(os.path.join(index_dir, META_FILENAME), 'w') as f:
                json.dump(meta, f, indent=2)

            save_index_hashes(index_dir, file_hashes_map)
            save_index_integrity(index_dir)

            self.set_progress(95)
            self.log(f"Indexed into '{name}' ({storage_type}) - {len(all_chunks)} chunks from {len(file_hashes_map)} files.")
            self.log("Done.")

        except Exception as e:
            self.log(f"ERROR during indexing: {e}")
            import traceback
            self.log(traceback.format_exc()[-500:])
        finally:
            self.indexing = False
            self.set_progress(100)


class BookAdderWindow(tk.Tk):
    """Main window. Assembles small widgets (Ch 10: small classes)."""

    def log(self, msg):
        # default until overridden by log_panel
        try:
            print('[GUI log]', msg)
        except Exception as e:
            logger.debug("Log fallback failed: %s", e)

    def set_progress(self, val):
        try:
            print(f'[GUI progress] {int(val)}%')
        except Exception as e:
            logger.debug("Progress fallback failed: %s", e)

    def __init__(self):
        super().__init__()
        self.title("RAG-Narock")
        self.geometry("700x580")

        # Apply theme early (fixes "themes nonexistent")
        gui_theme.apply_dark_norse_theme(self)

        self.shutdown = threading.Event()
        self._index_thread = None

        self._build_ui()

        self.controller = BookAdderController(
            log_callback=self._thread_safe_log,
            progress_callback=self._thread_safe_progress,
            shutdown_event=self.shutdown,
        )

        # Add a simple menu bar (fixes "menus ... do not exist")
        self._add_menus()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(250, self._show_backend_dependency_warnings)

    def _build_ui(self):
        # Target Index
        idx_frame = ttk.LabelFrame(self, text="Target Index", padding=8)
        idx_frame.pack(fill='x', padx=10, pady=5)

        from .widgets.index_selector import IndexSelector
        self.index_sel = IndexSelector(idx_frame, lambda: get_indexes())
        self.index_sel.pack(side='left')

        # File Queue with its own buttons (modular)
        queue_frame = ttk.LabelFrame(self, text="File Queue", padding=8)
        queue_frame.pack(fill='both', expand=True, padx=10, pady=5)

        from .widgets.file_queue import FileQueueWidget
        from ingestion.registry import EXTRACTORS
        self.file_queue_widget = FileQueueWidget(
            queue_frame,
            supported_exts=tuple(EXTRACTORS.keys()),
            log_fn=self.log
        )
        self.file_queue_widget.pack(fill='both', expand=True)

        # Options bar (OCR, GPU, etc.)
        from .widgets.options_bar import OptionsBar
        self.options = OptionsBar(self)
        self.options.pack(fill='x', padx=10, pady=2)

        # Additional options to match original (OCR backend, lang, etc.)
        opts2 = ttk.Frame(self)
        opts2.pack(fill='x', padx=10)
        
        ttk.Label(opts2, text="OCR lang:").pack(side='left')
        self.ocr_lang_var = tk.StringVar(value=rag_settings.get('ocr_lang') or 'eng')
        self.ocr_lang_var.trace_add("write", lambda *a: rag_settings.set('ocr_lang', self.ocr_lang_var.get()))
        ttk.Entry(opts2, textvariable=self.ocr_lang_var, width=8).pack(side='left', padx=2)
        
        self.split_var = tk.BooleanVar(value=bool(rag_settings.get('split_spreads')))
        ttk.Checkbutton(
            opts2, text="Split Spreads", variable=self.split_var,
            command=lambda: rag_settings.set('split_spreads', self.split_var.get())
        ).pack(side='left')
        
        self.neg_var = tk.BooleanVar(value=bool(rag_settings.get('ocr_negative')))
        ttk.Checkbutton(
            opts2, text="Negative", variable=self.neg_var,
            command=lambda: rag_settings.set('ocr_negative', self.neg_var.get())
        ).pack(side='left')

        # Log panel + progress
        from .widgets.log_panel import LogPanel
        self.log_panel = LogPanel(self)
        self.log_panel.pack(fill='both', expand=True, padx=10, pady=5)

        # Action buttons area (restored from original for usability)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=10, pady=5)

        self.index_btn = ttk.Button(
            btn_frame, text="Add to Index", command=self.do_index, style='Primary.TButton',
        )
        self.index_btn.pack(side='right', padx=2)
        ttk.Button(btn_frame, text="Settings", command=self._open_settings).pack(side='right', padx=2)
        ttk.Button(btn_frame, text="Editor", command=self._open_editor).pack(side='right', padx=2)

        # Note: splash was in original; can be added in run_gui if wanted
        # For now, theme + menus + buttons are present and styled.

    def _show_backend_dependency_warnings(self):
        warnings = get_missing_backend_warnings()
        if not warnings:
            return
        message = (
            "The app can still run, but these storage backends are disabled:\n\n"
            + "\n".join(f"- {warning}" for warning in warnings)
            + "\n\nFix: install the missing package in the active Python environment.\n"
            "Bypass: choose an installed storage backend in Settings."
        )
        show_copyable_warning(self, "Optional Storage Backends Missing", message)

    def _thread_safe_log(self, msg):
        """Schedule log output on the Tk main thread."""
        self.after(0, lambda m=msg: self.log_panel.log(m))

    def _thread_safe_progress(self, val):
        """Schedule progress updates on the Tk main thread."""
        self.after(0, lambda v=val: self.log_panel.set_progress(v))

    def _set_indexing_ui_state(self, active: bool):
        state = 'disabled' if active else 'normal'
        if hasattr(self, 'index_btn'):
            self.index_btn.config(state=state)

    def _on_close(self):
        if self.controller.indexing:
            self.shutdown.set()
        self.destroy()

    def do_index(self):
        if self.controller.indexing:
            self.log_panel.log("Indexing already in progress.")
            return

        storage_backend = rag_settings.get('storage_backend') or 'faiss'
        backend_info = get_backend_status().get(storage_backend)
        if backend_info and not backend_info["installed"]:
            show_copyable_warning(
                self,
                "Storage Backend Unavailable",
                (
                    f"The selected storage backend '{storage_backend}' is not installed.\n\n"
                    f"Fix: python -m pip install {backend_info['package']}\n"
                    "Bypass: open Settings and choose an installed storage backend."
                ),
            )
            return

        name = self.index_sel.get_name()
        files = self.file_queue_widget.get_files()
        if not files:
            self.log_panel.log("Nothing to index — add files first.")
            return

        opts = self.options.get_options() if hasattr(self, 'options') else {}
        ocr_opts = _build_gui_ocr_options(opts)
        queue_may_need_ocr = any(os.path.splitext(path)[1].lower() in OCR_EXTENSIONS for path in files)
        if queue_may_need_ocr and not ocr_opts.disabled and not ocr_backend_available(ocr_opts):
            if ocr_opts.force:
                show_copyable_warning(
                    self,
                    "OCR Backend Unavailable",
                    ocr_unavailable_message(),
                )
                return
            opts = dict(opts)
            opts['no_ocr'] = True
            opts['force_ocr'] = False
            show_copyable_warning(
                self,
                "OCR Backend Unavailable",
                ocr_unavailable_message()
                + "\n\nThis indexing run will continue with OCR disabled.",
            )

        gui_vars = {'new_name': self.index_sel.new_var.get()}

        self.shutdown.clear()
        self._set_indexing_ui_state(True)

        def worker():
            try:
                self.controller.do_index(name, files, opts, gui_vars)
            finally:
                self.after(0, lambda: self._set_indexing_ui_state(False))

        self._index_thread = threading.Thread(target=worker, daemon=True)
        self._index_thread.start()

    def run(self):
        self.mainloop()

    def _add_menus(self):
        """Simple menu bar for the app."""
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Add Folder", command=self._add_folder_action)
        file_menu.add_command(label="Add Files", command=self._add_files_action)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: show_copyable_info(self, "RAG-Narock", "Modular GUI (refactored)"))
        menubar.add_cascade(label="Help", menu=help_menu)

        # Settings menu / Tools (addresses missing settings menu button)
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Settings", command=self._open_settings)
        tools_menu.add_command(label="Index Editor", command=self._open_editor)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        self.config(menu=menubar)

    def _add_folder_action(self):
        if hasattr(self.file_queue_widget, 'add_folder'):
            self.file_queue_widget.add_folder()

    def _add_files_action(self):
        if hasattr(self.file_queue_widget, 'add_files'):
            self.file_queue_widget.add_files()

    def _open_settings(self):
        """Open settings dialog. Uses the existing (shimmed) implementation."""
        try:
            from rag_settings import SettingsDialog
            SettingsDialog(self)
            self.sync_from_settings()
        except Exception as e:
            show_copyable_error(self, "Settings Error", f"Could not open settings: {e}")

    def sync_from_settings(self):
        """Sync all main window variables and widgets from settings."""
        if hasattr(self, 'options'):
            self.options.sync_from_settings()
        if hasattr(self, 'ocr_lang_var'):
            self.ocr_lang_var.set(rag_settings.get('ocr_lang') or 'eng')
        if hasattr(self, 'split_var'):
            self.split_var.set(bool(rag_settings.get('split_spreads')))
        if hasattr(self, 'neg_var'):
            self.neg_var.set(bool(rag_settings.get('ocr_negative')))

    def _open_editor(self):
        """Open index editor dialog."""
        try:
            from rag_editor import EditorDialog
            EditorDialog(self)
        except Exception as e:
            show_copyable_error(self, "Editor Error", f"Could not open editor: {e}\nTry: python rag.py editor gui")



def run_gui(args):
    """Thin entry to modular GUI."""
    # splash logic could be here or in __main__
    app = BookAdderWindow()
    app.run()
    return 0
