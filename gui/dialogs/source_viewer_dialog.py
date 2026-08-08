"""
gui/dialogs/source_viewer_dialog.py — Scrollable themed viewer and editor for document text.
"""

import tkinter as tk
from tkinter import ttk
import gui.theme as theme
from core.index_manager import update_source_in_index

class SourceViewerDialog:
    """Themed modal text viewer and editor dialog for documents within the RAG editor."""

    def __init__(self, parent, source_name, text, index_name=None, on_save=None):
        self.parent = parent
        self.index_name = index_name
        self.source_name = source_name
        self.original_text = text
        self.on_save_callback = on_save

        self.win = tk.Toplevel(parent)
        self.win.title(f"Viewer: {source_name}")
        self.win.geometry("750x550")
        self.win.minsize(400, 300)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.configure(bg=theme.BG)

        # Header Frame
        header = tk.Frame(self.win, bg=theme.BG)
        header.pack(fill='x', padx=15, pady=(15, 5))
        
        tk.Label(header, text=source_name, bg=theme.BG, fg=theme.ACCENT,
                 font=('sans-serif', 12, 'bold')).pack(side='left')

        # Text container frame
        txt_frame = tk.Frame(self.win, bg=theme.BG3, bd=1, relief='solid')
        txt_frame.pack(fill='both', expand=True, padx=15, pady=5)

        # Scrollbar and text widget
        scrollbar = ttk.Scrollbar(txt_frame)
        scrollbar.pack(side='right', fill='y')

        # Text widget
        self.text_widget = tk.Text(
            txt_frame, bg=theme.BG, fg=theme.FG, insertbackground=theme.FG,
            yscrollcommand=scrollbar.set, relief='flat', wrap='word',
            font=('monospace', 10), padx=8, pady=8
        )
        self.text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.text_widget.yview)

        # Populate and disable editing initially
        self.text_widget.insert('1.0', text)
        self.text_widget.config(state='disabled')

        # Bottom buttons
        self.btn_frame = tk.Frame(self.win, bg=theme.BG)
        self.btn_frame.pack(fill='x', padx=15, pady=15)

        # Status Label
        self.status_label = tk.Label(
            self.btn_frame, text="", bg=theme.BG, fg=theme.FG_DIM,
            font=('sans-serif', 10)
        )
        self.status_label.pack(side='left')

        # Standard close button
        self.close_btn = tk.Button(
            self.btn_frame, text="Close", command=self.win.destroy,
            bg=theme.BG3, fg=theme.FG, activebackground=theme.BG2,
            activeforeground=theme.FG, relief='flat', padx=12, pady=4
        )
        self.close_btn.pack(side='right')

        # Edit button (only if index_name is provided to allow editing)
        self.edit_btn = None
        if self.index_name:
            self.edit_btn = tk.Button(
                self.btn_frame, text="Edit Source", command=self._toggle_edit,
                bg=theme.ACCENT, fg='#ffffff', activebackground=theme.ACCENT2,
                activeforeground='#ffffff', relief='flat', padx=12, pady=4
            )
            self.edit_btn.pack(side='right', padx=(0, 10))

        # Edit controls (initially hidden)
        self.save_btn = tk.Button(
            self.btn_frame, text="Save Changes", command=self._save_edit,
            bg=theme.ACCENT, fg='#ffffff', activebackground=theme.ACCENT2,
            activeforeground='#ffffff', relief='flat', padx=12, pady=4
        )
        self.cancel_btn = tk.Button(
            self.btn_frame, text="Cancel", command=self._cancel_edit,
            bg=theme.BG3, fg=theme.FG, activebackground=theme.BG2,
            activeforeground=theme.FG, relief='flat', padx=12, pady=4
        )

        # Center on parent
        self.win.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.win.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.win.winfo_height()) // 2
        self.win.geometry(f"+{max(0,px)}+{max(0,py)}")

        self.win.win = self.win # keep a reference
        self.win.wait_window()

    def _toggle_edit(self):
        self.text_widget.config(state='normal')
        if self.edit_btn:
            self.edit_btn.pack_forget()
        self.close_btn.pack_forget()
        self.cancel_btn.pack(side='right')
        self.save_btn.pack(side='right', padx=(0, 10))
        self.text_widget.focus_set()

    def _cancel_edit(self):
        self.text_widget.config(state='normal')
        self.text_widget.delete('1.0', 'end')
        self.text_widget.insert('1.0', self.original_text)
        self.text_widget.config(state='disabled')
        self.save_btn.pack_forget()
        self.cancel_btn.pack_forget()
        self.close_btn.pack(side='right')
        if self.edit_btn:
            self.edit_btn.pack(side='right', padx=(0, 10))
        self.status_label.config(text="")

    def _save_edit(self):
        new_text = self.text_widget.get('1.0', 'end-1c')
        if new_text == self.original_text:
            self._cancel_edit()
            return

        if not theme.ask_copyable_yesno(self.win, "Confirm Save", "Are you sure you want to save changes?\n\nThis will remove the old vectors, re-chunk the text, and generate new vector embeddings. This might take a few moments."):
            return

        self.status_label.config(text="Saving and re-indexing...", fg=theme.ACCENT)
        self.save_btn.config(state='disabled')
        self.cancel_btn.config(state='disabled')
        self.text_widget.config(state='disabled')
        self.win.update()

        try:
            update_source_in_index(self.index_name, self.source_name, new_text)
            self.original_text = new_text
            self.status_label.config(text="Saved successfully!", fg=theme.GREEN)
            
            # Re-enable inputs
            self.save_btn.config(state='normal')
            self.cancel_btn.config(state='normal')
            
            # Revert UI state back to viewing
            self.win.after(1000, self._exit_edit_mode_after_save)
            
            # Fire callback
            if self.on_save_callback:
                self.on_save_callback()
        except Exception as e:
            self.status_label.config(text="Save failed!", fg=theme.FIRE_RED)
            theme.show_copyable_error(self.win, "Save Error", f"Failed to save and re-index document: {e}")
            self.save_btn.config(state='normal')
            self.cancel_btn.config(state='normal')
            self.text_widget.config(state='normal')

    def _exit_edit_mode_after_save(self):
        self.status_label.config(text="")
        self._cancel_edit()
