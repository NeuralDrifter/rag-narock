"""
gui/dialogs/settings_dialog.py — SettingsDialog for Tk GUI.
Settings dialog implementation. Uses config.settings for load/save.
"""

import tkinter as tk
from tkinter import ttk
import gui.theme as theme
import config.settings as settings_mod
from gui.theme import show_copyable_info, show_copyable_error

class SettingsDialog:
    """Modal settings dialog. Edits main config keys and can launch ACL."""

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("RAG Settings")
        self.win.geometry("620x520") # Slightly wider/taller for options readability
        self.win.minsize(550, 450)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.configure(bg=theme.BG)

        self.cfg = settings_mod.load()

        # Notebook for tabs
        nb = ttk.Notebook(self.win)
        nb.pack(fill='both', expand=True, padx=12, pady=(12, 6))

        self.vars = {}
        self.choice_mappings = {}

        # Dynamically build tabs and fields from config TABS schema
        for tab_name, items in settings_mod.TABS:
            # Use tk.Frame with theme.BG to completely eliminate grey background/edges
            tab_frame = tk.Frame(nb, bg=theme.BG, padx=15, pady=15)
            nb.add(tab_frame, text=tab_name)

            # Configure column weights for resizing
            tab_frame.columnconfigure(0, weight=1)
            tab_frame.columnconfigure(1, weight=1)

            row = 0
            for key, label_text, typ, options, default in items:
                # Add label
                lbl = tk.Label(
                    tab_frame, text=label_text, bg=theme.BG, fg=theme.FG,
                    font=('sans-serif', 10), anchor='w'
                )
                lbl.grid(row=row, column=0, sticky='w', padx=6, pady=5)

                current_value = self.cfg.get(key, default)

                if typ == 'choice' and options:
                    # Choice field -> dropdown
                    display_labels = [opt[1] for opt in options]
                    val_to_label = {opt[0]: opt[1] for opt in options}
                    label_to_val = {opt[1]: opt[0] for opt in options}
                    self.choice_mappings[key] = label_to_val

                    # Numeric options allow manual input/typing, others are readonly dropdowns
                    is_numeric = all(isinstance(opt[0], (int, float)) or (isinstance(opt[0], str) and opt[0].isdigit()) for opt in options)
                    state_mode = 'normal' if is_numeric else 'readonly'

                    cb = ttk.Combobox(tab_frame, values=display_labels, state=state_mode, width=28)
                    initial_label = val_to_label.get(current_value, str(current_value))
                    cb.set(initial_label)
                    cb.grid(row=row, column=1, sticky='ew', padx=6, pady=5)
                    self.vars[key] = (cb, typ)

                elif typ == 'toggle' or typ == 'bool':
                    # Toggle field -> Checkbutton
                    var = tk.BooleanVar(value=bool(current_value))
                    chk = ttk.Checkbutton(tab_frame, variable=var)
                    chk.grid(row=row, column=1, sticky='w', padx=6, pady=5)
                    self.vars[key] = (var, typ)

                else:
                    # Text/Int/Str field -> Entry
                    var = tk.StringVar(value=str(current_value))
                    ent = ttk.Entry(tab_frame, textvariable=var, width=30)
                    ent.grid(row=row, column=1, sticky='ew', padx=6, pady=5)
                    self.vars[key] = (var, typ)

                row += 1

        # Bottom buttons panel
        btns = tk.Frame(self.win, bg=theme.BG)
        btns.pack(fill='x', side='bottom', pady=12, padx=12)

        # Theme-matched buttons
        tk.Button(
            btns, text="Save Settings", command=self._save,
            bg=theme.ACCENT, fg='#ffffff', activebackground=theme.ACCENT2,
            activeforeground='#ffffff', relief='flat', padx=15, pady=5,
            font=('sans-serif', 10, 'bold')
        ).pack(side='left', padx=6)

        tk.Button(
            btns, text="Cancel", command=self.win.destroy,
            bg=theme.BG3, fg=theme.FG, activebackground=theme.BG2,
            activeforeground=theme.FG, relief='flat', padx=15, pady=5,
            font=('sans-serif', 10)
        ).pack(side='left', padx=6)

        tk.Button(
            btns, text="Open ACL Manager", command=self._open_acl,
            bg=theme.BG3, fg=theme.FG, activebackground=theme.BG2,
            activeforeground=theme.FG, relief='flat', padx=15, pady=5,
            font=('sans-serif', 10)
        ).pack(side='right', padx=6)

        # Center on parent
        self.win.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.win.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.win.winfo_height()) // 2
        self.win.geometry(f"+{max(0,px)}+{max(0,py)}")

        self.win.wait_window()

    def _save(self):
        for key, (widget, typ) in self.vars.items():
            if typ == 'choice':
                val = widget.get()
                mapping = self.choice_mappings.get(key, {})
                raw_val = mapping.get(val, val)

                # Attempt conversion to numeric if standard default is int/float
                default = settings_mod.DEFAULTS.get(key)
                if isinstance(default, int):
                    try:
                        raw_val = int(raw_val)
                    except:
                        pass
                elif isinstance(default, float):
                    try:
                        raw_val = float(raw_val)
                    except:
                        pass
                self.cfg[key] = raw_val

            elif typ in ('toggle', 'bool'):
                self.cfg[key] = bool(widget.get())

            else:
                val = widget.get()
                default = settings_mod.DEFAULTS.get(key)
                if isinstance(default, int):
                    try:
                        val = int(val)
                    except:
                        pass
                elif isinstance(default, bool):
                    val = val.lower() == 'true'
                self.cfg[key] = val

        settings_mod.save(self.cfg)
        show_copyable_info(self.win, "Saved", "Settings saved successfully.")
        self.win.destroy()

    def _open_acl(self):
        try:
            from gui.dialogs.acl_dialog import ACLDialog
            ACLDialog(self.win)
        except Exception as e:
            show_copyable_error(self.win, "ACL Error", f"Could not open ACL: {e}")
