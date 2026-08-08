"""
gui/theme.py — Dark Norse theme (constants + application).

Clean Code: centralized styling, single source of truth for the look.
"""
import logging
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)

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
WARN_RED = '#e74c3c'


def apply_dark_norse_theme(root):
    """Apply the full Dark Norse ttk styling to the root window.
    Call this early in window __init__.
    """
    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use('clam')

    # Global defaults
    style.configure('.', background=BG, foreground=FG, fieldbackground=BG3,
                    bordercolor=BG3, darkcolor=BG, lightcolor=BG2,
                    troughcolor=BG2, selectbackground=ACCENT, selectforeground='#ffffff',
                    font=('sans-serif', 10))

    # Frames
    style.configure('TFrame', background=BG)
    style.configure('TLabelframe', background=BG, foreground=ACCENT,
                    bordercolor='#2a3a52')
    style.configure('TLabelframe.Label', background=BG, foreground=ACCENT,
                    font=('sans-serif', 10, 'bold'))

    # Labels
    style.configure('TLabel', background=BG, foreground=FG)
    style.configure('Info.TLabel', foreground=ICE)
    style.configure('Count.TLabel', foreground=FG_DIM)
    style.configure('Title.TLabel', foreground=ACCENT, font=('sans-serif', 11, 'bold'))

    # Buttons — fire orange
    style.configure('TButton', background=ACCENT, foreground='#ffffff',
                    bordercolor=ACCENT, font=('sans-serif', 9, 'bold'), padding=(8, 4))
    style.map('TButton',
              background=[('active', ACCENT2), ('disabled', BG3)],
              foreground=[('disabled', FG_DIM)])

    # Primary action button
    style.configure('Primary.TButton', background='#c0581e', foreground='#ffffff',
                    font=('sans-serif', 10, 'bold'), padding=(12, 5))
    style.map('Primary.TButton',
              background=[('active', ACCENT), ('disabled', BG3)])

    # Entry / Combobox
    style.configure('TEntry', fieldbackground=BG3, foreground=FG,
                    insertcolor=FG, bordercolor='#2a3a52')
    style.configure('TCombobox', fieldbackground=BG3, foreground=FG,
                    arrowcolor=ACCENT, bordercolor='#2a3a52')
    style.map('TCombobox', fieldbackground=[('readonly', BG3)],
              selectbackground=[('readonly', ACCENT)])
    root.option_add('*TCombobox*Listbox.background', BG3)
    root.option_add('*TCombobox*Listbox.foreground', FG)
    root.option_add('*TCombobox*Listbox.selectBackground', ACCENT)
    root.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')

    # Checkbutton
    style.configure('TCheckbutton', background=BG, foreground=FG,
                    indicatorcolor=BG3)
    style.map('TCheckbutton',
              indicatorcolor=[('selected', ACCENT), ('!selected', BG3)],
              background=[('active', BG2)])

    # Notebook / Tabs style
    style.configure('TNotebook', background=BG, bordercolor=BG3, darkcolor=BG, lightcolor=BG2)
    style.configure('TNotebook.Tab', background=BG3, foreground=FG, bordercolor=BG3,
                    padding=(12, 4), font=('sans-serif', 10, 'bold'))
    style.map('TNotebook.Tab',
              background=[('selected', ACCENT), ('active', BG2)],
              foreground=[('selected', '#ffffff'), ('active', FG)])

    # Progressbar
    style.configure('TProgressbar', background=ACCENT, troughcolor=BG3,
                    bordercolor=BG3, darkcolor=ACCENT, lightcolor=ACCENT2)

    # Scrollbar
    style.configure('TScrollbar', background=ACCENT, troughcolor=BG,
                    arrowcolor=FG, bordercolor=BG, lightcolor=ACCENT2, darkcolor=ACCENT)
    style.map('TScrollbar', background=[('active', ACCENT2)])


# --- Copyable popup helpers ---
# All error/info/warning/yesno popups should use these so the text
# can be selected and copied (very useful for bug reports).

def _show_copyable_popup(parent, title, message, ask=False):
    """Show a themed, copyable dialog.
    The message is in a Text widget (selectable + copyable).
    """
    if parent is None:
        try:
            parent = tk._default_root
        except Exception as e:
            logger.debug("Default root access failed: %s", e)
            parent = None
    if parent is None:
        parent = tk.Tk()
        parent.withdraw()

    win = tk.Toplevel(parent)
    win.title(title)
    win.resizable(True, False)
    win.transient(parent)
    win.grab_set()

    try:
        win.configure(bg=BG)
    except Exception as e:
        logger.debug("Win configure bg failed: %s", e)

    # Copyable message area
    msg_frame = tk.Frame(win, bg=BG, padx=10, pady=10)
    msg_frame.pack(fill='both', expand=True)

    txt = tk.Text(msg_frame, wrap=tk.WORD, height=8, width=55,
                  bg=BG3, fg=FG,
                  selectbackground=ACCENT, selectforeground='#ffffff',
                  relief='solid', borderwidth=1, padx=6, pady=4,
                  font=('sans-serif', 10))
    txt.insert('1.0', message)
    txt.configure(state='disabled')  # still fully selectable for copy
    txt.pack(fill='both', expand=True)

    # Buttons
    btn_frame = tk.Frame(win, bg=BG)
    btn_frame.pack(fill='x', padx=10, pady=(0, 10))

    def do_copy():
        try:
            win.clipboard_clear()
            win.clipboard_append(message)
        except Exception as e:
            logger.debug("Clipboard copy failed: %s", e)

    ttk.Button(btn_frame, text="Copy", command=do_copy).pack(side='left', padx=4)

    if ask:
        result = [False]
        def yes():
            result[0] = True
            win.destroy()
        def no():
            result[0] = False
            win.destroy()
        ttk.Button(btn_frame, text="Yes", command=yes).pack(side='right', padx=4)
        ttk.Button(btn_frame, text="No", command=no).pack(side='right', padx=4)
        win.wait_window()
        return result[0]
    else:
        ttk.Button(btn_frame, text="OK", command=win.destroy).pack(side='right', padx=4)
        win.wait_window()
        return None


def show_copyable_info(parent, title, message):
    return _show_copyable_popup(parent, title, message)

def show_copyable_warning(parent, title, message):
    return _show_copyable_popup(parent, title, message)

def show_copyable_error(parent, title, message):
    return _show_copyable_popup(parent, title, message)

def ask_copyable_yesno(parent, title, message):
    return _show_copyable_popup(parent, title, message, ask=True)
