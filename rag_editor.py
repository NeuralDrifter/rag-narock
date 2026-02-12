#!/usr/bin/env python3
"""
RAG-Narock Index Editor — TUI (curses) and GUI (tkinter) interfaces
for managing indexes: view sources, remove sources, delete indexes,
lock/unlock indexes.
"""

import os, sys, json

RAG_HOME = os.path.expanduser("~/.local/share/rag")

# Lazy import from rag.py to avoid circular imports
def _rag():
    import rag
    return rag

def _backends():
    import rag_backends
    return rag_backends

# Flame banner (shared with SettingsTUI)
from rag_settings import BANNER, BANNER_H, BANNER_W


# ── Curses TUI ──────────────────────────────────────────────────────────────

class EditorTUI:
    """Full-screen curses index editor with two-mode navigation."""

    MODE_INDEX = 0
    MODE_SOURCE = 1

    def __init__(self):
        self.mode = self.MODE_INDEX
        self.cursor = 0
        self.scroll_offset = 0
        self.indexes = []
        self.sources = []  # list of (source_name, chunk_count)
        self.selected_index = None
        self.status_msg = ""
        self.status_color = 6  # yellow
        self.has_colors = False
        self.has_flame = False
        self.confirming = False
        self.confirm_cursor = 1  # 0=Yes, 1=No (default No)
        self.confirm_action = None
        self.confirm_msg_lines = []
        self._refresh_indexes()

    def _refresh_indexes(self):
        rag = _rag()
        backends = _backends()
        self.indexes = []
        for name in rag.get_indexes():
            index_dir = os.path.join(RAG_HOME, name)
            meta = backends.get_index_meta_with_defaults(index_dir)
            locked = rag.is_index_locked(name)
            n_chunks = meta.get('n_chunks', 0)
            n_files = meta.get('n_files', 0)
            storage = meta.get('storage_backend', 'faiss')
            emb_model = meta.get('embedding_model', '?')
            emb_backend = meta.get('embedding_backend', 'local')
            self.indexes.append({
                'name': name, 'n_chunks': n_chunks,
                'n_files': n_files, 'locked': locked,
                'storage': storage, 'emb_model': emb_model,
                'emb_backend': emb_backend,
            })

    def _refresh_sources(self):
        if not self.selected_index:
            self.sources = []
            return
        rag = _rag()
        src_dict = rag.get_index_sources(self.selected_index)
        self.sources = sorted(src_dict.items(), key=lambda x: x[0])

    def run(self):
        import curses, struct, fcntl, termios, time
        try:
            packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b'\x00' * 8)
            orig_rows, orig_cols = struct.unpack('HHHH', packed)[:2]
        except Exception:
            orig_rows, orig_cols = 0, 0
        sys.stdout.write('\033[8;24;105t')
        sys.stdout.flush()
        time.sleep(0.05)
        try:
            return curses.wrapper(self._main)
        except KeyboardInterrupt:
            return False
        finally:
            if orig_rows > 0 and orig_cols > 0:
                sys.stdout.write(f'\033[8;{orig_rows};{orig_cols}t')
                sys.stdout.flush()

    def _main(self, stdscr):
        import curses

        self.stdscr = stdscr
        curses.curs_set(0)
        stdscr.timeout(-1)

        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # active tab
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)   # selected item
            curses.init_pair(3, curses.COLOR_CYAN, -1)                    # title
            curses.init_pair(4, curses.COLOR_GREEN, -1)                   # success
            curses.init_pair(5, curses.COLOR_RED, -1)                     # error/warning
            curses.init_pair(6, curses.COLOR_YELLOW, -1)                  # hints/locked
            curses.init_pair(7, curses.COLOR_WHITE, -1)                   # border
            curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_RED)     # confirm overlay
            self.has_colors = True

            # Flame gradient
            if curses.COLORS >= 256:
                flame_colors = [196, 202, 208, 214, 220, 231]
                for i, c in enumerate(flame_colors):
                    curses.init_pair(10 + i, c, -1)
                self.has_flame = True
            else:
                flame_fallback = [
                    curses.COLOR_RED, curses.COLOR_RED, curses.COLOR_YELLOW,
                    curses.COLOR_YELLOW, curses.COLOR_WHITE, curses.COLOR_WHITE,
                ]
                for i, c in enumerate(flame_fallback):
                    curses.init_pair(10 + i, c, -1)
                self.has_flame = True
        except Exception:
            self.has_colors = False

        while True:
            self._draw()
            ch = stdscr.getch()

            if self.confirming:
                self._handle_confirm(ch)
            else:
                action = self._handle_nav(ch)
                if action == 'quit':
                    return True

        return True

    def _attr(self, pair, extra=0):
        import curses
        if self.has_colors:
            return curses.color_pair(pair) | extra
        return extra

    def _items(self):
        if self.mode == self.MODE_INDEX:
            return self.indexes
        return self.sources

    def _visible_rows(self, h, content_start):
        return max(1, (h - content_start - 5) // 2)

    def _clamp_cursor(self):
        items = self._items()
        n = len(items)
        if n == 0:
            self.cursor = 0
            self.scroll_offset = 0
            return
        self.cursor = max(0, min(self.cursor, n - 1))

    def _handle_nav(self, ch):
        import curses
        items = self._items()
        n = len(items)

        if ch in (ord('q'), 27):  # q or Esc
            if self.mode == self.MODE_SOURCE:
                self.mode = self.MODE_INDEX
                self.cursor = 0
                self.scroll_offset = 0
                self.selected_index = None
                self.status_msg = ""
                return True
            return 'quit'

        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if self.mode == self.MODE_SOURCE:
                self.mode = self.MODE_INDEX
                self.cursor = 0
                self.scroll_offset = 0
                self.selected_index = None
                self.status_msg = ""
                return True

        if ch == curses.KEY_UP:
            self.cursor = max(0, self.cursor - 1)
            return True

        if ch == curses.KEY_DOWN:
            self.cursor = min(n - 1, self.cursor + 1) if n > 0 else 0
            return True

        if ch in (curses.KEY_ENTER, 10, 13):
            if self.mode == self.MODE_INDEX and n > 0:
                idx = self.indexes[self.cursor]
                self.selected_index = idx['name']
                self._refresh_sources()
                self.mode = self.MODE_SOURCE
                self.cursor = 0
                self.scroll_offset = 0
                self.status_msg = ""
                return True

        if ch in (ord('d'), curses.KEY_DC):  # Delete
            if n == 0:
                return True
            if self.mode == self.MODE_INDEX:
                idx = self.indexes[self.cursor]
                if idx['locked']:
                    self.status_msg = f"Index '{idx['name']}' is LOCKED. Press 'u' to unlock first."
                    self.status_color = 5  # red
                    return True
                self.confirm_msg_lines = [
                    "*** WARNING ***",
                    "",
                    f"PERMANENTLY delete index '{idx['name']}'",
                    f"({idx['n_chunks']:,} chunks from {idx['n_files']} files).",
                    "",
                    "CANNOT BE UNDONE.",
                ]
                self.confirm_action = ('delete_index', idx['name'])
                self.confirm_cursor = 1  # default No
                self.confirming = True
                return True

            elif self.mode == self.MODE_SOURCE:
                if not self.sources:
                    return True
                src_name, src_count = self.sources[self.cursor]
                # Check if parent index is locked
                if _rag().is_index_locked(self.selected_index):
                    self.status_msg = f"Index '{self.selected_index}' is LOCKED. Go back and press 'u' to unlock."
                    self.status_color = 5
                    return True
                self.confirm_msg_lines = [
                    "*** WARNING ***",
                    "",
                    f"Remove '{src_name}'",
                    f"({src_count:,} chunks) from '{self.selected_index}'.",
                    "Index will be rebuilt.",
                    "",
                    "CANNOT BE UNDONE.",
                ]
                self.confirm_action = ('remove_source', self.selected_index, src_name)
                self.confirm_cursor = 1
                self.confirming = True
                return True

        if ch == ord('u') and self.mode == self.MODE_INDEX and n > 0:
            idx = self.indexes[self.cursor]
            if idx['locked']:
                lock_file = os.path.join(RAG_HOME, idx['name'], ".locked")
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                self._refresh_indexes()
                self._clamp_cursor()
                self.status_msg = f"Unlocked '{idx['name']}'"
                self.status_color = 4  # green
            else:
                self.status_msg = f"'{idx['name']}' is already unlocked"
                self.status_color = 6
            return True

        if ch == ord('l') and self.mode == self.MODE_INDEX and n > 0:
            idx = self.indexes[self.cursor]
            if not idx['locked']:
                lock_file = os.path.join(RAG_HOME, idx['name'], ".locked")
                with open(lock_file, 'w') as f:
                    f.write("locked\n")
                self._refresh_indexes()
                self._clamp_cursor()
                self.status_msg = f"Locked '{idx['name']}'"
                self.status_color = 4
            else:
                self.status_msg = f"'{idx['name']}' is already locked"
                self.status_color = 6
            return True

        return True

    def _handle_confirm(self, ch):
        import curses

        if ch == 27:  # Esc — cancel
            self.confirming = False
            self.status_msg = "Cancelled."
            self.status_color = 6
            return

        if ch == curses.KEY_LEFT:
            self.confirm_cursor = 0
            return

        if ch == curses.KEY_RIGHT:
            self.confirm_cursor = 1
            return

        if ch in (curses.KEY_ENTER, 10, 13):
            self.confirming = False
            if self.confirm_cursor == 1:  # No
                self.status_msg = "Cancelled."
                self.status_color = 6
                return

            # Yes — execute action
            action = self.confirm_action
            try:
                if action[0] == 'delete_index':
                    _rag().delete_index(action[1], force=True)
                    self._refresh_indexes()
                    self._clamp_cursor()
                    self.status_msg = f"Deleted index '{action[1]}'"
                    self.status_color = 4
                elif action[0] == 'remove_source':
                    result = _rag().remove_source_from_index(action[1], action[2])
                    self._refresh_sources()
                    self._refresh_indexes()  # update counts
                    self._clamp_cursor()
                    self.status_msg = (f"Removed '{action[2]}' ({result['removed_chunks']:,} chunks). "
                                       f"{result['remaining_chunks']:,} chunks remain.")
                    self.status_color = 4
            except Exception as e:
                self.status_msg = f"ERROR: {e}"
                self.status_color = 5

    def _draw(self):
        import curses
        stdscr = self.stdscr
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        if h < 10 or w < 40:
            stdscr.addstr(0, 0, "Terminal too small")
            stdscr.refresh()
            return

        border_attr = self._attr(7, curses.A_DIM)

        # Top border
        top = "+" + "=" * (w - 2) + "+"
        stdscr.addstr(0, 0, top[:w], border_attr)

        # Side borders
        for row in range(1, h - 1):
            try:
                stdscr.addstr(row, 0, "|", border_attr)
                stdscr.addstr(row, w - 1, "|", border_attr)
            except curses.error:
                pass

        # Bottom border
        try:
            stdscr.addstr(h - 1, 0, ("+" + "=" * (w - 2) + "+")[:w-1], border_attr)
        except curses.error:
            pass

        # Banner
        show_banner = (h >= BANNER_H + 18 and w >= BANNER_W + 6)
        if show_banner:
            x_off = max(2, (w - BANNER_W) // 2)
            for bi, line in enumerate(BANNER):
                if self.has_flame:
                    attr = curses.color_pair(10 + bi) | curses.A_BOLD
                else:
                    attr = self._attr(6, curses.A_BOLD)
                try:
                    stdscr.addstr(1 + bi, x_off, line[:w-4], attr)
                except curses.error:
                    pass
            content_start = 1 + BANNER_H + 1
        else:
            title_bar = " RAG-Narock Index Editor "
            if w > len(title_bar) + 4:
                stdscr.addstr(0, 2, title_bar, self._attr(6, curses.A_BOLD))
            content_start = 2

        # Title row
        row = content_start
        if self.mode == self.MODE_INDEX:
            title = "INDEX EDITOR"
        else:
            idx_info = None
            for ix in self.indexes:
                if ix['name'] == self.selected_index:
                    idx_info = ix
                    break
            if idx_info:
                lock_str = " [LOCKED]" if idx_info['locked'] else ""
                title = f"{idx_info['name']} ({idx_info['n_files']} files, {idx_info['n_chunks']:,} chunks){lock_str}"
            else:
                title = self.selected_index or "Sources"

        try:
            stdscr.addstr(row, 3, title[:w-6], self._attr(3, curses.A_BOLD))
        except curses.error:
            pass

        # Separator
        row = content_start + 1
        sep = "-" * (w - 4)
        try:
            stdscr.addstr(row, 2, sep[:w-4], border_attr)
        except curses.error:
            pass

        # Items
        items = self._items()
        n = len(items)
        item_start = content_start + 2
        vis = self._visible_rows(h, content_start)

        # Adjust scroll offset
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + vis:
            self.scroll_offset = self.cursor - vis + 1
        self.scroll_offset = max(0, min(self.scroll_offset, max(0, n - vis)))

        # Scroll-up indicator
        if self.scroll_offset > 0:
            try:
                stdscr.addstr(item_start, w - 15, f"^ {self.scroll_offset} more", self._attr(6))
            except curses.error:
                pass

        for vi in range(vis):
            idx = self.scroll_offset + vi
            if idx >= n:
                break
            r = item_start + vi * 2
            if r >= h - 4:
                break

            selected = (idx == self.cursor)
            ptr = ">" if selected else " "
            ptr_attr = self._attr(6, curses.A_BOLD) if selected else 0
            try:
                stdscr.addstr(r, 3, ptr, ptr_attr)
            except curses.error:
                pass

            if self.mode == self.MODE_INDEX:
                ix = items[idx]
                lock_str = " [LOCKED]" if ix['locked'] else ""
                extra = f" [{ix.get('storage', 'faiss')}, {ix.get('emb_model', '?')}]"
                line = f"{ix['name']}  ({ix['n_chunks']:,}ch, {ix['n_files']}f){lock_str}{extra}"
                if selected:
                    lbl_attr = self._attr(2)
                elif ix['locked']:
                    lbl_attr = self._attr(6)
                else:
                    lbl_attr = curses.A_NORMAL
                try:
                    stdscr.addstr(r, 5, line[:w-8], lbl_attr)
                except curses.error:
                    pass
            else:
                src_name, src_count = items[idx]
                line = f"{src_name}  ({src_count:,} chunks)"
                lbl_attr = self._attr(2) if selected else curses.A_NORMAL
                try:
                    stdscr.addstr(r, 5, line[:w-8], lbl_attr)
                except curses.error:
                    pass

        # Scroll-down indicator
        remaining = n - self.scroll_offset - vis
        if remaining > 0:
            ind_row = item_start + vis * 2 - 1
            if ind_row < h - 4:
                try:
                    stdscr.addstr(ind_row, w - 15, f"v {remaining} more", self._attr(6))
                except curses.error:
                    pass

        # Empty state
        if n == 0:
            msg = "No indexes found." if self.mode == self.MODE_INDEX else "No sources in this index."
            try:
                stdscr.addstr(item_start + 1, 5, msg, self._attr(6))
            except curses.error:
                pass

        # Status message
        status_row = h - 4
        if self.status_msg:
            try:
                stdscr.addstr(status_row, 3, self.status_msg[:w-6], self._attr(self.status_color, curses.A_BOLD))
            except curses.error:
                pass

        # Help separator + keys
        help_row = h - 2
        try:
            stdscr.addstr(help_row - 1, 2, "-" * (w - 4), border_attr)
        except curses.error:
            pass

        if self.mode == self.MODE_INDEX:
            help_text = "  Enter=view  d=delete  l=lock  u=unlock  q/Esc=exit"
        else:
            help_text = "  d=remove source  Backspace=back  q/Esc=exit"
        try:
            stdscr.addstr(help_row, 2, help_text[:w-4], self._attr(6))
        except curses.error:
            pass

        # Confirmation overlay
        if self.confirming:
            self._draw_confirm(h, w)

        stdscr.refresh()

    def _draw_confirm(self, h, w):
        import curses
        lines = self.confirm_msg_lines
        box_w = min(50, w - 4)
        box_h = len(lines) + 6  # padding + buttons
        box_y = max(1, (h - box_h) // 2)
        box_x = max(1, (w - box_w) // 2)

        red_attr = self._attr(8, curses.A_BOLD)

        # Draw box
        top = "+" + "=" * (box_w - 2) + "+"
        bot = "+" + "=" * (box_w - 2) + "+"
        mid = "|" + " " * (box_w - 2) + "|"

        try:
            self.stdscr.addstr(box_y, box_x, top[:box_w], red_attr)
            for r in range(1, box_h - 1):
                self.stdscr.addstr(box_y + r, box_x, mid[:box_w], red_attr)
            self.stdscr.addstr(box_y + box_h - 1, box_x, bot[:box_w], red_attr)
        except curses.error:
            pass

        # Text lines
        for i, line in enumerate(lines):
            try:
                self.stdscr.addstr(box_y + 1 + i, box_x + 2, line[:box_w-4], red_attr)
            except curses.error:
                pass

        # Buttons
        btn_row = box_y + len(lines) + 2
        yes_label = "[ Yes ]"
        no_label = "[  No  ]"
        total_btn = len(yes_label) + 5 + len(no_label)
        btn_x = box_x + (box_w - total_btn) // 2

        if self.confirm_cursor == 0:
            yes_attr = self._attr(2, curses.A_BOLD)
            no_attr = red_attr
        else:
            yes_attr = red_attr
            no_attr = self._attr(2, curses.A_BOLD)

        try:
            self.stdscr.addstr(btn_row, btn_x, yes_label, yes_attr)
            self.stdscr.addstr(btn_row, btn_x + len(yes_label) + 5, no_label, no_attr)
        except curses.error:
            pass


# ── Tkinter GUI Dialog ──────────────────────────────────────────────────────

class EditorDialog:
    """Modal tkinter index editor dialog."""

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
                                      borderwidth=1, relief='flat')
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
                                       borderwidth=1, relief='flat')
        src_scroll = ttk.Scrollbar(right_frame, orient='vertical', command=self.source_list.yview)
        self.source_list.configure(yscrollcommand=src_scroll.set)
        self.source_list.pack(side='left', fill='both', expand=True)
        src_scroll.pack(side='right', fill='y')

        # Buttons
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill='x', padx=10, pady=5)

        self.delete_btn = ttk.Button(btn_frame, text="Delete Index", style='Danger.TButton',
                                      command=self._delete_index)
        self.delete_btn.pack(side='left', padx=2)

        self.lock_btn = ttk.Button(btn_frame, text="Lock/Unlock", style='Editor.TButton',
                                    command=self._toggle_lock)
        self.lock_btn.pack(side='left', padx=2)

        self.remove_btn = ttk.Button(btn_frame, text="Remove Source", style='Danger.TButton',
                                      command=self._remove_source)
        self.remove_btn.pack(side='right', padx=2)

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
        rag = _rag()
        backends = _backends()
        self.index_list.delete(0, 'end')
        self._indexes = []
        for name in rag.get_indexes():
            index_dir = os.path.join(RAG_HOME, name)
            meta = backends.get_index_meta_with_defaults(index_dir)
            locked = rag.is_index_locked(name)
            n_chunks = meta.get('n_chunks', 0)
            n_files = meta.get('n_files', 0)
            storage = meta.get('storage_backend', 'faiss')
            emb_model = meta.get('embedding_model', '?')
            lock_str = " [LOCKED]" if locked else ""
            self.index_list.insert('end', f"  {name}  ({n_chunks:,}ch, {n_files}f){lock_str} [{storage}, {emb_model}]")
            self._indexes.append({
                'name': name, 'n_chunks': n_chunks,
                'n_files': n_files, 'locked': locked,
            })
        self.source_list.delete(0, 'end')
        self.right_frame.configure(text="Sources")

    def _on_index_select(self, event=None):
        sel = self.index_list.curselection()
        if not sel:
            return
        idx = self._indexes[sel[0]]
        rag = _rag()
        sources = rag.get_index_sources(idx['name'])
        self.source_list.delete(0, 'end')
        self._sources = sorted(sources.items(), key=lambda x: x[0])
        for src_name, count in self._sources:
            self.source_list.insert('end', f"  {src_name}  ({count:,} chunks)")
        self.right_frame.configure(text=f"Sources in: {idx['name']}")

    def _selected_index(self):
        sel = self.index_list.curselection()
        if not sel:
            return None
        return self._indexes[sel[0]]

    def _notify_change(self):
        if self.on_change:
            self.on_change()

    def _delete_index(self):
        from tkinter import messagebox
        idx = self._selected_index()
        if not idx:
            return
        if idx['locked']:
            messagebox.showwarning("Locked",
                f"Index '{idx['name']}' is LOCKED.\nUnlock it first.",
                parent=self.win)
            return
        confirm = messagebox.askyesno("Delete Index",
            f"PERMANENTLY delete index '{idx['name']}'\n"
            f"({idx['n_chunks']:,} chunks from {idx['n_files']} files).\n\n"
            f"CANNOT BE UNDONE.",
            icon='warning', default='no', parent=self.win)
        if not confirm:
            return
        try:
            _rag().delete_index(idx['name'], force=True)
            self.status_var.set(f"Deleted index '{idx['name']}'")
            self._refresh_indexes()
            self._notify_change()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.win)

    def _toggle_lock(self):
        idx = self._selected_index()
        if not idx:
            return
        lock_file = os.path.join(RAG_HOME, idx['name'], ".locked")
        if idx['locked']:
            if os.path.exists(lock_file):
                os.remove(lock_file)
            self.status_var.set(f"Unlocked '{idx['name']}'")
        else:
            with open(lock_file, 'w') as f:
                f.write("locked\n")
            self.status_var.set(f"Locked '{idx['name']}'")
        self._refresh_indexes()
        self._notify_change()

    def _remove_source(self):
        from tkinter import messagebox
        idx = self._selected_index()
        if not idx:
            return
        if idx['locked']:
            messagebox.showwarning("Locked",
                f"Index '{idx['name']}' is LOCKED.\nUnlock it first.",
                parent=self.win)
            return
        sel = self.source_list.curselection()
        if not sel:
            return
        src_name, src_count = self._sources[sel[0]]
        confirm = messagebox.askyesno("Remove Source",
            f"Remove '{src_name}'\n"
            f"({src_count:,} chunks) from '{idx['name']}'.\n"
            f"Index will be rebuilt.\n\n"
            f"CANNOT BE UNDONE.",
            icon='warning', default='no', parent=self.win)
        if not confirm:
            return
        try:
            result = _rag().remove_source_from_index(idx['name'], src_name)
            self.status_var.set(
                f"Removed '{src_name}' ({result['removed_chunks']:,} chunks). "
                f"{result['remaining_chunks']:,} remain.")
            self._refresh_indexes()
            # Re-select same index to refresh sources
            for i, ix in enumerate(self._indexes):
                if ix['name'] == idx['name']:
                    self.index_list.selection_set(i)
                    self._on_index_select()
                    break
            self._notify_change()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.win)


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    """Run TUI editor from command line."""
    tui = EditorTUI()
    tui.run()

if __name__ == '__main__':
    main()
