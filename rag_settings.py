#!/usr/bin/env python3
"""
RAG Settings — unified config with curses TUI and tkinter GUI dialog.

Settings file: ~/.local/share/rag/settings.json
Usage:
    from rag_settings import get, load, save
    model = get('embedding_model')       # returns current value
    cfg = load()                         # returns full dict
    save(cfg)                            # writes to disk

    # TUI (terminal)
    from rag_settings import SettingsTUI
    SettingsTUI().run()

    # GUI (tkinter)
    from rag_settings import SettingsDialog
    SettingsDialog(parent_window)
"""

import os, sys, json
from pathlib import Path
import rag_acl

BANNER = r"""
(()/(   )\    )\ )     )\())   ) (           ( /(
 /(_)|(((_)( (()/(  __((_)\ ( /( )(   (   (  )\())
(_))  )\ _ )\ /(_))|___|((_))(_)|()\  )\  )\((_)\
| _ \ (_)_\(_|_)) __| | \| ((_)_ ((_)((_)((_) |(_)
|   /  / _ \   | (_ | | .` / _` | '_/ _ Y _|| / /
|_|_\ /_/ \_\   \___| |_|\_\__,_|_| \___|__||_\_\
""".strip('\n').splitlines()
BANNER_H = len(BANNER)
BANNER_W = max(len(l) for l in BANNER)

SETTINGS_DIR = os.path.expanduser("~/.local/share/rag")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")

# ── Schema ──────────────────────────────────────────────────────────────────

# Each item: (key, label, type, options_or_none, default, description)
# type: 'choice', 'toggle', 'text'
# For 'choice': options is list of (value, label) tuples
# For 'toggle': options is None, values are True/False
# For 'text': options is None, value is string

TABS = [
    ("Models", [
        ("embedding_backend", "Backend", "choice", [
            ("local",    "Local (CPU)"),
            ("ollama",   "Ollama (GPU)"),
            ("lmstudio", "LM Studio (GPU)"),
        ], "local"),
        ("gpu_indexing",  "GPU Indexing",  "toggle", None, False),
        ("conda_env",     "Conda Env",    "text",   None, "ai-env"),
        ("embedding_model", "Local Model", "choice", [
            ("all-MiniLM-L6-v2",                        "all-MiniLM-L6-v2 (English, 384d, fast)"),
            ("all-MiniLM-L12-v2",                       "all-MiniLM-L12-v2 (English, 384d, balanced)"),
            ("paraphrase-multilingual-MiniLM-L12-v2",   "multilingual-MiniLM-L12 (50+ langs, 384d)"),
            ("all-mpnet-base-v2",                       "all-mpnet-base-v2 (English, 768d, best quality)"),
        ], "all-MiniLM-L6-v2"),
        ("api_model",    "API Model",    "text", None, "nomic-embed-text"),
        ("ollama_url",   "Ollama URL",   "text", None, "http://localhost:11434"),
        ("lmstudio_url", "LM Studio URL","text", None, "http://localhost:1234"),
    ]),
    ("OCR", [
        ("disable_ocr",    "Disable OCR",      "toggle", None, False),
        ("ocr_backend",    "OCR Backend",      "choice", [
            ("tesseract", "Tesseract (CPU)"),
            ("easyocr",   "EasyOCR (GPU/XPU)"),
        ], "tesseract"),
        ("ocr_lang",       "OCR Language",     "text",   None, "eng"),
        ("force_ocr",      "Force OCR",        "toggle", None, False),
        ("ocr_negative",   "OCR Negative",     "toggle", None, False),
        ("split_spreads",  "Split Spreads",    "toggle", None, False),
        ("render_dpi",     "Render DPI",       "choice", [
            (150, "150"), (200, "200"), (300, "300"), (400, "400"),
        ], 300),
        ("min_image_size", "Min Image Size",   "choice", [
            (100, "100 px"), (200, "200 px"), (300, "300 px"), (400, "400 px"),
        ], 200),
    ]),
    ("RAG", [
        ("data_dir", "Data Directory", "text", None, "~/.local/share/rag"),
        ("storage_backend", "Storage Backend",  "choice", [
            ("faiss",       "FAISS (default)"),
            ("sqlite-vec",  "SQLite-vec"),
            ("sqlite-doc",  "SQLite-doc (document-aware)"),
        ], "faiss"),
        ("chunk_size",      "Chunk Size",       "choice", [
            (500, "500"), (1000, "1000"), (1500, "1500"), (2000, "2000"), (3000, "3000"),
        ], 1500),
        ("overlap",         "Overlap",          "choice", [
            (50, "50"), (100, "100"), (200, "200"), (300, "300"),
        ], 200),
        ("top_k",           "Top K Results",    "choice", [
            (3, "3"), (5, "5"), (10, "10"), (15, "15"), (20, "20"),
        ], 5),
        ("min_chunk_length","Min Chunk Length",  "choice", [
            (20, "20"), (50, "50"), (100, "100"),
        ], 50),
        ("code_chunk_size", "Code Chunk Size",  "choice", [
            (1500, "1500"), (2000, "2000"), (3000, "3000"), (4000, "4000"), (5000, "5000"),
        ], 3000),
        ("code_overlap",    "Code Overlap",     "choice", [
            (100, "100"), (200, "200"), (300, "300"), (500, "500"),
        ], 200),
    ]),
]

# Flat defaults for quick access
DEFAULTS = {}
SCHEMA = {}  # key -> (label, type, options, default, tab_name)
for tab_name, items in TABS:
    for key, label, typ, options, default in items:
        DEFAULTS[key] = default
        SCHEMA[key] = (label, typ, options, default, tab_name)


# Settings whose change makes existing embeddings incompatible
DANGEROUS_KEYS = {'embedding_backend', 'embedding_model', 'api_model', 'storage_backend'}

# ── Load / Save / Get ───────────────────────────────────────────────────────

def load():
    """Load settings, merging with defaults for any missing keys."""
    cfg = dict(DEFAULTS)
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            # Only accept known keys with valid types
            for key, val in saved.items():
                if key in SCHEMA:
                    cfg[key] = val
            # Preserve external_indexes list (not in schema but managed programmatically)
            if 'external_indexes' in saved and isinstance(saved['external_indexes'], list):
                cfg['external_indexes'] = saved['external_indexes']
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save(cfg):
    """Save settings to disk."""
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)


def get(key):
    """Get a single setting value (loads from disk each time)."""
    cfg = load()
    return cfg.get(key, DEFAULTS.get(key))


def get_data_dir():
    """Return resolved absolute data directory path."""
    raw = get('data_dir')
    return os.path.abspath(os.path.expanduser(raw)) if raw else os.path.expanduser("~/.local/share/rag")


def get_external_indexes():
    """Return list of registered external index absolute paths."""
    cfg = load()
    return cfg.get('external_indexes', [])


def add_external_index(path):
    """Register an external index path in settings. Returns True if added."""
    path = os.path.abspath(os.path.expanduser(path))
    cfg = load()
    externals = cfg.get('external_indexes', [])
    if path in externals:
        return False
    externals.append(path)
    cfg['external_indexes'] = externals
    save(cfg)
    return True


def remove_external_index(path):
    """Unregister an external index path from settings. Returns True if removed."""
    path = os.path.abspath(os.path.expanduser(path))
    cfg = load()
    externals = cfg.get('external_indexes', [])
    if path not in externals:
        return False
    externals.remove(path)
    cfg['external_indexes'] = externals
    save(cfg)
    return True


# ── Curses TUI ──────────────────────────────────────────────────────────────

class SettingsTUI:
    """Full-screen curses settings editor with tabbed navigation."""

    def __init__(self):
        self.cfg = load()
        self.saved_cfg = dict(self.cfg)  # snapshot for change detection
        self.tab_idx = 0
        self.item_idx = 0
        self.editing_text = False
        self.edit_buffer = ""
        self.edit_cursor = 0
        self.has_colors = False
        self.has_flame = False  # 256-color flame gradient
        self.warning_msg = ""
        self.acl_clients = []
        self.acl_sel = 0
        self.acl_naming = False
        self.acl_name_buf = ""
        self.acl_name_cur = 0
        self.acl_editing = False
        self.acl_edit_key = None
        self.acl_edit_privs = {}
        self.acl_edit_idx = 0
        self.acl_status = ""

    def run(self):
        """Launch the TUI. Returns True if settings were saved."""
        import curses, struct, fcntl, termios, time
        # Save original terminal size, then resize to 105x24
        try:
            packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b'\x00' * 8)
            orig_rows, orig_cols = struct.unpack('HHHH', packed)[:2]
        except Exception:
            orig_rows, orig_cols = 0, 0
        sys.stdout.write('\033[8;24;105t')
        sys.stdout.flush()
        time.sleep(0.05)  # let terminal process the resize
        try:
            return curses.wrapper(self._main)
        except KeyboardInterrupt:
            return False
        finally:
            # Restore original terminal size
            if orig_rows > 0 and orig_cols > 0:
                sys.stdout.write(f'\033[8;{orig_rows};{orig_cols}t')
                sys.stdout.flush()

    def _main(self, stdscr):
        import curses

        self.stdscr = stdscr
        curses.curs_set(0)
        stdscr.timeout(-1)

        # Colors
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # active tab
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)   # selected item
            curses.init_pair(3, curses.COLOR_CYAN, -1)                    # choice values
            curses.init_pair(4, curses.COLOR_GREEN, -1)                   # toggle on
            curses.init_pair(5, curses.COLOR_RED, -1)                     # toggle off
            curses.init_pair(6, curses.COLOR_YELLOW, -1)                  # arrows/hints
            curses.init_pair(7, curses.COLOR_WHITE, -1)                   # border dim
            self.has_colors = True

            # Flame gradient: pairs 10-15 for banner lines 0-5
            # 256-color: red(196) → orange-red(202) → orange(208) | gold(214) → yellow(220) → white(231)
            if curses.COLORS >= 256:
                flame_colors = [196, 202, 208, 214, 220, 231]
                for i, c in enumerate(flame_colors):
                    curses.init_pair(10 + i, c, -1)
                self.has_flame = True
            else:
                # 8-color fallback: red flames, yellow transition, white letters
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

            if self.acl_naming:
                self._handle_acl_naming(ch)
                continue
            if self.acl_editing:
                self._handle_acl_editing(ch)
                continue
            if self.editing_text:
                if self._handle_text_edit(ch):
                    continue
            else:
                if self._handle_nav(ch):
                    continue
                if ch in (ord('q'), 27):  # q or Esc → save and exit
                    save(self.cfg)
                    return True

        return False

    def _cur_tab_items(self):
        if self.tab_idx >= len(TABS):
            return []
        return TABS[self.tab_idx][1]

    def _cur_item(self):
        items = self._cur_tab_items()
        if 0 <= self.item_idx < len(items):
            return items[self.item_idx]
        return None

    def _handle_nav(self, ch):
        import curses
        total_tabs = len(TABS) + 1  # +1 for ACL tab

        if ch == ord('\t') or ch == 9:
            self.tab_idx = (self.tab_idx + 1) % total_tabs
            if self.tab_idx < len(TABS):
                self.item_idx = min(self.item_idx, len(self._cur_tab_items()) - 1)
            else:
                self._refresh_acl()
            return True

        if ch == curses.KEY_BTAB:
            self.tab_idx = (self.tab_idx - 1) % total_tabs
            if self.tab_idx < len(TABS):
                self.item_idx = min(self.item_idx, len(self._cur_tab_items()) - 1)
            else:
                self._refresh_acl()
            return True

        # ACL tab: delegate to ACL-specific handler
        if self.tab_idx >= len(TABS):
            return self._handle_acl_keys(ch)

        items = self._cur_tab_items()
        item = self._cur_item()

        if ch == curses.KEY_UP:
            self.item_idx = max(0, self.item_idx - 1)
            return True

        if ch == curses.KEY_DOWN:
            self.item_idx = min(len(items) - 1, self.item_idx + 1)
            return True

        if item is None:
            return True

        key, label, typ, options, default = item

        if ch == curses.KEY_LEFT:
            if typ == 'choice':
                self._cycle_choice(key, options, -1)
            elif typ == 'toggle':
                self.cfg[key] = not self.cfg[key]
            else:
                self.tab_idx = (self.tab_idx - 1) % total_tabs
                if self.tab_idx < len(TABS):
                    self.item_idx = min(self.item_idx, len(self._cur_tab_items()) - 1)
                else:
                    self._refresh_acl()
            return True

        if ch == curses.KEY_RIGHT:
            if typ == 'choice':
                self._cycle_choice(key, options, 1)
            elif typ == 'toggle':
                self.cfg[key] = not self.cfg[key]
            else:
                self.tab_idx = (self.tab_idx + 1) % total_tabs
                if self.tab_idx < len(TABS):
                    self.item_idx = min(self.item_idx, len(self._cur_tab_items()) - 1)
                else:
                    self._refresh_acl()
            return True

        if ch == ord(' '):
            if typ == 'choice':
                self._cycle_choice(key, options, 1)
            elif typ == 'toggle':
                self.cfg[key] = not self.cfg[key]
            return True

        if ch in (curses.KEY_ENTER, 10, 13):
            if typ == 'text':
                self.editing_text = True
                self.edit_buffer = str(self.cfg[key])
                self.edit_cursor = len(self.edit_buffer)
                curses.curs_set(1)
            elif typ == 'choice':
                self._cycle_choice(key, options, 1)
            elif typ == 'toggle':
                self.cfg[key] = not self.cfg[key]
            return True

        if ch in (ord('q'), 27):
            return False

        return True

    def _handle_text_edit(self, ch):
        import curses
        item = self._cur_item()
        if item is None:
            self.editing_text = False
            curses.curs_set(0)
            return True

        key = item[0]

        if ch in (curses.KEY_ENTER, 10, 13):
            # Confirm edit
            self.cfg[key] = self.edit_buffer
            self.editing_text = False
            curses.curs_set(0)
            return True

        if ch == 27:  # Esc — cancel edit
            self.editing_text = False
            curses.curs_set(0)
            return True

        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if self.edit_cursor > 0:
                self.edit_buffer = self.edit_buffer[:self.edit_cursor-1] + self.edit_buffer[self.edit_cursor:]
                self.edit_cursor -= 1
            return True

        if ch == curses.KEY_DC:  # Delete key
            if self.edit_cursor < len(self.edit_buffer):
                self.edit_buffer = self.edit_buffer[:self.edit_cursor] + self.edit_buffer[self.edit_cursor+1:]
            return True

        if ch == curses.KEY_LEFT:
            self.edit_cursor = max(0, self.edit_cursor - 1)
            return True

        if ch == curses.KEY_RIGHT:
            self.edit_cursor = min(len(self.edit_buffer), self.edit_cursor + 1)
            return True

        if ch == curses.KEY_HOME:
            self.edit_cursor = 0
            return True

        if ch == curses.KEY_END:
            self.edit_cursor = len(self.edit_buffer)
            return True

        # Printable character
        if 32 <= ch < 127:
            self.edit_buffer = self.edit_buffer[:self.edit_cursor] + chr(ch) + self.edit_buffer[self.edit_cursor:]
            self.edit_cursor += 1
            return True

        return True

    def _cycle_choice(self, key, options, direction):
        cur = self.cfg[key]
        values = [o[0] for o in options]
        try:
            idx = values.index(cur)
        except ValueError:
            idx = 0
        idx = (idx + direction) % len(values)
        self.cfg[key] = values[idx]

    def _check_warnings(self):
        """Set warning_msg if any dangerous settings were changed."""
        for key in DANGEROUS_KEYS:
            if self.cfg.get(key) != self.saved_cfg.get(key):
                self.warning_msg = ("WARNING: Changing the embedding model will make NEW embeddings\n"
                                   "incompatible with existing indexes. Queries may fail or return\n"
                                   "poor results. Re-indexing required.")
                return
        self.warning_msg = ""

    def _attr(self, pair, extra=0):
        import curses
        if self.has_colors:
            return curses.color_pair(pair) | extra
        return extra

    def _draw(self):
        import curses
        stdscr = self.stdscr
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        if h < 10 or w < 40:
            stdscr.addstr(0, 0, "Terminal too small")
            stdscr.refresh()
            return

        # Box border
        border_attr = self._attr(7, curses.A_DIM) if self.has_colors else curses.A_DIM
        # Top border
        top = "+" + "-" * (w - 2) + "+"
        stdscr.addstr(0, 0, top[:w], border_attr)

        # Side borders
        for row in range(1, h - 1):
            try:
                stdscr.addstr(row, 0, "|", border_attr)
                stdscr.addstr(row, w - 1, "|", border_attr)
            except curses.error:
                pass

        # Bottom border
        bot = "+" + "-" * (w - 2) + "+"
        try:
            stdscr.addstr(h - 1, 0, bot[:w-1], border_attr)
        except curses.error:
            pass

        # Banner or fallback title
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
            title = " RAG-Narock Settings "
            if w > len(title) + 4:
                stdscr.addstr(0, 2, title, self._attr(6, curses.A_BOLD))
            content_start = 2

        # Tabs row
        row = content_start
        col = 3
        all_tab_names = [t[0] for t in TABS] + ["ACL"]
        for ti, tname in enumerate(all_tab_names):
            if ti == self.tab_idx:
                attr = self._attr(1, curses.A_BOLD)
                stdscr.addstr(row, col, f"[ {tname} ]", attr)
            else:
                stdscr.addstr(row, col, f"  {tname}  ", self._attr(7))
            col += len(tname) + 6

        # Separator
        row = content_start + 1
        sep = "-" * (w - 4)
        stdscr.addstr(row, 2, sep[:w-4], border_attr)

        # Content
        start_row = content_start + 3
        if self.tab_idx >= len(TABS):
            self._draw_acl_content(start_row, h, w, border_attr)
        items = self._cur_tab_items()
        for ii, (key, label, typ, options, default) in enumerate(items):
            r = start_row + ii * 2
            if r >= h - 3:
                break

            selected = (ii == self.item_idx)

            # Pointer
            ptr = ">" if selected else " "
            ptr_attr = self._attr(6, curses.A_BOLD) if selected else 0
            stdscr.addstr(r, 3, ptr, ptr_attr)

            # Label
            lbl_attr = self._attr(2) if selected else curses.A_NORMAL
            padded_label = f"{label:<22}"
            stdscr.addstr(r, 5, padded_label, lbl_attr)

            # Value area
            val_col = 28
            cur_val = self.cfg.get(key, default)

            if typ == 'choice':
                # Find display label
                disp = str(cur_val)
                for oval, olbl in options:
                    if oval == cur_val:
                        disp = olbl
                        break
                arrows = "<" if selected else " "
                arrows_r = ">" if selected else " "
                stdscr.addstr(r, val_col, arrows, self._attr(6))
                stdscr.addstr(r, val_col + 2, disp[:w-val_col-6], self._attr(3, curses.A_BOLD if selected else 0))
                end_col = val_col + 2 + len(disp[:w-val_col-6]) + 1
                if end_col < w - 2:
                    stdscr.addstr(r, end_col, arrows_r, self._attr(6))

            elif typ == 'toggle':
                if cur_val:
                    stdscr.addstr(r, val_col, "[ON]", self._attr(4, curses.A_BOLD))
                else:
                    stdscr.addstr(r, val_col, "[OFF]", self._attr(5, curses.A_BOLD))

            elif typ == 'text':
                if self.editing_text and selected:
                    # Show editable text with cursor
                    display = self.edit_buffer + " "
                    max_len = w - val_col - 4
                    stdscr.addstr(r, val_col, display[:max_len], self._attr(3, curses.A_UNDERLINE))
                    # Position cursor
                    cpos = val_col + min(self.edit_cursor, max_len - 1)
                    try:
                        stdscr.move(r, cpos)
                    except curses.error:
                        pass
                else:
                    stdscr.addstr(r, val_col, str(cur_val)[:w-val_col-4], self._attr(3, curses.A_BOLD if selected else 0))

        # Model-change warning (not shown on ACL tab)
        self._check_warnings()
        if self.warning_msg and self.tab_idx < len(TABS):
            warn_lines = self.warning_msg.split('\n')
            warn_start = h - 2 - len(warn_lines) - 1
            for wi, wl in enumerate(warn_lines):
                try:
                    stdscr.addstr(warn_start + wi, 3, wl[:w-6], self._attr(5, curses.A_BOLD))
                except curses.error:
                    pass

        # Help line
        help_row = h - 2
        if self.acl_editing:
            help_text = "  Up/Dn navigate  |  Space=toggle privilege  |  s=save  |  Esc/q=cancel"
        elif self.acl_naming:
            help_text = "  Enter=confirm  |  Esc=cancel"
        elif self.tab_idx >= len(TABS):
            help_text = "  t=toggle  n=new  d=revoke  r=rotate  e=edit privs  c=show key  |  Tab switch  |  q save+exit"
        elif self.editing_text:
            help_text = "  Type to edit  |  Enter confirm  |  Esc cancel"
        else:
            help_text = "  Up/Dn navigate  |  Left/Right/Space cycle  |  Enter edit text  |  Tab switch  |  q save+exit"
        try:
            stdscr.addstr(help_row, 2, "-" * (w - 4), border_attr)
            stdscr.addstr(help_row + 0, 2, help_text[:w-4], self._attr(6))
        except curses.error:
            pass

        stdscr.refresh()

    def _refresh_acl(self):
        self.acl_clients = rag_acl.list_clients()
        self.acl_sel = min(self.acl_sel, max(0, len(self.acl_clients) - 1))
        self.acl_status = ""

    def _handle_acl_keys(self, ch):
        """Handle keys when ACL tab is active. Returns True if handled."""
        import curses
        if ch == curses.KEY_UP:
            self.acl_sel = max(0, self.acl_sel - 1)
            return True
        if ch == curses.KEY_DOWN:
            self.acl_sel = min(max(0, len(self.acl_clients) - 1), self.acl_sel + 1)
            return True
        if ch == ord('t'):
            rag_acl.set_enabled(not rag_acl.is_enabled())
            self.acl_status = "ACL " + ("enabled" if rag_acl.is_enabled() else "disabled")
            return True
        if ch == ord('n'):
            self.acl_naming = True
            self.acl_name_buf = ""
            self.acl_name_cur = 0
            curses.curs_set(1)
            return True
        if ch == ord('d') and self.acl_clients:
            key, client = self.acl_clients[self.acl_sel]
            rag_acl.revoke_client(key)
            self._refresh_acl()
            self.acl_status = f"Revoked: {client['name']}"
            return True
        if ch == ord('r') and self.acl_clients:
            key, client = self.acl_clients[self.acl_sel]
            new_key = rag_acl.rotate_key(key)
            self._refresh_acl()
            self.acl_status = f"New key: {new_key}"
            return True
        if ch == ord('c') and self.acl_clients:
            key, _ = self.acl_clients[self.acl_sel]
            self.acl_status = f"Key: {key}"
            return True
        if ch == ord('e') and self.acl_clients:
            key, client = self.acl_clients[self.acl_sel]
            self.acl_edit_key = key
            self.acl_edit_privs = {p: p in client.get('privileges', []) for p in rag_acl.ALL_PRIVILEGES}
            self.acl_edit_idx = 0
            self.acl_editing = True
            return True
        if ch in (ord('q'), 27):
            return False
        return True

    def _handle_acl_naming(self, ch):
        import curses
        if ch in (curses.KEY_ENTER, 10, 13):
            name = self.acl_name_buf.strip()
            if name:
                key = rag_acl.create_client(name)
                self._refresh_acl()
                self.acl_status = f"Created: {key}"
            self.acl_naming = False
            curses.curs_set(0)
        elif ch == 27:
            self.acl_naming = False
            curses.curs_set(0)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if self.acl_name_cur > 0:
                self.acl_name_buf = self.acl_name_buf[:self.acl_name_cur-1] + self.acl_name_buf[self.acl_name_cur:]
                self.acl_name_cur -= 1
        elif ch == curses.KEY_LEFT:
            self.acl_name_cur = max(0, self.acl_name_cur - 1)
        elif ch == curses.KEY_RIGHT:
            self.acl_name_cur = min(len(self.acl_name_buf), self.acl_name_cur + 1)
        elif 32 <= ch < 127:
            self.acl_name_buf = self.acl_name_buf[:self.acl_name_cur] + chr(ch) + self.acl_name_buf[self.acl_name_cur:]
            self.acl_name_cur += 1

    def _handle_acl_editing(self, ch):
        """Handle keys when editing privileges for a client."""
        import curses
        privs = list(rag_acl.ALL_PRIVILEGES)
        if ch == curses.KEY_UP:
            self.acl_edit_idx = max(0, self.acl_edit_idx - 1)
        elif ch == curses.KEY_DOWN:
            self.acl_edit_idx = min(len(privs) - 1, self.acl_edit_idx + 1)
        elif ch in (ord(' '), curses.KEY_ENTER, 10, 13):
            p = privs[self.acl_edit_idx]
            self.acl_edit_privs[p] = not self.acl_edit_privs[p]
        elif ch == ord('s'):
            # Save
            new_privs = [p for p in privs if self.acl_edit_privs.get(p)]
            rag_acl.update_client(self.acl_edit_key, privileges=new_privs)
            self.acl_editing = False
            self._refresh_acl()
            self.acl_status = "Privileges updated"
        elif ch in (ord('q'), 27):
            # Cancel
            self.acl_editing = False

    def _draw_acl_content(self, start_row, h, w, border_attr):
        """Draw ACL tab content area (called from _draw)."""
        import curses
        stdscr = self.stdscr

        if self.acl_editing:
            self._draw_acl_edit(start_row, h, w, border_attr)
            return

        enabled = rag_acl.is_enabled()
        status_text = "ENABLED" if enabled else "DISABLED"
        status_attr = self._attr(4, curses.A_BOLD) if enabled else self._attr(5, curses.A_BOLD)
        stdscr.addstr(start_row, 5, "Status: ", self._attr(6))
        stdscr.addstr(start_row, 13, status_text, status_attr)

        row = start_row + 2
        for ci, (key, client) in enumerate(self.acl_clients):
            if row >= h - 5:
                break
            sel = (ci == self.acl_sel)
            ptr = ">" if sel else " "
            ptr_attr = self._attr(6, curses.A_BOLD) if sel else 0

            masked = rag_acl.mask_key(key)
            name = client.get('name', '?')
            line1 = f'{masked}  "{name}"'

            indexes = client.get('indexes', '*')
            idx_str = "*" if indexes == "*" else ", ".join(indexes) if indexes else "(none)"
            privs = ", ".join(client.get('privileges', []))
            line2 = f"  idx: {idx_str} | privs: {privs}"

            lbl_attr = self._attr(2) if sel else curses.A_NORMAL
            stdscr.addstr(row, 3, ptr, ptr_attr)
            stdscr.addstr(row, 5, line1[:w-7], lbl_attr)
            row += 1
            stdscr.addstr(row, 5, line2[:w-7], self._attr(3))
            row += 2

        if not self.acl_clients:
            stdscr.addstr(start_row + 2, 5, "(no clients configured)", self._attr(7))

        if self.acl_status:
            try:
                stdscr.addstr(h - 4, 3, self.acl_status[:w-6], self._attr(6))
            except curses.error:
                pass

        if self.acl_naming:
            prompt = "Client name: "
            buf_display = self.acl_name_buf + " "
            try:
                stdscr.addstr(h - 4, 3, " " * (w - 6), curses.A_NORMAL)
                stdscr.addstr(h - 4, 3, prompt + buf_display[:w-len(prompt)-6],
                              self._attr(3, curses.A_UNDERLINE))
                stdscr.move(h - 4, 3 + len(prompt) + self.acl_name_cur)
            except curses.error:
                pass

    def _draw_acl_edit(self, start_row, h, w, border_attr):
        """Draw privilege editor for a single client."""
        import curses
        stdscr = self.stdscr
        client = rag_acl.load()['clients'].get(self.acl_edit_key, {})
        name = client.get('name', '?')

        stdscr.addstr(start_row, 5, f'Editing privileges: "{name}"', self._attr(6, curses.A_BOLD))

        privs = list(rag_acl.ALL_PRIVILEGES)
        for pi, priv in enumerate(privs):
            row = start_row + 2 + pi
            if row >= h - 4:
                break
            sel = (pi == self.acl_edit_idx)
            on = self.acl_edit_privs.get(priv, False)

            ptr = ">" if sel else " "
            ptr_attr = self._attr(6, curses.A_BOLD) if sel else 0
            stdscr.addstr(row, 5, ptr, ptr_attr)

            box = "[X]" if on else "[ ]"
            box_attr = self._attr(4, curses.A_BOLD) if on else self._attr(5)
            lbl_attr = self._attr(2) if sel else curses.A_NORMAL
            stdscr.addstr(row, 7, box, box_attr)
            stdscr.addstr(row, 11, priv, lbl_attr)


# ── Tkinter GUI Dialog ──────────────────────────────────────────────────────

class SettingsDialog:
    """Modal tkinter settings dialog with tabbed notebook."""

    # Theme colors — must match rag.py GUI theme
    BG       = '#0f1626'
    BG2      = '#1a2332'
    BG3      = '#243044'
    FG       = '#d4d4dc'
    FG_DIM   = '#7a8599'
    ACCENT   = '#e8781e'
    ACCENT2  = '#ff9a3c'
    ICE      = '#5eb8d4'
    WARN_RED = '#e74c3c'

    def __init__(self, parent):
        import tkinter as tk
        from tkinter import ttk

        self.cfg = load()
        self.result = False  # True if saved
        self.widgets = {}

        self.win = tk.Toplevel(parent)
        self.win.title("RAG-Narock Settings")
        self.win.geometry("520x420")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.configure(bg=self.BG)

        # Style the notebook tabs to match the dark theme
        style = ttk.Style(self.win)
        style.configure('Settings.TNotebook', background=self.BG, bordercolor=self.BG3)
        style.configure('Settings.TNotebook.Tab', background=self.BG2, foreground=self.FG_DIM,
                        padding=(12, 4), font=('sans-serif', 9, 'bold'))
        style.map('Settings.TNotebook.Tab',
                  background=[('selected', self.BG3)],
                  foreground=[('selected', self.ACCENT)])

        nb = ttk.Notebook(self.win, style='Settings.TNotebook')
        nb.pack(fill='both', expand=True, padx=10, pady=(10, 5))

        for tab_name, items in TABS:
            frame = ttk.Frame(nb, padding=15)
            nb.add(frame, text=tab_name)

            for row_i, (key, label, typ, options, default) in enumerate(items):
                ttk.Label(frame, text=label + ":").grid(row=row_i, column=0, sticky='w', pady=6, padx=(0, 15))

                cur_val = self.cfg.get(key, default)

                if typ == 'choice':
                    display_map = {olbl: oval for oval, olbl in options}
                    reverse_map = {oval: olbl for oval, olbl in options}
                    values = [olbl for _, olbl in options]
                    var = tk.StringVar(value=reverse_map.get(cur_val, str(cur_val)))
                    cb = ttk.Combobox(frame, textvariable=var, values=values,
                                      state='readonly', width=42)
                    cb.grid(row=row_i, column=1, sticky='ew', pady=6)
                    self.widgets[key] = ('choice', var, display_map)

                elif typ == 'toggle':
                    var = tk.BooleanVar(value=cur_val)
                    chk = ttk.Checkbutton(frame, variable=var, text="Enabled")
                    chk.grid(row=row_i, column=1, sticky='w', pady=6)
                    self.widgets[key] = ('toggle', var, None)

                elif typ == 'text':
                    var = tk.StringVar(value=str(cur_val))
                    ent = ttk.Entry(frame, textvariable=var, width=45)
                    ent.grid(row=row_i, column=1, sticky='ew', pady=6)
                    self.widgets[key] = ('text', var, None)

            frame.columnconfigure(1, weight=1)

        # Warning label (hidden by default)
        self.warn_label = tk.Label(self.win, text="", bg=self.BG, fg=self.WARN_RED,
                                    font=('sans-serif', 9, 'bold'), wraplength=480,
                                    justify='left', anchor='w')
        self.warn_label.pack(fill='x', padx=15, pady=(0, 2))
        self.warn_label.pack_forget()  # hidden initially

        # Track model-related variable changes
        self.initial_cfg = dict(self.cfg)
        for key in DANGEROUS_KEYS:
            if key in self.widgets:
                _, var, _ = self.widgets[key]
                var.trace_add('write', lambda *a: self._check_model_warning())

        # Buttons
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill='x', padx=10, pady=(5, 10))

        ttk.Button(btn_frame, text="Defaults", command=self._reset_defaults).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Access Control", command=self._open_acl_dialog).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.win.destroy).pack(side='right', padx=5)
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side='right', padx=5)

        # Center on parent
        self.win.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.win.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.win.winfo_height()) // 2
        self.win.geometry(f"+{max(0,px)}+{max(0,py)}")

        self.win.wait_window()

    def _check_model_warning(self):
        """Show/hide warning when model-related settings change."""
        self._collect()
        changed = any(self.cfg.get(k) != self.initial_cfg.get(k) for k in DANGEROUS_KEYS)
        if changed:
            self.warn_label.config(
                text="WARNING: Changing the embedding model will make NEW embeddings "
                     "incompatible with existing indexes. Queries may fail or return "
                     "poor results. Re-indexing required.")
            self.warn_label.pack(fill='x', padx=15, pady=(0, 2))
        else:
            self.warn_label.pack_forget()

    def _collect(self):
        """Collect current widget values into cfg dict."""
        for key, (typ, var, extra) in self.widgets.items():
            if typ == 'choice':
                display_map = extra
                display_val = var.get()
                self.cfg[key] = display_map.get(display_val, display_val)
            elif typ == 'toggle':
                self.cfg[key] = var.get()
            elif typ == 'text':
                self.cfg[key] = var.get()

    def _save(self):
        from tkinter import messagebox
        self._collect()
        # Warn if model settings changed and indexes exist
        model_changed = any(self.cfg.get(k) != self.initial_cfg.get(k) for k in DANGEROUS_KEYS)
        if model_changed:
            try:
                import rag
                has_indexes = bool(rag.get_indexes())
            except Exception:
                has_indexes = False
            if has_indexes:
                confirm = messagebox.askyesno("Model Settings Changed",
                    "You changed embedding model settings.\n\n"
                    "NEW embeddings will be INCOMPATIBLE with existing indexes. "
                    "Queries may fail or return poor results. "
                    "Re-indexing will be required.\n\n"
                    "Save anyway?",
                    icon='warning', default='no', parent=self.win)
                if not confirm:
                    return
        save(self.cfg)
        self.result = True
        self.win.destroy()

    def _reset_defaults(self):
        """Reset all widgets to default values."""
        for key, (typ, var, extra) in self.widgets.items():
            default = DEFAULTS[key]
            if typ == 'choice':
                # Find display label for default
                _, _, options, _, _ = SCHEMA[key]
                for oval, olbl in options:
                    if oval == default:
                        var.set(olbl)
                        break
            elif typ == 'toggle':
                var.set(default)
            elif typ == 'text':
                var.set(str(default))

    def _open_acl_dialog(self):
        """Open the Access Control dialog."""
        ACLDialog(self.win)


# ── Access Control Dialogs ─────────────────────────────────────────────────

class ACLDialog:
    """Modal tkinter dialog for managing RAG access control."""

    BG       = '#0f1626'
    BG2      = '#1a2332'
    BG3      = '#243044'
    FG       = '#d4d4dc'
    FG_DIM   = '#7a8599'
    ACCENT   = '#e8781e'
    ICE      = '#5eb8d4'
    WARN_RED = '#e74c3c'
    GREEN    = '#2ecc71'

    def __init__(self, parent):
        import tkinter as tk
        from tkinter import ttk

        self.win = tk.Toplevel(parent)
        self.win.title("RAG Access Control")
        self.win.geometry("620x420")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.configure(bg=self.BG)

        # Enable/disable toggle
        top_frame = tk.Frame(self.win, bg=self.BG)
        top_frame.pack(fill='x', padx=15, pady=(15, 5))

        self.enabled_var = tk.BooleanVar(value=rag_acl.is_enabled())
        tk.Label(top_frame, text="Access Control:", bg=self.BG, fg=self.FG,
                 font=('sans-serif', 11, 'bold')).pack(side='left')
        self.toggle_btn = tk.Button(top_frame,
                                     text="ENABLED" if self.enabled_var.get() else "DISABLED",
                                     bg=self.GREEN if self.enabled_var.get() else self.WARN_RED,
                                     fg='white', font=('sans-serif', 9, 'bold'), relief='flat',
                                     padx=10, command=self._toggle_enabled)
        self.toggle_btn.pack(side='left', padx=10)

        # Treeview for clients
        tree_frame = tk.Frame(self.win, bg=self.BG)
        tree_frame.pack(fill='both', expand=True, padx=15, pady=5)

        style = ttk.Style(self.win)
        style.configure('ACL.Treeview', background=self.BG2, foreground=self.FG,
                         fieldbackground=self.BG2, rowheight=28)
        style.configure('ACL.Treeview.Heading', background=self.BG3, foreground=self.ACCENT,
                         font=('sans-serif', 9, 'bold'))

        columns = ('name', 'key', 'indexes', 'privileges')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                  style='ACL.Treeview', selectmode='browse')
        self.tree.heading('name', text='Name')
        self.tree.heading('key', text='Key')
        self.tree.heading('indexes', text='Indexes')
        self.tree.heading('privileges', text='Privileges')
        self.tree.column('name', width=100)
        self.tree.column('key', width=120)
        self.tree.column('indexes', width=120)
        self.tree.column('privileges', width=240)

        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Buttons
        btn_frame = tk.Frame(self.win, bg=self.BG)
        btn_frame.pack(fill='x', padx=15, pady=(5, 5))

        for text, cmd in [("New Key", self._create_key), ("Edit", self._edit_key),
                          ("Rotate", self._rotate_key), ("Copy Key", self._copy_key),
                          ("Revoke", self._revoke_key)]:
            tk.Button(btn_frame, text=text, command=cmd, bg=self.BG3, fg=self.FG,
                       relief='flat', padx=8, pady=3).pack(side='left', padx=3)

        tk.Button(btn_frame, text="Close", command=self.win.destroy, bg=self.BG3, fg=self.FG,
                   relief='flat', padx=8, pady=3).pack(side='right', padx=3)

        # Status bar
        self.status_var = tk.StringVar(value="")
        tk.Label(self.win, textvariable=self.status_var, bg=self.BG, fg=self.ICE,
                 font=('monospace', 8), anchor='w').pack(fill='x', padx=15, pady=(0, 10))

        self._refresh()

        # Center on parent
        self.win.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.win.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.win.winfo_height()) // 2
        self.win.geometry(f"+{max(0,px)}+{max(0,py)}")

        self.win.wait_window()

    def _refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.acl_clients = rag_acl.list_clients()
        for key, client in self.acl_clients:
            masked = rag_acl.mask_key(key)
            indexes = client.get('indexes', '*')
            idx_str = "*" if indexes == "*" else ", ".join(indexes) if indexes else "(none)"
            privs = ", ".join(client.get('privileges', []))
            self.tree.insert('', 'end', iid=key,
                              values=(client.get('name', '?'), masked, idx_str, privs))

    def _toggle_enabled(self):
        new_state = not rag_acl.is_enabled()
        rag_acl.set_enabled(new_state)
        self.enabled_var.set(new_state)
        self.toggle_btn.config(text="ENABLED" if new_state else "DISABLED",
                                bg=self.GREEN if new_state else self.WARN_RED)
        self.status_var.set("ACL " + ("enabled" if new_state else "disabled"))

    def _get_selected_key(self):
        sel = self.tree.selection()
        if not sel:
            self.status_var.set("No client selected")
            return None
        return sel[0]

    def _create_key(self):
        from tkinter import simpledialog, messagebox
        name = simpledialog.askstring("New API Key", "Client name:", parent=self.win)
        if not name:
            return
        key = rag_acl.create_client(name.strip())
        self._refresh()
        messagebox.showinfo("Key Created",
                             f"Name: {name.strip()}\nKey: {key}\n\n"
                             "Add to MCP client config:\n"
                             f'  "env": {{"RAG_API_KEY": "{key}"}}',
                             parent=self.win)
        self.status_var.set(f"Created key for {name.strip()}")

    def _edit_key(self):
        key = self._get_selected_key()
        if not key:
            return
        client = dict(rag_acl.load()['clients'].get(key, {}))
        if not client:
            return
        ACLEditDialog(self.win, key, client)
        self._refresh()

    def _rotate_key(self):
        from tkinter import messagebox
        key = self._get_selected_key()
        if not key:
            return
        client = rag_acl.load()['clients'].get(key, {})
        if not messagebox.askyesno("Rotate Key",
                                    f"Rotate key for '{client.get('name', '?')}'?\n"
                                    "The old key will stop working immediately.",
                                    parent=self.win):
            return
        new_key = rag_acl.rotate_key(key)
        self._refresh()
        messagebox.showinfo("Key Rotated", f"New key: {new_key}", parent=self.win)
        self.status_var.set(f"Key rotated: {new_key}")

    def _copy_key(self):
        key = self._get_selected_key()
        if not key:
            return
        self.win.clipboard_clear()
        self.win.clipboard_append(key)
        self.status_var.set(f"Copied: {key}")

    def _revoke_key(self):
        from tkinter import messagebox
        key = self._get_selected_key()
        if not key:
            return
        client = rag_acl.load()['clients'].get(key, {})
        if not messagebox.askyesno("Revoke Key",
                                    f"Revoke key for '{client.get('name', '?')}'?\n"
                                    "This cannot be undone.",
                                    parent=self.win):
            return
        rag_acl.revoke_client(key)
        self._refresh()
        self.status_var.set("Key revoked")


class ACLEditDialog:
    """Sub-dialog for editing a client's name, indexes, and privileges."""

    BG  = '#0f1626'
    BG2 = '#1a2332'
    BG3 = '#243044'
    FG  = '#d4d4dc'

    def __init__(self, parent, key, client):
        import tkinter as tk

        self.key = key
        self.win = tk.Toplevel(parent)
        self.win.title(f"Edit: {client.get('name', '?')}")
        self.win.geometry("400x350")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.configure(bg=self.BG)

        pad = {'padx': 15, 'pady': 5}

        tk.Label(self.win, text="Name:", bg=self.BG, fg=self.FG).pack(anchor='w', **pad)
        self.name_var = tk.StringVar(value=client.get('name', ''))
        tk.Entry(self.win, textvariable=self.name_var, width=40).pack(anchor='w', padx=15)

        tk.Label(self.win, text="Indexes (* for all, or comma-separated):",
                 bg=self.BG, fg=self.FG).pack(anchor='w', **pad)
        indexes = client.get('indexes', '*')
        idx_str = "*" if indexes == "*" else ", ".join(indexes)
        self.idx_var = tk.StringVar(value=idx_str)
        tk.Entry(self.win, textvariable=self.idx_var, width=40).pack(anchor='w', padx=15)

        tk.Label(self.win, text="Privileges:", bg=self.BG, fg=self.FG).pack(anchor='w', **pad)
        self.priv_vars = {}
        priv_frame = tk.Frame(self.win, bg=self.BG)
        priv_frame.pack(anchor='w', padx=15)
        current_privs = set(client.get('privileges', []))
        for priv in rag_acl.ALL_PRIVILEGES:
            var = tk.BooleanVar(value=priv in current_privs)
            tk.Checkbutton(priv_frame, text=priv, variable=var, bg=self.BG, fg=self.FG,
                            selectcolor=self.BG2, activebackground=self.BG).pack(anchor='w')
            self.priv_vars[priv] = var

        btn_frame = tk.Frame(self.win, bg=self.BG)
        btn_frame.pack(fill='x', padx=15, pady=15)
        tk.Button(btn_frame, text="Cancel", command=self.win.destroy, bg=self.BG3, fg=self.FG,
                   relief='flat', padx=10).pack(side='right', padx=3)
        tk.Button(btn_frame, text="Save", command=self._save, bg=self.BG3, fg=self.FG,
                   relief='flat', padx=10).pack(side='right', padx=3)

        self.win.wait_window()

    def _save(self):
        name = self.name_var.get().strip() or None
        idx_str = self.idx_var.get().strip()
        if idx_str == "*":
            indexes = "*"
        else:
            indexes = [i.strip() for i in idx_str.split(",") if i.strip()]
        privileges = [p for p, v in self.priv_vars.items() if v.get()]
        rag_acl.update_client(self.key, name=name, indexes=indexes, privileges=privileges)
        self.win.destroy()


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    """Run TUI settings editor from command line."""
    tui = SettingsTUI()
    if tui.run():
        print("Settings saved.", file=sys.stderr)
    else:
        print("Cancelled.", file=sys.stderr)


if __name__ == '__main__':
    main()
