"""
tui/editor.py — Full EditorTUI (curses) ported from original for complete functionality.
Includes: list indexes with details/integrity, drill to sources/entries counts,
delete, remove source, lock/unlock, rename, integrity hash, export specific source (e),
export whole index (E), confirm dialogs, rename input, scroll, status.
TUI for no-GUI/terminal use.
"""

import os
import re
import sys
import logging

from core.constants import LOCK_FILENAME
from core.index_manager import (
    get_indexes,
    resolve_index_dir,
    get_index_sources,
    delete_index,
    remove_source_from_index,
    update_source_in_index,
    export_source,
    export_index,
    is_index_locked,
)
from core.integrity import check_index_integrity, save_index_integrity
from storage.base import get_index_meta_with_defaults
from rag_backends import detect_backend, get_backend
from tui.common import BANNER, BANNER_H, BANNER_W

logger = logging.getLogger(__name__)


class EditorTUI:
    """Full-screen curses index editor with two-mode navigation.
    Ported and adapted to modular structure.
    """

    MODE_INDEX = 0
    MODE_SOURCE = 1
    MODE_READER = 2

    def __init__(self):
        self.mode = self.MODE_INDEX
        self.cursor = 0
        self.scroll_offset = 0
        self.indexes = []
        self.sources = []  # list of (source_name, chunk_count)
        self.selected_index = None
        self.reader_text = []
        self.reader_scroll = 0
        self.reader_source = ""
        self.status_msg = ""
        self.status_color = 6  # yellow
        self.has_colors = False
        self.has_flame = False
        self.confirming = False
        self.confirm_cursor = 1  # 0=Yes, 1=No (default No)
        self.confirm_action = None
        self.confirm_msg_lines = []
        self.renaming = False
        self.rename_buf = ""
        self.rename_cur = 0  # cursor position in buffer
        self.rename_old = ""  # original index name
        self._refresh_indexes()

    def _refresh_indexes(self):
        self.indexes = []
        for name in get_indexes():
            index_dir = resolve_index_dir(name)
            meta = get_index_meta_with_defaults(index_dir)
            locked = is_index_locked(name)
            n_chunks = meta.get('n_chunks', 0)
            n_files = meta.get('n_files', 0)
            storage = meta.get('storage_backend', 'faiss')
            emb_model = meta.get('embedding_model', '?')
            emb_backend = meta.get('embedding_backend', 'local')
            integrity = check_index_integrity(index_dir)
            tampered = not integrity['ok'] and not integrity.get('untracked')
            unverified = integrity.get('untracked', False)
            self.indexes.append({
                'name': name, 'n_chunks': n_chunks,
                'n_files': n_files, 'locked': locked,
                'storage': storage, 'emb_model': emb_model,
                'emb_backend': emb_backend, 'tampered': tampered,
                'unverified': unverified,
            })

    def _refresh_sources(self):
        if not self.selected_index:
            self.sources = []
            return
        src_dict = get_index_sources(self.selected_index)
        self.sources = sorted(src_dict.items(), key=lambda x: x[0])

    def run(self):
        """Launch the interactive editor. Returns when user quits."""
        try:
            import curses, struct, time, shutil
            orig_rows, orig_cols = 0, 0
            if os.name != 'nt':
                try:
                    import fcntl, termios
                    packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b'\x00' * 8)
                    orig_rows, orig_cols = struct.unpack('HHHH', packed)[:2]
                except Exception as e:
                    logger.debug("Terminal size query failed: %s", e)
            if orig_rows == 0 or orig_cols == 0:
                try:
                    ts = shutil.get_terminal_size()
                    orig_rows, orig_cols = ts.lines, ts.columns
                except Exception:
                    pass

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
        except Exception as e:
            print(f"TUI unavailable: {e}")
            print("TUI is the option for terminals and systems with no visual/GUI interface.")
            print("On Windows: pip install windows-curses   (then re-run for full interactive TUI)")
            print("GUI version (for systems with a desktop/GUI): python rag.py editor gui")
            print("\nCurrent indexes (non-interactive view):")
            self._print_list_fallback()
            print("\nIn a working TUI: arrows, Enter=drill, d=del/remove, e/E=export, l/u=lock, r=rename, h=hash, q=quit")
            return False

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
        except Exception as e:
            logger.debug("Color init failed: %s", e)
            self.has_colors = False

        while True:
            self._draw()
            ch = stdscr.getch()

            if self.renaming:
                self._handle_rename(ch)
            elif self.confirming:
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
        h, w = self.stdscr.getmaxyx()

        show_banner = (h >= BANNER_H + 18 and w >= BANNER_W + 6)
        content_start = 1 + BANNER_H + 1 if show_banner else 2
        vis_rows = h - content_start - 4

        if self.mode == self.MODE_READER:
            if ch in (ord('q'), 27):  # q or Esc
                self.mode = self.MODE_SOURCE
                self.status_msg = ""
                return True
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                self.mode = self.MODE_SOURCE
                self.status_msg = ""
                return True

            if ch == ord('e'):
                if is_index_locked(self.selected_index):
                    self.status_msg = f"Index '{self.selected_index}' is LOCKED. Go back and press 'u' to unlock."
                    self.status_color = 5  # red
                    return True

                import tempfile
                import subprocess
                import sys
                import curses

                # Get editor
                editor = os.environ.get('EDITOR') or os.environ.get('VISUAL')
                if not editor:
                    if sys.platform == 'win32':
                        editor = 'notepad'
                    else:
                        editor = 'nano'

                # Combine reader lines to full text
                old_text = "\n".join(self.reader_text)

                # Write text to a temporary file in the system temp directory or workspace temp
                temp_dir = os.path.join(os.path.dirname(resolve_index_dir(self.selected_index)), ".tmp")
                os.makedirs(temp_dir, exist_ok=True)
                
                fd, temp_path = tempfile.mkstemp(suffix=".txt", dir=temp_dir)
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(old_text)

                    # Temporarily suspend curses mode
                    curses.def_shell_mode()
                    curses.endwin()

                    # Call editor
                    try:
                        subprocess.call([editor, temp_path])
                    except Exception as e:
                        self.stdscr.refresh()
                        self.status_msg = f"Failed to launch editor: {e}"
                        self.status_color = 5  # red
                        return True

                    # Restore curses
                    self.stdscr.refresh()

                    # Read new text
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        new_text = f.read()

                    # If it has changed, perform save and re-index
                    if new_text != old_text:
                        h, w = self.stdscr.getmaxyx()
                        self.stdscr.erase()
                        self.stdscr.addstr(h // 2, max(0, (w - 45) // 2), "Re-vectorizing and re-indexing document...")
                        self.stdscr.refresh()
                        
                        try:
                            update_source_in_index(self.selected_index, self.reader_source, new_text)
                            self.reader_text = new_text.splitlines()
                            self.reader_scroll = 0
                            self._refresh_sources()
                            self.status_msg = "Source successfully updated and re-indexed."
                            self.status_color = 4  # green
                        except Exception as e:
                            self.status_msg = f"Re-indexing failed: {e}"
                            self.status_color = 5  # red
                    else:
                        self.status_msg = "No changes made."
                        self.status_color = 6  # yellow

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    try:
                        os.rmdir(temp_dir)
                    except:
                        pass

                return True

            num_lines = len(self.reader_text)
            if ch == curses.KEY_UP:
                self.reader_scroll = max(0, self.reader_scroll - 1)
                return True
            if ch == curses.KEY_DOWN:
                self.reader_scroll = min(max(0, num_lines - vis_rows), self.reader_scroll + 1)
                return True
            if ch == curses.KEY_PPAGE:  # Page Up
                self.reader_scroll = max(0, self.reader_scroll - vis_rows)
                return True
            if ch == curses.KEY_NPAGE:  # Page Down
                self.reader_scroll = min(max(0, num_lines - vis_rows), self.reader_scroll + vis_rows)
                return True
            return True

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
                if idx.get('tampered'):
                    self.status_msg = "WARNING: Integrity mismatch — data may have been modified outside RAG-Narock. Press Backspace, then 'h' to rehash."
                    self.status_color = 5  # red
                elif idx.get('unverified'):
                    self.status_msg = "UNVERIFIED: Press Backspace, then 'h' to compute integrity hashes for this index."
                    self.status_color = 6  # yellow
                else:
                    self.status_msg = ""
                return True
            elif self.mode == self.MODE_SOURCE and n > 0:
                src_name, src_count = self.sources[self.cursor]
                self.stdscr.erase()
                self.stdscr.addstr(h // 2, max(0, (w - 20) // 2), "Loading document...")
                self.stdscr.refresh()
                try:
                    index_dir = resolve_index_dir(self.selected_index)
                    backend_type = detect_backend(index_dir)
                    backend = get_backend(index_dir, backend_type)
                    doc = backend.get_document(src_name)
                    if doc and doc.get('full_text'):
                        text = doc['full_text']
                    else:
                        text = "No text available for this source."
                except Exception as e:
                    text = f"Error loading document: {e}"
                self.reader_text = text.splitlines()
                self.reader_scroll = 0
                self.reader_source = src_name
                self.mode = self.MODE_READER
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
                if is_index_locked(self.selected_index):
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

        if ch == ord('e') and self.mode == self.MODE_SOURCE:
            if not self.sources:
                return True
            src_name, src_count = self.sources[self.cursor]
            output_dir = os.path.expanduser(f"~/rag-export/{self.selected_index}")
            try:
                result = export_source(self.selected_index, src_name, output_dir)
                self.status_msg = f"Exported '{src_name}' -> {output_dir}"
                self.status_color = 4  # green
            except Exception as e:
                self.status_msg = f"Export failed: {e}"
                self.status_color = 5  # red
            return True

        if ch == ord('E') and self.mode == self.MODE_SOURCE:
            output_dir = os.path.expanduser(f"~/rag-export/{self.selected_index}")
            try:
                result = export_index(self.selected_index, output_dir)
                self.status_msg = f"Exported {result['sources_exported']} source(s) -> {output_dir}"
                self.status_color = 4  # green
                if result['skipped']:
                    self.status_msg += f" ({len(result['skipped'])} skipped)"
            except Exception as e:
                self.status_msg = f"Export all failed: {e}"
                self.status_color = 5  # red
            return True

        if ch == ord('u') and self.mode == self.MODE_INDEX and n > 0:
            idx = self.indexes[self.cursor]
            if idx['locked']:
                lock_file = os.path.join(resolve_index_dir(idx['name']), LOCK_FILENAME)
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
                lock_file = os.path.join(resolve_index_dir(idx['name']), LOCK_FILENAME)
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

        if ch == ord('h') and self.mode == self.MODE_INDEX and n > 0:
            idx = self.indexes[self.cursor]
            try:
                index_dir = resolve_index_dir(idx['name'])
                save_index_integrity(index_dir)
                self._refresh_indexes()
                self._clamp_cursor()
                self.status_msg = f"Integrity hashes computed for '{idx['name']}'"
                self.status_color = 4  # green
            except Exception as e:
                self.status_msg = f"ERROR: {e}"
                self.status_color = 5
            return True

        if ch == ord('r') and self.mode == self.MODE_INDEX and n > 0:
            idx = self.indexes[self.cursor]
            if idx['locked']:
                self.status_msg = f"Index '{idx['name']}' is LOCKED. Press 'u' to unlock first."
                self.status_color = 5
                return True
            self.renaming = True
            self.rename_old = idx['name']
            self.rename_buf = idx['name']
            self.rename_cur = len(self.rename_buf)
            self.status_msg = ""
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
                    delete_index(action[1], force=True)
                    self._refresh_indexes()
                    self._clamp_cursor()
                    self.status_msg = f"Deleted index '{action[1]}'"
                    self.status_color = 4
                elif action[0] == 'remove_source':
                    result = remove_source_from_index(action[1], action[2])
                    self._refresh_sources()
                    self._refresh_indexes()  # update counts
                    self._clamp_cursor()
                    self.status_msg = (f"Removed '{action[2]}' ({result['removed_chunks']:,} chunks). "
                                       f"{result['remaining_chunks']:,} chunks remain.")
                    self.status_color = 4
            except Exception as e:
                self.status_msg = f"ERROR: {e}"
                self.status_color = 5

    def _handle_rename(self, ch):
        import curses

        if ch == 27:  # Esc — cancel
            self.renaming = False
            self.status_msg = "Rename cancelled."
            self.status_color = 6
            return

        if ch in (curses.KEY_ENTER, 10, 13):
            self.renaming = False
            new_name = self.rename_buf.strip()
            if not new_name or new_name == self.rename_old:
                self.status_msg = "Rename cancelled." if not new_name else "Name unchanged."
                self.status_color = 6
                return
            if not re.match(r'^[A-Za-z0-9_-]+$', new_name):
                self.status_msg = "Invalid name. Use letters, digits, _ or - only."
                self.status_color = 5
                return
            existing = {ix['name'] for ix in self.indexes}
            if new_name in existing:
                self.status_msg = f"Index '{new_name}' already exists."
                self.status_color = 5
                return
            try:
                old_dir = resolve_index_dir(self.rename_old)
                new_dir = os.path.join(os.path.dirname(old_dir), new_name)
                os.rename(old_dir, new_dir)
                self._refresh_indexes()
                self._clamp_cursor()
                self.status_msg = f"Renamed '{self.rename_old}' -> '{new_name}'"
                self.status_color = 4
            except Exception as e:
                self.status_msg = f"ERROR: {e}"
                self.status_color = 5
            return

        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if self.rename_cur > 0:
                self.rename_buf = self.rename_buf[:self.rename_cur-1] + self.rename_buf[self.rename_cur:]
                self.rename_cur -= 1
            return

        if ch == curses.KEY_DC:
            if self.rename_cur < len(self.rename_buf):
                self.rename_buf = self.rename_buf[:self.rename_cur] + self.rename_buf[self.rename_cur+1:]
            return

        if ch == curses.KEY_LEFT:
            self.rename_cur = max(0, self.rename_cur - 1)
            return

        if ch == curses.KEY_RIGHT:
            self.rename_cur = min(len(self.rename_buf), self.rename_cur + 1)
            return

        if ch == curses.KEY_HOME or ch == 1:  # Home or Ctrl-A
            self.rename_cur = 0
            return

        if ch == curses.KEY_END or ch == 5:  # End or Ctrl-E
            self.rename_cur = len(self.rename_buf)
            return

        # Printable character
        if 32 <= ch <= 126:
            c = chr(ch)
            self.rename_buf = self.rename_buf[:self.rename_cur] + c + self.rename_buf[self.rename_cur:]
            self.rename_cur += 1

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
            title_attr = self._attr(3, curses.A_BOLD)
        elif self.mode == self.MODE_READER:
            title = f"READING: {self.reader_source}"
            title_attr = self._attr(3, curses.A_BOLD)
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
            title_attr = self._attr(3, curses.A_BOLD)

        try:
            stdscr.addstr(row, 3, title[:w-6], title_attr)
        except curses.error:
            pass

        # Separator
        row = content_start + 1
        sep = "-" * (w - 4)
        try:
            stdscr.addstr(row, 2, sep[:w-4], border_attr)
        except curses.error:
            pass

        # Items / Reader Viewport
        if self.mode == self.MODE_READER:
            lines = self.reader_text
            num_lines = len(lines)
            vis = h - content_start - 5
            
            if num_lines > vis:
                percent = int((self.reader_scroll / max(1, num_lines - vis)) * 100)
                indicator = f"Line {self.reader_scroll + 1}/{num_lines} ({percent}%)"
                try:
                    stdscr.addstr(content_start + 2, w - len(indicator) - 4, indicator, self._attr(6))
                except curses.error:
                    pass

            for vi in range(vis):
                line_idx = self.reader_scroll + vi
                if line_idx >= num_lines:
                    break
                r = content_start + 2 + vi
                if r >= h - 4:
                    break
                try:
                    stdscr.addstr(r, 4, lines[line_idx][:w-8])
                except curses.error:
                    pass
            if num_lines == 0:
                try:
                    stdscr.addstr(content_start + 2, 5, "No text available.", self._attr(6))
                except curses.error:
                    pass
        else:
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
                    if ix.get('tampered'):
                        integrity_str = " [TAMPERED]"
                    elif ix.get('unverified'):
                        integrity_str = " [UNVERIFIED]"
                    else:
                        integrity_str = ""
                    extra = f" [{ix.get('storage', 'faiss')}, {ix.get('emb_model', '?')}]"
                    line = f"{ix['name']}  ({ix['n_chunks']:,}ch, {ix['n_files']}f){lock_str}{integrity_str}{extra}"
                    if selected:
                        lbl_attr = self._attr(2)
                    elif ix.get('tampered'):
                        lbl_attr = self._attr(5)  # red for tampered
                    elif ix.get('unverified'):
                        lbl_attr = self._attr(6)  # yellow for unverified
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
            help_text = "  Enter=view  d=delete  r=rename  l=lock  u=unlock  h=hash  q/Esc=exit"
        elif self.mode == self.MODE_SOURCE:
            help_text = "  Enter=read  d=remove  e=export  E=export all  Backspace=back  q/Esc=exit"
        else: # MODE_READER
            help_text = "  Up/Down=scroll  PgUp/PgDn=page scroll  e=edit  Backspace=back  q/Esc=exit"
        try:
            stdscr.addstr(help_row, 2, help_text[:w-4], self._attr(6))
        except curses.error:
            pass

        # Overlays
        if self.confirming:
            self._draw_confirm(h, w)
        if self.renaming:
            self._draw_rename(h, w)
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

    def _draw_rename(self, h, w):
        import curses
        box_w = min(56, w - 4)
        box_h = 7
        box_y = max(1, (h - box_h) // 2)
        box_x = max(1, (w - box_w) // 2)

        accent_attr = self._attr(6, curses.A_BOLD)
        border_attr = self._attr(7, curses.A_DIM)

        # Draw box
        top = "+" + "=" * (box_w - 2) + "+"
        bot = "+" + "=" * (box_w - 2) + "+"
        mid = "|" + " " * (box_w - 2) + "|"

        try:
            self.stdscr.addstr(box_y, box_x, top[:box_w], accent_attr)
            for r in range(1, box_h - 1):
                self.stdscr.addstr(box_y + r, box_x, mid[:box_w], accent_attr)
            self.stdscr.addstr(box_y + box_h - 1, box_x, bot[:box_w], accent_attr)
        except curses.error:
            pass

        # Title
        title = f" Rename '{self.rename_old}' "
        try:
            self.stdscr.addstr(box_y, box_x + 2, title[:box_w-4], accent_attr)
        except curses.error:
            pass

        # Input field
        field_w = box_w - 6
        buf = self.rename_buf
        # Scroll the visible portion if buffer is wider than field
        vis_start = max(0, self.rename_cur - field_w + 1)
        vis_text = buf[vis_start:vis_start + field_w]
        cursor_x = self.rename_cur - vis_start

        try:
            self.stdscr.addstr(box_y + 2, box_x + 3, " " * field_w, curses.A_UNDERLINE)
            self.stdscr.addstr(box_y + 2, box_x + 3, vis_text[:field_w], curses.A_UNDERLINE | curses.A_BOLD)
            # Draw cursor
            if cursor_x < field_w:
                cursor_ch = buf[self.rename_cur] if self.rename_cur < len(buf) else " "
                self.stdscr.addstr(box_y + 2, box_x + 3 + cursor_x, cursor_ch,
                                   self._attr(2, curses.A_BOLD))
        except curses.error:
            pass

        # Hint
        hint = "Enter=confirm  Esc=cancel"
        try:
            self.stdscr.addstr(box_y + 4, box_x + 3, hint[:box_w-6], self._attr(6))
        except curses.error:
            pass

    def _print_list_fallback(self):
        """Non-interactive fallback list (used when curses unavailable)."""
        if not self.indexes:
            self._refresh_indexes()
        if not self.indexes:
            print("  (no indexes)")
            return
        for ix in self.indexes:
            lock_str = " [LOCKED]" if ix['locked'] else ""
            integrity_str = ""
            if ix.get('tampered'):
                integrity_str = " [TAMPERED]"
            elif ix.get('unverified'):
                integrity_str = " [UNVERIFIED]"
            extra = f" [{ix.get('storage', '?')}, {ix.get('emb_model', '?')}]"
            print(f"  {ix['name']}  ({ix['n_chunks']:,}ch, {ix['n_files']}f){lock_str}{integrity_str}{extra}")
