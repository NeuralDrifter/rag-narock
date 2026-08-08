#!/usr/bin/env python3
"""
RAG Editor — slim re-export shim.

Full EditorTUI (for no-GUI/terminal) from tui/editor.py
Full EditorDialog (for GUI/desktop) from gui/dialogs/editor_dialog.py
"""

import logging
from tui.editor import EditorTUI

try:
    from gui.dialogs.editor_dialog import EditorDialog
except Exception as e:
    logging.getLogger(__name__).debug("EditorDialog import failed: %s", e)
    EditorDialog = None


