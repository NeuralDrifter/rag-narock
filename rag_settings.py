#!/usr/bin/env python3
"""
RAG Settings — slim shim (Ch 11 boundaries).

Pure logic in config/settings.py
TUI moved to tui/
GUI dialogs in gui/dialogs/
"""

import logging
import sys
from config.settings import *
from tui.settings import SettingsTUI

try:
    from gui.dialogs.settings_dialog import SettingsDialog
except Exception as e:
    logging.getLogger(__name__).debug("SettingsDialog import failed: %s", e)
    SettingsDialog = None

try:
    from gui.dialogs.acl_dialog import ACLDialog
except Exception as e:
    logging.getLogger(__name__).debug("ACLDialog import failed: %s", e)
    ACLDialog = None


def main():
    """Run TUI settings editor from command line."""
    tui = SettingsTUI()
    if tui.run():
        print("Settings saved.", file=sys.stderr)
    else:
        print("Cancelled.", file=sys.stderr)


if __name__ == '__main__':
    main()
