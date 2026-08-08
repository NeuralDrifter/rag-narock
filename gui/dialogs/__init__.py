"""
gui/dialogs — modal dialogs for GUI.
"""
from .caption_picker import CaptionPickerDialog
from .settings_dialog import SettingsDialog
try:
    from .editor_dialog import EditorDialog
except Exception:
    EditorDialog = None

