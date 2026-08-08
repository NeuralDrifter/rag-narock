"""Shared PyMuPDF import with MuPDF stderr noise suppressed."""

_fitz = None


def _suppress_mupdf_messages(tools) -> None:
    """Disable MuPDF error/warning display (must use method calls, not assignment)."""
    try:
        display_errors = getattr(tools, "mupdf_display_errors", None)
        if callable(display_errors):
            display_errors(False)
        display_warnings = getattr(tools, "mupdf_display_warnings", None)
        if callable(display_warnings):
            display_warnings(False)
    except Exception:
        pass


def import_fitz():
    """Return the fitz module with MuPDF messages suppressed."""
    global _fitz
    if _fitz is not None:
        return _fitz
    import fitz

    _suppress_mupdf_messages(fitz.TOOLS)
    _fitz = fitz
    return fitz