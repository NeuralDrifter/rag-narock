import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("fitz") is None,
    reason="PyMuPDF not installed",
)
def test_import_fitz_suppresses_mupdf_messages():
    from ingestion.fitz_utils import import_fitz

    fitz = import_fitz()
    tools = fitz.TOOLS

    assert callable(tools.mupdf_display_errors)
    assert callable(tools.mupdf_display_warnings)
    assert tools.mupdf_display_errors() is False
    assert tools.mupdf_display_warnings() is False