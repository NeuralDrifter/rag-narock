import config.settings as rag_settings
from gui.book_adder import _build_gui_ocr_options, _format_ocr_status
from indexing.embedder import _resolve_embedding_device
from ingestion.ocr_options import OcrOptions


def test_settings_get_accepts_single_key_only():
    value = rag_settings.get('storage_backend')
    assert value in ('faiss', 'sqlite-vec', 'sqlite-doc')


def test_resolve_embedding_device_honors_gpu_flag():
    assert _resolve_embedding_device(False) == 'cpu'
    assert _resolve_embedding_device(True) in ('cpu', 'cuda', 'xpu')


def test_build_gui_ocr_options_disables_force_when_no_ocr():
    opts = _build_gui_ocr_options({'no_ocr': True, 'force_ocr': True})
    assert opts.disabled is True
    assert opts.force is False


def test_format_ocr_status_force():
    status = _format_ocr_status(OcrOptions(force=True))
    assert 'force-OCR' in status


def test_easyocr_gpu_available_returns_bool():
    from ingestion.ocr import _easyocr_gpu_available
    assert isinstance(_easyocr_gpu_available(), bool)