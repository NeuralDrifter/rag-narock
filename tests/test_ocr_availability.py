from PIL import Image


def test_no_ocr_backend_warning_is_emitted_once(monkeypatch, capsys):
    import ingestion.ocr as ocr

    monkeypatch.setattr(ocr, "HAS_TESSERACT", False)
    monkeypatch.setattr(ocr, "HAS_EASYOCR", False)
    ocr._warned_messages.clear()

    img = Image.new("RGB", (4, 4), "white")
    assert ocr._ocr_single(img) == ""
    assert ocr._ocr_single(img) == ""

    captured = capsys.readouterr()
    assert captured.err.count("No OCR backend available") == 1


def test_ocr_unavailable_message_has_fix_and_bypass_options():
    from ingestion.ocr import ocr_unavailable_message

    message = ocr_unavailable_message()
    assert "python -m pip install pytesseract" in message
    assert "python -m pip install easyocr" in message
    assert "disable OCR" in message


def test_tesseract_resolves_common_windows_path(monkeypatch):
    import ingestion.ocr as ocr

    expected = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr.os.path, "exists", lambda path: path == expected)

    assert ocr._resolve_tesseract_cmd() == expected


def test_easyocr_without_gpu_falls_back_to_tesseract(monkeypatch):
    import ingestion.ocr as ocr

    monkeypatch.setattr(ocr, "HAS_EASYOCR", True)
    monkeypatch.setattr(ocr, "HAS_TESSERACT", True)
    monkeypatch.setattr(ocr, "_easyocr_gpu_available", lambda: False)
    ocr._warned_messages.clear()

    assert ocr.get_ocr_backend(type("Opts", (), {"backend": "easyocr"})()) == "tesseract"
