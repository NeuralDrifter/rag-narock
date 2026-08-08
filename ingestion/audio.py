"""
ingestion/audio.py — audio extraction / transcription.
Full logic housed here. Clean import surface.
"""

import os
import sys
import gc
import logging
import importlib.util
import shutil
import config.settings as rag_settings
from ingestion.ocr import sanitize_text
from indexing.embedder import _detect_best_device, _prepare_env_for_device, _release_gpu_memory

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.aac'}

# ── Audio transcription backend availability detection ──

def _check_ffmpeg():
    return shutil.which('ffmpeg') is not None

def _check_faster_whisper():
    return importlib.util.find_spec('faster_whisper') is not None

def _check_openai_whisper():
    return importlib.util.find_spec('whisper') is not None

def _check_whisper_cli():
    return shutil.which('whisper') is not None

HAS_FFMPEG = _check_ffmpeg()
HAS_FASTER_WHISPER = _check_faster_whisper()
HAS_OPENAI_WHISPER = _check_openai_whisper()
HAS_WHISPER_CLI = _check_whisper_cli()

_transcription_backend_override = None
_transcription_model_override = None
_transcription_lang_override = None
_transcription_disabled_override = None

_faster_whisper_model = None
_faster_whisper_model_name = None
_faster_whisper_device = None
_openai_whisper_model = None
_openai_whisper_model_name = None
_openai_whisper_device = None

def transcription_disabled() -> bool:
    """Check if transcription is globally disabled via settings or runtime override."""
    if _transcription_disabled_override is not None:
        return _transcription_disabled_override
    return rag_settings.get('disable_transcription')

def get_transcription_model():
    if _transcription_model_override is not None:
        return _transcription_model_override
    return rag_settings.get('transcription_model')

def get_transcription_language():
    if _transcription_lang_override is not None:
        return _transcription_lang_override
    return rag_settings.get('transcription_language') or None

def get_transcription_device():
    device = rag_settings.get('transcription_device')
    if device == 'auto':
        return _detect_best_device()
    return device

def get_transcription_backend():
    """Return active transcription backend, respecting settings and availability."""
    device = get_transcription_device()

    def _supports(backend_name, requested_device):
        if backend_name == 'faster-whisper':
            return requested_device in ('cpu', 'cuda')
        if backend_name == 'whisper':
            return requested_device in ('cpu', 'cuda', 'xpu')
        if backend_name == 'whisper-cli':
            return requested_device in ('cpu', 'cuda')
        return False

    def _fallback():
        for backend_name, available in (
            ('whisper', HAS_OPENAI_WHISPER),
            ('faster-whisper', HAS_FASTER_WHISPER),
            ('whisper-cli', HAS_WHISPER_CLI),
        ):
            if available and _supports(backend_name, device):
                return backend_name
        for backend_name, available in (
            ('faster-whisper', HAS_FASTER_WHISPER),
            ('whisper', HAS_OPENAI_WHISPER),
            ('whisper-cli', HAS_WHISPER_CLI),
        ):
            if available:
                return backend_name
        return None

    if _transcription_backend_override is not None:
        requested = _transcription_backend_override
    else:
        requested = rag_settings.get('transcription_backend')

    if requested == 'auto':
        return _fallback()

    if requested == 'faster-whisper' and HAS_FASTER_WHISPER and _supports(requested, device):
        return requested
    if requested == 'whisper' and HAS_OPENAI_WHISPER and _supports(requested, device):
        return requested
    if requested == 'whisper-cli' and HAS_WHISPER_CLI and _supports(requested, device):
        return requested

    if requested != 'auto':
        print(f"WARNING: Transcription backend '{requested}' is unavailable for device '{device}', falling back to auto-detect", file=sys.stderr)
    return _fallback()

def _get_faster_whisper_model(model_name, device):
    global _faster_whisper_model, _faster_whisper_model_name, _faster_whisper_device
    if (_faster_whisper_model is not None and
            _faster_whisper_model_name == model_name and
            _faster_whisper_device == device):
        return _faster_whisper_model
    from faster_whisper import WhisperModel
    compute_type = 'float16' if device == 'cuda' else 'int8'
    _faster_whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
    _faster_whisper_model_name = model_name
    _faster_whisper_device = device
    return _faster_whisper_model

def _get_openai_whisper_model(model_name, device):
    global _openai_whisper_model, _openai_whisper_model_name, _openai_whisper_device
    if (_openai_whisper_model is not None and
            _openai_whisper_model_name == model_name and
            _openai_whisper_device == device):
        return _openai_whisper_model
    import whisper
    _openai_whisper_model = whisper.load_model(model_name, device=device)
    _openai_whisper_model_name = model_name
    _openai_whisper_device = device
    return _openai_whisper_model

def _unload_transcription_model():
    global _faster_whisper_model, _faster_whisper_model_name, _faster_whisper_device
    global _openai_whisper_model, _openai_whisper_model_name, _openai_whisper_device
    was_gpu = False
    if _faster_whisper_model is not None:
        was_gpu = was_gpu or (_faster_whisper_device not in (None, 'cpu'))
        try:
            if hasattr(_faster_whisper_model, 'model'):
                _faster_whisper_model.model.to("cpu")
        except Exception:
            pass
        del _faster_whisper_model
        _faster_whisper_model = None
        _faster_whisper_model_name = None
        _faster_whisper_device = None
    if _openai_whisper_model is not None:
        was_gpu = was_gpu or (_openai_whisper_device not in (None, 'cpu'))
        try:
            _openai_whisper_model.cpu()
        except Exception:
            pass
        del _openai_whisper_model
        _openai_whisper_model = None
        _openai_whisper_model_name = None
        _openai_whisper_device = None
    if was_gpu:
        _release_gpu_memory()
        _prepare_env_for_device('cpu')
    else:
        gc.collect()

def transcribe_audio(path: str):
    """Transcribe an audio file locally with an available Whisper backend."""
    if transcription_disabled():
        return (None, False)
    if not HAS_FFMPEG:
        print("WARNING: ffmpeg is required for audio transcription but was not found", file=sys.stderr)
        return (None, False)

    backend = get_transcription_backend()
    if backend is None:
        print("WARNING: No audio transcription backend available", file=sys.stderr)
        return (None, False)

    model_name = get_transcription_model()
    language = get_transcription_language()
    device = get_transcription_device()
    backend_device = device
    if backend in ('faster-whisper', 'whisper-cli') and device == 'xpu':
        print(f"WARNING: {backend} does not support XPU, falling back to CPU for transcription", file=sys.stderr)
        backend_device = 'cpu'

    if backend == 'faster-whisper':
        model = _get_faster_whisper_model(model_name, backend_device)
        segments, _ = model.transcribe(path, language=language, vad_filter=True)
        text = ' '.join(seg.text.strip() for seg in segments if seg.text.strip())
    elif backend == 'whisper':
        _prepare_env_for_device(backend_device)
        model = _get_openai_whisper_model(model_name, backend_device)
        result = model.transcribe(path, language=language, fp16=(backend_device == 'cuda'))
        text = result.get('text', '')
    else:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory(prefix='rag_whisper_') as tmpdir:
            cmd = ['whisper', path, '--model', model_name, '--task', 'transcribe',
                   '--output_format', 'txt', '--output_dir', tmpdir, '--fp16', 'False']
            if language:
                cmd.extend(['--language', language])
            if backend_device in ('cpu', 'cuda'):
                cmd.extend(['--device', backend_device])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "whisper CLI failed")
            out_path = os.path.join(tmpdir, os.path.splitext(os.path.basename(path))[0] + '.txt')
            if not os.path.exists(out_path):
                raise RuntimeError("whisper CLI did not produce a transcript")
            with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()

    text = sanitize_text(text)
    return (text, False) if text.strip() else (None, False)

def extract_audio(path: str):
    """Returns (text, False) by transcribing supported audio/video containers."""
    return transcribe_audio(path)

# Also expose for legacy_commands convenience
__all__ = ['extract_audio', 'transcribe_audio', '_get_faster_whisper_model', '_get_openai_whisper_model', 'AUDIO_EXTENSIONS']
