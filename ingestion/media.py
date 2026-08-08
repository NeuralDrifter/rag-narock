"""
ingestion/media.py — media metadata, subtitle-aware extraction.
"""

import os
import sys
import json
import config.settings as rag_settings
from ingestion.ocr import sanitize_text
from ingestion.audio import transcribe_audio, HAS_FFMPEG, AUDIO_EXTENSIONS
from ingestion.subtitles import (
    load_subtitle_file,
    extract_embedded_subtitle_candidate,
    download_external_subtitle,
    gather_video_caption_candidates,
    _format_ts,
    VIDEO_EXTENSIONS,
)
from ingestion.documents import detect_document_type

MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

def probe_media_metadata(path):
    meta = {'title': '', 'duration': 0.0, 'streams': []}
    if not HAS_FFMPEG:
        return meta
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-print_format', 'json',
             '-show_format', '-show_streams', path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not result.stdout.strip():
            return meta
        data = json.loads(result.stdout)
    except Exception:
        return meta
    fmt = data.get('format', {}) or {}
    meta['title'] = (fmt.get('tags', {}) or {}).get('title', '')
    try:
        meta['duration'] = float(fmt.get('duration') or 0.0)
    except Exception:
        meta['duration'] = 0.0
    meta['streams'] = data.get('streams', []) or []
    return meta

def extract_media_text(path, selection=None):
    ext = os.path.splitext(path)[1].lower()
    if ext not in MEDIA_EXTENSIONS:
        from .registry import extract_file
        return extract_file(path)
    selection = selection or {'kind': 'transcription'}
    kind = selection.get('kind')
    if kind == 'sidecar':
        return (selection.get('text') or load_subtitle_file(selection['path'])['text'], False)
    if kind == 'embedded':
        resolved = selection if selection.get('text') else extract_embedded_subtitle_candidate(path, selection)
        return (resolved.get('text'), False)
    if kind == 'external':
        resolved = selection if selection.get('text') else download_external_subtitle(selection)
        return (resolved.get('text'), False)
    return transcribe_audio(path)

def auto_select_caption_candidate(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in MEDIA_EXTENSIONS:
        return None
    candidates = gather_video_caption_candidates(path)
    for kind in ('sidecar', 'embedded'):
        for candidate in candidates:
            if candidate.get('kind') == kind:
                return candidate
    return {'kind': 'transcription'}

def decorate_media_text(path, text, selection=None):
    ext = os.path.splitext(path)[1].lower()
    if ext not in MEDIA_EXTENSIONS or not text:
        return text
    meta = probe_media_metadata(path)
    lines = [f"# media: {os.path.basename(path)}", f"# type: {detect_document_type(path)}"]
    if meta.get('title'):
        lines.append(f"# title: {meta['title']}")
    if meta.get('duration'):
        lines.append(f"# duration: {_format_ts(meta['duration'])}")
    if selection:
        lines.append(f"# transcript_source: {selection.get('kind', 'transcription')}")
    header = '\n'.join(lines) + "\n\n"
    return header + text

__all__ = ['probe_media_metadata', 'extract_media_text', 'decorate_media_text', 'auto_select_caption_candidate']
