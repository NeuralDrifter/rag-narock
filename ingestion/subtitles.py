"""
ingestion/subtitles.py — subtitle parsing and external candidate fetch.
"""

import os
import re
import json
import sys
import config.settings as rag_settings
from ingestion.ocr import sanitize_text
from ingestion.audio import get_transcription_language, get_transcription_backend, HAS_FFMPEG

SUBTITLE_EXTENSIONS = {'.srt', '.vtt', '.ass', '.ssa'}
VIDEO_EXTENSIONS = {'.mp4', '.m4v', '.mov', '.webm', '.mkv', '.avi'}

def _format_ts(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    total_ms = int(round(seconds * 1000))
    hrs, rem = divmod(total_ms, 3600000)
    mins, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

def _parse_timecode(text):
    text = text.strip().replace(',', '.')
    parts = text.split(':')
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(parts[0])
    except Exception:
        return 0.0

def _clean_subtitle_markup(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{\\[^}]+\}', '', text)
    text = text.replace('\\N', '\n').replace('\\n', '\n')
    return sanitize_text(text)

def _entries_to_transcript(entries):
    lines = []
    for entry in entries:
        body = entry.get('text', '').strip()
        if not body:
            continue
        start = entry.get('start')
        end = entry.get('end')
        if start is not None and end is not None:
            lines.append(f"[{_format_ts(start)} --> {_format_ts(end)}] {body}")
        elif start is not None:
            lines.append(f"[{_format_ts(start)}] {body}")
        else:
            lines.append(body)
    return sanitize_text('\n'.join(lines))

def _parse_srt_text(text):
    entries = []
    blocks = re.split(r'\n\s*\n', text.replace('\r\n', '\n').replace('\r', '\n'))
    for block in blocks:
        lines = [line.strip('\ufeff') for line in block.split('\n') if line.strip()]
        if len(lines) < 2:
            continue
        if '-->' in lines[0]:
            timing = lines[0]
            body_lines = lines[1:]
        elif len(lines) >= 3 and '-->' in lines[1]:
            timing = lines[1]
            body_lines = lines[2:]
        else:
            continue
        start_text, end_text = [part.strip() for part in timing.split('-->', 1)]
        body = _clean_subtitle_markup('\n'.join(body_lines))
        if body:
            entries.append({
                'start': _parse_timecode(start_text),
                'end': _parse_timecode(end_text),
                'text': body,
            })
    return entries

def _parse_vtt_text(text):
    text = re.sub(r'^WEBVTT.*?\n+', '', text.replace('\r\n', '\n').replace('\r', '\n'), flags=re.DOTALL)
    return _parse_srt_text(text)

def _parse_ass_text(text):
    entries = []
    for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        if not line.startswith('Dialogue:'):
            continue
        parts = line.split(',', 9)
        if len(parts) < 10:
            continue
        start_text = parts[1].strip()
        end_text = parts[2].strip()
        body = _clean_subtitle_markup(parts[9].strip())
        if body:
            entries.append({
                'start': _parse_timecode(start_text),
                'end': _parse_timecode(end_text),
                'text': body,
            })
    return entries

def parse_subtitle_text(text, ext):
    ext = ext.lower()
    if ext == '.srt':
        entries = _parse_srt_text(text)
    elif ext == '.vtt':
        entries = _parse_vtt_text(text)
    elif ext in ('.ass', '.ssa'):
        entries = _parse_ass_text(text)
    else:
        entries = []
    transcript = _entries_to_transcript(entries) if entries else sanitize_text(text)
    return transcript, entries

def load_subtitle_file(path, max_bytes=50_000_000):
    size = os.path.getsize(path)
    if size > max_bytes:
        raise RuntimeError(f"Subtitle file too large ({size} bytes, limit {max_bytes})")
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    transcript, entries = parse_subtitle_text(raw, os.path.splitext(path)[1])
    return {
        'text': transcript,
        'entries': entries,
        'preview': transcript[:800],
        'path': path,
    }

def _subtitle_language_matches(lang):
    wanted = (rag_settings.get('subtitle_language') or '').strip().lower()
    if not wanted:
        return True
    if not lang:
        return False
    lang = lang.lower()
    return lang == wanted or lang.startswith(wanted + '-') or wanted.startswith(lang + '-')

def find_sidecar_subtitles(media_path):
    base, _ = os.path.splitext(media_path)
    folder = os.path.dirname(media_path)
    stem = os.path.basename(base).lower()
    candidates = []
    for entry in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, entry)
        if not os.path.isfile(fpath):
            continue
        root, ext = os.path.splitext(entry)
        if ext.lower() not in SUBTITLE_EXTENSIONS:
            continue
        root_l = root.lower()
        if not (root_l == stem or root_l.startswith(stem + '.') or root_l.startswith(stem + ' ')):
            continue
        info = load_subtitle_file(fpath)
        lang_guess = ''
        suffix = root_l[len(stem):].strip(' ._-')
        if suffix:
            lang_guess = suffix.split('.')[0]
        candidates.append({
            'id': f"sidecar:{entry}",
            'kind': 'sidecar',
            'label': entry,
            'language': lang_guess,
            'title': entry,
            'preview': info['preview'],
            'text': info['text'],
            'entries': info['entries'],
            'path': fpath,
            'provenance': {'kind': 'sidecar', 'path': fpath},
        })
    preferred = [c for c in candidates if _subtitle_language_matches(c.get('language'))]
    return preferred or candidates

def find_embedded_subtitles(media_path):
    from ingestion.media import probe_media_metadata
    if not HAS_FFMPEG or not os.path.splitext(media_path)[1].lower() in VIDEO_EXTENSIONS:
        return []
    meta = probe_media_metadata(media_path)
    candidates = []
    for stream in meta.get('streams', []):
        if stream.get('codec_type') != 'subtitle':
            continue
        tags = stream.get('tags', {}) or {}
        lang = tags.get('language', '')
        title = tags.get('title', '') or f"Subtitle stream {stream.get('index', '?')}"
        if rag_settings.get('subtitle_language') and not _subtitle_language_matches(lang):
            continue
        candidates.append({
            'id': f"embedded:{stream.get('index')}",
            'kind': 'embedded',
            'stream_index': stream.get('index'),
            'language': lang,
            'title': title,
            'label': f"{title} [{lang or 'unknown'}]",
            'preview': '',
            'text': '',
            'entries': [],
            'provenance': {'kind': 'embedded', 'stream_index': stream.get('index'), 'language': lang, 'title': title},
        })
    return candidates

def extract_embedded_subtitle_candidate(media_path, candidate):
    import subprocess, tempfile
    stream_index = candidate['stream_index']
    with tempfile.TemporaryDirectory(prefix='rag_subs_') as tmpdir:
        out_path = os.path.join(tmpdir, 'captions.vtt')
        cmd = ['ffmpeg', '-y', '-i', media_path, '-map', f'0:{stream_index}', out_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            # Fallback to direct copy if ffmpeg could not infer webvtt
            out_path = os.path.join(tmpdir, 'captions.srt')
            cmd = ['ffmpeg', '-y', '-i', media_path, '-map', f'0:{stream_index}', '-c:s', 'copy', out_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError(result.stderr.strip() or "ffmpeg subtitle extraction failed")
        info = load_subtitle_file(out_path)
    enriched = dict(candidate)
    enriched['text'] = info['text']
    enriched['entries'] = info['entries']
    enriched['preview'] = info['preview']
    return enriched

def search_subdl_subtitles(media_path):
    from ingestion.media import probe_media_metadata
    if not rag_settings.get('external_subtitles'):
        return []
    if rag_settings.get('subtitle_provider') != 'subdl':
        return []
    api_key = (rag_settings.get('subdl_api_key') or '').strip()
    if not api_key:
        return []
    import urllib.parse
    import urllib.request
    meta = probe_media_metadata(media_path)
    query = os.path.splitext(os.path.basename(media_path))[0]
    params = {'api_key': api_key, 'film_name': query}
    lang = (rag_settings.get('subtitle_language') or '').strip()
    if lang:
        params['languages'] = lang
    try:
        url = "https://api.subdl.com/api/v1/subtitles?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
    except Exception:
        return []

    items = payload.get('subtitles') or payload.get('results') or []
    candidates = []
    for item in items[:10]:
        release = item.get('release_name') or item.get('name') or query
        language = item.get('language') or item.get('lang') or ''
        download_url = item.get('url') or item.get('download_url') or item.get('link')
        if not download_url:
            continue
        duration = item.get('duration')
        score = 0
        if meta.get('title') and meta['title'].lower() in release.lower():
            score += 2
        if language and _subtitle_language_matches(language):
            score += 1
        candidates.append({
            'id': f"external:subdl:{item.get('subtitle_id') or item.get('id') or len(candidates)}",
            'kind': 'external',
            'provider': 'subdl',
            'label': f"{release} [{language or 'unknown'}]",
            'title': release,
            'language': language,
            'duration': duration,
            'download_url': download_url,
            'preview': item.get('comment') or item.get('release_info') or '',
            'score': score,
            'provenance': {'kind': 'external', 'provider': 'subdl', 'url': download_url, 'title': release, 'language': language},
        })
    return sorted(candidates, key=lambda c: (-c.get('score', 0), c['label']))

def download_external_subtitle(candidate):
    import io
    import urllib.request
    import tempfile
    import zipfile
    req = urllib.request.Request(candidate['download_url'], headers={'User-Agent': 'RAG-Narock/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with tempfile.TemporaryDirectory(prefix='rag_ext_sub_') as tmpdir:
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = [n for n in zf.namelist() if os.path.splitext(n)[1].lower() in SUBTITLE_EXTENSIONS]
                if not names:
                    raise RuntimeError("external subtitle archive contained no supported subtitle files")
                tmp_path = zf.extract(names[0], tmpdir)
        else:
            tmp_path = os.path.join(tmpdir, 'subtitle.srt')
            with open(tmp_path, 'wb') as f:
                f.write(data)
        try:
            info = load_subtitle_file(tmp_path)
        except Exception:
            text = sanitize_text(data.decode('utf-8', errors='replace'))
            info = {'text': text, 'entries': [], 'preview': text[:800]}
    enriched = dict(candidate)
    enriched['text'] = info['text']
    enriched['entries'] = info['entries']
    enriched['preview'] = info['preview']
    return enriched

def gather_video_caption_candidates(media_path):
    candidates = []
    if rag_settings.get('prefer_subtitles'):
        candidates.extend(find_sidecar_subtitles(media_path))
        for candidate in find_embedded_subtitles(media_path):
            try:
                candidates.append(extract_embedded_subtitle_candidate(media_path, candidate))
            except Exception:
                continue
        candidates.extend(search_subdl_subtitles(media_path))
    candidates.append({
        'id': 'transcribe',
        'kind': 'transcription',
        'label': 'Audio transcription',
        'title': 'Audio transcription',
        'language': get_transcription_language() or '',
        'preview': 'Transcribe the media audio track with the configured Whisper backend.',
        'text': '',
        'entries': [],
        'provenance': {'kind': 'transcription', 'backend': get_transcription_backend()},
    })
    return candidates

# Also expose for legacy_commands convenience
__all__ = ['load_subtitle_file', 'parse_subtitle_text', 'search_subdl_subtitles', 'gather_video_caption_candidates', 'extract_embedded_subtitle_candidate']
