# Legacy cmd_ code moved from god rag.py
# Full imports to make commands self-contained and runnable (no NameError from old god scope)
import os
import sys
import json
import gc
import logging
import re
import unicodedata
import hashlib
import signal
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Callable
from core.constants import SETTING_STORAGE_BACKEND, SETTING_CHUNK_SIZE, SETTING_OVERLAP, SETTING_TOP_K

logger = logging.getLogger(__name__)

import numpy as np

# Modular imports (config, core, ingestion) - prefer over old shims where possible
import config.settings as rag_settings
import rag_backends  # storage shim (detect/get_backend)
from core.constants import META_FILENAME, CHUNKS_FILENAME, INDEX_DB_FILENAME, LOCK_FILENAME
from core.hashing import file_hash, load_index_hashes, save_index_hashes
from core.integrity import (
    compute_index_integrity,
    save_index_integrity,
    check_index_integrity,
    suppress_index_integrity,
    _cli_integrity_gate,
)
from core.index_manager import (
    resolve_index_dir,
    get_indexes,
    get_index_info,
    get_index_sources,
    is_index_locked,
    remove_source_from_index,
    delete_index,
    _export_filename,
    export_source,
    export_index,
)
from ingestion.registry import EXTRACTORS, EXTRACTORS_OCR, extract_file
from ingestion.chunking import chunk_text, chunk_code

# Modular imports from specific domain files
from indexing.embedder import (
    embed_texts,
    resolve_embedding_for_index,
    _save_index_metadata,
    _auto_lock_index,
    _unload_embedding_model,
    _resolve_embedding_device,
    _resolve_embedding_model_name,
)
from ingestion.ocr import (
    sanitize_text,
    get_ocr_backend,
    ocr_backend_available,
    ocr_unavailable_message,
    _IMAGE_EXTRACTORS,
    associate_images_with_chunks,
)
from ingestion.documents import (
    extract_url,
    detect_document_type,
    _probe_needs_ocr,
)
from ingestion.media import (
    auto_select_caption_candidate,
    extract_media_text,
    decorate_media_text,
)
from ingestion.ocr import set_ocr_lang, _unload_easyocr
from ingestion.ocr_options import OcrOptions

OCR_NOTE = "[NOTE: This text was produced by OCR and may contain recognition errors — misspellings, garbled characters, or missing words.]\n"

# Also expose some used by gui etc when imported
__all__ = ['cmd_index', 'cmd_query', 'cmd_list', 'cmd_gui', 'cmd_settings', 'cmd_editor', 'cmd_add', 'cmd_add_url']


# ── Shared indexing pipeline (DRY — Ch 3, G5) ─────────────────────────────────

def _resolve_indexing_options(args) -> dict:
    """Resolve all indexing options from args + settings, returning a normalized dict."""
    opts = {}
    opts['chunk_size'] = args.chunk_size if getattr(args, 'chunk_size', None) is not None else rag_settings.get('chunk_size')
    opts['overlap'] = args.overlap if getattr(args, 'overlap', None) is not None else rag_settings.get('overlap')
    opts['storage_type'] = getattr(args, 'storage_backend', None) or rag_settings.get('storage_backend')
    opts['force_ocr'] = getattr(args, 'ocr', None)
    if opts['force_ocr'] is None:
        opts['force_ocr'] = rag_settings.get('force_ocr')
    opts['no_ocr'] = getattr(args, 'no_ocr', None)
    if opts['no_ocr'] is None:
        opts['no_ocr'] = rag_settings.get('disable_ocr')
    if opts['no_ocr']:
        opts['force_ocr'] = False
    return opts


def _discover_and_deduplicate_files(source_dir: str, index_dir: str, append: bool) -> List[str]:
    """Find all supported files in source_dir and filter out duplicates by hash and name."""
    files = []
    for root, dirs, fnames in os.walk(source_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in EXTRACTORS:
                files.append(os.path.join(root, f))

    if not files:
        print(f"No supported files in {source_dir}")
        return []

    # load existing hashes for duplicate detection
    existing_hashes = load_index_hashes(index_dir) if append else {}
    existing_sources = set()
    if append and os.path.exists(os.path.join(index_dir, CHUNKS_FILENAME)):
        with open(os.path.join(index_dir, CHUNKS_FILENAME), 'r') as f:
            existing_sources = set(c['source'] for c in json.load(f))

    # deduplicate files by hash and name
    deduped_files = []
    dup_files = []
    new_hashes = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        fhash = file_hash(fpath)

        if fhash in existing_hashes:
            dup_files.append((fname, f"duplicate of '{existing_hashes[fhash]}' (same content)"))
            continue
        relname = os.path.relpath(fpath, source_dir)
        if relname in existing_sources:
            dup_files.append((fname, "already in index (same name)"))
            continue
        if fhash in new_hashes:
            dup_files.append((fname, f"duplicate of '{new_hashes[fhash]}' in this batch"))
            continue

        deduped_files.append(fpath)
        new_hashes[fhash] = fname

    if dup_files:
        print(f"Skipping {len(dup_files)} duplicate(s):")
        for name, reason in dup_files:
            print(f"  - {name}: {reason}")

    if not deduped_files:
        print("All files already in index. Nothing to add.")

    return deduped_files


def _build_ocr_options(args) -> OcrOptions:
    """Build and return an OcrOptions object from command arguments and settings."""
    no_ocr = args.no_ocr if args.no_ocr is not None else rag_settings.get('disable_ocr')
    ocr_backend_arg = getattr(args, 'ocr_backend', None)
    force_ocr = args.ocr if args.ocr is not None else rag_settings.get('force_ocr')
    if no_ocr:
        force_ocr = False
    ocr_lang = args.ocr_lang if args.ocr_lang is not None else rag_settings.get('ocr_lang')
    ocr_neg = args.ocr_negative if args.ocr_negative is not None else rag_settings.get('ocr_negative')
    split_sp = args.split_spreads if args.split_spreads is not None else rag_settings.get('split_spreads')

    return OcrOptions(
        disabled=no_ocr, force=force_ocr, backend=ocr_backend_arg,
        language=ocr_lang, negative=ocr_neg, split_spreads=split_sp,
    )


def _print_ocr_status(ocr_opts: OcrOptions):
    """Print human-readable OCR status message based on options."""
    ocr_status = []
    if ocr_opts.disabled:
        ocr_status.append("OCR disabled")
    elif ocr_opts.force:
        ocr_status.append("force-OCR")
    active_backend = get_ocr_backend(ocr_opts)
    if active_backend and not ocr_opts.disabled:
        ocr_status.append(f"backend={active_backend}")
    if ocr_opts.negative and not ocr_opts.disabled:
        ocr_status.append("negative")
    if ocr_opts.split_spreads:
        ocr_status.append("split-spreads")
    if ocr_status:
        print(f"OCR mode: {', '.join(ocr_status)}")


def _storage_backend_available(storage_type: str) -> bool:
    status = rag_backends.get_backend_status().get(storage_type)
    if status is None or status["installed"]:
        return True
    print(
        f"Storage backend '{storage_type}' is unavailable because package "
        f"'{status['package']}' is not installed.",
        file=sys.stderr,
    )
    print(f"Fix: python -m pip install {status['package']}", file=sys.stderr)
    print("Bypass: choose another installed backend with --storage-backend or in Settings.", file=sys.stderr)
    return False


def _extract_and_save_images(result: dict, source_dir: str) -> int:
    """Extract images from documents and save to backend if configured and supported."""
    if not rag_settings.get('extract_images') or result['storage_type'] != 'sqlite-doc':
        return 0

    print("Extracting images...", end="", file=sys.stderr)
    all_images = []
    for doc in result['all_documents']:
        source = doc['source']
        fpath = os.path.join(source_dir, source)
        ext = os.path.splitext(fpath)[1].lower()
        extractor = _IMAGE_EXTRACTORS.get(ext)
        if not extractor:
            continue
        try:
            doc_images = extractor(fpath)
            doc_chunks = [c for c in result['all_chunks'] if c['source'] == source]
            doc_images = associate_images_with_chunks(doc_images, doc_chunks, doc['full_text'])
            for img in doc_images:
                img['source'] = source
            all_images.extend(doc_images)
        except Exception as e:
            print(f"\n  Image extraction failed for {source}: {e}", file=sys.stderr)

    image_count = 0
    if all_images:
        doc_id_map = result['backend'].get_doc_id_map()
        chunk_id_map = result['backend'].get_chunk_id_map()
        result['backend'].save_images(all_images, doc_id_map, chunk_id_map)
        image_count = len(all_images)
    print(f" {image_count} images stored.", file=sys.stderr)
    return image_count


def _index_documents(
    sources: List[dict],
    index_name: str,
    opts: dict,
    *,
    existing_hashes: dict = None,
    gpu: bool = False,
    append: bool = False,
    ocr_options: OcrOptions = None,
) -> dict:
    """Shared indexing pipeline: extract -> chunk -> embed -> save to backend.

    Each source dict supports two modes:
      - Pre-extracted: {path, rel_path, fname, text, is_ocr, hash?, doc_type?, doc_extra?}
      - Needs extraction: {path, rel_path, fname}  (uses extract_file + caption detection)

    opts: output of _resolve_indexing_options()

    Returns dict with pipeline results for metadata construction.
    Returns {'ok': False} on failure.
    """
    if existing_hashes is None:
        existing_hashes = {}

    index_dir = resolve_index_dir(index_name)
    os.makedirs(index_dir, exist_ok=True)

    storage_type = opts['storage_type']
    if append:
        storage_type = rag_backends.detect_backend(index_dir)
    if not _storage_backend_available(storage_type):
        return {'ok': False}

    # 1. Extract + chunk
    all_chunks = []
    all_documents = []
    skipped_files = []
    file_hashes_map = {}

    for i, src in enumerate(sources):
        fname = src['fname']
        fpath = src['path']
        rel_path = src.get('rel_path', fname)
        print(f"[{i+1}/{len(sources)}] {fname}", end="")

        # Pre-extracted text or live extraction (with media caption support)
        if 'text' in src:
            text = src['text']
            is_ocr = src.get('is_ocr', False)
        else:
            selection = auto_select_caption_candidate(fpath)
            if selection:
                text, is_ocr = extract_media_text(fpath, selection=selection)
                text = decorate_media_text(fpath, text, selection)
            else:
                kwargs = {'force_ocr': opts.get('force_ocr', False)}
                if ocr_options is not None:
                    kwargs['ocr_options'] = ocr_options
                text, is_ocr = extract_file(fpath, **kwargs)

        if not text:
            skipped_files.append(fname)
            print(" - SKIP (no text)")
            continue

        # Determine document type
        doc_type = src.get('doc_type')
        if doc_type is None:
            try:
                doc_type = detect_document_type(fpath)
            except Exception:
                doc_type = 'book'

        chunks = chunk_text(text, opts['chunk_size'], opts['overlap'])
        ocr_tag = " [OCR]" if is_ocr else ""
        for ci, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'source': rel_path,
                'chunk': ci,
                'of': len(chunks),
                'ocr': is_ocr,
            })

        # Document record for sqlite-doc backend
        doc = {
            'source': rel_path,
            'full_text': text,
            'doc_type': doc_type,
            'language': None,
            'ocr': is_ocr,
        }
        doc.update(src.get('doc_extra', {}))
        all_documents.append(doc)

        # Hash for dedup tracking
        fhash = src.get('hash')
        if fhash is None:
            fhash = file_hash(fpath)
        file_hashes_map[fhash] = fname
        print(f" - {len(chunks)} chunks{ocr_tag}")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} file(s) with no extractable text:")
        for f in skipped_files:
            print(f"  - {f}")

    if not all_chunks:
        print("No content extracted!")
        return {'ok': False}

    # 2. Embed
    device = _resolve_embedding_device(gpu)
    print(f"\nNew: {len(all_chunks)} chunks. Embedding ({device.upper()})...")

    texts = [c['text'] for c in all_chunks]
    embeddings = embed_texts(texts, device=device)
    _unload_embedding_model()

    dim = embeddings.shape[1]

    # 3. Save to backend
    backend = rag_backends.get_backend(index_dir, storage_type)

    all_hashes = {**existing_hashes, **file_hashes_map}
    docs_arg = all_documents if storage_type == 'sqlite-doc' else None

    if append and backend.exists():
        existing_chunks = backend.get_chunks()
        if storage_type == 'sqlite-doc':
            backend.append(all_chunks, embeddings, file_hashes_map, documents=docs_arg)
        else:
            backend.append(all_chunks, embeddings, file_hashes_map)
        all_chunks = existing_chunks + all_chunks
    else:
        if storage_type == 'sqlite-doc':
            backend.save(all_chunks, embeddings, all_hashes, documents=docs_arg)
        else:
            backend.save(all_chunks, embeddings, all_hashes)

    # 4. Return result for caller metadata construction
    n_sources = len(set(c['source'] for c in all_chunks))
    emb_backend, emb_model = _resolve_embedding_model_name()
    return {
        'ok': True,
        'all_chunks': all_chunks,
        'all_documents': all_documents,
        'file_hashes_map': file_hashes_map,
        'backend': backend,
        'dim': dim,
        'emb_backend': emb_backend,
        'emb_model': emb_model,
        'storage_type': storage_type,
        'n_sources': n_sources,
        'n_chunks_total': len(all_chunks),
        'chunk_size': opts['chunk_size'],
        'overlap': opts['overlap'],
    }


def _finalize_index(index_dir: str, result: dict, source_dir: str, **extra) -> dict:
    """Build metadata from pipeline result, save it, and auto-lock the index."""
    meta = {
        'source_dir': source_dir,
        'chunk_size': result['chunk_size'],
        'overlap': result['overlap'],
        'n_chunks': result['n_chunks_total'],
        'n_files': result['n_sources'],
        'dim': result['dim'],
        'storage_backend': result['storage_type'],
        'embedding_backend': result['emb_backend'],
        'embedding_model': result['emb_model'],
    }
    meta.update(extra)
    _save_index_metadata(index_dir, meta)
    _auto_lock_index(index_dir)
    return meta

def cmd_index(args):
    source_dir = os.path.abspath(args.path)
    if not os.path.isdir(source_dir):
        print(f"Not a directory: {source_dir}")
        return 1

    index_dir = os.path.join(rag_settings.get_data_dir(), args.name)

    # protect locked indexes from accidental overwrite (append is always OK)
    lock_file = os.path.join(index_dir, LOCK_FILENAME)
    if os.path.exists(lock_file) and not args.append:
        if not args.force:
            print(f"Index '{args.name}' is LOCKED. Use --force to overwrite, or --append to add.")
            return 1
        else:
            print(f"WARNING: Overwriting locked index '{args.name}'")

    os.makedirs(index_dir, exist_ok=True)

    deduped_files = _discover_and_deduplicate_files(source_dir, index_dir, args.append)
    if not deduped_files:
        return 0

    ocr_opts = _build_ocr_options(args)
    if ocr_opts.language != rag_settings.get('ocr_lang'):
        if not set_ocr_lang(ocr_opts.language):
            return 1

    _print_ocr_status(ocr_opts)

    # Sort: text-layer files first, OCR-needing files last
    if not ocr_opts.disabled:
        print("Probing files for text layers...", end="", flush=True)
        ocr_flags = {f: _probe_needs_ocr(f) for f in deduped_files}
        deduped_files.sort(key=lambda f: ocr_flags[f])
        n_ocr = sum(1 for v in ocr_flags.values() if v)
        print(f" {len(deduped_files) - n_ocr} text-layer, {n_ocr} likely OCR")
        if n_ocr and not ocr_backend_available(ocr_opts):
            print(ocr_unavailable_message(), file=sys.stderr)
            if ocr_opts.force:
                return 1
            print("Continuing this indexing run with OCR disabled.", file=sys.stderr)
            ocr_opts = OcrOptions(
                disabled=True,
                force=False,
                backend=ocr_opts.backend,
                language=ocr_opts.language,
                negative=ocr_opts.negative,
                split_spreads=ocr_opts.split_spreads,
            )

    print(f"Indexing {len(deduped_files)} files from {source_dir}")

    # Build sources for the shared pipeline
    sources = []
    for fpath in deduped_files:
        sources.append({
            'path': fpath,
            'rel_path': os.path.relpath(fpath, source_dir),
            'fname': os.path.basename(fpath),
        })

    pipeline_opts = _resolve_indexing_options(args)

    result = _index_documents(
        sources, args.name, pipeline_opts,
        existing_hashes=load_index_hashes(index_dir) if args.append else {},
        gpu=args.gpu,
        append=args.append,
        ocr_options=ocr_opts,
    )
    if not result['ok']:
        _unload_easyocr()
        return 1

    image_count = _extract_and_save_images(result, source_dir)

    # Finalize: metadata + lock
    source_dir_label = source_dir
    if args.append and os.path.exists(os.path.join(index_dir, META_FILENAME)):
        source_dir_label = 'multiple'
    _finalize_index(index_dir, result, source_dir_label, has_images=image_count > 0)

    _unload_easyocr()

    print(f"\nDone! Index '{args.name}' -> {index_dir} [AUTO-LOCKED]")
    print(f"  {result['n_chunks_total']} chunks from {result['n_sources']} files, {result['dim']}-dim embeddings")
    return 0

def cmd_add(args):
    """Add a single file to an index (create index if needed)."""
    fpath = os.path.abspath(args.file)
    if not os.path.isfile(fpath):
        print(f"File not found: {fpath}")
        return 1

    ext = os.path.splitext(fpath)[1].lower()
    if ext not in EXTRACTORS:
        print(f"Unsupported file type: {ext}")
        print(f"Supported: {', '.join(sorted(EXTRACTORS.keys()))}")
        return 1

    fname = os.path.basename(fpath)
    index_dir = os.path.join(rag_settings.get_data_dir(), args.name)
    meta_path = os.path.join(index_dir, META_FILENAME)
    index_exists = os.path.exists(meta_path)

    os.makedirs(index_dir, exist_ok=True)

    # Duplicate check by hash
    fhash = file_hash(fpath)
    if index_exists:
        existing_hashes = load_index_hashes(index_dir)
        if fhash in existing_hashes:
            print(f"Already indexed: {fname} (duplicate of '{existing_hashes[fhash]}')")
            return 0

    # Extract text (with media caption support)
    selection = auto_select_caption_candidate(fpath)
    if selection:
        text, is_ocr = extract_media_text(fpath, selection=selection)
        text = decorate_media_text(fpath, text, selection)
    else:
        text, is_ocr = extract_file(fpath, force_ocr=False)
    if not text:
        print(f"No text extracted from {fname}")
        return 1

    # Build source dict with pre-extracted text
    source = {
        'path': fpath,
        'rel_path': fname,
        'fname': fname,
        'text': text,
        'is_ocr': is_ocr,
        'hash': fhash,
        'doc_type': detect_document_type(fpath),
    }

    pipeline_opts = _resolve_indexing_options(args)

    result = _index_documents(
        [source], args.name, pipeline_opts,
        existing_hashes={},
        gpu=args.gpu,
        append=index_exists,
    )
    if not result['ok']:
        return 1

    _finalize_index(index_dir, result, 'single-file')

    ocr_tag = " [OCR]" if is_ocr else ""
    action = "Appended" if index_exists else "Created"
    print(f"{action} '{args.name}': {fname} -> {result['n_chunks_total']} chunks{ocr_tag}")
    return 0


def cmd_add_url(args):
    """Fetch a URL and add its content to an index."""
    from urllib.parse import urlparse

    url = args.url.strip()
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        print(f"Invalid URL: {url}")
        return 1

    # Derive a source name from the URL
    source_name = args.source
    if not source_name:
        path_part = parsed.path.strip('/').replace('/', '_') or 'index'
        source_name = f"{parsed.netloc}_{path_part}"
        source_name = re.sub(r'[^\w\-.]', '_', source_name)
        if not source_name.endswith('.html'):
            source_name += '.html'

    index_dir = os.path.join(rag_settings.get_data_dir(), args.name)
    meta_path = os.path.join(index_dir, META_FILENAME)
    index_exists = os.path.exists(meta_path)

    os.makedirs(index_dir, exist_ok=True)

    # Duplicate check by URL hash
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    if index_exists:
        existing_hashes = load_index_hashes(index_dir)
        if url_hash in existing_hashes:
            print(f"Already indexed: {url} (as '{existing_hashes[url_hash]}')")
            return 0

    # Check source name collision
    if index_exists:
        backend = rag_backends.get_backend(index_dir, rag_backends.detect_backend(index_dir))
        if backend.exists():
            existing_sources = set(c['source'] for c in backend.get_chunks())
            if source_name in existing_sources:
                print(f"Source '{source_name}' already in index. Use --source to specify a different name.")
                return 1

    print(f"Fetching {url} ...")
    text, is_ocr = extract_url(url)
    if not text:
        print(f"No text extracted from {url}")
        return 1

    print(f"  Extracted {len(text)} characters")

    # Build source dict with pre-extracted text
    source = {
        'path': url,
        'rel_path': source_name,
        'fname': source_name,
        'text': text,
        'is_ocr': False,
        'hash': url_hash,
        'doc_type': 'url',
        'doc_extra': {'url': url},
    }

    pipeline_opts = _resolve_indexing_options(args)

    result = _index_documents(
        [source], args.name, pipeline_opts,
        existing_hashes={},
        gpu=args.gpu,
        append=index_exists,
    )
    if not result['ok']:
        return 1

    _finalize_index(index_dir, result, 'url')

    action = "Appended" if index_exists else "Created"
    print(f"{action} '{args.name}': {source_name} -> {result['n_chunks_total']} chunks")
    return 0


def cmd_query(args):
    import numpy as np

    # Resolve settings fallback
    if args.top_k is None:
        args.top_k = rag_settings.get('top_k')

    context = getattr(args, 'context', 0) or 0
    source_filter = getattr(args, 'source', '') or ''

    index_dir = resolve_index_dir(args.name)
    if not _cli_integrity_gate(args.name, index_dir):
        return 1

    # Detect backend and check existence
    backend_type = rag_backends.detect_backend(index_dir)
    backend = rag_backends.get_backend(index_dir, backend_type)

    if not backend.exists():
        print(f"Index '{args.name}' not found. Run: rag.py index /path --name {args.name}")
        return 1

    # Auto-resolve embedding model for this index
    resolved = resolve_embedding_for_index(args.name)
    if resolved['warning'] and resolved['warning'].startswith('ERROR'):
        print(resolved['warning'], file=sys.stderr)
        return 1
    if resolved['warning']:
        print(resolved['warning'], file=sys.stderr)

    q_emb = embed_texts([args.query],
                        override_backend=resolved['backend'],
                        override_model=resolved['model'],
                        override_url=resolved['api_url'])
    _unload_embedding_model()

    OCR_NOTE = "[NOTE: This text was produced by OCR and may contain recognition errors — misspellings, garbled characters, or missing words.]\n"

    ctx_results = backend.search_with_context(q_emb, args.top_k,
                                               context=context,
                                               source_filter=source_filter)
    results = []
    for rank, hit in enumerate(ctx_results):
        is_ocr = hit.get('ocr', False)
        entry = {
            'rank': rank + 1,
            'score': hit['score'],
            'source': hit['source'],
            'chunk': f"{hit['chunk']+1}/{hit['of']}",
            'text': hit['text'],
            'ocr': is_ocr,
            'adjacent': hit.get('adjacent', []),
        }
        results.append(entry)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"=== RAG: \"{args.query}\" ({len(results)} results) ===\n")
        for r in results:
            ocr_str = " [OCR]" if r['ocr'] else ""
            print(f"--- [{r['rank']}] {r['source']} (chunk {r['chunk']}, score {r['score']:.3f}){ocr_str} ---")
            if r['ocr']:
                print(OCR_NOTE)
            for adj in r['adjacent']:
                if adj['chunk'] < int(r['chunk'].split('/')[0]) - 1:
                    print(f"  [context: chunk {adj['chunk']+1}]")
                    print(f"  {adj['text'][:1500]}")
                    print()
            print(r['text'][:3000])
            for adj in r['adjacent']:
                if adj['chunk'] > int(r['chunk'].split('/')[0]) - 1:
                    print()
                    print(f"  [context: chunk {adj['chunk']+1}]")
                    print(f"  {adj['text'][:1500]}")
            print()

    return 0


def cmd_list(args):
    names = get_indexes()
    if not names:
        print("No indexes.")
        return 0
    data_dir = rag_settings.get_data_dir()
    externals = {os.path.abspath(p) for p in rag_settings.get_external_indexes()}
    for name in names:
        index_dir = resolve_index_dir(name)
        locked = os.path.exists(os.path.join(index_dir, LOCK_FILENAME))
        lock_icon = " [LOCKED]" if locked else ""
        is_ext = os.path.abspath(index_dir) in externals
        ext_tag = " [EXTERNAL]" if is_ext else ""
        meta = rag_backends.get_index_meta_with_defaults(index_dir)
        storage = meta.get('storage_backend', 'faiss')
        emb_backend = meta.get('embedding_backend', 'local')
        emb_model = meta.get('embedding_model', 'all-MiniLM-L6-v2')
        print(f"  {name}: {meta['n_chunks']} chunks from {meta['n_files']} files{lock_icon}{ext_tag}")
        loc = index_dir if is_ext else meta.get('source_dir', '?')
        print(f"    Source: {loc}  |  Storage: {storage}  |  Model: {emb_model} ({emb_backend})")
    return 0

def cmd_lock(args):
    index_dir = resolve_index_dir(args.name)
    if not os.path.exists(os.path.join(index_dir, META_FILENAME)):
        print(f"Index '{args.name}' not found")
        return 1
    lock_file = os.path.join(index_dir, LOCK_FILENAME)
    with open(lock_file, 'w') as f:
        f.write("locked\n")
    print(f"Index '{args.name}' is now LOCKED. Use 'index --force' to overwrite.")
    return 0

def cmd_unlock(args):
    index_dir = resolve_index_dir(args.name)
    lock_file = os.path.join(index_dir, LOCK_FILENAME)
    if os.path.exists(lock_file):
        os.remove(lock_file)
        print(f"Index '{args.name}' is now unlocked.")
    else:
        print(f"Index '{args.name}' was not locked.")
    return 0

def cmd_delete(args):
    import shutil
    index_dir = resolve_index_dir(args.name)
    if not os.path.exists(index_dir):
        print(f"Index '{args.name}' not found")
        return 0
    lock_file = os.path.join(index_dir, LOCK_FILENAME)
    if os.path.exists(lock_file) and not args.force:
        print(f"Index '{args.name}' is LOCKED. Use --force to delete.")
        return 1
    shutil.rmtree(index_dir)
    # If it was an external index, unregister it
    rag_settings.remove_external_index(index_dir)
    print(f"Deleted index '{args.name}'")
    return 0

def cmd_add_external(args):
    """Register an external index that lives outside the data directory."""
    path = os.path.abspath(os.path.expanduser(args.path))
    meta_path = os.path.join(path, META_FILENAME)
    if not os.path.exists(meta_path):
        print(f"Not a valid index: {path}")
        print(f"  (no meta.json found)")
        return 1
    name = os.path.basename(path)
    # Check for name collision with local indexes
    data_dir = rag_settings.get_data_dir()
    if os.path.exists(os.path.join(data_dir, name, META_FILENAME)):
        print(f"Name collision: '{name}' already exists in {data_dir}")
        print(f"  Rename the external directory or remove the local index first.")
        return 1
    if rag_settings.add_external_index(path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Registered external index '{name}' from {path}")
        print(f"  {meta.get('n_chunks', '?')} chunks, {meta.get('n_files', '?')} files")
    else:
        print(f"Already registered: {path}")
    return 0

def cmd_move(args):
    """Move an index into the data directory."""
    import shutil
    name = args.name
    data_dir = rag_settings.get_data_dir()
    dest = os.path.join(data_dir, name)

    # Find the source index
    src = resolve_index_dir(name)
    if not os.path.exists(os.path.join(src, META_FILENAME)):
        print(f"Index '{name}' not found")
        return 1

    # Already in data_dir?
    if os.path.abspath(src) == os.path.abspath(dest):
        print(f"Index '{name}' is already in {data_dir}")
        return 0

    # Check destination doesn't already exist
    if os.path.exists(dest) and not args.force:
        print(f"Destination already exists: {dest}")
        print(f"  Use --force to overwrite.")
        return 1

    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(dest) and args.force:
        shutil.rmtree(dest)

    shutil.move(src, dest)
    # Unregister from externals if it was there
    rag_settings.remove_external_index(src)
    print(f"Moved '{name}' -> {dest}")
    return 0

def cmd_gui(args):
    """Thin delegation (Ch 11: separate construction from use).
    The real GUI implementation is now in gui/book_adder.py (modular widgets + controller).
    This keeps exact launch behavior while eliminating the god-function smell (G30).
    """
    from gui.book_adder import run_gui
    return run_gui(args)


def cmd_settings(args):
    """Thin wrapper for settings. Delegates to config + tui/gui."""
    if getattr(args, 'gui', False):
        from rag_settings import SettingsDialog
        root = None
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            SettingsDialog(root)
        finally:
            if root:
                root.destroy()
        return 0
    else:
        from rag_settings import SettingsTUI  # or from tui
        tui = SettingsTUI()
        tui.run()
        return 0


def cmd_editor(args):
    """Thin wrapper for editor."""
    if getattr(args, 'gui', False) or getattr(args, 'editor_cmd', None) == 'gui' or getattr(args, 'editor_sub', None) == 'gui':
        # GUI editor (for systems with desktop/GUI)
        from rag_editor import EditorDialog
        root = None
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            EditorDialog(root)
        finally:
            if root:
                root.destroy()
        return 0
    else:
        # TUI editor (for terminals / no visual interface)
        from rag_editor import EditorTUI
        tui = EditorTUI()
        tui.run()
        return 0


# (prior duplicate removed)


