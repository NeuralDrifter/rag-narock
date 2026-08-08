"""
core/constants.py — canonical filenames and magic values used across the project.

Single source of truth (G25: no magic strings). Import from here instead of
writing "meta.json", "chunks.json", etc. as string literals.
"""

# ── Index file layout ─────────────────────────────────────────────────────────
META_FILENAME       = "meta.json"
CHUNKS_FILENAME     = "chunks.json"
INDEX_DB_FILENAME   = "index.db"
INDEX_FAISS_FILENAME = "index.faiss"
LOCK_FILENAME       = ".locked"
INTEGRITY_FILENAME  = ".integrity"
HASHES_FILENAME     = "hashes.json"

# ── DB table names ────────────────────────────────────────────────────────────
TABLE_CHUNKS    = "chunks"
TABLE_DOCUMENTS = "documents"
TABLE_VEC_INDEX = "vec_index"
TABLE_HASHES    = "hashes"
TABLE_IMAGES    = "images"

# ── Score conversion (sqlite-vec cosine → similarity) ─────────────────────────
# sqlite-vec cosine distance: 0 = identical, 2 = opposite.
# Convert to [0,1] similarity so all backends return comparable scores.
COSINE_TO_SIMILARITY_SCALE = 2.0

# ── Settings keys used programmatically ───────────────────────────────────────
SETTING_STORAGE_BACKEND  = "storage_backend"
SETTING_CHUNK_SIZE       = "chunk_size"
SETTING_OVERLAP          = "overlap"
SETTING_TOP_K            = "top_k"
SETTING_MIN_CHUNK_LENGTH = "min_chunk_length"
SETTING_CODE_CHUNK_SIZE  = "code_chunk_size"
SETTING_CODE_OVERLAP     = "code_overlap"
SETTING_DATA_DIR         = "data_dir"
SETTING_EMBEDDING_BACKEND = "embedding_backend"
SETTING_EMBEDDING_MODEL  = "embedding_model"
SETTING_OCR_BACKEND      = "ocr_backend"
SETTING_DISABLE_OCR      = "disable_ocr"
SETTING_FORCE_OCR        = "force_ocr"
SETTING_OCR_LANG         = "ocr_lang"
SETTING_OCR_NEGATIVE     = "ocr_negative"
SETTING_SPLIT_SPREADS    = "split_spreads"
SETTING_EXTRACT_IMAGES   = "extract_images"
