# RAG System Guide

Local retrieval-augmented generation (RAG) system at `/home/mike/tools/rag.py`. Converts books/documents into searchable vector indexes. Designed for LLM assistants to query via CLI or MCP — returns relevant text passages that the LLM uses as context. **Everything runs on-device. Nothing leaves the machine.**

## How It Works

1. **Index**: Point at a folder of books → extracts text → chunks into ~1500 char passages → embeds with `all-MiniLM-L6-v2` (384-dim) → stores FAISS index + metadata on disk
2. **Query**: Embed the search query → find nearest chunks via cosine similarity → return top-K passages with source attribution
3. **No generation model** — the RAG just retrieves. The LLM calling it does the reasoning.

## Two Ways to Use It

### 1. MCP Server (Recommended for LLM CLIs)

The RAG is available as a local MCP server at `/home/mike/tools/rag_mcp.py`. This gives LLMs native tool access — no bash commands needed.

**Add to Claude Code:**
```bash
claude mcp add rag -s user --transport stdio -- /home/mike/miniforge3/envs/ai-env/bin/python3 /home/mike/tools/rag_mcp.py
```

**Or use the system wrapper (if installed):**
```bash
claude mcp add rag -s user --transport stdio -- rag-mcp
```

**For other MCP-compatible CLIs**, the pattern is the same — point at the server with stdio transport.

**MCP Tools:**

| Tool | Description |
|------|-------------|
| `rag_list` | List all available indexes with stats. **Call this first** to discover indexes. |
| `rag_query` | Search an index by natural language. Requires `query` and `index_name`. |
| `rag_sources` | List all book/document titles in an index. |

The MCP server freshly scans for indexes on every call, so new indexes created via the GUI or CLI are immediately discoverable. If `index_name` is omitted or invalid, all tools return the list of available indexes.

**MCP Details:**
- Transport: **stdio** (local only, no network listener)
- Embedding model: loads on first `rag_query` call, stays resident for server lifetime
- CPU-only, offline — same isolation as the CLI tool

### 2. CLI (for scripts, automation, and manual use)

```bash
# If in the ai-env conda environment:
python3 /home/mike/tools/rag.py <command>

# From anywhere (system wrapper):
rag <command>
```

## Requirements

- Python environment with: `sentence-transformers`, `faiss-cpu`, `PyMuPDF`, `ebooklib`, `beautifulsoup4`, `mobi`, `mcp` (for MCP server)
- All installed in the `ai-env` conda environment
- The embedding model (`all-MiniLM-L6-v2`, ~80MB) is cached locally at `~/.cache/huggingface/` after first use
- **Offline-first**: the model loads from local cache with no network contact. Only downloads once on first-ever use.

## Supported Formats

| Extension | Library Used |
|-----------|-------------|
| `.pdf` | PyMuPDF (fitz) |
| `.epub` | ebooklib + BeautifulSoup |
| `.mobi` | mobi + BeautifulSoup |
| `.txt`, `.md`, `.rst` | Plain text reader |

Image-only PDFs will be skipped (no OCR). Use `bulk_convert.py` to OCR them first, then add the resulting text files.

## CLI Commands

### Index a folder of documents

```bash
rag index /path/to/folder --name index_name
```

Options:
- `--name NAME` — Name for the index (default: `default`). Use descriptive names like `tech_books`, `Creative_Writing`, etc.
- `--chunk-size N` — Characters per chunk (default: 1500). Larger = more context per result but less precise matching.
- `--overlap N` — Character overlap between chunks (default: 200). Prevents losing context at chunk boundaries.
- `--append` — Add new files to an existing index without overwriting. Skips files already present. **This is the safe way to add books.**
- `--force` — Override a locked index (required to overwrite a locked index).

Indexes are **auto-locked** after every successful write. You don't need to manually lock them.

### Query an index

```bash
rag query "your search terms" --name index_name
```

Options:
- `--name NAME` — Which index to search (default: `default`)
- `--top-k N` — Number of results to return (default: 5)
- `--json` — Output as JSON instead of formatted text (useful for programmatic consumption)

Examples:
```bash
# Human-readable output
rag query "sub-group shuffle reduction SYCL" --name tech_books --top-k 5

# JSON output for parsing
rag query "local memory barrier" --name tech_books --json

# Suppress model loading noise (recommended for LLM use via bash)
rag query "GPU memory bandwidth" --name tech_books --top-k 5 2>/dev/null
```

### List all indexes

```bash
rag list
```

Shows index name, chunk count, file count, lock status, and source directory for each index.

### Lock / Unlock an index

```bash
# Lock — prevents overwriting (indexes auto-lock, so this is rarely needed)
rag lock index_name

# Unlock
rag unlock index_name
```

Locked indexes:
- **Cannot** be overwritten by `index` (without `--force`)
- **Cannot** be deleted by `delete` (without `--force`)
- **CAN** be appended to with `index --append` (append is always safe)
- Show `[LOCKED]` in `list` output
- **Auto-locked** after every successful CLI or GUI index operation

### Delete an index

```bash
rag delete index_name
rag delete index_name --force   # required if locked
```

Permanently removes the index directory from `~/.local/share/rag/<name>/`.

### GUI

```bash
rag gui
```

Opens a tkinter GUI with:
- Index selector dropdown (auto-discovers all indexes)
- Add Folder / Add Files buttons
- File queue with duplicate detection (SHA-256 hash + filename matching)
- Progress bar and log
- "Add to Index" button — appends without overwriting, auto-locks when done

## Current Indexes

| Index Name | Contents | Chunks | Files | Status |
|------------|----------|--------|-------|--------|
| `tech_books` | 342 technical books — C++, SYCL, DPC++, Python, Java, JavaScript, SQL, AI/ML, systems programming, security, and more | 158,473 | 342 | LOCKED |
| `Creative_Writing` | 76 creative writing books | 24,167 | 76 | LOCKED |

## Adding Books to an Existing Index

### SAFE: Use --append (recommended)

```bash
# Put new books in a folder and append — existing content is preserved
rag index /path/to/new/books --name tech_books --append
```

This will:
- Extract and embed only the NEW files
- Skip any files whose source name or content hash already exists in the index
- Merge new embeddings with existing ones
- Works even on LOCKED indexes
- Auto-locks the index after completion

### SAFE: Use the GUI

```bash
rag gui
```

Select your target index, add files/folders, click "Add to Index". The GUI handles deduplication and auto-locking.

### ALTERNATIVE: Create a separate index

```bash
rag index /path/to/new/books --name gpu_optimization
```

Then query both:
```bash
rag query "topic" --name tech_books --top-k 3
rag query "topic" --name gpu_optimization --top-k 3
```

### DANGEROUS: Full re-index (overwrites everything)

```bash
# Only works with --force since indexes are auto-locked
rag index /path/to/all/books --name tech_books --force
```

This deletes all existing content and rebuilds from scratch. **Don't do this unless you intend to replace the entire index.**

## Best Practices

1. **Indexes auto-lock** after every successful write. You don't need to manually lock them.

2. **Use --append to add books**, never re-index from scratch unless necessary.

3. **Create separate indexes** for different topics if the material is unrelated. This keeps searches focused and avoids polluting results.

4. **Use descriptive index names** — `tech_books`, `Creative_Writing`, `gpu_optimization`, not `default` or `test`.

5. **For MCP users**: call `rag_list` first to discover available indexes before querying. Indexes can be added at any time.

6. **For CLI/bash users**: add `2>/dev/null` to suppress model loading messages.

## Important Design Details

### CPU-Only, No GPU Interference

The script forces CPU execution via environment variables before any ML imports:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ONEAPI_DEVICE_SELECTOR"] = "opencl:cpu"
```
This ensures the embedding model never touches the GPUs, which matters when GPU workloads (like LLM inference testing) are running simultaneously.

### Offline-First, No Network Contact

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```
The embedding model is cached locally at `~/.cache/huggingface/`. After the one-time initial download, the tool never contacts Hugging Face or any external service. All embedding computation runs locally in your Python process. **Your book content never leaves the machine.**

### Model Loading

- **CLI**: Model loads on-demand for `index` or `query`, is explicitly unloaded + garbage collected immediately after. Does NOT run as a background service.
- **MCP server**: Model loads on first `rag_query` call and stays resident for the server lifetime (avoids reload latency on repeated queries).
- **GUI**: Model only loads when you click "Add to Index". Never loads at GUI startup.

### Auto-Locking

All indexes are automatically locked after every successful index operation (CLI or GUI). This prevents accidental overwrites. The `--append` flag bypasses the lock (append is always safe). Use `--force` only if you truly intend to overwrite.

### Duplicate Detection

Three layers of deduplication:
1. **SHA-256 hash vs index** — catches identical content under different filenames
2. **SHA-256 hash vs queue** — catches duplicates within the same batch (GUI)
3. **Filename vs index sources** — catches same-named files even if content differs slightly

### Storage Layout

```
~/.local/share/rag/
└── <index_name>/
    ├── index.faiss    # FAISS flat inner-product index
    ├── meta.json      # Index metadata (source dir, chunk/file counts, dimensions)
    ├── chunks.json    # All chunk text + source attribution
    ├── hashes.json    # SHA-256 hashes for duplicate detection
    └── .locked        # Lock file (present if index is locked)
```

### Chunking Strategy

- Splits on paragraph boundaries (double newline)
- Default 1500 chars per chunk, 200 char overlap
- Oversized paragraphs get force-split
- Fragments under 50 chars are discarded
- Overlap prevents losing context that spans chunk boundaries

### Embedding Details

- Model: `sentence-transformers/all-MiniLM-L6-v2` (22M params, 80MB)
- Dimension: 384
- Embeddings are L2-normalized, search uses inner product (equivalent to cosine similarity)
- FAISS IndexFlatIP — brute force, exact search. Fast enough for <200K chunks.

## Tips for LLM Usage

1. **MCP is preferred** over CLI — use the `rag_query` tool directly instead of shelling out to bash.
2. **Always call `rag_list` first** to discover available indexes. New ones can appear at any time.
3. **Query with natural language**, not keywords — the embedding model understands semantic meaning ("how does memory coalescing work" beats "memory coalescing").
4. **Score interpretation**: Scores range 0-1 (cosine similarity). Above 0.5 is a strong match, 0.3-0.5 is relevant, below 0.3 is likely noise.
5. **Combine with web search**: The RAG covers textbook fundamentals but not bleeding-edge specs or hardware-specific docs. Use web search for the latest documentation.
6. **Multiple indexes**: Query different indexes for different topics. `tech_books` for programming, `Creative_Writing` for fiction/writing craft.

## Quick Reference

**MCP (recommended):**
```
rag_list()                                          # discover indexes
rag_query("search terms", index_name="tech_books")  # search
rag_sources(index_name="tech_books")                # list books
```

**CLI:**
```bash
rag list
rag query "search terms" --name tech_books --top-k 5 2>/dev/null
rag index /path/to/new/books --name tech_books --append
rag gui
```

**Add MCP server to Claude Code:**
```bash
claude mcp add rag -s user --transport stdio -- /home/mike/miniforge3/envs/ai-env/bin/python3 /home/mike/tools/rag_mcp.py
```

## File Locations

| File | Purpose |
|------|---------|
| `/home/mike/tools/rag.py` | Main RAG tool (CLI + GUI) |
| `/home/mike/tools/rag_mcp.py` | MCP server wrapper |
| `/usr/local/bin/rag` | System PATH wrapper for CLI |
| `/usr/local/bin/rag-mcp` | System PATH wrapper for MCP server |
| `~/.local/share/rag/` | Index storage directory |
| `~/.cache/huggingface/` | Cached embedding model (local, offline) |
