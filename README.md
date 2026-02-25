<div align="center">

<img src="rag_splash.png" width="400" />

# RAG Narock

Local RAG system for searching indexed books, documents, and codebases.
Everything runs on-device. Nothing leaves the machine.

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" />
<img src="https://img.shields.io/badge/MCP-0d1117?style=flat-square&logo=anthropic&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />

</div>

---

## Features

- **Offline-first** — indexes and searches locally with no cloud dependency
- **MCP server** — plug directly into Claude Code or any MCP-compatible assistant
- **Multiple backends** — FAISS (fast), SQLite-vec (compact), SQLite-doc (document-aware with context expansion)
- **OCR built-in** — automatically OCRs scanned PDFs and images via Tesseract
- **Integrity verification** — SHA-256 hashing detects tampering or corruption
- **Code indexing** — language-aware chunking for codebases
- **TUI editor** — curses-based interface for managing indexes, sources, and locks
- **Configurable embeddings** — local CPU models, or GPU via Ollama / LM Studio

## Supported Formats

| Documents | Images (OCR) | Code |
|-----------|-------------|------|
| PDF, EPUB, MOBI, DJVU | PNG, JPG, TIFF, BMP, WebP | All text-based languages |
| TXT, Markdown, RST | PBM, PGM, PPM | Build files auto-detected |

## Installation

```bash
git clone https://github.com/NeuralDrifter/rag-narock.git
cd rag-narock
pip install sentence-transformers faiss-cpu sqlite-vec mcp
```

Optional for OCR:
```bash
sudo apt install tesseract-ocr
pip install pytesseract pdf2image Pillow
```

Optional for EPUB/MOBI:
```bash
pip install ebooklib mobi
```

## Quick Start

**Index a folder of books:**
```bash
python rag.py index /path/to/books --name MyBooks
```

**Index a codebase (with document-aware backend):**
```bash
python rag.py code /path/to/project --name MyCode
```

**Search from the command line:**
```bash
python rag.py query "how does authentication work" --name MyCode --top-k 5
```

**Open the TUI editor to manage indexes:**
```bash
python rag.py editor
```

**Open the settings panel:**
```bash
python rag.py settings
```

## MCP Server

Register with Claude Code:
```bash
claude mcp add rag -s user --transport stdio -- python3 /path/to/rag_mcp.py
```

### Tools

| Tool | Description |
|------|-------------|
| `rag_list` | List available indexes with stats |
| `rag_sources` | List documents in an index, with optional name filter |
| `rag_query` | Search for relevant passages with auto source detection |
| `rag_read` | Retrieve full document text (sqlite-doc) |
| `rag_chunk` | Get a specific chunk by number (sqlite-doc) |
| `rag_context` | Get a chunk with neighboring chunks (sqlite-doc) |
| `rag_read_range` | Read sequential chunks as continuous text (sqlite-doc) |

**Workflow:** `rag_list` → `rag_sources` → `rag_query` with `source_filter`

The MCP server auto-detects when a query mentions a source name and filters automatically. Results include source attribution, chunk positions, and relevance scores. Unverified or tampered indexes are refused.

## Embedding Models

| Model | Dimensions | Languages | Speed |
|-------|-----------|-----------|-------|
| `all-MiniLM-L6-v2` (default) | 384 | English | Fast |
| `all-MiniLM-L12-v2` | 384 | English | Balanced |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ | Balanced |
| `all-mpnet-base-v2` | 768 | English | Best quality |

Or use any model served by Ollama or LM Studio for GPU-accelerated embeddings.

## Storage Backends

| Backend | Format | Best For |
|---------|--------|----------|
| **FAISS** | index.faiss + JSON | Speed, large indexes |
| **SQLite-vec** | Single .db file | Compact storage |
| **SQLite-doc** | Single .db file | Full document retrieval, context expansion, chunk navigation |

## CLI Reference

```
python rag.py <command>

Indexing:
  index <path>        Index a folder of documents (PDF, EPUB, MOBI, DJVU, TXT)
  code <path>         Index a codebase with sqlite-doc backend
  query <text>        Search an index
  read                Retrieve full document text
  chunk               Get a specific chunk by number
  sources             List all sources in an index

Management:
  list                List all indexes
  lock / unlock       Protect or unprotect an index
  delete              Remove an index
  move                Move an index into the data directory
  add-external        Register an external index directory
  editor              Open the TUI index manager
  gui                 Open the GUI book adder

Integrity:
  verify              SHA-256 integrity check
  integrity           Accept or rehash an index

Settings:
  settings            Open the settings panel (TUI or GUI)
```

## Configuration

Run `python rag.py settings` to configure:

- **Embedding backend** — local CPU, Ollama, or LM Studio
- **Storage backend** — FAISS, SQLite-vec, or SQLite-doc
- **Chunk size / overlap** — tune for your content type
- **OCR settings** — language, DPI, invert colors, split spreads
- **Data directory** — where indexes are stored

## License

[MIT](LICENSE) — Copyright (c) 2026 Michael Burgus
