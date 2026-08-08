"""
ingestion/chunking.py — text and code-aware chunkers.

Clean Code: extracted for SRP, small focused module for chunking logic.
Uses config for settings.
"""

import re
import os

from config import settings as cfg


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    if chunk_size is None:
        chunk_size = cfg.get('chunk_size')
    if overlap is None:
        overlap = cfg.get('overlap')

    # Phase 1: Split into paragraphs while respecting Math/Code block boundaries
    lines = text.split('\n')
    paragraphs = []
    current_para = []
    in_code = False
    in_math = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code = not in_code
        if '$$' in line:
            if line.count('$$') % 2 == 1:
                in_math = not in_math
        if '\\[' in line and '\\]' not in line:
            in_math = True
        if '\\]' in line and '\\[' not in line:
            in_math = False
            
        if not in_code and not in_math and not stripped:
            if current_para:
                paragraphs.append('\n'.join(current_para).strip())
                current_para = []
        else:
            current_para.append(line)
            
    if current_para:
        paragraphs.append('\n'.join(current_para).strip())

    chunks = []
    current = ""
    for para in paragraphs:
        if not para:
            continue
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + para if len(current) > overlap else para
        else:
            current = (current + "\n\n" + para) if current else para

    if current.strip():
        chunks.append(current.strip())

    final = []
    for c in chunks:
        if len(c) > chunk_size * 2:
            # Semantic fallback: split large blocks by lines instead of raw characters
            sub_lines = c.split('\n')
            current_sub = ""
            for sl in sub_lines:
                if len(current_sub) + len(sl) + 1 > chunk_size and current_sub:
                    final.append(current_sub.strip())
                    current_sub = current_sub[-overlap:] + "\n" + sl if len(current_sub) > overlap else sl
                else:
                    current_sub = (current_sub + "\n" + sl) if current_sub else sl
            if current_sub:
                final.append(current_sub.strip())
        else:
            final.append(c)

    # If any chunk is STILL too big (e.g. a single 5000 character minified line), hard split chars
    super_final = []
    for c in final:
        if len(c) > chunk_size * 2:
            for i in range(0, len(c), chunk_size - overlap):
                sub = c[i:i + chunk_size]
                if sub.strip():
                    super_final.append(sub.strip())
        else:
            super_final.append(c)

    min_len = cfg.get('min_chunk_length')
    return [c for c in super_final if len(c) > min_len]


# ── Code indexing support ──

CODE_EXTENSIONS = {
    '.py', '.pyx', '.pyi',
    '.js', '.jsx', '.mjs', '.cjs',
    '.ts', '.tsx', '.mts',
    '.c', '.h',
    '.cpp', '.hpp', '.cc', '.hh', '.cxx',
    '.rs',
    '.go',
    '.java',
    '.rb',
    '.ex', '.exs',
    '.zig',
    '.lua',
    '.hs',
    '.jl',
    '.pl', '.pm',
    '.r', '.R',
    '.swift',
    '.kt', '.kts',
    '.scala',
    '.cs',
    '.fs', '.fsx',
    '.clj', '.cljs',
    '.erl', '.hrl',
    '.ml', '.mli',
    '.nim',
    '.v', '.sv',
    '.sh', '.bash', '.zsh', '.fish',
    '.ps1',
    '.sql',
    '.mojo',
    '.toml', '.yaml', '.yml', '.json',
    '.xml',
    '.ini', '.cfg', '.conf',
    '.html', '.htm', '.css', '.scss', '.sass', '.less',
    '.vue', '.svelte',
    '.md', '.rst', '.txt',
    '.cmake', '.mk',
}

CODE_FILENAMES = {
    'Makefile', 'Dockerfile', 'Containerfile',
    'CMakeLists.txt', 'Cargo.toml', 'go.mod', 'package.json',
    'pyproject.toml', 'setup.py', 'setup.cfg',
    '.env.example',
}

SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', 'build', 'dist', 'vendor',
    '.venv', 'venv', '.tox', 'target', '.next', 'coverage',
    '.mypy_cache', '.pytest_cache', '.ruff_cache', '.cache',
}

LANG_MAP = {
    '.py': 'python', '.pyx': 'python', '.pyi': 'python',
    '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript', '.mts': 'typescript',
    '.c': 'c', '.h': 'c',
    '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp', '.hh': 'cpp', '.cxx': 'cpp',
    '.rs': 'rust', '.go': 'go', '.java': 'java', '.rb': 'ruby',
    '.ex': 'elixir', '.exs': 'elixir', '.zig': 'zig', '.lua': 'lua',
    '.hs': 'haskell', '.jl': 'julia', '.pl': 'perl', '.pm': 'perl',
    '.r': 'r', '.R': 'r', '.swift': 'swift', '.kt': 'kotlin', '.kts': 'kotlin',
    '.scala': 'scala', '.cs': 'csharp', '.fs': 'fsharp', '.fsx': 'fsharp',
    '.clj': 'clojure', '.cljs': 'clojure', '.erl': 'erlang', '.hrl': 'erlang',
    '.ml': 'ocaml', '.mli': 'ocaml', '.nim': 'nim',
    '.sh': 'bash', '.bash': 'bash', '.zsh': 'zsh', '.fish': 'fish',
    '.ps1': 'powershell', '.sql': 'sql', '.mojo': 'mojo',
    '.html': 'html', '.htm': 'html', '.css': 'css',
    '.scss': 'scss', '.sass': 'sass', '.less': 'less',
    '.vue': 'vue', '.svelte': 'svelte',
    '.toml': 'toml', '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json',
    '.xml': 'xml', '.md': 'markdown', '.rst': 'rst', '.txt': 'text',
}

_SPLIT_PATTERNS = {
    'python': re.compile(r'^(?:def |class |async def )', re.MULTILINE),
    'javascript': re.compile(r'^(?:function |class |export |const \w+ = |let \w+ = |var \w+ = )', re.MULTILINE),
    'typescript': re.compile(r'^(?:function |class |export |const \w+ = |interface |type |enum )', re.MULTILINE),
    'c': re.compile(r'^(?:\w[\w\s*&]*\w+\s*\(|struct |enum |typedef )', re.MULTILINE),
    'cpp': re.compile(r'^(?:\w[\w\s*&:<>]*\w+\s*\(|struct |enum |class |namespace |template)', re.MULTILINE),
    'rust': re.compile(r'^(?:fn |pub fn |struct |enum |impl |trait |mod |pub struct |pub enum )', re.MULTILINE),
    'go': re.compile(r'^(?:func |type |var |const )', re.MULTILINE),
    'java': re.compile(r'^(?:\s*(?:public |private |protected |static )*(?:class |interface |enum |void |int |String |boolean |\w+ ))', re.MULTILINE),
}


def _detect_language(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1]
    return LANG_MAP.get(ext, '')


def _chunk_python_ast(text: str, chunk_size: int, overlap: int) -> list[str]:
    import ast
    try:
        tree = ast.parse(text)
    except Exception:
        return []
    
    pts = [0]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pts.append(node.lineno - 1)
            if isinstance(node, ast.ClassDef):
                for subnode in node.body:
                    if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        pts.append(subnode.lineno - 1)
    pts = sorted(list(set(pts)))
    
    lines = text.split('\n')
    pts.append(len(lines))
    segments = []
    for i in range(len(pts)-1):
        start = pts[i]
        end = pts[i+1]
        if start < end:
            seg = '\n'.join(lines[start:end]).strip()
            if seg:
                segments.append(seg)
                
    chunks = []
    current = ""
    for seg in segments:
        if len(current) + len(seg) + 2 > chunk_size and current:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + seg if len(current) > overlap else seg
        else:
            current = (current + "\n\n" + seg) if current else seg
    if current:
        chunks.append(current.strip())
        
    return chunks


def _chunk_treesitter_ast(text: str, language: str, chunk_size: int, overlap: int) -> list[str]:
    try:
        import tree_sitter
    except ImportError:
        return []
        
    lang_module = None
    try:
        if language == 'rust':
            import tree_sitter_rust as lang_module
        elif language == 'c':
            import tree_sitter_c as lang_module
        elif language == 'cpp':
            import tree_sitter_cpp as lang_module
        elif language == 'go':
            import tree_sitter_go as lang_module
        elif language == 'ruby':
            import tree_sitter_ruby as lang_module
        elif language == 'javascript':
            import tree_sitter_javascript as lang_module
        elif language == 'typescript':
            import tree_sitter_typescript as lang_module
        elif language == 'java':
            import tree_sitter_java as lang_module
        else:
            return []
    except ImportError:
        return []

    try:
        TS_LANGUAGE = tree_sitter.Language(lang_module.language())
        parser = tree_sitter.Parser(TS_LANGUAGE)
        tree = parser.parse(bytes(text, "utf8"))
    except Exception:
        return []

    pts = [0]
    for child in tree.root_node.children:
        row = getattr(child.start_point, 'row', child.start_point[0] if isinstance(child.start_point, tuple) else 0)
        pts.append(row)
        
        start_row = row
        end_row = getattr(child.end_point, 'row', child.end_point[0] if isinstance(child.end_point, tuple) else 0)
        if end_row - start_row > 20 and len(child.children) > 0:
            for subchild in child.children:
                sub_start = getattr(subchild.start_point, 'row', subchild.start_point[0] if isinstance(subchild.start_point, tuple) else 0)
                sub_end = getattr(subchild.end_point, 'row', subchild.end_point[0] if isinstance(subchild.end_point, tuple) else 0)
                if sub_end - sub_start > 2:
                    pts.append(sub_start)

    pts = sorted(list(set(pts)))

    lines = text.split('\n')
    pts.append(len(lines))
    segments = []
    for i in range(len(pts)-1):
        start = pts[i]
        end = pts[i+1]
        if start < end:
            seg = '\n'.join(lines[start:end]).strip()
            if seg:
                segments.append(seg)
                
    chunks = []
    current = ""
    for seg in segments:
        if len(current) + len(seg) + 2 > chunk_size and current:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + seg if len(current) > overlap else seg
        else:
            current = (current + "\n\n" + seg) if current else seg
    if current:
        chunks.append(current.strip())
        
    return chunks


def _get_import_block(text: str, language: str) -> str:
    lines = text.split('\n')
    import_lines = []
    patterns = {
        'python': re.compile(r'^(import |from |#)'),
        'javascript': re.compile(r'^(import |const .* = require|//|/\*)'),
        'typescript': re.compile(r'^(import |const .* = require|//|/\*)'),
        'c': re.compile(r'^(#include |#define |#pragma |//)'),
        'cpp': re.compile(r'^(#include |#define |#pragma |using |//|namespace)'),
        'rust': re.compile(r'^(use |mod |extern |//)'),
        'go': re.compile(r'^(import |package |//)'),
        'java': re.compile(r'^(import |package |//)'),
    }
    pat = patterns.get(language)
    if not pat:
        return ''
    for line in lines:
        stripped = line.strip()
        if not stripped:
            import_lines.append('')
            continue
        if pat.match(stripped):
            import_lines.append(line)
        else:
            break
    while import_lines and not import_lines[-1].strip():
        import_lines.pop()
    return '\n'.join(import_lines)


def chunk_code(text: str, filepath: str, chunk_size: int = None,
               overlap: int = None) -> list[str]:
    if chunk_size is None:
        chunk_size = cfg.get('code_chunk_size')
    if overlap is None:
        overlap = cfg.get('code_overlap')

    language = _detect_language(filepath)
    header = f"# file: {filepath}\n\n"
    import_block = _get_import_block(text, language)
    prefix = header
    if import_block:
        prefix += import_block + "\n\n"

    if len(text) <= chunk_size:
        return [header + text]

    if language == 'python':
        ast_chunks = _chunk_python_ast(text, chunk_size, overlap)
    else:
        ast_chunks = _chunk_treesitter_ast(text, language, chunk_size, overlap)
        
    if ast_chunks:
        final_ast = []
        for c in ast_chunks:
            if len(c) > chunk_size * 2:
                sub = chunk_text(c, chunk_size, overlap)
                for sc in sub:
                    final_ast.append(prefix + sc)
            else:
                final_ast.append(prefix + c)
        min_len = cfg.get('min_chunk_length')
        return [c for c in final_ast if len(c) > min_len]

    split_pat = _SPLIT_PATTERNS.get(language)
    if split_pat:
        matches = list(split_pat.finditer(text))
        if len(matches) > 1:
            chunks = []
            for mi, m in enumerate(matches):
                start = m.start()
                end = matches[mi + 1].start() if mi + 1 < len(matches) else len(text)
                segment = text[start:end].rstrip()
                if not segment.strip():
                    continue
                if len(segment) > chunk_size * 2:
                    sub_chunks = chunk_text(segment, chunk_size, overlap)
                    for sc in sub_chunks:
                        chunks.append(prefix + sc)
                else:
                    chunks.append(prefix + segment)

            merged = []
            buf = ""
            for c in chunks:
                body = c[len(prefix):]
                if buf and len(buf) + len(body) + 2 <= chunk_size:
                    buf += "\n\n" + body
                else:
                    if buf:
                        merged.append(prefix + buf)
                    buf = body
            if buf:
                merged.append(prefix + buf)

            if merged:
                min_len = cfg.get('min_chunk_length')
                return [c for c in merged if len(c) > min_len]

    raw_chunks = chunk_text(text, chunk_size, overlap)
    return [prefix + c for c in raw_chunks]
