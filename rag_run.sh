#!/usr/bin/env bash
# rag_run.sh — launcher for RAG-Narock that activates the correct conda env
# Usage: rag_run.sh <script.py> [args...]
#   e.g. rag_run.sh /home/mike/tools/rag_mcp.py
#        rag_run.sh /home/mike/tools/rag.py index /path --gpu
#
# Reads conda_env from ~/.local/share/rag/settings.json (default: ai-env)
# The conda env's own activation scripts handle oneAPI setup if needed

SETTINGS_FILE="$HOME/.local/share/rag/settings.json"
CONDA_ROOT="$HOME/miniforge3"

# Read a setting from settings.json (jq-free)
read_setting() {
    local key="$1" default="$2"
    if [ -f "$SETTINGS_FILE" ]; then
        val=$("$CONDA_ROOT/bin/python3" -c "
import json
try:
    cfg = json.load(open('$SETTINGS_FILE'))
    print(cfg.get('$key', '$default'))
except: print('$default')
" 2>/dev/null)
        echo "${val:-$default}"
    else
        echo "$default"
    fi
}

CONDA_ENV=$(read_setting conda_env ai-env)

# Activate conda env (ai-xpu auto-sources oneAPI setvars via activation scripts)
# Redirect stdout to /dev/null — activation only sets env vars, stdout is noise
# (oneAPI setvars.sh prints :: messages that would break MCP stdio protocol)
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" >/dev/null 2>&1

exec python3 "$@"
