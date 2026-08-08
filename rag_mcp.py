#!/usr/bin/env python3
"""
RAG MCP Server — Model Context Protocol server for RAG-Narock.

Launch with: python rag_mcp.py
Or configure in Claude Code / other MCP clients as a stdio server.

Implementation in mcp_server/ package.
"""
import sys

try:
    from mcp_server.server import main
except ImportError:
    print("MCP server not yet implemented in mcp_server/ package.", file=sys.stderr)
    print("The MCP tools are available via rag.py CLI commands.", file=sys.stderr)
    sys.exit(1)

if __name__ == '__main__':
    main()
