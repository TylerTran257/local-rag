"""Entrypoint: ``python -m app.mcp [--transport stdio|streamable-http]``."""
from __future__ import annotations

import argparse

from app.core.logging import configure_logging
from app.mcp.server import mcp


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Local-RAG MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport to serve (default: stdio).",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
