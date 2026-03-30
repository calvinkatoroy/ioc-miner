from __future__ import annotations

import sys

from .base import BaseIngestor


class PlaintextIngestor(BaseIngestor):
    """Reads from a file path or stdin ('-')."""

    source_type = "plaintext"

    def ingest(self, source: str) -> str:
        if source == "-":
            return sys.stdin.read()
        with open(source, encoding="utf-8", errors="replace") as f:
            return f.read()
