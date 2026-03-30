from __future__ import annotations

from abc import ABC, abstractmethod


class BaseIngestor(ABC):
    """All ingestors return a plain string — preprocessing happens downstream."""

    @abstractmethod
    def ingest(self, source: str) -> str:
        """
        Args:
            source: file path, URL, channel username, etc.
        Returns:
            Raw text content.
        """
        ...

    @property
    @abstractmethod
    def source_type(self) -> str: ...
