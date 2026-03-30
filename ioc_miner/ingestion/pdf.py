from __future__ import annotations

from .base import BaseIngestor


class PDFIngestor(BaseIngestor):
    """Extracts text from PDF using pdfplumber (handles complex layouts)."""

    source_type = "pdf"

    def ingest(self, source: str) -> str:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required: pip install pdfplumber")

        pages: list[str] = []
        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
