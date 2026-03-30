from __future__ import annotations

from .base import BaseIngestor

# Tags whose content is never useful for IOC extraction
_SKIP_TAGS = {"script", "style", "nav", "footer", "head", "noscript", "svg"}


class HTMLIngestor(BaseIngestor):
    """
    Fetches a URL or reads a local HTML file and extracts readable text.
    Strips boilerplate (nav, footer, scripts) before returning.
    """

    source_type = "html"

    def __init__(self, timeout: int = 15, user_agent: str | None = None):
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; ioc-miner/0.1; +https://github.com/calvinkatoroy/ioc-miner)"
        )

    def ingest(self, source: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4 lxml")

        html = self._fetch(source)
        return self._extract_text(html)

    def _fetch(self, source: str) -> str:
        if source.startswith(("http://", "https://")):
            import requests

            resp = requests.get(
                source,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
            resp.raise_for_status()
            return resp.text
        else:
            with open(source, encoding="utf-8", errors="replace") as f:
                return f.read()

    def _extract_text(self, html: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(_SKIP_TAGS):
            tag.decompose()

        # Prefer article/main content if present
        main = soup.find("article") or soup.find("main") or soup.body or soup
        lines = []
        for element in main.descendants:
            if element.name in ("p", "li", "td", "th", "pre", "code", "h1", "h2", "h3"):
                text = element.get_text(separator=" ", strip=True)
                if text:
                    lines.append(text)

        return "\n".join(lines)
