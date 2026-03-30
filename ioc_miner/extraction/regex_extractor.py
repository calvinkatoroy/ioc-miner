"""
Regex-based IOC extractor — fast baseline, no model required.

Handles defanged variants that survive after preprocessing (belt-and-suspenders),
and excludes known false positives (private IPs in benign ranges, example.com, etc.).
"""

from __future__ import annotations

import re

from ioc_miner.models.ioc import IOC, IOCType

# ─── Patterns ─────────────────────────────────────────────────────────────────

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)"
_IPV4 = re.compile(rf"\b(?:{_OCTET}\.){{3}}{_OCTET}\b")

# Domains: at least one label + a known or plausible TLD (2-6 chars)
_DOMAIN = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)"
    r"+(?:com|net|org|io|gov|edu|mil|int|info|biz|ru|cn|de|uk|fr|jp|br|in|nl|au"
    r"|onion|xyz|top|site|club|online|tech|app|dev|sh|cc|co|me|tv|tk|ml|ga|cf|gq"
    r"|[a-z]{2,6})\b",
    re.IGNORECASE,
)

_URL = re.compile(
    r"https?://"
    r"(?:[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%])+",
    re.IGNORECASE,
)

_MD5 = re.compile(r"\b[0-9a-fA-F]{32}\b")
_SHA1 = re.compile(r"\b[0-9a-fA-F]{40}\b")
_SHA256 = re.compile(r"\b[0-9a-fA-F]{64}\b")
_SHA512 = re.compile(r"\b[0-9a-fA-F]{128}\b")

_CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

_EMAIL = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)

# Windows and Unix paths
_FILEPATH = re.compile(
    r"(?:[A-Za-z]:\\(?:[^\\\s/:*?\"<>|\r\n]+\\)*[^\\\s/:*?\"<>|\r\n]*)"  # Windows
    r"|(?:/(?:[^\s/]+/)*[^\s/]+)",  # Unix
)

# ─── Allowlist / denylist ──────────────────────────────────────────────────────

# Private/reserved IPv4 ranges to skip
_PRIVATE_RANGES = [
    re.compile(r"^10\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^172\.(?:1[6-9]|2\d|3[01])\."),
    re.compile(r"^127\."),
    re.compile(r"^0\.0\.0\.0$"),
    re.compile(r"^255\.255\.255\.255$"),
    re.compile(r"^169\.254\."),   # link-local
]

# Benign example domains to skip
_BENIGN_DOMAINS = {
    "example.com", "example.org", "example.net",
    "test.com", "localhost",
    "google.com", "microsoft.com", "apple.com",  # unlikely IOCs in most reports
}

# Common false-positive hashes (empty file, known-clean)
_BENIGN_HASHES = {
    "d41d8cd98f00b204e9800998ecf8427e",   # MD5 of empty string
    "da39a3ee5e6b4b0d3255bfef95601890afd80709",  # SHA1 of empty
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # SHA256 of empty
}


def _is_private_ip(ip: str) -> bool:
    return any(p.match(ip) for p in _PRIVATE_RANGES)


def _looks_like_version(match: re.Match, text: str) -> bool:
    """Heuristic: skip hex strings that are likely version numbers (v1.0a3b...)."""
    start = max(0, match.start() - 2)
    prefix = text[start:match.start()]
    return bool(re.search(r"[vV]$", prefix.strip()))


# ─── Extractor ────────────────────────────────────────────────────────────────

class RegexExtractor:
    """
    Extracts IOCs from a single sentence using regex patterns.

    Returns a list of IOC objects with verdict=UNKNOWN (context classifier
    runs downstream to determine malicious/benign).
    """

    def __init__(self, skip_private_ips: bool = True, skip_benign_domains: bool = True):
        self.skip_private_ips = skip_private_ips
        self.skip_benign_domains = skip_benign_domains

    def extract(self, sentence: str, source: str = "") -> list[IOC]:
        results: list[IOC] = []

        results.extend(self._extract_urls(sentence, source))
        results.extend(self._extract_ips(sentence, source))
        results.extend(self._extract_domains(sentence, source, seen_urls={u.value for u in results}))
        results.extend(self._extract_hashes(sentence, source))
        results.extend(self._extract_cves(sentence, source))
        results.extend(self._extract_emails(sentence, source))
        results.extend(self._extract_filepaths(sentence, source))

        return results

    def extract_all(self, sentences: list[str], source: str = "") -> list[IOC]:
        """Deduplicate across sentences, keep first occurrence context."""
        seen: set[tuple[IOCType, str]] = set()
        results: list[IOC] = []
        for sentence in sentences:
            for ioc in self.extract(sentence, source):
                key = (ioc.type, ioc.value)
                if key not in seen:
                    seen.add(key)
                    results.append(ioc)
        return results

    # ── per-type helpers ──────────────────────────────────────────────────────

    def _make_ioc(self, ioc_type: IOCType, value: str, sentence: str, source: str) -> IOC:
        return IOC(
            type=ioc_type,
            value=value,
            context=sentence,
            source=source,
            extracted_by="regex",
        )

    def _extract_urls(self, sentence: str, source: str) -> list[IOC]:
        results = []
        for m in _URL.finditer(sentence):
            results.append(self._make_ioc(IOCType.URL, m.group(), sentence, source))
        return results

    def _extract_ips(self, sentence: str, source: str) -> list[IOC]:
        results = []
        for m in _IPV4.finditer(sentence):
            ip = m.group()
            if self.skip_private_ips and _is_private_ip(ip):
                continue
            results.append(self._make_ioc(IOCType.IP, ip, sentence, source))
        return results

    def _extract_domains(self, sentence: str, source: str, seen_urls: set[str]) -> list[IOC]:
        results = []
        for m in _DOMAIN.finditer(sentence):
            domain = m.group().lower()
            if self.skip_benign_domains and domain in _BENIGN_DOMAINS:
                continue
            # Skip if this domain is already covered by a URL match
            if any(domain in url for url in seen_urls):
                continue
            results.append(self._make_ioc(IOCType.DOMAIN, domain, sentence, source))
        return results

    def _extract_hashes(self, sentence: str, source: str) -> list[IOC]:
        results = []
        # Order matters: try longer hashes first to avoid partial matches
        for pattern, ioc_type in [
            (_SHA512, IOCType.SHA512),
            (_SHA256, IOCType.SHA256),
            (_SHA1, IOCType.SHA1),
            (_MD5, IOCType.MD5),
        ]:
            for m in pattern.finditer(sentence):
                value = m.group().lower()
                if value in _BENIGN_HASHES:
                    continue
                if _looks_like_version(m, sentence):
                    continue
                results.append(self._make_ioc(ioc_type, value, sentence, source))
        return results

    def _extract_cves(self, sentence: str, source: str) -> list[IOC]:
        return [
            self._make_ioc(IOCType.CVE, m.group().upper(), sentence, source)
            for m in _CVE.finditer(sentence)
        ]

    def _extract_emails(self, sentence: str, source: str) -> list[IOC]:
        return [
            self._make_ioc(IOCType.EMAIL, m.group().lower(), sentence, source)
            for m in _EMAIL.finditer(sentence)
        ]

    def _extract_filepaths(self, sentence: str, source: str) -> list[IOC]:
        results = []
        for m in _FILEPATH.finditer(sentence):
            path = m.group()
            # Skip very short Unix paths that are likely false positives
            if path.startswith("/") and len(path) < 4:
                continue
            results.append(self._make_ioc(IOCType.FILEPATH, path, sentence, source))
        return results
