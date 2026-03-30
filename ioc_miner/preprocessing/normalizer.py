"""
Defanging reversal and text normalization for threat intel content.

Threat intel analysts "defang" IOCs to prevent accidental clicks:
  hxxp://  → http://
  [.]      → .
  (dot)    → .
  [at]     → @
  1.2[.]3  → 1.2.3
"""

from __future__ import annotations

import re
import unicodedata


# Defang patterns ordered from most specific to least
_DEFANG_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Protocol defanging
    (re.compile(r"hxxps?", re.IGNORECASE), lambda m: m.group().replace("xx", "tt")),
    (re.compile(r"h\[tt\]ps?", re.IGNORECASE), lambda m: "http" + m.group()[6:]),
    # Dot substitutions
    (re.compile(r"\[\.\]"), "."),
    (re.compile(r"\(dot\)", re.IGNORECASE), "."),
    (re.compile(r"\[dot\]", re.IGNORECASE), "."),
    (re.compile(r"\{\.\}"), "."),
    # At substitutions
    (re.compile(r"\[at\]", re.IGNORECASE), "@"),
    (re.compile(r"\[@\]"), "@"),
    (re.compile(r"\(at\)", re.IGNORECASE), "@"),
    # Bracket-wrapped TLDs: example[.]com
    (re.compile(r"\[([a-z]{2,6})\]", re.IGNORECASE), r"\1"),
]


def refang(text: str) -> str:
    """Reverse defanging transformations in threat intel text."""
    for pattern, replacement in _DEFANG_PATTERNS:
        if callable(replacement):
            text = pattern.sub(replacement, text)
        else:
            text = pattern.sub(replacement, text)
    return text


def normalize_encoding(text: str) -> str:
    """Normalize unicode to NFC, replace smart quotes and em-dashes."""
    text = unicodedata.normalize("NFC", text)
    # Smart quotes → straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Em/en dash → hyphen
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    # Non-breaking space → space
    text = text.replace("\u00a0", " ")
    return text


def remove_noise(text: str) -> str:
    """Remove PDF artifacts, repeated whitespace, null bytes."""
    # Null bytes and control chars (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Repeated whitespace on same line
    text = re.sub(r"[ \t]{2,}", " ", text)
    # More than 2 consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess(text: str) -> str:
    """Full preprocessing pipeline: encoding → noise → defang."""
    text = normalize_encoding(text)
    text = remove_noise(text)
    text = refang(text)
    return text


def sentence_tokenize(text: str) -> list[str]:
    """
    Split text into sentences. Uses nltk if available, falls back to
    a regex-based splitter that handles IP addresses and CVEs correctly.
    """
    try:
        import nltk

        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except ImportError:
        pass

    # Fallback: split on sentence-ending punctuation but not inside
    # patterns like "1.2.3.4" or "CVE-2024-1234" or "v1.0.2"
    sentences = []
    # Protect dots inside IP-like patterns and version numbers
    protected = re.sub(r"(\d+)\.(\d+)", r"\1<<<DOT>>>\2", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
    for part in parts:
        sentences.append(part.replace("<<<DOT>>>", ".").strip())
    return [s for s in sentences if s]
