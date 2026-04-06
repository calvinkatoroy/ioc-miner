"""
SecBERT-based NER extractor.

Loads a fine-tuned token classification model and maps predicted labels
to IOCType. Falls back gracefully if torch/transformers are unavailable.

Label scheme from fine_tune_secbert.py (BIO format):
  B-IOC / I-IOC       → generic IOC, refined to IP/DOMAIN/URL/HASH by regex post-processing
  B-CVE / I-CVE       → CVE identifier
  B-MALWARE / I-MALWARE → malware name (tagged but not an extractable IOC type — used for context)
  O
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ioc_miner.models.ioc import IOC, IOCType

logger = logging.getLogger(__name__)

# Map model label names → IOCType
# IOC is a broad label — we refine it by value pattern after extraction
_LABEL_MAP: dict[str, IOCType | None] = {
    "IOC": IOCType.SHA256,   # placeholder; refined below by _refine_ioc_type()
    "CVE": IOCType.CVE,
    "MALWARE": None,         # not a structured IOC type, skip
    # Legacy fine-grained labels (for forward-compat if model is retrained)
    "IP": IOCType.IP,
    "DOMAIN": IOCType.DOMAIN,
    "URL": IOCType.URL,
    "HASH": IOCType.SHA256,
    "MD5": IOCType.MD5,
    "SHA1": IOCType.SHA1,
    "SHA256": IOCType.SHA256,
    "SHA512": IOCType.SHA512,
    "EMAIL": IOCType.EMAIL,
    "FILEPATH": IOCType.FILEPATH,
}

# Patterns for refining the generic IOC label
_RE_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_RE_DOMAIN = re.compile(r"^[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)+$")
_RE_URL = re.compile(r"^https?://")
_RE_EMAIL = re.compile(r"^[^@]+@[^@]+\.[^@]+$")
_RE_MD5 = re.compile(r"^[0-9a-fA-F]{32}$")
_RE_SHA1 = re.compile(r"^[0-9a-fA-F]{40}$")
_RE_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_RE_SHA512 = re.compile(r"^[0-9a-fA-F]{128}$")

_DEFAULT_MODEL = "jackaduma/SecBERT"


def _refine_ioc_type(value: str) -> IOCType | None:
    """
    Refine a generic IOC label to a specific type based on value pattern.
    Returns None if the value doesn't match any known pattern (will be skipped).
    """
    v = value.strip()
    if _RE_URL.match(v):
        return IOCType.URL
    if _RE_EMAIL.match(v):
        return IOCType.EMAIL
    if _RE_SHA512.match(v):
        return IOCType.SHA512
    if _RE_SHA256.match(v):
        return IOCType.SHA256
    if _RE_SHA1.match(v):
        return IOCType.SHA1
    if _RE_MD5.match(v):
        return IOCType.MD5
    if _RE_IPV4.match(v):
        return IOCType.IP
    if _RE_DOMAIN.match(v):
        return IOCType.DOMAIN
    # Could be a filepath or other indicator — return as filepath if it looks like a path
    if v.startswith("/") or (len(v) > 2 and v[1] == ":"):
        return IOCType.FILEPATH
    return None  # unrecognized — skip


class NERExtractor:
    """
    Wraps a HuggingFace token-classification pipeline for IOC NER.

    Args:
        model_path: HuggingFace model ID or local path to fine-tuned model.
                    Defaults to base SecBERT (useful before fine-tuning).
        device: 'cpu', 'cuda', or 'mps'. Auto-detected if None.
        batch_size: Number of sentences per forward pass.
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 16,
    ):
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self._pipeline = None
        self._device = device

    def _load(self) -> None:
        """Lazy-load the pipeline on first use."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required: pip install transformers torch"
            )

        if self._device is None:
            import torch
            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1  # CPU
        else:
            device = self._device

        logger.info("Loading NER model: %s", self.model_path)
        self._pipeline = pipeline(
            "token-classification",
            model=self.model_path,
            aggregation_strategy="simple",  # merge B/I tokens automatically
            device=device,
        )

    def extract(self, sentence: str, source: str = "") -> list[IOC]:
        return self.extract_batch([sentence], source)

    def extract_all(self, sentences: list[str], source: str = "") -> list[IOC]:
        return self.extract_batch(sentences, source)

    def extract_batch(self, sentences: list[str], source: str = "") -> list[IOC]:
        self._load()
        results: list[IOC] = []

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            predictions = self._pipeline(batch)  # type: ignore[misc]

            # When batch_size=1, pipeline may return a flat list instead of list-of-lists
            if batch and not isinstance(predictions[0], list):
                predictions = [predictions]

            for sentence, preds in zip(batch, predictions):
                for pred in preds:
                    label_raw = pred["entity_group"].upper()
                    # Strip B-/I- prefix if aggregation_strategy didn't remove it
                    if label_raw.startswith(("B-", "I-")):
                        label = label_raw[2:]
                    else:
                        label = label_raw
                    if label not in _LABEL_MAP:
                        continue

                    value = pred["word"].strip().lower()
                    ioc_type = _LABEL_MAP[label]

                    # Generic IOC label → refine to specific type by value pattern
                    if label == "IOC":
                        ioc_type = _refine_ioc_type(value)

                    # Skip if type is None (MALWARE label or unrecognized IOC pattern)
                    if ioc_type is None:
                        continue

                    results.append(
                        IOC(
                            type=ioc_type,
                            value=value,
                            context=sentence,
                            source=source,
                            extracted_by="secbert",
                            confidence=float(pred["score"]),
                        )
                    )

        return results
