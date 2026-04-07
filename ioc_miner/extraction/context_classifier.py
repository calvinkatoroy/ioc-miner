"""
Context classifier — determines whether an extracted IOC is a malicious
indicator, a benign mention, or a sinkhole/example.

Architecture:
  1. Rule-based pass  (fast, high precision — handles clear-cut cases)
  2. Zero-shot NLI    (ML fallback for UNKNOWN cases, no training data needed)

The NLI model reasons: given the sentence context, does it entail that the
IOC is "a malicious indicator used in an attack" vs "a benign or example mention"?

Rule-based signals:
  MALICIOUS  → "C2", "beacon", "dropper", "exfiltrate", "command and control", ...
  BENIGN     → "sinkhole", "example", "researcher", "dns resolver", ...
  SINKHOLE   → "sinkhole(d)" near the IOC value
"""

from __future__ import annotations

import logging
import re

from ioc_miner.models.ioc import IOC, IOCVerdict

logger = logging.getLogger(__name__)

# ─── Rule-based signals ───────────────────────────────────────────────────────

_MALICIOUS_SIGNALS = re.compile(
    r"\b(?:"
    r"c2|command.and.control|command\s*&\s*control|c&c"
    r"|beacon|beaconing"
    r"|dropper|downloader|loader"
    r"|malware|ransomware|trojan|rat|backdoor|rootkit|spyware|adware|worm|botnet"
    r"|exfiltrat|exfil"
    r"|payload|shellcode|exploit|exploit(?:ed|ing|ation)"
    r"|attacker|threat.actor|adversary|apt|campaign"
    r"|c2.server|c2.infrastructure|c2.channel"
    r"|lateral.movement|privilege.escalation|persistence"
    r"|staging.server|pivot"
    r")\b",
    re.IGNORECASE,
)

_BENIGN_SIGNALS = re.compile(
    r"\b(?:"
    r"sinkhole|sinkhol"
    r"|example|placeholder|redacted|sanitized|anonymized"
    r"|legitimate|legit\b"
    r"|researcher|research(?:er)?|analyst|vendor"
    r"|whitelist|allowlist|trusted"
    r"|dns.resolver|recursive.resolver|public.dns"
    r"|scan(?:ner|ning)|shodan|censys|greynoise"
    r"|benign|clean|safe"
    r"|test(?:ing)?\b|lab\b"
    r")\b",
    re.IGNORECASE,
)

_SINKHOLE_SIGNALS = re.compile(r"\bsinkhole(?:d|ing)?\b", re.IGNORECASE)

# ─── NLI hypothesis templates ─────────────────────────────────────────────────

_NLI_HYPOTHESES = [
    "This indicator is a malicious IOC used in a cyberattack or threat campaign.",
    "This indicator is a benign, example, or sinkholed address not used maliciously.",
]
_NLI_LABELS = [IOCVerdict.MALICIOUS, IOCVerdict.BENIGN]
# Confidence threshold — below this, keep UNKNOWN
_NLI_THRESHOLD = 0.65


def _window(ioc_value: str, context: str, chars: int = 150) -> str:
    """Return text within `chars` characters of the IOC value in context."""
    idx = context.find(ioc_value)
    if idx == -1:
        return context
    start = max(0, idx - chars)
    end = min(len(context), idx + len(ioc_value) + chars)
    return context[start:end]


_RE_BARE_PREFIX = re.compile(r"^[Ii]ndicator\s*:?\s*")

def _has_meaningful_context(ioc_value: str, window: str, min_extra: int = 30) -> bool:
    """
    Return True if window contains enough context beyond the IOC value itself.

    Prevents NLI from running on bare "Indicator: <value>" entries where the
    model has no signal and defaults to benign regardless of the actual label.
    """
    extra = window.replace(ioc_value, "", 1)
    extra = _RE_BARE_PREFIX.sub("", extra).strip()
    return len(extra) >= min_extra


# ─── Zero-shot NLI classifier ─────────────────────────────────────────────────

class _NLIClassifier:
    """
    Wraps a zero-shot classification pipeline.
    Lazy-loaded on first use to avoid import overhead when ML is not needed.

    Default model: cross-encoder/nli-MiniLM2-L6-H768
      - 22M params, fast on CPU (~50ms/sentence)
      - Better precision than BART-large-mnli for short security sentences
    """

    _DEFAULT_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self._DEFAULT_MODEL
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers is required: pip install transformers")

        logger.info("Loading NLI model: %s", self.model_name)
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=-1,  # CPU — fast enough for short sentences
        )

    def classify(self, text: str) -> tuple[IOCVerdict, float]:
        """
        Returns (verdict, confidence). verdict is UNKNOWN if below threshold.
        """
        self._load()
        result = self._pipeline(  # type: ignore[misc]
            text,
            candidate_labels=[h for h in _NLI_HYPOTHESES],
            hypothesis_template="{}",
            multi_label=False,
        )
        # result["labels"] is sorted by score descending
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        if top_score < _NLI_THRESHOLD:
            return IOCVerdict.UNKNOWN, top_score

        # Map hypothesis back to verdict
        idx = _NLI_HYPOTHESES.index(top_label)
        return _NLI_LABELS[idx], top_score

    def classify_batch(self, texts: list[str]) -> list[tuple[IOCVerdict, float]]:
        self._load()
        results = self._pipeline(  # type: ignore[misc]
            texts,
            candidate_labels=_NLI_HYPOTHESES,
            hypothesis_template="{}",
            multi_label=False,
        )
        if isinstance(results, dict):
            results = [results]

        output = []
        for result in results:
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            if top_score < _NLI_THRESHOLD:
                output.append((IOCVerdict.UNKNOWN, top_score))
            else:
                idx = _NLI_HYPOTHESES.index(top_label)
                output.append((_NLI_LABELS[idx], top_score))
        return output


# ─── Public classifier ────────────────────────────────────────────────────────

class ContextClassifier:
    """
    Classifies each IOC as MALICIOUS / BENIGN / SINKHOLE / UNKNOWN.

    Args:
        use_ml: if True, load zero-shot NLI model to resolve UNKNOWN cases.
                Defaults to False (rule-based only, no model download).
        ml_model: override the NLI model ID (default: MiniLM cross-encoder).
    """

    def __init__(self, use_ml: bool = False, ml_model: str | None = None):
        self.use_ml = use_ml
        self._nli: _NLIClassifier | None = _NLIClassifier(ml_model) if use_ml else None

    def classify(self, ioc: IOC) -> IOC:
        """Mutates ioc.verdict and ioc.verdict_confidence in place, returns ioc."""
        window = _window(ioc.value, ioc.context)

        # ── Rule-based pass (fast, high precision) ────────────────────────────
        if _SINKHOLE_SIGNALS.search(window):
            ioc.verdict = IOCVerdict.SINKHOLE
            ioc.verdict_confidence = 0.90
        elif _BENIGN_SIGNALS.search(window):
            ioc.verdict = IOCVerdict.BENIGN
            ioc.verdict_confidence = 0.80
        elif _MALICIOUS_SIGNALS.search(window):
            ioc.verdict = IOCVerdict.MALICIOUS
            ioc.verdict_confidence = 0.85
        else:
            ioc.verdict = IOCVerdict.UNKNOWN
            ioc.verdict_confidence = 0.0

        # ── ML fallback for UNKNOWN ───────────────────────────────────────────
        if ioc.verdict == IOCVerdict.UNKNOWN and self._nli is not None:
            if _has_meaningful_context(ioc.value, window):
                verdict, conf = self._nli.classify(window)
                ioc.verdict = verdict
                ioc.verdict_confidence = conf

        return ioc

    def classify_all(self, iocs: list[IOC]) -> list[IOC]:
        """
        Batch-aware: collects all UNKNOWN IOCs and runs NLI in one batch
        instead of one call per IOC (significantly faster on CPU).
        """
        # Rule-based pass first
        for ioc in iocs:
            window = _window(ioc.value, ioc.context)
            if _SINKHOLE_SIGNALS.search(window):
                ioc.verdict = IOCVerdict.SINKHOLE
                ioc.verdict_confidence = 0.90
            elif _BENIGN_SIGNALS.search(window):
                ioc.verdict = IOCVerdict.BENIGN
                ioc.verdict_confidence = 0.80
            elif _MALICIOUS_SIGNALS.search(window):
                ioc.verdict = IOCVerdict.MALICIOUS
                ioc.verdict_confidence = 0.85
            else:
                ioc.verdict = IOCVerdict.UNKNOWN
                ioc.verdict_confidence = 0.0

        # ML batch pass for remaining UNKNOWNs — only when context is meaningful
        if self._nli is not None:
            unknowns = [i for i in iocs if i.verdict == IOCVerdict.UNKNOWN]
            unknowns = [i for i in unknowns if _has_meaningful_context(i.value, _window(i.value, i.context))]
            if unknowns:
                windows = [_window(i.value, i.context) for i in unknowns]
                verdicts = self._nli.classify_batch(windows)
                for ioc, (verdict, conf) in zip(unknowns, verdicts):
                    ioc.verdict = verdict
                    ioc.verdict_confidence = conf

        return iocs
