"""
Convert CyNER / APTnotes datasets to ioc-miner benchmark JSONL format.

Output format (one sentence per line):
    {"text": "...", "iocs": [{"type": "ip", "value": "1.2.3.4", "verdict": "unknown"}]}

Sources
-------
cyner   — naorm/malware-text-db-cyner + naorm/dnrti-cyner (HuggingFace)
aptref  — APTnotes GitHub corpus (requires local clone, see --aptdir)

Usage
-----
    # CyNER test split only (downloads ~10 MB from HuggingFace)
    python scripts/prepare_eval_data.py --source cyner --output data/eval/cyner_test.jsonl

    # APTnotes plain-text files in a directory
    python scripts/prepare_eval_data.py --source aptref \\
        --aptdir data/aptref/text --output data/eval/aptref.jsonl

    # Both combined
    python scripts/prepare_eval_data.py --source all \\
        --aptdir data/aptref/text --output data/eval/combined.jsonl

Notes
-----
- CyNER "Indicator" spans are refined to a specific IOCType using the same
  regex patterns as RegexExtractor (IP, domain, URL, hash, email).
- CyNER "Vulnerability" spans are mapped to CVE.
- CyNER "Malware" spans are dropped (not a structured IOC type).
- Verdicts are not available in CyNER — all default to "unknown". To add
  verdict labels for classifier evaluation, annotate the output JSONL manually
  or use scripts/annotate_verdicts.py (interactive tool).
- APTnotes IOC extraction is purely regex-based (no ground-truth spans),
  so the output is suitable for regex recall measurement only.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── IOC type refinement (mirrors NERExtractor._refine_ioc_type) ──────────────

_RE_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_RE_DOMAIN = re.compile(r"^[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)+$")
_RE_URL = re.compile(r"^https?://", re.IGNORECASE)
_RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_RE_MD5 = re.compile(r"^[0-9a-fA-F]{32}$")
_RE_SHA1 = re.compile(r"^[0-9a-fA-F]{40}$")
_RE_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_RE_SHA512 = re.compile(r"^[0-9a-fA-F]{128}$")
_RE_CVE = re.compile(r"^CVE-\d{4}-\d{4,7}$", re.IGNORECASE)


def _refine_type(value: str) -> str | None:
    """Return IOCType string or None if value doesn't match a known pattern."""
    v = value.strip()
    if _RE_URL.match(v):
        return "url"
    if _RE_EMAIL.match(v):
        return "email"
    if _RE_CVE.match(v):
        return "cve"
    if _RE_SHA512.match(v):
        return "sha512"
    if _RE_SHA256.match(v):
        return "sha256"
    if _RE_SHA1.match(v):
        return "sha1"
    if _RE_MD5.match(v):
        return "md5"
    if _RE_IPV4.match(v):
        return "ip"
    if _RE_DOMAIN.match(v):
        return "domain"
    return None


# ─── CyNER source ─────────────────────────────────────────────────────────────

_CYNER_DATASETS = [
    "naorm/malware-text-db-cyner",
    "naorm/dnrti-cyner",
]

_CYNER_TYPE_MAP = {
    "Indicator": None,       # refined by value pattern
    "Vulnerability": "cve",
    "Malware": "__skip__",
}


def _load_cyner(split: str = "train") -> list[dict]:
    """
    Load CyNER span-level datasets, group by sentence, and convert to
    benchmark JSONL records.

    Both CyNER datasets use the same schema:
        Original Sentence   — the full sentence text
        Text / Fixed Text   — the entity span text
        Type                — Indicator | Vulnerability | Malware | ...

    Returns list of {"text": ..., "iocs": [...]} dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("pip install datasets")
        sys.exit(1)

    grouped: dict[str, dict] = {}  # sentence_key → {text, iocs}

    for ds_name in _CYNER_DATASETS:
        log.info("Loading %s split=%s ...", ds_name, split)
        try:
            ds = load_dataset(ds_name, split=split)
        except Exception as e:
            log.warning("Could not load %s: %s — skipping", ds_name, e)
            continue

        cols = ds.column_names
        for row in ds:
            sentence = row["Original Sentence"].strip()
            key = f"{ds_name}::{hash(sentence)}"

            if key not in grouped:
                grouped[key] = {"text": sentence, "iocs": []}

            entity_text = (row.get("Fixed Text") or row.get("Text", "")).strip()
            raw_type = row.get("Type", "")
            mapped = _CYNER_TYPE_MAP.get(raw_type, "__skip__")

            if mapped == "__skip__" or not entity_text:
                continue

            if mapped is None:
                # Indicator → refine by value pattern
                ioc_type = _refine_type(entity_text)
            else:
                ioc_type = mapped

            if ioc_type is None:
                continue

            grouped[key]["iocs"].append({
                "type": ioc_type,
                "value": entity_text.lower(),
                "verdict": "unknown",
            })

    # Deduplicate IOCs within each sentence (same type+value)
    records = []
    for entry in grouped.values():
        seen: set[tuple] = set()
        deduped = []
        for ioc in entry["iocs"]:
            key = (ioc["type"], ioc["value"])
            if key not in seen:
                seen.add(key)
                deduped.append(ioc)
        records.append({"text": entry["text"], "iocs": deduped})

    log.info("CyNER: %d sentences, %d total IOC spans",
             len(records), sum(len(r["iocs"]) for r in records))
    return records


# ─── APTnotes source ──────────────────────────────────────────────────────────

def _load_aptref(aptdir: Path) -> list[dict]:
    """
    Regex-extract IOCs from APTnotes plain-text files.

    APTnotes doesn't have span-level ground truth, so this is suitable for
    measuring regex extractor recall against a real-world corpus. The
    'verdicts' field is left as "unknown".

    Each .txt file is sentence-split and processed independently.
    """
    from ioc_miner.preprocessing.normalizer import preprocess, sentence_tokenize
    from ioc_miner.extraction.regex_extractor import RegexExtractor

    txt_files = sorted(aptdir.glob("*.txt"))
    if not txt_files:
        log.warning("No .txt files found in %s", aptdir)
        return []

    log.info("APTnotes: found %d .txt files in %s", len(txt_files), aptdir)
    extractor = RegexExtractor()
    records: list[dict] = []

    for fpath in txt_files:
        try:
            raw = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.warning("Could not read %s: %s", fpath, e)
            continue

        clean = preprocess(raw)
        sentences = sentence_tokenize(clean)

        for sentence in sentences:
            iocs = extractor.extract(sentence, source=str(fpath))
            records.append({
                "text": sentence,
                "iocs": [
                    {"type": i.type.value, "value": i.value.lower(), "verdict": "unknown"}
                    for i in iocs
                ],
            })

    log.info("APTnotes: %d sentences, %d total IOC spans",
             len(records), sum(len(r["iocs"]) for r in records))
    return records


# ─── Writer ───────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), output)


# ─── Stats summary ────────────────────────────────────────────────────────────

def print_stats(records: list[dict]) -> None:
    from collections import Counter
    type_counts: Counter = Counter()
    verdict_counts: Counter = Counter()
    sentences_with_iocs = 0
    for r in records:
        if r["iocs"]:
            sentences_with_iocs += 1
        for ioc in r["iocs"]:
            type_counts[ioc["type"]] += 1
            verdict_counts[ioc["verdict"]] += 1

    print(f"\n{'─'*50}")
    print(f"Total sentences : {len(records)}")
    print(f"With IOCs       : {sentences_with_iocs}")
    print(f"Total IOC spans : {sum(type_counts.values())}")
    print("\nBy type:")
    for t, n in sorted(type_counts.items()):
        print(f"  {t:<12} {n}")
    print("\nBy verdict:")
    for v, n in sorted(verdict_counts.items()):
        print(f"  {v:<12} {n}")
    print(f"{'─'*50}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare evaluation JSONL datasets for ioc-miner benchmarking"
    )
    parser.add_argument(
        "--source",
        choices=["cyner", "aptref", "all"],
        default="cyner",
        help="Dataset source to convert (default: cyner)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="CyNER dataset split to use: train | validation | test (default: train)",
    )
    parser.add_argument(
        "--aptdir",
        type=Path,
        default=Path("data/aptref/text"),
        help="Directory containing APTnotes .txt files (default: data/aptref/text)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/cyner_eval.jsonl"),
        help="Output JSONL path (default: data/eval/cyner_eval.jsonl)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics after writing",
    )
    args = parser.parse_args()

    records: list[dict] = []

    if args.source in ("cyner", "all"):
        records.extend(_load_cyner(split=args.split))

    if args.source in ("aptref", "all"):
        if not args.aptdir.exists():
            log.error(
                "--aptdir %s does not exist. Clone APTnotes and extract text files first.",
                args.aptdir,
            )
            if args.source == "aptref":
                sys.exit(1)
        else:
            records.extend(_load_aptref(args.aptdir))

    if not records:
        log.error("No records produced. Check your --source and --aptdir arguments.")
        sys.exit(1)

    write_jsonl(records, args.output)

    if args.stats:
        print_stats(records)


if __name__ == "__main__":
    main()
