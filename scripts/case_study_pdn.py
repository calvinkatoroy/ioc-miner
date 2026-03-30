"""
Case Study: 2024 PDN Ransomware Incident (Brain Cipher)

The Pusat Data Nasional (PDN) attack in June 2024 is the largest publicly
documented ransomware incident in Indonesian history. Brain Cipher encrypted
PDN servers, disrupting 282 government agencies and demanding $8M ransom.
BSSN and multiple threat intelligence vendors published IOC reports.

This script runs ioc-miner across all available PDN-related report files and
produces a consolidated IOC set with verdict labels, suitable for:
  1. Comparing extracted IOCs against BSSN's official published indicators
  2. Demonstrating real-world pipeline performance in the paper's case study

Data sources (collect manually before running):
─────────────────────────────────────────────────────────────────────────────
  BSSN advisory (PDF):
    https://bssn.go.id/wp-content/uploads/2024/06/ADVISORY-BRAIN-CIPHER.pdf

  Recorded Future report (PDF or HTML):
    Search: "Brain Cipher PDN Recorded Future 2024 IOC"

  SentinelOne blog (HTML):
    Search: "Brain Cipher ransomware SentinelOne 2024"

  Any public STIX/MISP bundles from PDN incident
─────────────────────────────────────────────────────────────────────────────

Usage
-----
    # Place report files in data/case_study/pdn/ (PDF, HTML, or TXT)
    python scripts/case_study_pdn.py --input data/case_study/pdn/ \\
        --output results/pdn_iocs.csv

    # With SecBERT NER model
    python scripts/case_study_pdn.py --input data/case_study/pdn/ \\
        --model models/secbert-ner --output results/pdn_iocs.csv

    # Compare against BSSN ground truth
    python scripts/case_study_pdn.py --input data/case_study/pdn/ \\
        --ground-truth data/eval/pdn_bssn_gt.jsonl \\
        --output results/pdn_iocs.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Known Brain Cipher / PDN IOC seeds for cross-referencing
# Source: publicly disclosed in threat intel blogs post-incident
_KNOWN_PDN_IOCS: dict[str, str] = {
    # Hashes associated with Brain Cipher ransomware payloads (SHA256 fragments — illustrative)
    # Replace with actual published hashes from BSSN advisory
    # "actual_sha256_here": "brain_cipher_payload",
}

_PDN_MALICIOUS_KEYWORDS = {
    "brain cipher", "brainciper", "pdns", "pdn ransomware",
    "lockbit", "esxi",  # Brain Cipher is LockBit 3.0 fork
    "ransom note", "!!!read me", "how_to_decrypt",
}


def _ingest_file(path: Path, model: str | None) -> list:
    """Ingest a single file and return classified IOCs."""
    from ioc_miner.ingestion.html import HTMLIngestor
    from ioc_miner.ingestion.pdf import PDFIngestor
    from ioc_miner.ingestion.plaintext import PlaintextIngestor
    from ioc_miner.preprocessing.normalizer import preprocess, sentence_tokenize
    from ioc_miner.extraction.regex_extractor import RegexExtractor
    from ioc_miner.extraction.context_classifier import ContextClassifier
    from ioc_miner.models.ioc import IOCVerdict

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        ingestor = PDFIngestor()
    elif suffix in (".html", ".htm"):
        ingestor = HTMLIngestor()
    else:
        ingestor = PlaintextIngestor()

    try:
        raw = ingestor.ingest(str(path))
    except Exception as e:
        log.warning("Could not ingest %s: %s", path.name, e)
        return []

    clean = preprocess(raw)
    sentences = sentence_tokenize(clean)

    extractor = RegexExtractor()
    iocs = extractor.extract_all(sentences, source=str(path))

    if model:
        from ioc_miner.extraction.ner_extractor import NERExtractor
        ner = NERExtractor(model_path=model)
        ner_iocs = ner.extract_batch(sentences, source=str(path))
        existing = {(i.type, i.value) for i in iocs}
        for ioc in ner_iocs:
            if (ioc.type, ioc.value) not in existing:
                iocs.append(ioc)

    classifier = ContextClassifier(use_ml=False)
    return classifier.classify_all(iocs)


def _print_summary(all_iocs: list, source_map: dict[str, int]) -> None:
    from ioc_miner.models.ioc import IOCVerdict

    total = len(all_iocs)
    verdict_counts = Counter(i.verdict.value for i in all_iocs)
    type_counts = Counter(i.type.value for i in all_iocs)

    print("\n" + "═" * 60)
    print("  PDN CASE STUDY — SUMMARY")
    print("═" * 60)
    print(f"  Reports processed : {len(source_map)}")
    print(f"  Total unique IOCs : {total}")
    print()
    print("  By verdict:")
    for v in ("malicious", "benign", "sinkhole", "unknown"):
        n = verdict_counts.get(v, 0)
        bar = "█" * (n * 30 // max(total, 1))
        print(f"    {v:<10} {n:>4}  {bar}")
    print()
    print("  By type:")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<12} {n:>4}")
    print()
    print("  By source:")
    for src, n in sorted(source_map.items(), key=lambda x: -x[1]):
        print(f"    {Path(src).name:<40} {n:>4} IOCs")
    print("═" * 60 + "\n")


def _save_csv(all_iocs: list, output: Path) -> None:
    from ioc_miner.output.csv_formatter import to_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_csv(all_iocs), encoding="utf-8")
    log.info("Saved %d IOCs to %s", len(all_iocs), output)


def _save_stix(all_iocs: list, output: Path) -> None:
    from ioc_miner.output.stix_formatter import to_stix_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_stix_json(all_iocs), encoding="utf-8")
    log.info("Saved STIX bundle to %s", output)


def _run_benchmark_comparison(all_iocs: list, gt_path: Path) -> None:
    """Compare extracted IOCs against BSSN ground truth."""
    from ioc_miner.evaluation.benchmark import BenchmarkEvaluator, load_ground_truth
    from ioc_miner.extraction.regex_extractor import RegexExtractor

    log.info("Loading BSSN ground truth: %s", gt_path)
    gt = load_ground_truth(gt_path)

    evaluator = BenchmarkEvaluator()

    # Build a mock extractor that returns our already-extracted IOCs
    class _PrecomputedExtractor:
        def __init__(self, iocs):
            self._iocs = iocs
        def extract_all(self, sentences, source=""):
            return self._iocs

    result = evaluator.evaluate_extraction(_PrecomputedExtractor(all_iocs), gt)

    print("\n" + "─" * 60)
    print("  COMPARISON vs. BSSN GROUND TRUTH")
    print("─" * 60)
    print(result.report_markdown())

    verdict_metrics = evaluator.evaluate_verdict(all_iocs, gt)
    if verdict_metrics.total > 0:
        print()
        print(verdict_metrics.report_markdown("Verdict Accuracy"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Case study: IOC extraction from PDN ransomware reports"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/case_study/pdn"),
        help="Directory containing PDN report files (.pdf, .html, .txt)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID or local path for SecBERT NER (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pdn_iocs.csv"),
        help="Output file path (.csv or .json for STIX)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        metavar="JSONL",
        help="BSSN ground truth JSONL for evaluation comparison (optional)",
    )
    parser.add_argument(
        "--no-benign",
        action="store_true",
        help="Exclude benign and sinkholed IOCs from output",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error(
            "Input directory not found: %s\n"
            "Create it and place PDN report files there:\n"
            "  mkdir -p %s\n"
            "  # Download BSSN advisory, vendor blogs, etc. into that directory",
            args.input, args.input,
        )
        sys.exit(1)

    report_files = [
        f for f in sorted(args.input.iterdir())
        if f.suffix.lower() in (".pdf", ".html", ".htm", ".txt")
    ]
    if not report_files:
        log.error("No .pdf/.html/.txt files found in %s", args.input)
        sys.exit(1)

    log.info("Found %d report files", len(report_files))

    # ── Ingest + extract ──────────────────────────────────────────────────────
    from ioc_miner.models.ioc import IOC, IOCVerdict

    all_iocs: list[IOC] = []
    source_map: dict[str, int] = {}
    seen: set[tuple] = set()

    for fpath in report_files:
        log.info("Processing: %s", fpath.name)
        iocs = _ingest_file(fpath, args.model)

        if args.no_benign:
            iocs = [i for i in iocs if i.verdict not in (IOCVerdict.BENIGN, IOCVerdict.SINKHOLE)]

        new = 0
        for ioc in iocs:
            key = (ioc.type, ioc.value)
            if key not in seen:
                seen.add(key)
                all_iocs.append(ioc)
                new += 1

        source_map[str(fpath)] = new
        log.info("  → %d new unique IOCs", new)

    if not all_iocs:
        log.warning("No IOCs extracted. Check report file contents.")
        sys.exit(0)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(all_iocs, source_map)

    # ── Save output ───────────────────────────────────────────────────────────
    if args.output.suffix == ".json":
        _save_stix(all_iocs, args.output)
    else:
        _save_csv(all_iocs, args.output)

    # ── Benchmark comparison ──────────────────────────────────────────────────
    if args.ground_truth:
        if not args.ground_truth.exists():
            log.error("Ground truth file not found: %s", args.ground_truth)
        else:
            _run_benchmark_comparison(all_iocs, args.ground_truth)


if __name__ == "__main__":
    main()
