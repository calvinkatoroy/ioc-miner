"""
Run the full ioc-miner benchmark evaluation.

Evaluates all extractors against a labeled JSONL ground truth file and
prints a side-by-side comparison table (P/R/F1 per IOC type).

Usage
-----
    # Regex only (no extra deps)
    python scripts/run_benchmark.py --eval data/eval/cyner_eval.jsonl

    # Regex + SecBERT NER (requires fine-tuned model)
    python scripts/run_benchmark.py --eval data/eval/cyner_eval.jsonl \\
        --model models/secbert-ner

    # All extractors including baselines (requires iocextract, ioc-finder)
    python scripts/run_benchmark.py --eval data/eval/cyner_eval.jsonl \\
        --baselines all

    # Save results to JSON for paper tables
    python scripts/run_benchmark.py --eval data/eval/cyner_eval.jsonl \\
        --baselines all --output results/benchmark.json

    # Include verdict accuracy (ContextClassifier)
    python scripts/run_benchmark.py --eval data/eval/cyner_eval.jsonl \\
        --verdict --use-ml

Output
------
Prints a Markdown comparison table to stdout. If --output is given,
also writes full JSON results (P/R/F1, TP/FP/FN per type, verdict confusion).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_extractor(name: str, model: str | None = None):
    """Resolve extractor name to an instance."""
    if name == "regex":
        from ioc_miner.extraction.regex_extractor import RegexExtractor
        return RegexExtractor()

    if name == "secbert":
        if not model:
            log.error("--model is required for --extractors secbert")
            sys.exit(1)
        from ioc_miner.extraction.ner_extractor import NERExtractor
        return NERExtractor(model_path=model)

    if name == "iocextract":
        from ioc_miner.evaluation.baselines import IocextractBaseline
        return IocextractBaseline()

    if name == "ioc-finder":
        from ioc_miner.evaluation.baselines import IocFinderBaseline
        return IocFinderBaseline()

    if name == "cacador":
        from ioc_miner.evaluation.baselines import CacadorBaseline
        return CacadorBaseline()

    log.error("Unknown extractor: %s", name)
    sys.exit(1)


def _resolve_extractor_names(names: list[str], model: str | None) -> list[str]:
    """Expand 'all' and validate names."""
    all_extractors = ["regex", "iocextract", "ioc-finder", "cacador"]
    if model:
        all_extractors.insert(1, "secbert")

    expanded: list[str] = []
    for name in names:
        if name == "all":
            expanded.extend(all_extractors)
        else:
            expanded.append(name)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ioc-miner extractors against labeled ground truth"
    )
    parser.add_argument(
        "--eval",
        type=Path,
        required=True,
        metavar="JSONL",
        help="Path to labeled evaluation JSONL (from scripts/prepare_eval_data.py)",
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=["regex"],
        metavar="NAME",
        help=(
            "Extractors to evaluate: regex, secbert, iocextract, ioc-finder, cacador, all "
            "(default: regex)"
        ),
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        metavar="NAME",
        help="Shorthand: add baseline extractors (all | iocextract | ioc-finder | cacador)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="HuggingFace model ID or local path for SecBERT NER extractor",
    )
    parser.add_argument(
        "--verdict",
        action="store_true",
        help="Also evaluate ContextClassifier verdict accuracy (requires verdict-labeled GT)",
    )
    parser.add_argument(
        "--use-ml",
        action="store_true",
        help="Enable NLI zero-shot classifier in ContextClassifier (slow on CPU)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="JSON",
        help="Save full results to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.disable(logging.INFO)

    # ── Load ground truth ─────────────────────────────────────────────────────
    if not args.eval.exists():
        log.error("Eval file not found: %s", args.eval)
        log.error("Run: python scripts/prepare_eval_data.py --output %s", args.eval)
        sys.exit(1)

    from ioc_miner.evaluation.benchmark import BenchmarkEvaluator, load_ground_truth

    log.info("Loading ground truth: %s", args.eval)
    gt = load_ground_truth(args.eval)
    log.info("Loaded %d sentences", len(gt))

    if not gt:
        log.error("Ground truth is empty")
        sys.exit(1)

    # ── Resolve extractors ────────────────────────────────────────────────────
    extractor_names = list(args.extractors)
    if args.baselines is not None:
        if len(args.baselines) == 0:
            # --baselines with no args → all
            extractor_names.extend(["iocextract", "ioc-finder", "cacador"])
        else:
            extractor_names.extend(args.baselines)

    extractor_names = _resolve_extractor_names(extractor_names, args.model)

    extractors: dict = {}
    for name in extractor_names:
        log.info("Loading extractor: %s", name)
        try:
            extractors[name] = _load_extractor(name, args.model)
        except (ImportError, RuntimeError) as e:
            log.warning("Skipping %s: %s", name, e)

    if not extractors:
        log.error("No extractors available")
        sys.exit(1)

    # ── Run extraction benchmark ──────────────────────────────────────────────
    evaluator = BenchmarkEvaluator()

    print("\n" + "═" * 72)
    print("  IOC EXTRACTION BENCHMARK")
    print(f"  Eval set : {args.eval}  ({len(gt)} sentences)")
    print(f"  Extractors: {', '.join(extractors)}")
    print("═" * 72 + "\n")

    extraction_results: dict = {}
    for name, extractor in extractors.items():
        log.info("Evaluating: %s", name)
        try:
            result = evaluator.evaluate_extraction(extractor, gt)
            extraction_results[name] = result
        except Exception as e:
            log.warning("Evaluation failed for %s: %s", name, e)

    if len(extraction_results) == 1:
        name, result = next(iter(extraction_results.items()))
        print(result.report_markdown(title=name))
    else:
        print(BenchmarkEvaluator.comparison_markdown(extraction_results))

    # ── Verdict benchmark ─────────────────────────────────────────────────────
    verdict_results: dict = {}
    if args.verdict:
        print("\n" + "─" * 72)
        print("  VERDICT CLASSIFICATION BENCHMARK  (ContextClassifier)")
        print("─" * 72 + "\n")

        from ioc_miner.extraction.context_classifier import ContextClassifier
        from ioc_miner.extraction.regex_extractor import RegexExtractor

        sentences = [s.text for s in gt]
        classifier = ContextClassifier(use_ml=args.use_ml)

        # Run regex + classifier (primary pipeline)
        iocs = RegexExtractor().extract_all(sentences, source=str(args.eval))
        classified = classifier.classify_all(iocs)
        vm = evaluator.evaluate_verdict(classified, gt)
        verdict_results["ioc-miner"] = vm

        print(vm.report_markdown(title="ioc-miner (regex + ContextClassifier)"))

        if vm.total == 0:
            print(
                "\n[!] No verdict-labeled IOCs found in ground truth.\n"
                "    Add verdict fields ('malicious'|'benign'|'sinkhole') to your\n"
                "    eval JSONL to enable verdict evaluation.\n"
            )

    # ── Save results ──────────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out: dict = {
            "eval_file": str(args.eval),
            "num_sentences": len(gt),
            "extraction": {name: r.to_dict() for name, r in extraction_results.items()},
        }
        if verdict_results:
            out["verdict"] = {name: v.to_dict() for name, v in verdict_results.items()}

        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        log.info("Results saved to %s", args.output)

    print()


if __name__ == "__main__":
    main()
