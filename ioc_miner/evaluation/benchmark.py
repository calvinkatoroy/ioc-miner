"""
Evaluation framework for ioc-miner.

Ground truth format (JSONL — one sentence per line):
    {"text": "Malware beaconed to 1.2.3.4.", "iocs": [{"type": "ip", "value": "1.2.3.4", "verdict": "malicious"}]}

Usage:
    from ioc_miner.evaluation.benchmark import BenchmarkEvaluator, load_ground_truth
    from ioc_miner.extraction.regex_extractor import RegexExtractor
    from ioc_miner.extraction.context_classifier import ContextClassifier

    gt = load_ground_truth("data/eval/annotated.jsonl")
    evaluator = BenchmarkEvaluator()

    extraction = evaluator.evaluate_extraction(RegexExtractor(), gt)
    print(extraction.report_markdown("RegexExtractor"))

    iocs = RegexExtractor().extract_all([s.text for s in gt])
    classified = ContextClassifier().classify_all(iocs)
    verdict = evaluator.evaluate_verdict(classified, gt)
    print(verdict.report_markdown("ContextClassifier"))

    comparison = evaluator.compare_extractors(gt, {
        "ioc-miner": RegexExtractor(),
        "iocextract": IocextractBaseline(),
        "ioc-finder": IocFinderBaseline(),
    })
    print(BenchmarkEvaluator.comparison_markdown(comparison))
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ioc_miner.models.ioc import IOC, IOCType, IOCVerdict


# ─── Ground truth schema ──────────────────────────────────────────────────────

@dataclass
class GroundTruthIOC:
    type: IOCType
    value: str
    verdict: IOCVerdict = IOCVerdict.UNKNOWN


@dataclass
class GroundTruthSentence:
    text: str
    iocs: list[GroundTruthIOC]


def load_ground_truth(path: str | Path) -> list[GroundTruthSentence]:
    """
    Load annotated JSONL file. Each line must be:
        {"text": "...", "iocs": [{"type": "ip", "value": "1.2.3.4", "verdict": "malicious"}, ...]}

    - "verdict" is optional; defaults to "unknown" (sentence skipped in verdict eval).
    - Lines with invalid JSON or missing "text" raise ValueError.
    """
    samples: list[GroundTruthSentence] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {lineno}: invalid JSON — {e}") from e
            if "text" not in obj:
                raise ValueError(f"Line {lineno}: missing 'text' field")

            iocs: list[GroundTruthIOC] = []
            for raw in obj.get("iocs", []):
                try:
                    ioc_type = IOCType(raw["type"].lower())
                except (KeyError, ValueError):
                    continue
                verdict = IOCVerdict(raw.get("verdict", "unknown").lower())
                iocs.append(GroundTruthIOC(
                    type=ioc_type,
                    value=raw["value"].lower(),
                    verdict=verdict,
                ))

            samples.append(GroundTruthSentence(text=obj["text"], iocs=iocs))

    return samples


# ─── Metrics dataclasses ──────────────────────────────────────────────────────

@dataclass
class ExtractionMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0

    def __iadd__(self, other: ExtractionMetrics) -> ExtractionMetrics:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class EvalResults:
    per_type: dict[IOCType, ExtractionMetrics] = field(default_factory=dict)
    micro: ExtractionMetrics = field(default_factory=ExtractionMetrics)

    def report_markdown(self, title: str = "") -> str:
        lines: list[str] = []
        if title:
            lines.append(f"## {title}\n")
        lines.append("| IOC Type   | Precision | Recall | F1    | TP | FP | FN |")
        lines.append("|------------|-----------|--------|-------|----|----|----|")
        for ioc_type in sorted(self.per_type, key=lambda t: t.value):
            m = self.per_type[ioc_type]
            lines.append(
                f"| {ioc_type.value:<10} | {m.precision:.3f}     | {m.recall:.3f}  "
                f"| {m.f1:.3f} | {m.tp:>2} | {m.fp:>2} | {m.fn:>2} |"
            )
        m = self.micro
        lines.append(
            f"| **micro**  | **{m.precision:.3f}** | **{m.recall:.3f}** "
            f"| **{m.f1:.3f}** | {m.tp:>2} | {m.fp:>2} | {m.fn:>2} |"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "per_type": {t.value: m.to_dict() for t, m in self.per_type.items()},
            "micro": self.micro.to_dict(),
        }


@dataclass
class VerdictMetrics:
    # confusion[true_verdict][pred_verdict] = count
    confusion: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def report_markdown(self, title: str = "") -> str:
        lines: list[str] = []
        if title:
            lines.append(f"## {title}\n")
        lines.append(f"**Accuracy: {self.accuracy:.3f}** ({self.correct}/{self.total})\n")

        labels = sorted(
            {v for row in self.confusion.values() for v in row} | set(self.confusion.keys())
        )
        if not labels:
            lines.append("*(no verdict-labeled IOCs in ground truth)*")
            return "\n".join(lines)

        lines.append("**Confusion matrix** (rows = true, cols = predicted):\n")
        header = "| True \\ Pred | " + " | ".join(f"{l:10}" for l in labels) + " |"
        lines.append(header)
        lines.append("|" + "-------------|" * (len(labels) + 1))
        for true in labels:
            row = f"| {true:<12} | "
            row += " | ".join(
                str(self.confusion.get(true, {}).get(pred, 0)).rjust(10) for pred in labels
            )
            row += " |"
            lines.append(row)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "correct": self.correct,
            "total": self.total,
            "confusion": {k: dict(v) for k, v in self.confusion.items()},
        }


# ─── Extractor protocol ───────────────────────────────────────────────────────

class ExtractorProtocol(Protocol):
    def extract_all(self, sentences: list[str], source: str = "") -> list[IOC]: ...


# ─── Evaluator ────────────────────────────────────────────────────────────────

class BenchmarkEvaluator:
    """
    Evaluates IOC extraction and verdict classification against labeled ground truth.

    Extraction evaluation is corpus-level: all (type, value) pairs predicted
    across the full sentence set are compared against all annotated ground truth pairs.
    This matches standard practice in IOC extraction papers.
    """

    def evaluate_extraction(
        self,
        extractor: ExtractorProtocol,
        ground_truth: list[GroundTruthSentence],
        source: str = "eval",
    ) -> EvalResults:
        """
        Compute precision, recall, and F1 per IOC type and micro-averaged.

        Match criterion: exact (IOCType, lowercased value) pair.
        """
        sentences = [s.text for s in ground_truth]
        predicted = extractor.extract_all(sentences, source=source)

        # Build corpus-level sets
        gt_by_type: dict[IOCType, set[str]] = defaultdict(set)
        for sample in ground_truth:
            for g in sample.iocs:
                gt_by_type[g.type].add(g.value.lower())

        pred_by_type: dict[IOCType, set[str]] = defaultdict(set)
        for ioc in predicted:
            pred_by_type[ioc.type].add(ioc.value.lower())

        all_types = set(gt_by_type.keys()) | set(pred_by_type.keys())
        per_type: dict[IOCType, ExtractionMetrics] = {}
        micro = ExtractionMetrics()

        for ioc_type in all_types:
            gt_set = gt_by_type[ioc_type]
            pred_set = pred_by_type[ioc_type]
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            per_type[ioc_type] = ExtractionMetrics(tp=tp, fp=fp, fn=fn)
            micro.tp += tp
            micro.fp += fp
            micro.fn += fn

        return EvalResults(per_type=per_type, micro=micro)

    def evaluate_verdict(
        self,
        classified_iocs: list[IOC],
        ground_truth: list[GroundTruthSentence],
    ) -> VerdictMetrics:
        """
        Compare predicted verdict labels against ground truth verdict labels.

        Only evaluates IOCs where the ground truth verdict is not UNKNOWN
        (i.e., explicitly annotated). Unmatched predictions are ignored.
        """
        gt_verdicts: dict[tuple[IOCType, str], IOCVerdict] = {}
        for sample in ground_truth:
            for g in sample.iocs:
                if g.verdict != IOCVerdict.UNKNOWN:
                    gt_verdicts[(g.type, g.value.lower())] = g.verdict

        metrics = VerdictMetrics()
        for ioc in classified_iocs:
            key = (ioc.type, ioc.value.lower())
            if key not in gt_verdicts:
                continue
            true_verdict = gt_verdicts[key]
            pred_verdict = ioc.verdict
            metrics.confusion[true_verdict.value][pred_verdict.value] += 1
            metrics.total += 1
            if true_verdict == pred_verdict:
                metrics.correct += 1

        return metrics

    def compare_extractors(
        self,
        ground_truth: list[GroundTruthSentence],
        extractors: dict[str, ExtractorProtocol],
        source: str = "eval",
    ) -> dict[str, EvalResults]:
        """Run evaluate_extraction for each named extractor."""
        return {
            name: self.evaluate_extraction(ext, ground_truth, source=source)
            for name, ext in extractors.items()
        }

    @staticmethod
    def comparison_markdown(comparison: dict[str, EvalResults]) -> str:
        """
        Side-by-side F1 comparison table — one column per extractor, one row per IOC type.
        Suitable for pasting directly into a paper table.
        """
        names = list(comparison.keys())
        lines = [
            "| IOC Type   | " + " | ".join(f"{n} P" for n in names)
            + " | " + " | ".join(f"{n} R" for n in names)
            + " | " + " | ".join(f"{n} F1" for n in names) + " |"
        ]
        lines.append("|------------|" + "--------|" * (len(names) * 3))

        all_types = sorted(
            {t for r in comparison.values() for t in r.per_type},
            key=lambda t: t.value,
        )
        for ioc_type in all_types:
            row = f"| {ioc_type.value:<10} |"
            for results in comparison.values():
                m = results.per_type.get(ioc_type, ExtractionMetrics())
                row += f" {m.precision:.3f}  |"
            for results in comparison.values():
                m = results.per_type.get(ioc_type, ExtractionMetrics())
                row += f" {m.recall:.3f} |"
            for results in comparison.values():
                m = results.per_type.get(ioc_type, ExtractionMetrics())
                row += f" {m.f1:.3f} |"
            lines.append(row)

        # Micro-average row
        row = "| **micro**  |"
        for results in comparison.values():
            row += f" {results.micro.precision:.3f}  |"
        for results in comparison.values():
            row += f" {results.micro.recall:.3f} |"
        for results in comparison.values():
            row += f" **{results.micro.f1:.3f}** |"
        lines.append(row)

        return "\n".join(lines)

    @staticmethod
    def save_results(results: dict[str, EvalResults], path: str | Path) -> None:
        """Serialize all EvalResults to JSON."""
        out = {name: r.to_dict() for name, r in results.items()}
        Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
