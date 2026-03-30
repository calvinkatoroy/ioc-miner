"""
Tests for the evaluation framework.

Uses synthetic in-memory ground truth — no external files or libraries required.
"""

import json
import tempfile
from pathlib import Path

import pytest

from ioc_miner.evaluation.benchmark import (
    BenchmarkEvaluator,
    EvalResults,
    ExtractionMetrics,
    GroundTruthIOC,
    GroundTruthSentence,
    VerdictMetrics,
    load_ground_truth,
)
from ioc_miner.extraction.context_classifier import ContextClassifier
from ioc_miner.extraction.regex_extractor import RegexExtractor
from ioc_miner.models.ioc import IOC, IOCType, IOCVerdict


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def evaluator():
    return BenchmarkEvaluator()


@pytest.fixture
def simple_gt() -> list[GroundTruthSentence]:
    return [
        GroundTruthSentence(
            text="The malware C2 server is at 185.220.101.47.",
            iocs=[GroundTruthIOC(type=IOCType.IP, value="185.220.101.47", verdict=IOCVerdict.MALICIOUS)],
        ),
        GroundTruthSentence(
            text="Dropper downloaded from https://evil.example.ru/payload.exe.",
            iocs=[GroundTruthIOC(type=IOCType.URL, value="https://evil.example.ru/payload.exe", verdict=IOCVerdict.MALICIOUS)],
        ),
        GroundTruthSentence(
            text="File hash: aabbccdd" * 4 + "aabbccdd",  # 40 hex chars — SHA1
            iocs=[],
        ),
    ]


# ─── ExtractionMetrics ────────────────────────────────────────────────────────

class TestExtractionMetrics:
    def test_perfect(self):
        m = ExtractionMetrics(tp=10, fp=0, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_zero_predictions(self):
        m = ExtractionMetrics(tp=0, fp=0, fn=5)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_no_ground_truth(self):
        m = ExtractionMetrics(tp=0, fp=3, fn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0

    def test_partial(self):
        m = ExtractionMetrics(tp=2, fp=1, fn=1)
        assert m.precision == pytest.approx(2 / 3)
        assert m.recall == pytest.approx(2 / 3)
        assert m.f1 == pytest.approx(2 / 3)

    def test_iadd(self):
        a = ExtractionMetrics(tp=2, fp=1, fn=1)
        b = ExtractionMetrics(tp=3, fp=2, fn=0)
        a += b
        assert a.tp == 5
        assert a.fp == 3
        assert a.fn == 1


# ─── load_ground_truth ────────────────────────────────────────────────────────

class TestLoadGroundTruth:
    def _write_jsonl(self, lines: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        for obj in lines:
            tmp.write(json.dumps(obj) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_basic_load(self):
        path = self._write_jsonl([
            {"text": "C2 at 1.2.3.4.", "iocs": [{"type": "ip", "value": "1.2.3.4", "verdict": "malicious"}]},
            {"text": "No IOCs here.", "iocs": []},
        ])
        gt = load_ground_truth(path)
        assert len(gt) == 2
        assert gt[0].iocs[0].type == IOCType.IP
        assert gt[0].iocs[0].verdict == IOCVerdict.MALICIOUS
        assert gt[1].iocs == []

    def test_missing_verdict_defaults_unknown(self):
        path = self._write_jsonl([
            {"text": "Seen 1.2.3.4.", "iocs": [{"type": "ip", "value": "1.2.3.4"}]},
        ])
        gt = load_ground_truth(path)
        assert gt[0].iocs[0].verdict == IOCVerdict.UNKNOWN

    def test_skips_unknown_ioc_type(self):
        path = self._write_jsonl([
            {"text": "X.", "iocs": [{"type": "bitcoin_address", "value": "1abc"}]},
        ])
        gt = load_ground_truth(path)
        assert gt[0].iocs == []

    def test_invalid_json_raises(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.write("not json\n")
        tmp.close()
        with pytest.raises(ValueError, match="Line 1"):
            load_ground_truth(tmp.name)

    def test_missing_text_raises(self):
        path = self._write_jsonl([{"iocs": []}])
        with pytest.raises(ValueError, match="missing 'text'"):
            load_ground_truth(path)

    def test_skips_blank_lines(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.write('\n{"text": "X.", "iocs": []}\n\n')
        tmp.close()
        gt = load_ground_truth(tmp.name)
        assert len(gt) == 1


# ─── BenchmarkEvaluator.evaluate_extraction ───────────────────────────────────

class TestEvaluateExtraction:
    def test_perfect_recall(self, evaluator):
        gt = [
            GroundTruthSentence(
                text="C2 server is 185.220.101.47.",
                iocs=[GroundTruthIOC(IOCType.IP, "185.220.101.47")],
            )
        ]
        results = evaluator.evaluate_extraction(RegexExtractor(), gt)
        assert results.per_type[IOCType.IP].recall == 1.0
        assert results.per_type[IOCType.IP].tp == 1

    def test_false_positive_counted(self, evaluator):
        # Ground truth has no IOCs, but text contains an IP
        gt = [GroundTruthSentence(text="Server 8.8.8.8 is a DNS resolver.", iocs=[])]
        results = evaluator.evaluate_extraction(RegexExtractor(), gt)
        assert results.per_type.get(IOCType.IP, ExtractionMetrics()).fp >= 1

    def test_false_negative_counted(self, evaluator):
        gt = [
            GroundTruthSentence(
                text="No IP in this text.",
                iocs=[GroundTruthIOC(IOCType.IP, "185.220.101.47")],
            )
        ]
        results = evaluator.evaluate_extraction(RegexExtractor(), gt)
        assert results.per_type[IOCType.IP].fn == 1
        assert results.per_type[IOCType.IP].tp == 0

    def test_micro_average_aggregates_types(self, evaluator):
        gt = [
            GroundTruthSentence(
                text="C2 at 185.220.101.47. Hash: " + "a" * 64,
                iocs=[
                    GroundTruthIOC(IOCType.IP, "185.220.101.47"),
                    GroundTruthIOC(IOCType.SHA256, "a" * 64),
                ],
            )
        ]
        results = evaluator.evaluate_extraction(RegexExtractor(), gt)
        assert results.micro.tp == 2

    def test_empty_ground_truth(self, evaluator):
        results = evaluator.evaluate_extraction(RegexExtractor(), [])
        assert results.micro.tp == 0
        assert results.micro.f1 == 0.0

    def test_value_matching_is_case_insensitive(self, evaluator):
        gt = [
            GroundTruthSentence(
                text="CVE-2021-44228 exploited.",
                iocs=[GroundTruthIOC(IOCType.CVE, "cve-2021-44228")],  # lowercase in GT
            )
        ]
        results = evaluator.evaluate_extraction(RegexExtractor(), gt)
        # RegexExtractor uppercases CVEs; benchmark lowercases both sides — should match
        assert results.per_type[IOCType.CVE].tp == 1


# ─── BenchmarkEvaluator.evaluate_verdict ──────────────────────────────────────

class TestEvaluateVerdict:
    def _make_classified_ioc(self, ioc_type, value, verdict) -> IOC:
        return IOC(
            type=ioc_type,
            value=value,
            context="",
            source="test",
            extracted_by="test",
            verdict=verdict,
        )

    def test_perfect_verdict_accuracy(self, evaluator):
        gt = [
            GroundTruthSentence(
                text="",
                iocs=[GroundTruthIOC(IOCType.IP, "1.2.3.4", IOCVerdict.MALICIOUS)],
            )
        ]
        iocs = [self._make_classified_ioc(IOCType.IP, "1.2.3.4", IOCVerdict.MALICIOUS)]
        metrics = evaluator.evaluate_verdict(iocs, gt)
        assert metrics.accuracy == 1.0
        assert metrics.total == 1
        assert metrics.correct == 1

    def test_wrong_verdict(self, evaluator):
        gt = [
            GroundTruthSentence(
                text="",
                iocs=[GroundTruthIOC(IOCType.IP, "1.2.3.4", IOCVerdict.MALICIOUS)],
            )
        ]
        iocs = [self._make_classified_ioc(IOCType.IP, "1.2.3.4", IOCVerdict.BENIGN)]
        metrics = evaluator.evaluate_verdict(iocs, gt)
        assert metrics.accuracy == 0.0
        assert metrics.confusion["malicious"]["benign"] == 1

    def test_unknown_gt_verdict_skipped(self, evaluator):
        gt = [
            GroundTruthSentence(
                text="",
                iocs=[GroundTruthIOC(IOCType.IP, "1.2.3.4", IOCVerdict.UNKNOWN)],
            )
        ]
        iocs = [self._make_classified_ioc(IOCType.IP, "1.2.3.4", IOCVerdict.MALICIOUS)]
        metrics = evaluator.evaluate_verdict(iocs, gt)
        assert metrics.total == 0

    def test_unmatched_prediction_ignored(self, evaluator):
        gt = [GroundTruthSentence(text="", iocs=[])]
        iocs = [self._make_classified_ioc(IOCType.IP, "9.9.9.9", IOCVerdict.MALICIOUS)]
        metrics = evaluator.evaluate_verdict(iocs, gt)
        assert metrics.total == 0


# ─── compare_extractors ───────────────────────────────────────────────────────

class TestCompareExtractors:
    def test_returns_result_per_extractor(self, evaluator, simple_gt):
        comparison = evaluator.compare_extractors(
            simple_gt,
            {"regex": RegexExtractor(), "regex2": RegexExtractor()},
        )
        assert set(comparison.keys()) == {"regex", "regex2"}
        assert isinstance(comparison["regex"], EvalResults)

    def test_comparison_markdown_has_all_names(self, evaluator, simple_gt):
        comparison = evaluator.compare_extractors(
            simple_gt,
            {"regex": RegexExtractor()},
        )
        md = BenchmarkEvaluator.comparison_markdown(comparison)
        assert "regex" in md
        assert "micro" in md


# ─── Integration: regex + context classifier ──────────────────────────────────

class TestIntegration:
    def test_full_pipeline_on_malicious_sentence(self, evaluator):
        gt = [
            GroundTruthSentence(
                text="The C2 server beaconed to 185.220.101.47 over port 443.",
                iocs=[GroundTruthIOC(IOCType.IP, "185.220.101.47", IOCVerdict.MALICIOUS)],
            )
        ]
        extractor = RegexExtractor()
        classifier = ContextClassifier(use_ml=False)

        extraction = evaluator.evaluate_extraction(extractor, gt)
        assert extraction.per_type[IOCType.IP].tp == 1

        iocs = extractor.extract_all([s.text for s in gt])
        classified = classifier.classify_all(iocs)
        verdict = evaluator.evaluate_verdict(classified, gt)
        # Rule-based classifier should detect "C2" / "beacon" → MALICIOUS
        assert verdict.total == 1
        assert verdict.correct == 1
