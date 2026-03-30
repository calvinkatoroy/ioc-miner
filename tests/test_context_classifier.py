import pytest
from ioc_miner.extraction.context_classifier import ContextClassifier
from ioc_miner.models.ioc import IOC, IOCType, IOCVerdict


@pytest.fixture
def clf():
    return ContextClassifier()


def _ioc(value: str, context: str) -> IOC:
    return IOC(type=IOCType.IP, value=value, context=context, source="test", extracted_by="regex")


def test_malicious_c2(clf):
    ioc = _ioc("185.220.101.47", "The C2 server at 185.220.101.47 received beacon callbacks.")
    clf.classify(ioc)
    assert ioc.verdict == IOCVerdict.MALICIOUS


def test_benign_dns_resolver(clf):
    ioc = _ioc("8.8.8.8", "The actor used 8.8.8.8 as a DNS resolver to avoid detection.")
    clf.classify(ioc)
    assert ioc.verdict == IOCVerdict.BENIGN


def test_sinkhole(clf):
    ioc = _ioc("1.2.3.4", "The domain was sinkholed, pointing to 1.2.3.4.")
    clf.classify(ioc)
    assert ioc.verdict == IOCVerdict.SINKHOLE


def test_unknown_no_signals(clf):
    ioc = _ioc("1.2.3.4", "The IP 1.2.3.4 was observed in network logs.")
    clf.classify(ioc)
    assert ioc.verdict == IOCVerdict.UNKNOWN


def test_classify_all(clf):
    iocs = [
        _ioc("185.1.1.1", "Malware beaconed to 185.1.1.1."),
        _ioc("8.8.8.8", "Used 8.8.8.8 as a DNS resolver."),
    ]
    result = clf.classify_all(iocs)
    assert result[0].verdict == IOCVerdict.MALICIOUS
    assert result[1].verdict == IOCVerdict.BENIGN
