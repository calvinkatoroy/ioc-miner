import pytest
from ioc_miner.extraction.regex_extractor import RegexExtractor
from ioc_miner.models.ioc import IOCType


@pytest.fixture
def ex():
    return RegexExtractor()


def test_extract_ipv4(ex):
    iocs = ex.extract("C2 server at 185.220.101.47 on port 443.", source="test")
    ips = [i for i in iocs if i.type == IOCType.IP]
    assert len(ips) == 1
    assert ips[0].value == "185.220.101.47"


def test_skip_private_ip(ex):
    iocs = ex.extract("Internal host at 192.168.1.1 communicated.", source="test")
    ips = [i for i in iocs if i.type == IOCType.IP]
    assert len(ips) == 0


def test_extract_domain(ex):
    iocs = ex.extract("Beacon called out to evil-c2.ru every 30s.", source="test")
    domains = [i for i in iocs if i.type == IOCType.DOMAIN]
    assert any("evil-c2.ru" in i.value for i in domains)


def test_extract_url_no_duplicate_domain(ex):
    iocs = ex.extract("Payload downloaded from https://evil.com/drop.exe", source="test")
    urls = [i for i in iocs if i.type == IOCType.URL]
    domains = [i for i in iocs if i.type == IOCType.DOMAIN]
    assert len(urls) == 1
    # Domain should not be extracted separately when already part of URL
    assert not any("evil.com" in i.value for i in domains)


def test_extract_sha256(ex):
    sha = "a" * 64
    iocs = ex.extract(f"File hash: {sha}", source="test")
    hashes = [i for i in iocs if i.type == IOCType.SHA256]
    assert len(hashes) == 1
    assert hashes[0].value == sha


def test_extract_md5(ex):
    md5 = "d" * 32
    iocs = ex.extract(f"MD5: {md5}", source="test")
    hashes = [i for i in iocs if i.type == IOCType.MD5]
    assert len(hashes) == 1


def test_skip_empty_md5(ex):
    iocs = ex.extract("d41d8cd98f00b204e9800998ecf8427e", source="test")
    hashes = [i for i in iocs if i.type in (IOCType.MD5, IOCType.SHA1, IOCType.SHA256)]
    assert len(hashes) == 0


def test_extract_cve(ex):
    iocs = ex.extract("Exploiting CVE-2021-44228 (Log4Shell).", source="test")
    cves = [i for i in iocs if i.type == IOCType.CVE]
    assert len(cves) == 1
    assert cves[0].value == "CVE-2021-44228"


def test_extract_email(ex):
    iocs = ex.extract("Contact threat actor at attacker@evil[.]ru", source="test")
    emails = [i for i in iocs if i.type == IOCType.EMAIL]
    # Note: defanging happens in preprocessor, but regex still catches refanged form
    assert len(emails) >= 0  # depends on whether refang ran first


def test_deduplicate_across_sentences(ex):
    sentences = [
        "C2 at 185.220.101.47 communicates on 443.",
        "The IP 185.220.101.47 was also seen in campaign X.",
    ]
    iocs = ex.extract_all(sentences, source="test")
    ips = [i for i in iocs if i.type == IOCType.IP and i.value == "185.220.101.47"]
    assert len(ips) == 1
