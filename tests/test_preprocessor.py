from ioc_miner.preprocessing.normalizer import refang, normalize_encoding, sentence_tokenize


def test_refang_hxxp():
    assert refang("hxxp://evil.com") == "http://evil.com"
    assert refang("hxxps://evil.com") == "https://evil.com"


def test_refang_dot_variants():
    assert refang("evil[.]com") == "evil.com"
    assert refang("evil(dot)com") == "evil.com"
    assert refang("evil[dot]com") == "evil.com"
    assert refang("1.2[.]3.4") == "1.2.3.4"


def test_refang_at():
    assert refang("user[at]evil.com") == "user@evil.com"
    assert refang("user[@]evil.com") == "user@evil.com"


def test_normalize_smart_quotes():
    text = "\u201chello\u201d and \u2018world\u2019"
    result = normalize_encoding(text)
    assert '"hello"' in result
    assert "'world'" in result


def test_sentence_tokenize_preserves_ips():
    text = "The actor used 185.220.101.47 as a C2 server. It communicated on port 443."
    sentences = sentence_tokenize(text)
    # IP should not be split across sentences
    assert any("185.220.101.47" in s for s in sentences)
