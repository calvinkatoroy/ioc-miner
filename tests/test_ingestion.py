import tempfile
import os
import pytest
from ioc_miner.ingestion.plaintext import PlaintextIngestor


def test_plaintext_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("The C2 server at 185.220.101.47 exfiltrated data.")
        path = f.name
    try:
        ingestor = PlaintextIngestor()
        text = ingestor.ingest(path)
        assert "185.220.101.47" in text
    finally:
        os.unlink(path)


def test_plaintext_missing_file():
    ingestor = PlaintextIngestor()
    with pytest.raises(FileNotFoundError):
        ingestor.ingest("/nonexistent/path/file.txt")
