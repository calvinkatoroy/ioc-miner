"""
Baseline extractor wrappers for benchmark comparison.

Each wraps an external IOC extraction library and exposes the same
    extract_all(sentences, source) -> list[IOC]
interface as RegexExtractor, so BenchmarkEvaluator can treat them uniformly.

Install extras:
    pip install iocextract          # IocextractBaseline
    pip install ioc-finder          # IocFinderBaseline
    # cacador: go install github.com/sroberts/cacador@latest  # CacadorBaseline
"""

from __future__ import annotations

import json
import logging
import subprocess

from ioc_miner.models.ioc import IOC, IOCType, IOCVerdict

logger = logging.getLogger(__name__)


def _make(
    ioc_type: IOCType,
    value: str,
    sentences: list[str],
    source: str,
    extractor_name: str,
) -> IOC:
    v_lower = value.lower()
    context = next((s for s in sentences if v_lower in s.lower()), sentences[0] if sentences else "")
    return IOC(
        type=ioc_type,
        value=v_lower,
        context=context,
        source=source,
        extracted_by=extractor_name,
    )


class IocextractBaseline:
    """
    Wraps the `iocextract` library (https://github.com/InQuest/python-iocextract).

    Install: pip install iocextract

    Extracts: URLs, IPs, MD5/SHA1/SHA256/SHA512, emails.
    Does not extract: domains, CVEs, filepaths.
    """

    name = "iocextract"

    def extract_all(self, sentences: list[str], source: str = "") -> list[IOC]:
        try:
            import iocextract
        except ImportError:
            raise ImportError("pip install iocextract")

        text = " ".join(sentences)
        results: list[IOC] = []
        seen: set[tuple[IOCType, str]] = set()

        def _add(ioc_type: IOCType, value: str) -> None:
            key = (ioc_type, value.lower())
            if key not in seen:
                seen.add(key)
                results.append(_make(ioc_type, value, sentences, source, self.name))

        for v in iocextract.extract_urls(text, refang=True):
            _add(IOCType.URL, v)
        for v in iocextract.extract_ips(text, refang=True):
            _add(IOCType.IP, v)
        for v in iocextract.extract_md5_hashes(text):
            _add(IOCType.MD5, v)
        for v in iocextract.extract_sha1_hashes(text):
            _add(IOCType.SHA1, v)
        for v in iocextract.extract_sha256_hashes(text):
            _add(IOCType.SHA256, v)
        for v in iocextract.extract_sha512_hashes(text):
            _add(IOCType.SHA512, v)
        for v in iocextract.extract_emails(text, refang=True):
            _add(IOCType.EMAIL, v)

        return results


class IocFinderBaseline:
    """
    Wraps the `ioc-finder` library (https://github.com/fhightower/ioc-finder).

    Install: pip install ioc-finder

    Extracts: IPs, URLs, domains, MD5/SHA1/SHA256/SHA512, emails, CVEs.
    """

    name = "ioc-finder"

    _TYPE_MAP: dict[str, IOCType] = {
        "ipv4s": IOCType.IP,
        "urls": IOCType.URL,
        "domains": IOCType.DOMAIN,
        "md5s": IOCType.MD5,
        "sha1s": IOCType.SHA1,
        "sha256s": IOCType.SHA256,
        "sha512s": IOCType.SHA512,
        "email_addresses": IOCType.EMAIL,
        "cves": IOCType.CVE,
    }

    def extract_all(self, sentences: list[str], source: str = "") -> list[IOC]:
        try:
            from ioc_finder import find_iocs
        except ImportError:
            raise ImportError("pip install ioc-finder")

        text = " ".join(sentences)
        found = find_iocs(text)

        results: list[IOC] = []
        seen: set[tuple[IOCType, str]] = set()

        for key, ioc_type in self._TYPE_MAP.items():
            for value in found.get(key, []):
                v = str(value).lower()
                k = (ioc_type, v)
                if k not in seen:
                    seen.add(k)
                    results.append(_make(ioc_type, v, sentences, source, self.name))

        return results


class CacadorBaseline:
    """
    Wraps the cacador Go binary (https://github.com/sroberts/cacador).

    cacador must be installed and on PATH:
        go install github.com/sroberts/cacador@latest

    Input is piped via stdin; output is JSON.
    """

    name = "cacador"

    _TYPE_MAP: dict[str, IOCType] = {
        "domains": IOCType.DOMAIN,
        "ips": IOCType.IP,
        "urls": IOCType.URL,
        "md5s": IOCType.MD5,
        "sha1s": IOCType.SHA1,
        "sha256s": IOCType.SHA256,
        "emails": IOCType.EMAIL,
        "cves": IOCType.CVE,
    }

    def extract_all(self, sentences: list[str], source: str = "") -> list[IOC]:
        text = "\n".join(sentences)
        try:
            proc = subprocess.run(
                ["cacador"],
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "cacador binary not found on PATH. "
                "Install: go install github.com/sroberts/cacador@latest"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("cacador timed out after 30s")

        if proc.returncode != 0:
            logger.warning("cacador exited %d: %s", proc.returncode, proc.stderr.strip())

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"cacador returned invalid JSON: {e}") from e

        results: list[IOC] = []
        seen: set[tuple[IOCType, str]] = set()

        for key, ioc_type in self._TYPE_MAP.items():
            for value in data.get(key, []):
                v = str(value).lower()
                k = (ioc_type, v)
                if k not in seen:
                    seen.add(k)
                    results.append(_make(ioc_type, v, sentences, source, self.name))

        return results
