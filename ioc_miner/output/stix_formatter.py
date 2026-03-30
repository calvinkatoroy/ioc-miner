"""
STIX 2.1 output formatter.

Maps IOCType → appropriate STIX 2.1 object:
  IP       → ipv4-addr + indicator
  DOMAIN   → domain-name + indicator
  URL      → url + indicator
  HASH     → file (with hash property) + indicator
  CVE      → vulnerability
  EMAIL    → email-addr + indicator
  FILEPATH → file + indicator
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ioc_miner.models.ioc import IOC, IOCType, IOCVerdict


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _indicator_pattern(ioc: IOC) -> str | None:
    v = ioc.value.replace("'", "\\'")
    match ioc.type:
        case IOCType.IP:
            return f"[ipv4-addr:value = '{v}']"
        case IOCType.DOMAIN:
            return f"[domain-name:value = '{v}']"
        case IOCType.URL:
            return f"[url:value = '{v}']"
        case IOCType.MD5:
            return f"[file:hashes.MD5 = '{v}']"
        case IOCType.SHA1:
            return f"[file:hashes.'SHA-1' = '{v}']"
        case IOCType.SHA256:
            return f"[file:hashes.'SHA-256' = '{v}']"
        case IOCType.SHA512:
            return f"[file:hashes.'SHA-512' = '{v}']"
        case IOCType.EMAIL:
            return f"[email-addr:value = '{v}']"
        case IOCType.FILEPATH:
            return f"[file:name = '{v}']"
        case _:
            return None


def to_stix_bundle(iocs: list[IOC], identity_name: str = "ioc-miner") -> dict:
    """Convert a list of IOCs to a STIX 2.1 bundle dict."""
    try:
        import stix2
    except ImportError:
        raise ImportError("stix2 is required: pip install stix2")

    identity = stix2.Identity(name=identity_name, identity_class="tool")
    objects: list = [identity]
    ts = _now()

    for ioc in iocs:
        if ioc.type == IOCType.CVE:
            obj = stix2.Vulnerability(
                name=ioc.value,
                external_references=[
                    {
                        "source_name": "cve",
                        "external_id": ioc.value,
                        "url": f"https://nvd.nist.gov/vuln/detail/{ioc.value}",
                    }
                ],
                created_by_ref=identity.id,
            )
            objects.append(obj)
            continue

        pattern = _indicator_pattern(ioc)
        if pattern is None:
            continue

        labels = [ioc.verdict.value] if ioc.verdict != IOCVerdict.UNKNOWN else ["unknown"]
        indicator = stix2.Indicator(
            name=ioc.value,
            description=ioc.context[:500] if ioc.context else "",
            pattern=pattern,
            pattern_type="stix",
            valid_from=ts,
            labels=labels,
            confidence=int(ioc.verdict_confidence * 100),
            created_by_ref=identity.id,
            external_references=(
                [{"source_name": "source", "description": ioc.source}]
                if ioc.source
                else []
            ),
        )
        objects.append(indicator)

    bundle = stix2.Bundle(*objects, spec_version="2.1")
    return json.loads(bundle.serialize())


def to_stix_json(iocs: list[IOC], **kwargs) -> str:
    return json.dumps(to_stix_bundle(iocs, **kwargs), indent=2)
