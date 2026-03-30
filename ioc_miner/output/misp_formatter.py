"""
MISP (Malware Information Sharing Platform) attribute format.

Produces a MISP event JSON with attributes — compatible with MISP import
via the /events/add API endpoint or direct JSON import in the UI.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ioc_miner.models.ioc import IOC, IOCType

# MISP attribute type mapping
_MISP_TYPE: dict[IOCType, str] = {
    IOCType.IP: "ip-dst",
    IOCType.DOMAIN: "domain",
    IOCType.URL: "url",
    IOCType.MD5: "md5",
    IOCType.SHA1: "sha1",
    IOCType.SHA256: "sha256",
    IOCType.SHA512: "sha512",
    IOCType.CVE: "vulnerability",
    IOCType.EMAIL: "email-src",
    IOCType.FILEPATH: "filename",
}

_MISP_CATEGORY: dict[IOCType, str] = {
    IOCType.IP: "Network activity",
    IOCType.DOMAIN: "Network activity",
    IOCType.URL: "Network activity",
    IOCType.MD5: "Artifacts dropped",
    IOCType.SHA1: "Artifacts dropped",
    IOCType.SHA256: "Artifacts dropped",
    IOCType.SHA512: "Artifacts dropped",
    IOCType.CVE: "External analysis",
    IOCType.EMAIL: "Attribution",
    IOCType.FILEPATH: "Artifacts dropped",
}


def to_misp_event(iocs: list[IOC], event_info: str = "IOC Miner extraction") -> dict:
    now = int(datetime.now(timezone.utc).timestamp())
    attributes = []

    for ioc in iocs:
        misp_type = _MISP_TYPE.get(ioc.type)
        if misp_type is None:
            continue
        attributes.append(
            {
                "uuid": str(uuid.uuid4()),
                "type": misp_type,
                "category": _MISP_CATEGORY.get(ioc.type, "Other"),
                "value": ioc.value,
                "comment": ioc.context[:500] if ioc.context else "",
                "to_ids": ioc.verdict.value == "malicious",
                "timestamp": str(now),
            }
        )

    return {
        "Event": {
            "uuid": str(uuid.uuid4()),
            "info": event_info,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "threat_level_id": "2",
            "analysis": "1",
            "distribution": "0",
            "Attribute": attributes,
        }
    }


def to_misp_json(iocs: list[IOC], **kwargs) -> str:
    return json.dumps(to_misp_event(iocs, **kwargs), indent=2)
