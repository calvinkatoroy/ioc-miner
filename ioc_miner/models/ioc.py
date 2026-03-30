from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IOCType(str, Enum):
    IP = "ip"
    DOMAIN = "domain"
    URL = "url"
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    CVE = "cve"
    EMAIL = "email"
    FILEPATH = "filepath"


class IOCVerdict(str, Enum):
    MALICIOUS = "malicious"
    BENIGN = "benign"
    SINKHOLE = "sinkhole"
    UNKNOWN = "unknown"


@dataclass
class IOC:
    type: IOCType
    value: str
    # raw sentence the IOC was extracted from
    context: str
    source: str
    extracted_by: str  # "regex" | "secbert"
    confidence: float = 1.0
    verdict: IOCVerdict = IOCVerdict.UNKNOWN
    verdict_confidence: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "value": self.value,
            "context": self.context,
            "source": self.source,
            "extracted_by": self.extracted_by,
            "confidence": self.confidence,
            "verdict": self.verdict.value,
            "verdict_confidence": self.verdict_confidence,
            "tags": self.tags,
        }

    def __hash__(self) -> int:
        return hash((self.type, self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IOC):
            return NotImplemented
        return self.type == other.type and self.value == other.value
