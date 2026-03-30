from __future__ import annotations

import csv
import io

from ioc_miner.models.ioc import IOC

_FIELDS = ["type", "value", "verdict", "verdict_confidence", "confidence", "source", "extracted_by", "context"]


def to_csv(iocs: list[IOC], include_context: bool = True) -> str:
    buf = io.StringIO()
    fields = _FIELDS if include_context else [f for f in _FIELDS if f != "context"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for ioc in iocs:
        row = ioc.to_dict()
        writer.writerow({k: row[k] for k in fields})
    return buf.getvalue()
