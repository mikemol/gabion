from __future__ import annotations

from gabion.analysis.foundation.wire_types import WireValue
from gabion.runtime import stable_encode


def normalize_alt_evidence_payload(evidence: WireValue) -> WireValue:
    canonical = stable_encode.stable_wire_value(
        evidence,
        source="aspf_evidence.normalize_alt_evidence_payload",
    )
    return canonical
