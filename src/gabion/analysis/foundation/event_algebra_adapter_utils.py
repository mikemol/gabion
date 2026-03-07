from __future__ import annotations

from hashlib import sha1
from typing import Mapping

from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.runtime import stable_encode


def mapping_payload_to_json_object(payload: Mapping[str, object]) -> JSONObject:
    check_deadline()
    return {str(key): payload[key] for key in payload}


def payload_sha1_digest(payload: Mapping[str, object]) -> str:
    check_deadline()
    canonical = stable_encode.stable_compact_text(payload).encode("utf-8")
    return sha1(canonical).hexdigest()


__all__ = ["mapping_payload_to_json_object", "payload_sha1_digest"]
