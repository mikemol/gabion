from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def stable_docflow_compliance_row_id(row: Mapping[str, object]) -> str:
    identity = {
        "row_kind": row.get("row_kind"),
        "status": row.get("status"),
        "path": row.get("path"),
        "invariant": row.get("invariant"),
        "source_row_kind": row.get("source_row_kind"),
        "dep": row.get("dep"),
        "anchor": row.get("anchor"),
        "qual": row.get("qual"),
        "field": row.get("field"),
        "missing": row.get("missing"),
        "evidence_id": row.get("evidence_id"),
        "detail": row.get("detail"),
    }
    digest = hashlib.sha1(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    prefix = str(row.get("row_kind", "row")).strip() or "row"
    return f"{prefix}:{digest}"


__all__ = ["stable_docflow_compliance_row_id"]
