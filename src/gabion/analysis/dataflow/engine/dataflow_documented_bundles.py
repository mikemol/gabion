# gabion:boundary_normalization_module
from __future__ import annotations

"""Canonical documented-bundle marker iteration helpers."""

import re
from pathlib import Path

from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _forbid_adhoc_bundle_discovery,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once

_BUNDLE_MARKER = re.compile(r"dataflow-bundle:\s*(.*)")


def _iter_documented_bundles(path: Path) -> set[tuple[str, ...]]:
    """Return bundles documented via '# dataflow-bundle: a, b' markers."""
    check_deadline()
    _forbid_adhoc_bundle_discovery("_iter_documented_bundles")
    bundles: set[tuple[str, ...]] = set()
    try:
        text = path.read_text()
    except (OSError, UnicodeError):
        return bundles
    for line in text.splitlines():
        check_deadline()
        match = _BUNDLE_MARKER.search(line)
        if not match:
            continue
        payload = match.group(1)
        if not payload:
            continue
        parts = [p.strip() for p in re.split(r"[,\s]+", payload) if p.strip()]
        if len(parts) < 2:
            continue
        bundles.add(
            tuple(
                sort_once(
                    parts,
                    source=(
                        "gabion.analysis.dataflow_documented_bundles."
                        "_iter_documented_bundles.site_1"
                    ),
                )
            )
        )
    return bundles


__all__ = ["_iter_documented_bundles"]
