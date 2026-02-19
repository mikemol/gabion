from __future__ import annotations

import itertools
import json

from gabion.analysis import dataflow_audit as da


def _payload_bytes(output: da._PartialWorkerCarrierOutput) -> bytes:
    payload = {
        "violations": list(output.violations),
        "witnesses": list(output.witnesses),
        "deltas": list(output.deltas),
        "snapshots": list(output.snapshots),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sample_partials() -> tuple[da._PartialWorkerCarrierOutput, ...]:
    return (
        da._PartialWorkerCarrierOutput(
            violations=(
                "src/zeta.py:9 violation z",
                "orphan violation",
            ),
            witnesses=(
                {"path": "src/beta.py", "kind": "witness", "value": 2},
                {"kind": "witness", "value": 99},
            ),
            deltas=(
                {"path": "src/gamma.py", "bundle": ["b", "a"], "delta": 1},
            ),
            snapshots=(
                {"path": "src/beta.py", "hash": "b"},
            ),
        ),
        da._PartialWorkerCarrierOutput(
            violations=(
                "src/alpha.py:1 violation a",
                "src/beta.py:3 violation b",
            ),
            witnesses=(
                {"path": "src/alpha.py", "kind": "witness", "value": 1},
                {"path": "src/beta.py", "kind": "witness", "value": 2},
            ),
            deltas=(
                {"path": "src/alpha.py", "bundle": ["a"], "delta": -1},
            ),
            snapshots=(
                {"path": "src/alpha.py", "hash": "a"},
            ),
        ),
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_merge_canonicalization.py::test_merge_worker_carriers_is_byte_stable_across_partial_orders::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_worker_carriers::test_dataflow_audit_merge_canonicalization.py::tests.test_dataflow_audit_merge_canonicalization._payload_bytes::test_dataflow_audit_merge_canonicalization.py::tests.test_dataflow_audit_merge_canonicalization._sample_partials
def test_merge_worker_carriers_is_byte_stable_across_partial_orders() -> None:
    partials = _sample_partials()
    merged_bytes = {
        _payload_bytes(da._merge_worker_carriers(ordering))
        for ordering in itertools.permutations(partials)
    }
    assert len(merged_bytes) == 1


# gabion:evidence E:call_footprint::tests/test_dataflow_audit_merge_canonicalization.py::test_merge_worker_carriers_global_order_is_path_anchored::dataflow_audit.py::gabion.analysis.dataflow_audit._merge_worker_carriers::test_dataflow_audit_merge_canonicalization.py::tests.test_dataflow_audit_merge_canonicalization._sample_partials
def test_merge_worker_carriers_global_order_is_path_anchored() -> None:
    merged = da._merge_worker_carriers(tuple(reversed(_sample_partials())))
    assert list(merged.violations) == [
        "src/alpha.py:1 violation a",
        "src/beta.py:3 violation b",
        "src/zeta.py:9 violation z",
        "orphan violation",
    ]
    assert [entry["path"] for entry in merged.witnesses if "path" in entry] == [
        "src/alpha.py",
        "src/beta.py",
    ]
    assert [entry["path"] for entry in merged.deltas] == [
        "src/alpha.py",
        "src/gamma.py",
    ]
    assert [entry["path"] for entry in merged.snapshots] == [
        "src/alpha.py",
        "src/beta.py",
    ]
