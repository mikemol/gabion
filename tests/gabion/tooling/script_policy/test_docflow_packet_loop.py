from __future__ import annotations

import json
from pathlib import Path

from scripts.policy import docflow_packet_enforce, docflow_packetize


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# gabion:behavior primary=desired
def test_docflow_packetize_classifies_metadata_and_semantic_updates(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "in").mkdir(parents=True)

    (root / "in" / "in-54.md").write_text(
        """---
doc_id: in_54
doc_owner: maintainer
---

# in/in-54.md
metadata packet
""",
        encoding="utf-8",
    )
    (root / "in" / "in-46.md").write_text(
        """---
doc_id: in_46
doc_owner: maintainer
---

# in/in-46.md
`src/gabion/analysis/legacy_dataflow_monolith.py::_compute_fingerprint_provenance`
""",
        encoding="utf-8",
    )

    compliance_path = root / "artifacts" / "out" / "docflow_compliance.json"
    section_reviews_path = root / "artifacts" / "out" / "docflow_section_reviews.json"
    out_path = root / "artifacts" / "out" / "docflow_warning_doc_packets.json"
    summary_path = root / "artifacts" / "out" / "docflow_warning_doc_packet_summary.json"

    _write_json(
        compliance_path,
        {
            "rows": [],
            "summary": {"contradicts": 0},
            "version": 1,
        },
    )
    _write_json(
        section_reviews_path,
        {
            "rows": [
                {
                    "row_kind": "doc_section_review",
                    "status": "stale_dep",
                    "path": "in/in-54.md",
                    "dep": "POLICY_SEED.md#policy_seed",
                    "dep_version": 48,
                    "expected_dep_version": 2,
                    "anchor": "in_in_54",
                },
                {
                    "row_kind": "doc_section_review",
                    "status": "missing_review",
                    "path": "in/in-46.md",
                    "dep": "glossary.md#suite_site",
                    "dep_version": None,
                    "expected_dep_version": 1,
                    "anchor": "in_in_46",
                },
            ]
        },
    )

    rc = docflow_packetize.run(
        root=root,
        compliance_path=compliance_path,
        section_reviews_path=section_reviews_path,
        out_path=out_path,
        summary_out_path=summary_path,
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    packets = payload["packets"]
    classes = {packet["path"]: packet["classification"] for packet in packets}
    assert classes["in/in-54.md"] == "metadata_only"
    assert classes["in/in-46.md"] == "needs_semantic_update"

    packet_46 = next(packet for packet in packets if packet["path"] == "in/in-46.md")
    assert packet_46["stale_anchor_hints"]


# gabion:behavior primary=verboten facets=drift
def test_docflow_packet_enforce_ratchets_new_drift_and_scope(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True)

    packets_path = root / "artifacts" / "out" / "docflow_warning_doc_packets.json"
    baseline_path = root / "docs" / "baselines" / "docflow_packet_baseline.json"
    out_path = root / "artifacts" / "out" / "docflow_packet_enforcement.json"
    debt_out_path = root / "artifacts" / "out" / "docflow_packet_debt_ledger.json"

    _write_json(
        packets_path,
        {
            "packets": [
                {
                    "path": "in/in-54.md",
                    "classification": "metadata_only",
                    "doc_owner": "maintainer",
                    "touch_set": ["in/in-54.md"],
                    "proving_tests": [],
                    "rows": [{"row_id": "docflow:row-1"}],
                    "stale_anchor_hints": [],
                }
            ]
        },
    )
    _write_json(baseline_path, {"entries": []})

    rc_new = docflow_packet_enforce.run(
        root=root,
        packets_path=packets_path,
        baseline_path=baseline_path,
        out_path=out_path,
        debt_out_path=debt_out_path,
        check=True,
        write_baseline=False,
        max_age_days=14,
        changed_paths=[],
        base_sha=None,
        head_sha=None,
        scope_allowlist=set(),
        run_proving_tests=False,
    )
    assert rc_new == 1

    _write_json(
        baseline_path,
        {
            "entries": [
                {
                    "row_id": "docflow:row-1",
                    "path": "in/in-54.md",
                    "classification": "metadata_only",
                    "first_seen": "2000-01-01",
                }
            ]
        },
    )
    rc_drift = docflow_packet_enforce.run(
        root=root,
        packets_path=packets_path,
        baseline_path=baseline_path,
        out_path=out_path,
        debt_out_path=debt_out_path,
        check=True,
        write_baseline=False,
        max_age_days=1,
        changed_paths=[],
        base_sha=None,
        head_sha=None,
        scope_allowlist=set(),
        run_proving_tests=False,
    )
    assert rc_drift == 1

    _write_json(
        baseline_path,
        {
            "entries": [
                {
                    "row_id": "docflow:row-1",
                    "path": "in/in-54.md",
                    "classification": "metadata_only",
                    "first_seen": "2100-01-01",
                }
            ]
        },
    )
    rc_scope = docflow_packet_enforce.run(
        root=root,
        packets_path=packets_path,
        baseline_path=baseline_path,
        out_path=out_path,
        debt_out_path=debt_out_path,
        check=True,
        write_baseline=False,
        max_age_days=365,
        changed_paths=["in/in-99.md"],
        base_sha=None,
        head_sha=None,
        scope_allowlist=set(),
        run_proving_tests=False,
    )
    assert rc_scope == 1


# gabion:behavior primary=desired
def test_docflow_packet_enforce_skips_scope_guard_without_active_packet_debt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True)

    packets_path = root / "artifacts" / "out" / "docflow_warning_doc_packets.json"
    baseline_path = root / "docs" / "baselines" / "docflow_packet_baseline.json"
    out_path = root / "artifacts" / "out" / "docflow_packet_enforcement.json"
    debt_out_path = root / "artifacts" / "out" / "docflow_packet_debt_ledger.json"

    _write_json(packets_path, {"packets": []})
    _write_json(baseline_path, {"entries": []})

    rc = docflow_packet_enforce.run(
        root=root,
        packets_path=packets_path,
        baseline_path=baseline_path,
        out_path=out_path,
        debt_out_path=debt_out_path,
        check=True,
        write_baseline=False,
        max_age_days=14,
        changed_paths=["AGENTS.md", "docs/governance_control_loops.md"],
        base_sha=None,
        head_sha=None,
        scope_allowlist=set(),
        run_proving_tests=False,
    )
    assert rc == 0
