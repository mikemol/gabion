from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_rules import aspf_normalization_idempotence_rule as rule
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


# gabion:behavior primary=desired
def test_collect_violations_flags_duplicate_pre_core_normalization_class(
    tmp_path: Path,
) -> None:
    trace_payload = {
        "trace_id": "aspf-trace:test",
        "one_cells": [
            {
                "source": "ingress:payload",
                "target": "carrier:payload",
                "representative": "first",
                "basis_path": ["ingress", "parse"],
                "kind": "boundary_parse",
                "surface": "",
                "metadata": {"normalization_class": "parse"},
            },
            {
                "source": "ingress:payload",
                "target": "carrier:payload",
                "representative": "second",
                "basis_path": ["ingress", "parse"],
                "kind": "boundary_parse",
                "surface": "",
                "metadata": {"normalization_class": "parse"},
            },
            {
                "source": "runtime:inputs",
                "target": "analysis:engine",
                "representative": "start",
                "basis_path": ["analysis", "call", "start"],
                "kind": "analysis_call_start",
                "surface": "",
                "metadata": {},
            },
        ],
    }
    _write_json(tmp_path / "artifacts/out/aspf_trace.json", trace_payload)
    _write_jsonl(
        tmp_path / "artifacts/out/aspf_delta.jsonl",
        [
            {
                "one_cell_ref": "one_cells.1",
                "phase": "ingress",
            },
            {
                "one_cell_ref": "one_cells.2",
                "phase": "ingress",
            },
        ],
    )

    batch = build_policy_scan_batch(root=tmp_path, target_globs=())
    violations = rule.collect_violations(batch=batch)
    assert len(violations) == 1
    assert violations[0].kind == "duplicate_normalization_class_pre_core"
    assert violations[0].normalization_class == "parse"
    assert violations[0].fiber_trace
    assert violations[0].applicability_bounds is not None
    assert (
        violations[0].applicability_bounds.violation_applies_when_boundary_before_ordinal_gt
        == 2
    )
    assert violations[0].counterfactual_boundary is not None
    assert (
        violations[0].counterfactual_boundary.suggested_boundary_before_ordinal == 2
    )


# gabion:behavior primary=desired
def test_collect_violations_ignores_post_core_duplicates(tmp_path: Path) -> None:
    trace_payload = {
        "trace_id": "aspf-trace:test",
        "one_cells": [
            {
                "source": "runtime:inputs",
                "target": "analysis:engine",
                "representative": "start",
                "basis_path": ["analysis", "call", "start"],
                "kind": "analysis_call_start",
                "surface": "",
                "metadata": {},
            },
            {
                "source": "analysis:engine",
                "target": "analysis:engine",
                "representative": "first",
                "basis_path": ["analysis", "normalize"],
                "kind": "normalize_payload",
                "surface": "",
                "metadata": {"normalization_class": "normalize"},
            },
            {
                "source": "analysis:engine",
                "target": "analysis:engine",
                "representative": "second",
                "basis_path": ["analysis", "normalize"],
                "kind": "normalize_payload",
                "surface": "",
                "metadata": {"normalization_class": "normalize"},
            },
        ],
    }
    _write_json(tmp_path / "artifacts/out/aspf_trace.json", trace_payload)

    batch = build_policy_scan_batch(root=tmp_path, target_globs=())
    violations = rule.collect_violations(batch=batch)
    assert violations == []


# gabion:behavior primary=allowed_unwanted facets=legacy
def test_load_baseline_rejects_mixed_legacy_and_structured_entries(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baselines/aspf_normalization_idempotence_policy_baseline.json"
    _write_json(
        baseline_path,
        {
            "version": 1,
            "violations": [
                {
                    "path": "artifacts/out/aspf_trace.json",
                    "qualname": "flow:path:1.2.3",
                    "line": 7,
                    "kind": "duplicate_normalization_class_pre_core",
                },
                {
                    "path": "artifacts/out/aspf_trace.json",
                    "qualname": "flow:path:1.2.3",
                    "kind": "duplicate_normalization_class_pre_core",
                    "structured_hash": "abc123",
                },
            ],
        },
    )
    keys = rule._load_baseline(baseline_path)
    assert keys == set()


# gabion:behavior primary=verboten facets=strict
def test_load_baseline_accepts_strict_structured_entries(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baselines/aspf_normalization_idempotence_policy_baseline.json"
    _write_json(
        baseline_path,
        {
            "version": 1,
            "violations": [
                {
                    "path": "artifacts/out/aspf_trace.json",
                    "qualname": "flow:path:1.2.3",
                    "kind": "duplicate_normalization_class_pre_core",
                    "structured_hash": "abc123",
                },
            ],
        },
    )
    keys = rule._load_baseline(baseline_path)
    assert keys == {
        "artifacts/out/aspf_trace.json:flow:path:1.2.3:"
        "duplicate_normalization_class_pre_core:abc123"
    }


# gabion:behavior primary=verboten facets=invalid
def test_collect_ingress_violations_reports_invalid_trace_json(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "artifacts/out/aspf_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("{bad", encoding="utf-8")

    violations = rule.collect_ingress_violations(root=tmp_path)
    assert len(violations) == 1
    assert violations[0].kind == "invalid_trace_document_json"
    assert violations[0].path == trace_path.as_posix()


# gabion:behavior primary=verboten facets=invalid
def test_collect_ingress_violations_reports_invalid_delta_line_shape(
    tmp_path: Path,
) -> None:
    trace_payload = {
        "trace": {
            "one_cells": [],
            "controls": {"aspf_delta_jsonl": "artifacts/out/custom.delta.jsonl"},
        }
    }
    _write_json(tmp_path / "artifacts/out/aspf_trace.json", trace_payload)
    _write_jsonl(
        tmp_path / "artifacts/out/custom.delta.jsonl",
        [
            {"phase": "ingress"},
        ],
    )

    violations = rule.collect_ingress_violations(root=tmp_path)
    assert len(violations) == 1
    assert violations[0].kind == "invalid_delta_jsonl_line_shape"
    assert violations[0].path == (tmp_path / "artifacts/out/custom.delta.jsonl").as_posix()


# gabion:behavior primary=verboten facets=missing
def test_collect_ingress_violations_ignores_missing_external_controls_delta_path(
    tmp_path: Path,
) -> None:
    trace_payload = {
        "trace": {
            "one_cells": [],
            "controls": {"aspf_delta_jsonl": "/tmp/nonexistent-gabion.delta.jsonl"},
        }
    }
    _write_json(tmp_path / "artifacts/out/aspf_trace.json", trace_payload)

    violations = rule.collect_ingress_violations(root=tmp_path)
    assert violations == []


# gabion:behavior primary=desired
def test_collect_ingress_violations_ignores_default_delta_jsonl_event_noise(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "artifacts/out/aspf_trace.json", {"one_cells": []})
    _write_jsonl(
        tmp_path / "artifacts/out/aspf_delta.jsonl",
        [
            {
                "seq": 1,
                "event_kind": "semantic_surface_projection",
                "one_cell_ref": None,
                "phase": "post",
            },
        ],
    )

    violations = rule.collect_ingress_violations(root=tmp_path)
    assert violations == []


# gabion:behavior primary=verboten facets=invalid
def test_collect_ingress_violations_reports_invalid_baseline_payload(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baselines/aspf_normalization_idempotence_policy_baseline.json"
    _write_json(
        baseline_path,
        {
            "version": 1,
            "violations": [
                {
                    "path": "artifacts/out/aspf_trace.json",
                    "qualname": "flow:path:1.2.3",
                    "kind": "duplicate_normalization_class_pre_core",
                },
            ],
        },
    )

    violations = rule.collect_ingress_violations(
        root=tmp_path,
        baseline_path=baseline_path,
    )
    assert len(violations) == 1
    assert violations[0].kind == "invalid_baseline_payload"
    assert violations[0].path == baseline_path.as_posix()


# gabion:behavior primary=desired
def test_collect_violations_defaults_to_current_run_trace_only(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts/out/aspf_trace.json",
        {
            "one_cells": [
                {
                    "source": "ingress:payload",
                    "target": "carrier:payload",
                    "basis_path": ["ingress", "parse"],
                    "kind": "boundary_parse",
                    "surface": "",
                    "metadata": {"normalization_class": "parse"},
                },
                {
                    "source": "runtime:inputs",
                    "target": "analysis:engine",
                    "basis_path": ["analysis", "call", "start"],
                    "kind": "analysis_call_start",
                    "surface": "",
                    "metadata": {},
                },
            ]
        },
    )
    snapshot_path = (
        tmp_path
        / "artifacts/out/aspf_state/session-20260306T220000Z/0002_ci-local.snapshot.json"
    )
    _write_json(
        snapshot_path,
        {
            "one_cells": [
                {
                    "source": "ingress:payload",
                    "target": "carrier:payload",
                    "basis_path": ["ingress", "parse"],
                    "kind": "boundary_parse",
                    "surface": "",
                    "metadata": {"normalization_class": "parse"},
                },
                {
                    "source": "ingress:payload",
                    "target": "carrier:payload",
                    "basis_path": ["ingress", "parse"],
                    "kind": "boundary_parse",
                    "surface": "",
                    "metadata": {"normalization_class": "parse"},
                },
            ]
        },
    )

    batch = build_policy_scan_batch(root=tmp_path, target_globs=())
    default_violations = rule.collect_violations(batch=batch)
    assert default_violations == []

    archive_violations = rule.collect_violations(
        batch=batch,
        include_snapshot_archive=True,
    )
    assert len(archive_violations) == 1
    assert archive_violations[0].path == snapshot_path.relative_to(tmp_path).as_posix()


# gabion:behavior primary=desired
def test_collect_ingress_violations_snapshot_archive_is_opt_in(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts/out/aspf_trace.json",
        {
            "trace": {
                "one_cells": [],
            }
        },
    )
    snapshot_path = (
        tmp_path
        / "artifacts/out/aspf_state/session-20260306T220500Z/0003_ci-local.snapshot.json"
    )
    _write_json(snapshot_path, {})

    default_violations = rule.collect_ingress_violations(root=tmp_path)
    assert default_violations == []

    archive_violations = rule.collect_ingress_violations(
        root=tmp_path,
        include_snapshot_archive=True,
    )
    assert len(archive_violations) == 1
    assert archive_violations[0].kind == "invalid_trace_payload_shape"
    assert archive_violations[0].path == snapshot_path.as_posix()
