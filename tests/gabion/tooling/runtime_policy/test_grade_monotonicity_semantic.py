from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_substrate import grade_monotonicity_semantic
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_grade_monotonicity_semantic.py::test_grade_monotonicity_equal_grade_edge_passes::grade_monotonicity_semantic.py::gabion.tooling.policy_substrate.grade_monotonicity_semantic.collect_grade_monotonicity
# gabion:behavior primary=desired
def test_grade_monotonicity_equal_grade_edge_passes(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "good.py",
        "def leaf(value: int) -> int:\n"
        "    return value + 1\n\n"
        "def caller(value: int) -> int:\n"
        "    return leaf(value)\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)

    assert len(report.witnesses) == 1
    assert report.witnesses[0].edge_resolution_status == "resolved"
    assert report.witnesses[0].monotone is True
    assert report.violations == ()
    payload = report.policy_data()
    assert payload["violation_count"] == 0
    assert payload["witness_rows"][0]["caller_grade"]["type_domain_cardinality"] == 1


# gabion:evidence E:call_footprint::tests/test_grade_monotonicity_semantic.py::test_grade_monotonicity_looser_callee_fails_on_ambiguity_axes::grade_monotonicity_semantic.py::gabion.tooling.policy_substrate.grade_monotonicity_semantic.collect_grade_monotonicity
# gabion:behavior primary=desired
def test_grade_monotonicity_looser_callee_fails_on_ambiguity_axes(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "bad.py",
        "def looser(value: int | None) -> int:\n"
        "    if value is None:\n"
        "        return 0\n"
        "    return value\n\n"
        "def caller(value: int) -> int:\n"
        "    return looser(value)\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)

    rule_ids = {item.rule_id for item in report.violations}
    assert {"GMP-001", "GMP-002", "GMP-004"}.issubset(rule_ids)
    witness = report.witnesses[0]
    assert witness.monotone is False
    assert "GMP-001" in witness.failure_rule_ids
    assert witness.caller_grade.nullable_domain_cardinality == 1
    assert witness.callee_grade.nullable_domain_cardinality == 2
    nullable_violation = next(
        item for item in report.violations if item.rule_id == "GMP-001"
    )
    assert nullable_violation.details["guidance"] == {
        "why": "a callee accepts nullable or sentinel-bearing carriers after a stricter caller.",
        "prefer": "normalize nullability once at ingress; keep downstream callees strict",
        "avoid": [
            "do not reintroduce Optional or sentinel-bearing contracts downstream",
        ],
        "playbook_ref": "docs/policy_rules/grade_monotonicity.md#gmp-001",
    }


# gabion:evidence E:call_footprint::tests/test_grade_monotonicity_semantic.py::test_grade_monotonicity_boundary_marker_allows_cost_escalation::grade_monotonicity_semantic.py::gabion.tooling.policy_substrate.grade_monotonicity_semantic.collect_grade_monotonicity
# gabion:behavior primary=desired
def test_grade_monotonicity_boundary_marker_allows_cost_escalation(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "aggregate.py",
        "# gabion:grade_boundary kind=aggregation_materialization name=flatten_groups\n"
        "def flatten_groups(groups: list[list[int]]) -> list[int]:\n"
        "    return [item for group in groups for item in group]\n\n"
        "def caller(groups: list[list[int]]) -> list[int]:\n"
        "    return flatten_groups(groups)\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)

    assert len(report.witnesses) == 1
    witness = report.witnesses[0]
    assert witness.edge_kind == "boundary"
    assert witness.boundary_marker is not None
    assert witness.boundary_marker.kind.value == "aggregation_materialization"
    assert "GMP-006" not in witness.failure_rule_ids
    assert "GMP-007" not in witness.failure_rule_ids


# gabion:behavior primary=desired
def test_grade_monotonicity_decorator_boundary_allows_whole_function_scope(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "decorate.py",
        "from gabion.invariants import grade_boundary\n\n"
        "@grade_boundary(kind='aggregation_materialization', name='flatten_groups')\n"
        "def flatten_groups(groups: list[list[int]]) -> list[int]:\n"
        "    return [item for group in groups for item in group]\n\n"
        "def caller(groups: list[list[int]]) -> list[int]:\n"
        "    return flatten_groups(groups)\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)

    witness = report.witnesses[0]
    assert witness.edge_kind == "boundary"
    assert witness.boundary_marker is not None
    assert witness.boundary_marker.source == "function_decorator"
    assert "GMP-006" not in witness.failure_rule_ids
    assert "GMP-007" not in witness.failure_rule_ids


# gabion:behavior primary=desired
def test_grade_monotonicity_callsite_boundary_scope_is_local(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "scoped.py",
        "from gabion.invariants import grade_boundary\n\n"
        "def fanout(values: list[int]) -> list[int]:\n"
        "    items = [value for value in values]\n"
        "    return items\n\n"
        "def caller(values: list[int]) -> tuple[list[int], list[int]]:\n"
        "    with grade_boundary(kind='aggregation_materialization', name='scoped_fanout'):\n"
        "        allowed = fanout(values)\n"
        "    blocked = fanout(values)\n"
        "    return (allowed, blocked)\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)

    fanout_witnesses = [
        witness
        for witness in report.witnesses
        if witness.callee_qualname.endswith(".fanout")
    ]
    assert len(fanout_witnesses) == 2
    boundary_witness = next(
        witness for witness in fanout_witnesses if witness.boundary_marker is not None
    )
    ordinary_witness = next(
        witness for witness in fanout_witnesses if witness.boundary_marker is None
    )
    assert boundary_witness.boundary_marker is not None
    assert boundary_witness.boundary_marker.source == "callsite_with"
    assert "GMP-006" not in boundary_witness.failure_rule_ids
    assert "GMP-007" not in boundary_witness.failure_rule_ids
    assert "GMP-006" in ordinary_witness.failure_rule_ids
    assert "GMP-007" in ordinary_witness.failure_rule_ids


# gabion:behavior primary=desired
def test_grade_monotonicity_semantic_carrier_boundary_suppresses_adapter_edges(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "adapter.py",
        "import json\n\n"
        "# gabion:grade_boundary kind=semantic_carrier_adapter name=semantic_adapter\n"
        "def build_row(raw: str) -> dict[str, object]:\n"
        "    return {'payload': json.loads(raw), 'present': bool(raw)}\n\n"
        "def caller(raw: str) -> dict[str, object]:\n"
        "    return build_row(raw)\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)

    build_row_witnesses = [
        witness
        for witness in report.witnesses
        if witness.callee_qualname.endswith(".build_row")
    ]
    assert len(build_row_witnesses) == 1
    witness = build_row_witnesses[0]
    assert witness.edge_kind == "boundary"
    assert witness.boundary_marker is not None
    assert witness.boundary_marker.kind.value == "semantic_carrier_adapter"
    assert witness.failure_rule_ids == ()


# gabion:evidence E:call_footprint::tests/test_grade_monotonicity_semantic.py::test_grade_monotonicity_unresolved_external_targets_fail_closed::grade_monotonicity_semantic.py::gabion.tooling.policy_substrate.grade_monotonicity_semantic.collect_grade_monotonicity
# gabion:behavior primary=desired
def test_grade_monotonicity_unresolved_external_targets_fail_closed(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "external.py",
        "import json\n\n"
        "def caller(raw: str) -> object:\n"
        "    return json.loads(raw)\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)

    assert len(report.witnesses) == 1
    witness = report.witnesses[0]
    assert witness.edge_resolution_status == "unresolved_external"
    assert witness.callee_qualname.startswith("<unresolved_external:")
    assert witness.monotone is False
    assert witness.caller_grade.type_domain_cardinality == 1
    assert witness.caller_grade.shape_domain_cardinality == 1
    assert {
        "GMP-001",
        "GMP-002",
        "GMP-003",
        "GMP-004",
        "GMP-006",
        "GMP-007",
    }.issubset(set(witness.failure_rule_ids))


# gabion:evidence E:call_footprint::tests/test_grade_monotonicity_semantic.py::test_grade_monotonicity_materialized_local_output_counts_for_cardinality::grade_monotonicity_semantic.py::gabion.tooling.policy_substrate.grade_monotonicity_semantic.collect_grade_monotonicity
# gabion:behavior primary=desired
def test_grade_monotonicity_materialized_local_output_counts_for_cardinality(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "fanout.py",
        "def fanout(values: list[int]) -> list[int]:\n"
        "    items = [value for value in values]\n"
        "    return items\n\n"
        "def caller(values: list[int]) -> list[int]:\n"
        "    return fanout(values)\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)

    witness = report.witnesses[0]
    assert "GMP-006" in witness.failure_rule_ids
    assert "GMP-007" in witness.failure_rule_ids
    assert witness.callee_grade.output_cardinality_class.name == "LINEAR"
    assert witness.callee_grade.work_growth_class.name == "LINEAR"


# gabion:behavior primary=desired
def test_grade_monotonicity_keys_follow_structural_identity_not_line_motion(
    tmp_path: Path,
) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    source_a = (
        "def fanout(values: list[int]) -> list[int]:\n"
        "    items = [value for value in values]\n"
        "    return items\n\n"
        "def caller(values: list[int]) -> list[int]:\n"
        "    return fanout(values)\n"
    )
    source_b = (
        "def fanout(values: list[int]) -> list[int]:\n"
        "    items = [value for value in values]\n"
        "    return items\n\n"
        "def caller(values: list[int]) -> list[int]:\n"
        "    \n"
        "    return fanout(values)\n"
    )
    _write(repo_a / "src" / "gabion" / "analysis" / "identity.py", source_a)
    _write(repo_b / "src" / "gabion" / "analysis" / "identity.py", source_b)

    report_a = grade_monotonicity_semantic.collect_grade_monotonicity(
        batch=build_policy_scan_batch(
            root=repo_a,
            target_globs=("src/gabion/analysis/**/*.py",),
        )
    )
    report_b = grade_monotonicity_semantic.collect_grade_monotonicity(
        batch=build_policy_scan_batch(
            root=repo_b,
            target_globs=("src/gabion/analysis/**/*.py",),
        )
    )

    witness_a = report_a.witnesses[0]
    witness_b = report_b.witnesses[0]
    assert witness_a.witness_id == witness_b.witness_id
    assert witness_a.edge_site_identity != witness_b.edge_site_identity
    assert report_a.violations[0].key == report_b.violations[0].key


# gabion:evidence E:call_footprint::tests/test_grade_monotonicity_semantic.py::test_grade_monotonicity_protocol_discharge_uses_explicit_marker::grade_monotonicity_semantic.py::gabion.tooling.policy_substrate.grade_monotonicity_semantic.collect_grade_monotonicity
# gabion:behavior primary=desired
def test_grade_monotonicity_protocol_discharge_uses_explicit_marker(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "protocol.py",
        "from gabion.invariants import decision_protocol, never\n\n"
        "@decision_protocol\n"
        "def protocol(value: int) -> int:\n"
        "    match value:\n"
        "        case 0:\n"
        "            return 0\n"
        "        case _:\n"
        "            never('unreachable')\n",
    )
    batch = build_policy_scan_batch(
        root=tmp_path,
        target_globs=("src/gabion/analysis/**/*.py",),
    )

    report = grade_monotonicity_semantic.collect_grade_monotonicity(batch=batch)
    grade_by_qual = dict(report.callable_grades)

    protocol_grade = grade_by_qual["gabion.analysis.protocol.protocol"]
    assert int(protocol_grade.protocol_discharge_level) == 3
    assert protocol_grade.runtime_classification_count >= 1


# gabion:behavior primary=desired
def test_grade_monotonicity_governance_priority_ranks_follow_playbook() -> None:
    assert (
        grade_monotonicity_semantic.grade_monotonicity_governance_priority_rank(
            "GMP-001"
        )
        == 10
    )
    assert (
        grade_monotonicity_semantic.grade_monotonicity_governance_priority_rank(
            "GMP-007"
        )
        == 70
    )
    assert (
        grade_monotonicity_semantic.grade_monotonicity_governance_priority_rank(
            "grade_monotonicity.new_violations"
        )
        is None
    )
