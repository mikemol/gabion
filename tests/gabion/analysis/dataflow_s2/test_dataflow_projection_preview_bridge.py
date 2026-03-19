from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.dataflow.engine.dataflow_contracts import ReportCarrier
from gabion.analysis.dataflow.io import dataflow_projection_preview_bridge as bridge
from gabion.exceptions import NeverThrown


def _build_groups() -> dict[Path, dict[str, list[set[str]]]]:
    return {
        Path("pkg/a.py"): {
            "pkg.a.fn": [{"alpha", "beta"}, {"gamma"}],
        },
        Path("pkg/b.py"): {
            "pkg.b.fn": [set()],
        },
    }


def _build_rich_report(*, forest: Forest) -> ReportCarrier:
    return ReportCarrier(
        forest=forest,
        type_suggestions=["type_suggestion"],
        type_ambiguities=["type_ambiguity"],
        type_callsite_evidence=["type_callsite"],
        constant_smells=["constant_smell"],
        unused_arg_smells=["unused_arg_smell"],
        decision_surfaces=["decision_surface"],
        value_decision_surfaces=["value_decision_surface"],
        decision_warnings=["decision_warning", "decision_warning"],
        fingerprint_warnings=["fingerprint_warning"],
        fingerprint_matches=["fingerprint_match"],
        fingerprint_synth=["fingerprint_synth"],
        fingerprint_provenance=[{"kind": "prov"}],
        context_suggestions=["context_hint"],
        value_decision_rewrites=["rewrite_hint"],
        deadline_obligations=[
            {
                "site": {"path": "pkg/a.py", "function": "pkg.a.fn"},
                "status": "VIOLATION",
                "kind": "deadline",
                "detail": "missing deadline",
                "span": [1, 0, 1, 10],
            }
        ],
        parse_failure_witnesses=[
            {"path": "pkg/parse_b.py", "stage": "", "error_type": "SyntaxError"},
            {
                "path": "pkg/parse_a.py",
                "stage": "ingest",
                "error_type": "SyntaxError",
                "error": "bad syntax",
            },
        ],
        resumability_obligations=[
            {
                "status": "VIOLATION",
                "contract": "resumability",
                "kind": "checkpoint",
                "detail": "missing state",
                "phase": "emit",
            },
            {
                "status": "SATISFIED",
                "contract": "resumability",
                "kind": "checkpoint",
            },
            {
                "status": "PENDING",
                "contract": "resumability",
                "kind": "checkpoint",
            },
        ],
        incremental_report_obligations=[
            {
                "status": "VIOLATION",
                "contract": "incremental_report",
                "kind": "section",
                "section_id": "summary",
                "detail": "missing section",
            },
        ],
        deprecated_signals=(
            "deprecated.signal.one",
            "deprecated.signal.two",
        ),
    )


# gabion:behavior primary=desired
def test_preview_section_lines_rich_report_covers_all_sections() -> None:
    forest = Forest()
    file_node = forest.add_file_site("pkg/a.py")
    forest.add_alt("PreviewAlt", (file_node,))
    groups = _build_groups()
    report = _build_rich_report(forest=forest)

    for section_id in sorted(bridge._PREVIEW_BUILDERS):  # noqa: SLF001
        lines = bridge.preview_section_lines(
            section_id,
            report=report,
            groups_by_path=groups,
            project_root=Path("."),
        )
        assert lines
        assert lines[0].endswith("preview (provisional).")

    violations = bridge.preview_section_lines(
        "violations",
        report=report,
        groups_by_path=groups,
        project_root=Path("."),
    )
    assert any("Top known violations:" in line for line in violations)
    assert sum(1 for line in violations if line == "- decision_warning") == 1

    resumability = bridge.preview_section_lines(
        "resumability_obligations",
        report=report,
        groups_by_path=groups,
        project_root=Path("."),
    )
    assert any("`violations`: `1`" in line for line in resumability)
    assert any("`satisfied`: `1`" in line for line in resumability)
    assert any("`pending`: `1`" in line for line in resumability)
    assert any("`sample_violation`" in line for line in resumability)

    parse_failures = bridge.preview_section_lines(
        "parse_failure_witnesses",
        report=report,
        groups_by_path=groups,
        project_root=Path("."),
    )
    assert any("`stage[ingest]`: `1`" in line for line in parse_failures)
    assert any("`stage[unknown]`: `1`" in line for line in parse_failures)


# gabion:behavior primary=verboten facets=empty
def test_preview_section_lines_empty_paths_cover_empty_branches() -> None:
    report = ReportCarrier(forest=Forest())
    groups: dict[Path, dict[str, list[set[str]]]] = {}

    components = bridge.preview_section_lines(
        "components",
        report=report,
        groups_by_path=groups,
        project_root=Path("."),
    )
    assert not any("forest_alternatives" in line for line in components)

    violations = bridge.preview_section_lines(
        "violations",
        report=report,
        groups_by_path=groups,
        project_root=Path("."),
    )
    assert "- none observed yet" in violations

    deadline = bridge.preview_section_lines(
        "deadline_summary",
        report=report,
        groups_by_path=groups,
        project_root=Path("."),
    )
    assert "- no deadline obligations yet" in deadline

    deprecated = bridge.preview_section_lines(
        "deprecated_substrate",
        report=report,
        groups_by_path=groups,
        project_root=Path("."),
    )
    assert deprecated == [
        "Deprecated substrate preview (provisional).",
        "- `informational_signals`: `0`",
    ]


# gabion:behavior primary=desired
def test_known_violation_lines_merges_sources_and_dedupes() -> None:
    report = ReportCarrier(
        forest=Forest(),
        resumability_obligations=[
            {
                "status": "VIOLATION",
                "contract": "resumability",
                "kind": "checkpoint",
                "detail": "missing state",
            }
        ],
        incremental_report_obligations=[
            {
                "status": "VIOLATION",
                "contract": "resumability",
                "kind": "checkpoint",
                "detail": "missing state",
            }
        ],
        parse_failure_witnesses=[
            {
                "path": "pkg/parse.py",
                "stage": "ingest",
                "error_type": "SyntaxError",
                "error": "bad syntax",
            }
        ],
        decision_warnings=["warning-a", "warning-a"],
    )
    lines = bridge._known_violation_lines(report)  # noqa: SLF001
    assert "warning-a" in lines
    assert sum(1 for line in lines if line == "warning-a") == 1
    assert any("parse_failure" in line for line in lines)


# gabion:behavior primary=verboten facets=never,raises
def test_preview_unknown_section_raises_never() -> None:
    with pytest.raises(NeverThrown):
        bridge.preview_section_lines(
            "not-a-section",
            report=ReportCarrier(forest=Forest()),
            groups_by_path={},
            project_root=Path("."),
        )


# gabion:behavior primary=desired
def test_preview_sections_cover_no_sample_and_no_violation_paths() -> None:
    report = ReportCarrier(
        forest=Forest(),
        resumability_obligations=[{"status": "SATISFIED", "contract": "resumability", "kind": "checkpoint"}],
        type_suggestions=["a"],
        type_callsite_evidence=["b"],
    )

    type_flow = bridge.preview_section_lines(
        "type_flow",
        report=report,
        groups_by_path={},
            project_root=Path("."),
        )
    assert not any("sample_type_ambiguity" in line for line in type_flow)

    resumability = bridge.preview_section_lines(
        "resumability_obligations",
        report=report,
        groups_by_path={},
            project_root=Path("."),
        )
    assert not any("sample_violation" in line for line in resumability)

    for section_id in (
        "constant_smells",
        "unused_arg_smells",
        "decision_surfaces",
        "value_decision_surfaces",
        "fingerprint_warnings",
        "fingerprint_matches",
        "fingerprint_synthesis",
        "context_suggestions",
    ):
        lines = bridge.preview_section_lines(
            section_id,
            report=report,
            groups_by_path={},
            project_root=Path("."),
        )
        assert all("sample_" not in line for line in lines)
