from __future__ import annotations

from pathlib import Path
import sys
import textwrap

from gabion.analysis.timeout_context import Deadline, deadline_scope


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._internal_broad_type_lint_lines::annot
def test_internal_broad_type_str_linted(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def internal(x: str) -> str:
                return x

            def outer():
                return internal("hi")
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(project_root=tmp_path)
    with deadline_scope(Deadline.from_timeout_ticks(10_000, 1_000_000)):
        analysis = da.analyze_paths(
            [target],
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            include_deadness_witnesses=False,
            include_coherence_witnesses=False,
            include_rewrite_plans=False,
            include_exception_obligations=False,
            include_handledness_witnesses=False,
            include_never_invariants=False,
            include_decision_surfaces=False,
            include_value_decision_surfaces=False,
            include_invariant_propositions=False,
            include_lint_lines=True,
            include_ambiguities=False,
            include_bundle_forest=False,
            include_deadline_obligations=False,
            config=config,
        )
    assert any("GABION_BROAD_TYPE" in line for line in analysis.lint_lines)


def test_internal_broad_type_int_linted(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            def internal(x: int) -> int:
                return x + 1

            def outer():
                return internal(3)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(project_root=tmp_path)
    with deadline_scope(Deadline.from_timeout_ticks(10_000, 1_000_000)):
        analysis = da.analyze_paths(
            [target],
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            include_deadness_witnesses=False,
            include_coherence_witnesses=False,
            include_rewrite_plans=False,
            include_exception_obligations=False,
            include_handledness_witnesses=False,
            include_never_invariants=False,
            include_decision_surfaces=False,
            include_value_decision_surfaces=False,
            include_invariant_propositions=False,
            include_lint_lines=True,
            include_ambiguities=False,
            include_bundle_forest=False,
            include_deadline_obligations=False,
            config=config,
        )
    assert any("broad type 'int'" in line for line in analysis.lint_lines)


def test_internal_node_id_not_linted(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        textwrap.dedent(
            """
            from gabion.analysis.aspf import NodeId

            def internal(node: NodeId) -> NodeId:
                return node

            def outer():
                return internal(NodeId("Kind", ()))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(project_root=tmp_path)
    with deadline_scope(Deadline.from_timeout_ticks(10_000, 1_000_000)):
        analysis = da.analyze_paths(
            [target],
            recursive=True,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=0,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            include_deadness_witnesses=False,
            include_coherence_witnesses=False,
            include_rewrite_plans=False,
            include_exception_obligations=False,
            include_handledness_witnesses=False,
            include_never_invariants=False,
            include_decision_surfaces=False,
            include_value_decision_surfaces=False,
            include_invariant_propositions=False,
            include_lint_lines=True,
            include_ambiguities=False,
            include_bundle_forest=False,
            include_deadline_obligations=False,
            config=config,
        )
    assert not any("GABION_BROAD_TYPE" in line for line in analysis.lint_lines)


def test_broad_type_helpers_cover_edges(tmp_path: Path) -> None:
    da = _load()
    assert da._normalize_type_name("typing.Any") == "Any"
    assert da._normalize_type_name("builtins.int") == "int"
    assert da._is_broad_internal_type("None") is False
    assert da._is_broad_internal_type("Literal['x']") is True
    assert da._is_broad_internal_type("object") is True
    assert da._is_broad_internal_type("CustomType") is False


def test_internal_broad_type_skips_tests(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "test_sample.py"
    target.write_text(
        textwrap.dedent(
            """
            def internal(x: str) -> str:
                return x
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config = da.AuditConfig(project_root=tmp_path)
    with deadline_scope(Deadline.from_timeout_ticks(10_000, 1_000_000)):
        lines = da._internal_broad_type_lint_lines(
            [target],
            project_root=tmp_path,
            ignore_params=set(),
            strictness="high",
            external_filter=True,
        )
    assert lines == []
