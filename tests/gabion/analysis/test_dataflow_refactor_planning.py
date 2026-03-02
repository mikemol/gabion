from __future__ import annotations

from pathlib import Path

from gabion.analysis.dataflow_contracts import AuditConfig
from gabion.analysis.dataflow_refactor_planning import (
    build_refactor_plan,
    render_refactor_plan,
    render_reuse_lemma_stubs,
)


def test_build_refactor_plan_handles_empty_file_set(tmp_path: Path) -> None:
    plan = build_refactor_plan(
        groups_by_path={},
        paths=[],
        config=AuditConfig(project_root=tmp_path),
    )

    assert plan["bundles"] == []
    assert "No files available for refactor plan." in plan["warnings"]
    assert plan["forest_signature_partial"] is True


def test_build_refactor_plan_handles_groups_without_matching_function_info(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "\n".join(
            [
                "def known(arg):",
                "    return arg",
                "",
            ]
        ),
        encoding="utf-8",
    )

    plan = build_refactor_plan(
        groups_by_path={source: {"missing.fn": [{"arg"}]}},
        paths=[source],
        config=AuditConfig(project_root=tmp_path, external_filter=False),
    )

    assert plan["bundles"] == []
    assert "No bundle components available for refactor plan." in plan["warnings"]


def test_build_refactor_plan_builds_component_schedule_and_cycles(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "\n".join(
            [
                "def first(arg):",
                "    return second(arg)",
                "",
                "def second(arg):",
                "    return first(arg)",
                "",
                "def external(arg):",
                "    return len(arg)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    plan = build_refactor_plan(
        groups_by_path={
            source: {
                "first": [{"arg"}],
                "second": [{"arg"}],
                "external": [{"arg"}],
            }
        },
        paths=[source],
        config=AuditConfig(project_root=tmp_path, external_filter=False),
    )

    bundles = plan["bundles"]
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle["bundle"] == ["arg"]
    assert sorted(bundle["functions"]) == [
        "module.external",
        "module.first",
        "module.second",
    ]
    assert bundle["cycles"]


def test_render_refactor_plan_renders_bundle_order_cycles_and_warnings() -> None:
    rendered = render_refactor_plan(
        {
            "bundles": [
                {
                    "bundle": ["alpha", "beta"],
                    "order": ["callee.fn", "caller.fn"],
                    "cycles": [["caller.fn", "callee.fn"]],
                }
            ],
            "warnings": ["cycle requires manual split"],
        }
    )

    assert "### Bundle: alpha, beta" in rendered
    assert "Order (callee-first):" in rendered
    assert "- callee.fn" in rendered
    assert "Cycles:" in rendered
    assert "caller.fn, callee.fn" in rendered
    assert "Warnings:" in rendered
    assert "cycle requires manual split" in rendered


def test_render_refactor_plan_handles_empty_bundle_list() -> None:
    rendered = render_refactor_plan({"bundles": [], "warnings": []})
    assert "No refactoring plan available." in rendered
    assert "Warnings:" not in rendered


def test_render_reuse_lemma_stubs_handles_empty_and_filtered_entries() -> None:
    empty_rendered = render_reuse_lemma_stubs({})
    assert "# No lemma suggestions available." in empty_rendered

    rendered = render_reuse_lemma_stubs(
        {
            "suggested_lemmas": [
                {"suggested_name": "", "rewrite_plan_artifact": {"plan_id": "skip"}},
                {"kind": "reuse", "suggested_name": "lemma_skip", "rewrite_plan_artifact": "not-a-dict"},
                {
                    "kind": "reuse",
                    "suggested_name": "lemma_a",
                    "rewrite_plan_artifact": {"plan_id": "p2", "detail": "beta"},
                },
                {
                    "kind": "reuse",
                    "suggested_name": "lemma_b",
                    "rewrite_plan_artifact": {"plan_id": "p1", "detail": "alpha"},
                },
            ]
        }
    )

    assert '"artifact_kind": "reuse_rewrite_plan_bundle"' in rendered
    assert '"plan_id": "p1"' in rendered
    assert '"plan_id": "p2"' in rendered


def test_render_refactor_plan_renders_cycles_without_order_block() -> None:
    rendered = render_refactor_plan(
        {
            "bundles": [
                {
                    "bundle": ["arg"],
                    "order": [],
                    "cycles": [["module.first", "module.second"]],
                },
                {
                    "bundle": ["other"],
                    "order": [],
                    "cycles": [["module.third", "module.fourth"]],
                },
                {
                    "bundle": ["leaf"],
                    "order": [],
                    "cycles": [],
                }
            ],
            "warnings": [],
        }
    )
    assert "Order (callee-first):" not in rendered
    assert "Cycles:" in rendered
