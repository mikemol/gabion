from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import fiber_normalization_contract_rule as rule


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:behavior primary=desired
def test_fiber_rule_flags_duplicate_pre_core_narrowing(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/example_boundary.py",
        "\n".join(
            [
                "# gabion:boundary_normalization_module",
                "import gabion.example_core as example_core",
                "",
                "def run_boundary(value: object) -> str:",
                "    if isinstance(value, str):",
                "        pass",
                "    if isinstance(value, str):",
                "        pass",
                "    return example_core.run_core(value)",
            ]
        )
        + "\n",
    )

    violations = rule.collect_violations(root=tmp_path)
    assert len(violations) == 1
    assert violations[0].kind == "duplicate_normalization_before_core"
    assert violations[0].normalization_class == "narrow"
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
def test_fiber_rule_ignores_post_core_reapplication(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/example_boundary.py",
        "\n".join(
            [
                "# gabion:boundary_normalization_module",
                "import gabion.example_core as example_core",
                "",
                "def run_boundary(value: object) -> str:",
                "    if isinstance(value, str):",
                "        pass",
                "    result = example_core.run_core(value)",
                "    if isinstance(value, str):",
                "        pass",
                "    return result",
            ]
        )
        + "\n",
    )

    violations = rule.collect_violations(root=tmp_path)
    assert violations == []


# gabion:behavior primary=desired
def test_fiber_rule_reads_annotation_contract(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/example_boundary.py",
        "\n".join(
            [
                "# gabion:boundary_normalization_module",
                "import gabion.example_core as example_core",
                "",
                "def run_boundary(payload: str) -> str:",
                "    # gabion:taint_intro input=payload class=parse",
                "    # gabion:taint_intro input=payload class=parse",
                "    return example_core.run_core(payload)",
            ]
        )
        + "\n",
    )

    violations = rule.collect_violations(root=tmp_path)
    assert len(violations) == 1
    assert violations[0].normalization_class == "parse"
    assert violations[0].input_slot == "payload"
    assert violations[0].fiber_trace[0].phase_hint == "annotation"
