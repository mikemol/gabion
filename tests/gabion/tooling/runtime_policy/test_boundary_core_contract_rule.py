from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import boundary_core_contract_rule as rule


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_boundary_core_contract_passes_for_single_hop_boundary_to_core(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/example_boundary.py",
        "\n".join(
            [
                "# gabion:boundary_normalization_module",
                "from gabion.example_core import run_core",
                "",
                "def run_boundary(value: str) -> str:",
                "    normalized = value.strip()",
                "    return run_core(normalized)",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "src/gabion/example_core.py",
        "\n".join(
            [
                "def run_core(value: str) -> str:",
                "    return value",
            ]
        )
        + "\n",
    )

    violations = rule.collect_violations(root=tmp_path)
    assert violations == []


def test_boundary_core_contract_flags_missing_core_pair(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/example_boundary.py",
        "\n".join(
            [
                "# gabion:boundary_normalization_module",
                "def run_boundary(value: str) -> str:",
                "    return value",
            ]
        )
        + "\n",
    )

    violations = rule.collect_violations(root=tmp_path)
    assert len(violations) == 1
    assert violations[0].kind == "missing_paired_core_module"


def test_boundary_core_contract_flags_raw_ingress_types_in_core(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/example_boundary.py",
        "\n".join(
            [
                "# gabion:boundary_normalization_module",
                "import gabion.example_core as example_core",
                "",
                "def run_boundary(value: str) -> str:",
                "    return example_core.run_core(value)",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "src/gabion/example_core.py",
        "\n".join(
            [
                "from typing import Any",
                "",
                "def run_core(value: Any) -> str:",
                "    return str(value)",
            ]
        )
        + "\n",
    )

    violations = rule.collect_violations(root=tmp_path)
    assert any(item.kind == "raw_ingress_type_in_core" for item in violations)
