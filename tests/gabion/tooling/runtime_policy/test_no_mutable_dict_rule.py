from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import no_mutable_dict_rule as rule
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch_from_sources


# gabion:behavior primary=desired
def test_collect_violations_detects_mutable_dict_shapes() -> None:
    source = "\n".join(
        [
            "def make_dict(value):",
            "    payload = {'value': value}",
            "    return payload",
            "",
            "def make_dict_comp(values):",
            "    return {value: value for value in values}",
            "",
            "def make_dict_ctor(values):",
            "    return dict(values)",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/sample.py": source},
    )

    violations = rule.collect_violations(batch=batch)

    kinds = {item.kind for item in violations}
    assert "dict_literal" in kinds
    assert "dict_comprehension" in kinds
    assert "dict_constructor" in kinds
    assert violations[0].fiber_trace


# gabion:behavior primary=desired
def test_collect_violations_accepts_dto_only_shapes() -> None:
    source = "\n".join(
        [
            "from dataclasses import dataclass",
            "",
            "@dataclass(frozen=True)",
            "class EntryDTO:",
            "    key: str",
            "    value: int",
            "",
            "def make_entry(key: str, value: int) -> EntryDTO:",
            "    return EntryDTO(key=key, value=value)",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/sample.py": source},
    )

    violations = rule.collect_violations(batch=batch)

    assert violations == []
