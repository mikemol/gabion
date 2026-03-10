from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import no_scalar_conversion_boundary_rule as rule
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch_from_sources


# gabion:behavior primary=desired
def test_collect_violations_detects_scalar_conversion_outside_boundary() -> None:
    source = "\n".join(
        [
            "from functools import reduce",
            "",
            "def normalize(value):",
            "    label = str(value)",
            "    explicit = value.__str__()",
            "    formatted = f'value={value}'",
            "    rendered = '{}'.format(value)",
            "    combined = 'value=' + value",
            "    joined = ','.join([value, value])",
            "    reduced = reduce(lambda left, right: left, [value, value])",
            "    return label, explicit, formatted, rendered, combined, joined, reduced",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/analysis/foundation/sample.py": source},
    )

    violations = rule.collect_violations(batch=batch)

    kinds = {item.kind for item in violations}
    assert "scalar_cast" in kinds
    assert "dunder_str_call" in kinds
    assert "fstring_format" in kinds
    assert "string_format" in kinds
    assert "string_add" in kinds
    assert "string_join" in kinds
    assert "reduce_call" in kinds
    assert violations[0].fiber_trace


# gabion:behavior primary=desired
def test_collect_violations_allows_io_boundary_and_dunder_str() -> None:
    io_source = "\n".join(
        [
            "def render(value):",
            "    return str(value)",
            "",
        ]
    )
    dto_source = "\n".join(
        [
            "class LabelDTO:",
            "    def __init__(self, value):",
            "        self.value = value",
            "",
            "    def __str__(self):",
            "        return str(self.value)",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={
            "src/gabion/analysis/dataflow/io/sample.py": io_source,
            "src/gabion/analysis/foundation/dto.py": dto_source,
        },
    )

    violations = rule.collect_violations(batch=batch)

    assert violations == []


# gabion:behavior primary=desired
def test_collect_violations_detects_semantic_string_add_flow() -> None:
    source = "\n".join(
        [
            "def suffix() -> str:",
            "    return 'tail'",
            "",
            "def combine(prefix: str, value):",
            "    return prefix + value",
            "",
            "def combine_with_call(value):",
            "    prefix = suffix()",
            "    return prefix + value",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/analysis/foundation/sample.py": source},
    )

    violations = rule.collect_violations(batch=batch)

    string_add_violations = [item for item in violations if item.kind == "string_add"]
    assert len(string_add_violations) == 2
