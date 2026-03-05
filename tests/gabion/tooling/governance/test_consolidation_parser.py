from __future__ import annotations

from gabion_governance.consolidation_audit.parser import parse_lint_entry, parse_surface_line


def test_parse_surface_line_decision() -> None:
    line = "src/x.py:foo decision surface params: flag, mode (tier=2)"
    parsed = parse_surface_line(line, value_encoded=False)
    assert parsed is not None
    assert parsed.path == "src/x.py"
    assert parsed.qual == "foo"
    assert parsed.params == ("flag", "mode")
    assert parsed.meta == "tier=2"


def test_parse_surface_line_value_encoded() -> None:
    line = "src/x.py:foo value-encoded decision params: encoded_flag (tier=2)"
    parsed = parse_surface_line(line, value_encoded=True)
    assert parsed is not None
    assert parsed.params == ("encoded_flag",)


def test_parse_surface_line_rejects_malformed() -> None:
    assert parse_surface_line("src/x.py:foo params: x", value_encoded=False) is None


def test_parse_lint_entry_extracts_param() -> None:
    line = "src/x.py:10:2: GABION_DECISION_SURFACE param 'flag' (decision surface)"
    parsed = parse_lint_entry(line)
    assert parsed is not None
    assert parsed.code == "GABION_DECISION_SURFACE"
    assert parsed.param == "flag"


def test_parse_lint_entry_rejects_non_lint_shape() -> None:
    assert parse_lint_entry("not-a-lint-line") is None
