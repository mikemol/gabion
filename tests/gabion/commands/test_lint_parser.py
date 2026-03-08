from __future__ import annotations

from gabion import cli, server
from gabion.commands.lint_parser import parse_lint_line


# gabion:evidence E:call_footprint::tests/test_lint_parser.py::test_shared_lint_parser_contract_for_valid_line::lint_parser.py::gabion.commands.lint_parser.parse_lint_line::cli.py::gabion.cli._parse_lint_line::server.py::gabion.server._parse_lint_line_as_payload
# gabion:behavior primary=desired
def test_shared_lint_parser_contract_for_valid_line() -> None:
    line = "pkg/mod.py:12:34: DF001 message body"

    shared = parse_lint_line(line)
    server_parsed = server._parse_lint_line_as_payload(line)
    cli_parsed = cli._parse_lint_line(line)

    assert shared is not None
    expected = shared.model_dump()
    assert server_parsed == expected
    assert cli_parsed == {**expected, "severity": "warning"}


# gabion:evidence E:call_footprint::tests/test_lint_parser.py::test_shared_lint_parser_contract_for_malformed_lines::lint_parser.py::gabion.commands.lint_parser.parse_lint_line::cli.py::gabion.cli._parse_lint_line::server.py::gabion.server._parse_lint_line_as_payload
# gabion:behavior primary=desired
def test_shared_lint_parser_contract_for_malformed_lines() -> None:
    malformed_lines = ["not a lint row", "pkg/mod.py:1:2:", "pkg/mod.py:1:2:   "]
    for line in malformed_lines:
        assert parse_lint_line(line) is None
        assert server._parse_lint_line_as_payload(line) is None
        assert cli._parse_lint_line(line) is None
