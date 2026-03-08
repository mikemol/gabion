from __future__ import annotations

import ast
from pathlib import Path


CLI_PATH = Path("src/gabion/cli.py")


def _function_nodes() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"), filename=str(CLI_PATH))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


# gabion:evidence E:function_site::cli.py::gabion.cli._emit_lint_outputs E:function_site::cli.py::gabion.cli._check_gate_policy
# gabion:behavior primary=desired
def test_cli_runtime_helpers_are_thin_facades() -> None:
    functions = _function_nodes()
    facade_names = {
        "_parse_lint_line",
        "_collect_lint_entries",
        "_write_lint_jsonl",
        "_emit_lint_outputs",
        "_emit_timeout_profile_artifacts",
        "_default_check_artifact_flags",
        "_default_check_delta_options",
        "_check_help_or_exit",
        "_check_gate_policy",
        "_check_lint_mode",
        "_nonzero_exit_causes",
        "_emit_nonzero_exit_causes",
        "_emit_analysis_resume_summary",
    }
    missing = facade_names - functions.keys()
    assert not missing

    for name in sorted(facade_names):
        body = functions[name].body
        assert len(body) <= 2, f"{name} should stay a thin facade"


# gabion:evidence E:call_footprint::tests/test_cli_architecture_entrypoint.py::test_cli_runtime_logic_resides_in_support_modules::cli.py::gabion.cli._context_cli_deps
# gabion:behavior primary=desired
def test_cli_runtime_logic_resides_in_support_modules() -> None:
    source = CLI_PATH.read_text(encoding="utf-8")
    assert "result_emitters." in source
    assert "check_runtime_facade." in source
    assert "context_cli_runtime_deps(" in source


# gabion:evidence E:call_footprint::tests/test_cli_architecture_entrypoint.py::test_cli_composition_root_avoids_regex_business_logic
# gabion:behavior primary=desired
def test_cli_composition_root_avoids_regex_business_logic() -> None:
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"), filename=str(CLI_PATH))
    assert all(
        not isinstance(node, ast.Import) or all(alias.name != "re" for alias in node.names)
        for node in tree.body
    )
    assert all(
        not isinstance(node, ast.ImportFrom) or node.module != "re"
        for node in tree.body
    )
    assert "re.compile(" not in CLI_PATH.read_text(encoding="utf-8")
