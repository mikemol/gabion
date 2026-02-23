from __future__ import annotations

import importlib

from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import check_deadline, get_deadline_clock, get_forest


def _import_module(module_path: str):
    return importlib.import_module(module_path)

# gabion:evidence E:call_footprint::tests/test_script_scope_bindings.py::test_script_scope_helpers_bind_deadline_clock_and_forest::test_script_scope_bindings.py::tests.test_script_scope_bindings._import_module::timeout_context.py::gabion.analysis.timeout_context.check_deadline::timeout_context.py::gabion.analysis.timeout_context.get_deadline_clock::timeout_context.py::gabion.analysis.timeout_context.get_forest
def test_script_scope_helpers_bind_deadline_clock_and_forest() -> None:
    scope_functions = [
        ("scripts.audit_tools", "_audit_deadline_scope"),
        ("gabion.tooling.delta_advisory", "_deadline_scope"),
        ("scripts.audit_in_step_structure", "_deadline_scope"),
        ("scripts.ci_watch", "_deadline_scope"),
        ("gabion.tooling.delta_triplets", "_deadline_scope"),
        ("gabion.tooling.docflow_delta_emit", "_delta_deadline_scope"),
        ("scripts.docflow_promote_sections", "_promote_deadline_scope"),
        ("scripts.extract_test_evidence", "_deadline_scope_from_env"),
        ("scripts.pin_actions", "_deadline_scope"),
        ("scripts.policy_check", "_policy_deadline_scope"),
        ("scripts.refresh_baselines", "_deadline_scope"),
        ("scripts.sppf_sync", "_deadline_scope"),
    ]
    for module_path, scope_name in scope_functions:
        module = _import_module(module_path)
        scope = getattr(module, scope_name)
        with scope():
            assert isinstance(get_forest(), Forest)
            start_mark = get_deadline_clock().get_mark()
            check_deadline()
            end_mark = get_deadline_clock().get_mark()
            assert end_mark >= start_mark
