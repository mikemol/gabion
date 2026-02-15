from __future__ import annotations

import importlib
from pathlib import Path
import sys

from gabion.analysis.timeout_context import check_deadline


def _import_script_module(name: str):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    return importlib.import_module(name)


def test_script_scope_helpers_bind_deadline_clock_and_forest() -> None:
    scope_functions = [
        ("audit_tools", "_audit_deadline_scope"),
        ("ambiguity_delta_advisory", "_deadline_scope"),
        ("annotation_drift_delta_advisory", "_deadline_scope"),
        ("audit_in_step_structure", "_deadline_scope"),
        ("ci_watch", "_deadline_scope"),
        ("delta_triplets", "_deadline_scope"),
        ("docflow_delta_advisory", "_deadline_scope"),
        ("docflow_delta_emit", "_delta_deadline_scope"),
        ("docflow_promote_sections", "_promote_deadline_scope"),
        ("extract_test_evidence", "_deadline_scope_from_env"),
        ("obsolescence_delta_advisory", "_deadline_scope"),
        ("pin_actions", "_deadline_scope"),
        ("policy_check", "_policy_deadline_scope"),
        ("refresh_baselines", "_deadline_scope"),
        ("sppf_sync", "_deadline_scope"),
    ]
    for module_name, scope_name in scope_functions:
        module = _import_script_module(module_name)
        scope = getattr(module, scope_name)
        with scope():
            check_deadline()
