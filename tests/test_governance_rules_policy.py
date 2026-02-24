from __future__ import annotations

from pathlib import Path

import pytest

from gabion.tooling import governance_rules


# gabion:evidence E:function_site::tests/test_governance_rules_policy.py::test_governance_rules_define_required_gates
def test_governance_rules_define_required_gates() -> None:
    rules = governance_rules.load_governance_rules()
    assert {"obsolescence_opaque", "obsolescence_unmapped", "annotation_orphaned", "ambiguity", "docflow"}.issubset(rules.gates)
    assert rules.controller_drift.enforce_at_or_above in {"high", "critical", "medium", "low"}


# gabion:evidence E:function_site::tests/test_governance_rules_policy.py::test_governance_severity_thresholds_are_monotonic
def test_governance_severity_thresholds_are_monotonic() -> None:
    rules = governance_rules.load_governance_rules()
    for policy in rules.gates.values():
        assert policy.severity.warning_threshold <= policy.severity.blocking_threshold
        assert policy.correction.transitions
        assert "baseline_write_requires_explicit_flag" in policy.correction.bounded_steps


# gabion:evidence E:call_footprint::tests/test_governance_rules_policy.py::test_governance_rules_validation_failures::governance_rules.py::gabion.tooling.governance_rules._tuple_path::governance_rules.py::gabion.tooling.governance_rules._as_int::governance_rules.py::gabion.tooling.governance_rules._gate_from_mapping::governance_rules.py::gabion.tooling.governance_rules.load_governance_rules
def test_governance_rules_validation_failures(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        governance_rules._tuple_path("bad", field_name="field")

    with pytest.raises(ValueError):
        governance_rules._as_int("nope", field_name="num")

    with pytest.raises(ValueError):
        governance_rules._gate_from_mapping("x", {"severity": "bad", "correction": {}})
    with pytest.raises(ValueError):
        governance_rules._gate_from_mapping("x", {"severity": {}, "correction": "bad"})
    with pytest.raises(ValueError):
        governance_rules._gate_from_mapping(
            "x",
            {
                "severity": {"warning_threshold": 0, "blocking_threshold": 1},
                "correction": {"mode": 1, "transitions": [], "bounded_steps": []},
                "delta_keys": ["summary", "delta"],
                "before_keys": ["summary", "before"],
                "after_keys": ["summary", "after"],
            },
        )
    with pytest.raises(ValueError):
        governance_rules._gate_from_mapping(
            "x",
            {
                "severity": {"warning_threshold": 0, "blocking_threshold": 1},
                "correction": {"mode": "hard-fail", "transitions": "bad", "bounded_steps": []},
                "delta_keys": ["summary", "delta"],
                "before_keys": ["summary", "before"],
                "after_keys": ["summary", "after"],
            },
        )
    with pytest.raises(ValueError):
        governance_rules._gate_from_mapping(
            "x",
            {
                "severity": {"warning_threshold": 0, "blocking_threshold": 1},
                "correction": {"mode": "hard-fail", "transitions": [], "bounded_steps": "bad"},
                "delta_keys": ["summary", "delta"],
                "before_keys": ["summary", "before"],
                "after_keys": ["summary", "after"],
            },
        )

    root_bad = tmp_path / "root_bad.yaml"
    root_bad.write_text("- 1\n", encoding="utf-8")
    governance_rules.load_governance_rules.cache_clear()
    with pytest.raises(ValueError):
        governance_rules.load_governance_rules(root_bad)

    gates_bad = tmp_path / "gates_bad.yaml"
    gates_bad.write_text("override_token_env: TOKEN\ngates: {}\n", encoding="utf-8")
    governance_rules.load_governance_rules.cache_clear()
    with pytest.raises(ValueError):
        governance_rules.load_governance_rules(gates_bad)

    drift_bad = tmp_path / "drift_bad.yaml"
    drift_bad.write_text("override_token_env: TOKEN\ngates: {}\ncontroller_drift: []\n", encoding="utf-8")
    governance_rules.load_governance_rules.cache_clear()
    with pytest.raises(ValueError):
        governance_rules.load_governance_rules(drift_bad)

    gates_not_mapping = tmp_path / "gates_not_mapping.yaml"
    gates_not_mapping.write_text("override_token_env: TOKEN\ngates: []\ncontroller_drift: {}\n", encoding="utf-8")
    governance_rules.load_governance_rules.cache_clear()
    with pytest.raises(ValueError):
        governance_rules.load_governance_rules(gates_not_mapping)

    remediation_not_mapping = tmp_path / "remediation_not_mapping.yaml"
    remediation_not_mapping.write_text(
        """
override_token_env: TOKEN
gates: {}
controller_drift:
  severity_classes: [low]
  enforce_at_or_above: high
  remediation_by_severity: []
  consecutive_passes_required: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    governance_rules.load_governance_rules.cache_clear()
    with pytest.raises(ValueError):
        governance_rules.load_governance_rules(remediation_not_mapping)


# gabion:evidence E:call_footprint::tests/test_governance_rules_policy.py::test_governance_rules_loader_none_and_gate_filtering::governance_rules.py::gabion.tooling.governance_rules._yaml_loader::governance_rules.py::gabion.tooling.governance_rules.load_governance_rules
def test_governance_rules_loader_none_and_gate_filtering(tmp_path: Path) -> None:
    original_yaml = governance_rules.yaml
    try:
        governance_rules.yaml = None
        with pytest.raises(RuntimeError):
            governance_rules._yaml_loader()
    finally:
        governance_rules.yaml = original_yaml

    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(
        """
override_token_env: TOKEN
gates:
  1: not-a-mapping
  valid_gate:
    env_flag: GABION_GATE_TEST
    enabled_mode: default_true
    delta_keys: [summary, delta]
    before_keys: [summary, before]
    after_keys: [summary, after]
    severity:
      warning_threshold: 1
      blocking_threshold: 2
    correction:
      mode: hard-fail
      transitions: [advisory->ratchet]
      bounded_steps: [baseline_write_requires_explicit_flag]
controller_drift:
  severity_classes: [low, medium, high]
  enforce_at_or_above: high
  remediation_by_severity:
    high: temporary_override_or_fix
  consecutive_passes_required: 2
command_policies:
  1: not-a-mapping
  bad-shape: []
  valid.command:
    maturity: beta
    require_lsp_carrier: true
    parity_required: true
    probe_payload:
      a: 1
    parity_ignore_keys: [root]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    governance_rules.load_governance_rules.cache_clear()
    loaded = governance_rules.load_governance_rules(mixed)
    assert list(loaded.gates.keys()) == ["valid_gate"]
    assert list(loaded.command_policies.keys()) == ["valid.command"]
    non_mapping_commands = tmp_path / "non_mapping_commands.yaml"
    non_mapping_commands.write_text(
        """
override_token_env: TOKEN
gates:
  valid_gate:
    env_flag: GABION_GATE_TEST
    enabled_mode: default_true
    delta_keys: [summary, delta]
    before_keys: [summary, before]
    after_keys: [summary, after]
    severity:
      warning_threshold: 1
      blocking_threshold: 2
    correction:
      mode: hard-fail
      transitions: [advisory->ratchet]
      bounded_steps: [baseline_write_requires_explicit_flag]
controller_drift:
  severity_classes: [low, medium, high]
  enforce_at_or_above: high
  remediation_by_severity:
    high: temporary_override_or_fix
  consecutive_passes_required: 2
command_policies: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    governance_rules.load_governance_rules.cache_clear()
    loaded_non_mapping_commands = governance_rules.load_governance_rules(
        non_mapping_commands
    )
    assert loaded_non_mapping_commands.command_policies == {}
    governance_rules.load_governance_rules.cache_clear()


# gabion:evidence E:function_site::test_governance_rules_policy.py::tests.test_governance_rules_policy.test_governance_rules_as_bool_branches
def test_governance_rules_as_bool_branches() -> None:
    assert governance_rules._as_bool(True, field_name="f") is True
    assert governance_rules._as_bool(False, field_name="f") is False
    with pytest.raises(ValueError):
        governance_rules._as_bool("maybe", field_name="f")
    with pytest.raises(ValueError):
        governance_rules._as_bool(1, field_name="f")
