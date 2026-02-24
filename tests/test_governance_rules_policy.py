from __future__ import annotations

from pathlib import Path

import pytest

from gabion.tooling import governance_rules


# gabion:evidence E:function_site::tests/test_governance_rules_policy.py::test_governance_rules_define_required_gates
def test_governance_rules_define_required_gates() -> None:
    rules = governance_rules.load_governance_rules()
    assert {"obsolescence_opaque", "obsolescence_unmapped", "annotation_orphaned", "ambiguity", "docflow"}.issubset(rules.gates)


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
    gates_bad.write_text("override_token_env: TOKEN\ngates: []\n", encoding="utf-8")
    governance_rules.load_governance_rules.cache_clear()
    with pytest.raises(ValueError):
        governance_rules.load_governance_rules(gates_bad)


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
