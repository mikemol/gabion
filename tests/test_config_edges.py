from __future__ import annotations

from pathlib import Path

from gabion import config


def test_load_toml_missing_and_invalid(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    assert config._load_toml(missing) == {}

    invalid = tmp_path / "invalid.toml"
    invalid.write_text("not = [toml", encoding="utf-8")
    assert config._load_toml(invalid) == {}

    assert config._load_toml(tmp_path) == {}


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.load_config::config_path,root
def test_load_config_default_path(tmp_path: Path) -> None:
    cfg = tmp_path / config.DEFAULT_CONFIG_NAME
    cfg.write_text("[dataflow]\nstrictness = 'low'\n", encoding="utf-8")
    data = config.load_config(root=tmp_path, config_path=None)
    assert data["dataflow"]["strictness"] == "low"


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.decision_tier_map::section E:decision_surface/direct::config.py::gabion.config._normalize_name_list::value
def test_decision_tier_map_normalizes_inputs() -> None:
    assert config.decision_tier_map(None) == {}
    assert config.decision_tier_map("bad") == {}
    tiers = config.decision_tier_map(
        {"tier1": "a, b", "tier2": ["c", "d"], "tier3": ("e",)}
    )
    assert tiers["a"] == 1
    assert tiers["b"] == 1
    assert tiers["c"] == 2
    assert tiers["d"] == 2
    assert tiers["e"] == 3


# gabion:evidence E:decision_surface/direct::config.py::gabion.config.decision_ignore_list::section E:decision_surface/direct::config.py::gabion.config.decision_require_tiers::section E:decision_surface/direct::config.py::gabion.config.exception_never_list::section E:decision_surface/direct::config.py::gabion.config._as_bool::value E:decision_surface/direct::config.py::gabion.config._normalize_name_list::value
def test_config_helpers_cover_bool_and_lists() -> None:
    assert config._normalize_name_list(["a, b", "c"]) == ["a", "b", "c"]
    assert config._as_bool(True) is True
    assert config._as_bool(0) is False
    assert config._as_bool(2) is True
    assert config._as_bool("yes") is True
    assert config._as_bool("nope") is False

    assert config.decision_require_tiers(None) is False
    assert config.decision_require_tiers("bad") is False
    assert config.decision_require_tiers({"require_tiers": "on"}) is True

    assert config.decision_ignore_list(None) == []
    assert config.decision_ignore_list("bad") == []
    assert config.decision_ignore_list({"ignore_params": ["a", "b"]}) == ["a", "b"]

    assert config.exception_never_list(None) == []
    assert config.exception_never_list("bad") == []
    assert config.exception_never_list({"never": "A, B"}) == ["A", "B"]
