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


def test_load_config_default_path(tmp_path: Path) -> None:
    cfg = tmp_path / config.DEFAULT_CONFIG_NAME
    cfg.write_text("[dataflow]\nstrictness = 'low'\n", encoding="utf-8")
    data = config.load_config(root=tmp_path, config_path=None)
    assert data["dataflow"]["strictness"] == "low"


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
