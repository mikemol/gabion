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
