from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

DEFAULT_CONFIG_NAME = "gabion.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    try:
        data = tomllib.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_config(root: Path | None = None, config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        base = root if root is not None else Path.cwd()
        config_path = base / DEFAULT_CONFIG_NAME
    return _load_toml(config_path)


def dataflow_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> dict[str, Any]:
    data = load_config(root=root, config_path=config_path)
    section = data.get("dataflow", {})
    return section if isinstance(section, dict) else {}


def merge_payload(payload: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in payload.items():
        if value is None:
            continue
        merged[key] = value
    return merged
