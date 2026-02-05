from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from typing import TypeAlias
import tomllib

DEFAULT_CONFIG_NAME = "gabion.toml"

TomlScalar: TypeAlias = str | int | float | bool | None | date | datetime | time
TomlValue: TypeAlias = TomlScalar | list["TomlValue"] | dict[str, "TomlValue"]
TomlTable: TypeAlias = dict[str, TomlValue]


def _load_toml(path: Path) -> TomlTable:
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


def load_config(root: Path | None = None, config_path: Path | None = None) -> TomlTable:
    if config_path is None:
        base = root if root is not None else Path.cwd()
        config_path = base / DEFAULT_CONFIG_NAME
    return _load_toml(config_path)


def dataflow_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    section = data.get("dataflow", {})
    return section if isinstance(section, dict) else {}


def synthesis_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    section = data.get("synthesis", {})
    return section if isinstance(section, dict) else {}


def decision_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    section = data.get("decision", {})
    return section if isinstance(section, dict) else {}


def exception_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    section = data.get("exceptions", {})
    return section if isinstance(section, dict) else {}


def fingerprint_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    section = data.get("fingerprints", {})
    return section if isinstance(section, dict) else {}


def _normalize_name_list(value: TomlValue) -> list[str]:
    items: list[str] = []
    if value is None:
        return items
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
    return [item for item in items if item]


def _as_bool(value: TomlValue) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def decision_tier_map(section: TomlTable | None) -> dict[str, int]:
    if section is None:
        return {}
    if not isinstance(section, dict):
        return {}
    tiers: dict[str, int] = {}
    for tier, key in ((1, "tier1"), (2, "tier2"), (3, "tier3")):
        for name in _normalize_name_list(section.get(key)):
            tiers[name] = tier
    return tiers


def decision_require_tiers(section: TomlTable | None) -> bool:
    if section is None:
        return False
    if not isinstance(section, dict):
        return False
    return _as_bool(section.get("require_tiers"))


def decision_ignore_list(section: TomlTable | None) -> list[str]:
    if section is None:
        return []
    if not isinstance(section, dict):
        return []
    return _normalize_name_list(section.get("ignore_params"))


def exception_never_list(section: TomlTable | None) -> list[str]:
    if section is None:
        return []
    if not isinstance(section, dict):
        return []
    return _normalize_name_list(section.get("never"))


def merge_payload(payload: TomlTable, defaults: TomlTable) -> TomlTable:
    merged = dict(defaults)
    for key, value in payload.items():
        if value is None:
            continue
        merged[key] = value
    return merged
