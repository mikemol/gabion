from __future__ import annotations

from datetime import date, datetime, time
from functools import singledispatch
from pathlib import Path
from typing import TypeAlias
import tomllib
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never

DEFAULT_CONFIG_NAME = "gabion.toml"

TomlScalar: TypeAlias = str | int | float | bool | None | date | datetime | time
TomlValue: TypeAlias = TomlScalar | list["TomlValue"] | dict[str, "TomlValue"]
TomlTable: TypeAlias = dict[str, TomlValue]
_NONE_TYPE = type(None)


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
    return _toml_table_default_empty(data)


def load_config(root: Path | None = None, config_path: Path | None = None) -> TomlTable:
    if config_path is None:
        base = root if root is not None else Path.cwd()
        config_path = base / DEFAULT_CONFIG_NAME
    return _load_toml(config_path)


def dataflow_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    return _toml_table_default_empty(data.get("dataflow", {}))


def synthesis_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    return _toml_table_default_empty(data.get("synthesis", {}))


def decision_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    return _toml_table_default_empty(data.get("decision", {}))


def exception_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    return _toml_table_default_empty(data.get("exceptions", {}))


def fingerprint_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    return _toml_table_default_empty(data.get("fingerprints", {}))


def taint_defaults(
    root: Path | None = None, config_path: Path | None = None
) -> TomlTable:
    data = load_config(root=root, config_path=config_path)
    return _toml_table_default_empty(data.get("taint", {}))


def _split_name_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@singledispatch
def _normalize_name_item(value: object) -> list[str]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_normalize_name_item.register
def _sd_reg_1(value: str) -> list[str]:
    return _split_name_parts(value)


def _normalize_name_item_empty(value: object) -> list[str]:
    _ = value
    return []


for _runtime_type in (dict, list, tuple, set, int, float, bool, date, datetime, time, _NONE_TYPE):
    _normalize_name_item.register(_runtime_type)(_normalize_name_item_empty)


def _normalize_name_sequence(values: list[TomlValue] | tuple[TomlValue, ...] | set[TomlValue]) -> list[str]:
    check_deadline()
    items: list[str] = []
    for item in values:
        check_deadline()
        items.extend(_normalize_name_item(item))
    return [item for item in items if item]


@singledispatch
def _normalize_name_list(value: TomlValue) -> list[str]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_normalize_name_list.register(_NONE_TYPE)
def _sd_reg_2(value: None) -> list[str]:
    _ = value
    return []


@_normalize_name_list.register
def _sd_reg_3(value: str) -> list[str]:
    return _split_name_parts(value)


@_normalize_name_list.register(list)
def _sd_reg_4(value: list[TomlValue]) -> list[str]:
    return _normalize_name_sequence(value)


@_normalize_name_list.register(tuple)
def _sd_reg_5(value: tuple[TomlValue, ...]) -> list[str]:
    return _normalize_name_sequence(value)


@_normalize_name_list.register(set)
def _sd_reg_6(value: set[TomlValue]) -> list[str]:
    return _normalize_name_sequence(value)


def _normalize_name_list_empty(value: object) -> list[str]:
    _ = value
    return []


for _runtime_type in (dict, int, float, bool, date, datetime, time):
    _normalize_name_list.register(_runtime_type)(_normalize_name_list_empty)


@singledispatch
def _as_bool(value: TomlValue) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_as_bool.register
def _sd_reg_7(value: bool) -> bool:
    return value


@_as_bool.register
def _sd_reg_8(value: int) -> bool:
    return value != 0


@_as_bool.register
def _sd_reg_9(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_bool_false(value: object) -> bool:
    _ = value
    return False


for _runtime_type in (dict, list, tuple, set, float, date, datetime, time, _NONE_TYPE):
    _as_bool.register(_runtime_type)(_as_bool_false)


def _toml_table_default_empty(value: object) -> TomlTable:
    match value:
        case dict() as table:
            return {str(key): table[key] for key in table}
        case _:
            return {}


def decision_tier_map(section: TomlTable | None) -> dict[str, int]:
    check_deadline()
    section_table = _toml_table_default_empty(section)
    tiers: dict[str, int] = {}
    for tier, key in ((1, "tier1"), (2, "tier2"), (3, "tier3")):
        check_deadline()
        for name in _normalize_name_list(section_table.get(key)):
            check_deadline()
            tiers[name] = tier
    return tiers


def decision_require_tiers(section: TomlTable | None) -> bool:
    section_table = _toml_table_default_empty(section)
    return _as_bool(section_table.get("require_tiers"))


def decision_ignore_list(section: TomlTable | None) -> list[str]:
    section_table = _toml_table_default_empty(section)
    return _normalize_name_list(section_table.get("ignore_params"))


def exception_never_list(section: TomlTable | None) -> list[str]:
    section_table = _toml_table_default_empty(section)
    markers = exception_marker_families(section_table)
    if "never" in markers and markers["never"]:
        return markers["never"]
    return _normalize_name_list(section_table.get("never"))


def exception_marker_families(section: TomlTable | None) -> dict[str, list[str]]:
    section_table = _toml_table_default_empty(section)
    markers = _toml_table_default_empty(section_table.get("markers"))
    families: dict[str, list[str]] = {}
    for family, payload in markers.items():
        check_deadline()
        family_name = str(family).strip()
        if not family_name:
            continue
        families[family_name] = _normalize_name_list(payload)
    return families


def exception_marker_family(section: TomlTable | None, family: str) -> list[str]:
    if not family.strip():
        return []
    families = exception_marker_families(section)
    if family in families and families[family]:
        return families[family]
    if family == "never":
        return exception_never_list(section)
    return []


def taint_profile(section: TomlTable | None) -> str:
    section_table = _toml_table_default_empty(section)
    return str(section_table.get("profile", "observe") or "observe").strip().lower()


def taint_boundary_registry(section: TomlTable | None) -> list[dict[str, TomlValue]]:
    section_table = _toml_table_default_empty(section)
    boundaries = section_table.get("boundaries")
    normalized: list[dict[str, TomlValue]] = []
    match boundaries:
        case list() as boundary_rows:
            for row in boundary_rows:
                check_deadline()
                match row:
                    case dict() as row_table:
                        normalized.append({str(key): row_table[key] for key in row_table})
                    case _:
                        continue
        case _:
            return []
    return normalized


def dataflow_deadline_roots(section: TomlTable | None) -> list[str]:
    section_table = _toml_table_default_empty(section)
    return _normalize_name_list(section_table.get("deadline_roots"))


def dataflow_adapter_payload(section: TomlTable | None) -> TomlTable:
    section_table = _toml_table_default_empty(section)
    return _toml_table_default_empty(section_table.get("adapter"))


def dataflow_required_surfaces(section: TomlTable | None) -> list[str]:
    adapter = dataflow_adapter_payload(section)
    return _normalize_name_list(adapter.get("required_surfaces"))


def merge_payload(payload: TomlTable, defaults: TomlTable) -> TomlTable:
    check_deadline()
    merged = dict(defaults)
    for key, value in payload.items():
        check_deadline()
        if value is None:
            continue
        merged[key] = value
    return merged
