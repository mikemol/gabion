from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import Callable, Protocol


class SymbolTableLike(Protocol):
    imports: Mapping[tuple[str, str], str]
    internal_roots: Set[str]
    external_filter: bool
    star_imports: Mapping[str, Set[str]]
    module_exports: Mapping[str, Set[str]]
    module_export_map: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class SerializeSymbolTableForResumeDeps:
    sort_once_fn: Callable[..., object]


@dataclass(frozen=True)
class DeserializeSymbolTableForResumeDeps:
    symbol_table_ctor: Callable[..., object]
    sequence_or_none_fn: Callable[..., object]
    check_deadline_fn: Callable[[], None]
    str_set_from_sequence_fn: Callable[..., object]
    mapping_or_none_fn: Callable[..., object]
    mapping_or_empty_fn: Callable[..., object]


def serialize_symbol_table_for_resume(
    table: SymbolTableLike,
    *,
    deps: SerializeSymbolTableForResumeDeps,
):
    return {
        "imports": [
            [module, name, fqn]
            for (module, name), fqn in deps.sort_once_fn(
                table.imports.items(),
                source="_serialize_symbol_table_for_resume.imports",
            )
        ],
        "internal_roots": deps.sort_once_fn(
            table.internal_roots,
            source="_serialize_symbol_table_for_resume.internal_roots",
        ),
        "external_filter": bool(table.external_filter),
        "star_imports": {
            module: deps.sort_once_fn(
                names,
                source=f"_serialize_symbol_table_for_resume.star_imports.{module}",
            )
            for module, names in deps.sort_once_fn(
                table.star_imports.items(),
                source="_serialize_symbol_table_for_resume.star_imports",
            )
        },
        "module_exports": {
            module: deps.sort_once_fn(
                names,
                source=f"_serialize_symbol_table_for_resume.module_exports.{module}",
            )
            for module, names in deps.sort_once_fn(
                table.module_exports.items(),
                source="_serialize_symbol_table_for_resume.module_exports",
            )
        },
        "module_export_map": {
            module: {
                name: mapping[name]
                for name in deps.sort_once_fn(
                    mapping,
                    source=(
                        "_serialize_symbol_table_for_resume.module_export_map."
                        f"{module}"
                    ),
                )
            }
            for module, mapping in deps.sort_once_fn(
                table.module_export_map.items(),
                source="_serialize_symbol_table_for_resume.module_export_map",
            )
        },
    }


def deserialize_symbol_table_for_resume(
    payload: Mapping[str, object],
    *,
    deps: DeserializeSymbolTableForResumeDeps,
):
    table = deps.symbol_table_ctor(external_filter=bool(payload.get("external_filter", True)))
    raw_imports = deps.sequence_or_none_fn(payload.get("imports"))
    if raw_imports is not None:
        for entry in raw_imports:
            deps.check_deadline_fn()
            entry_sequence = deps.sequence_or_none_fn(entry)
            if entry_sequence is not None and len(entry_sequence) == 3:
                module, name, fqn = entry_sequence
                if type(module) is str and type(name) is str and type(fqn) is str:
                    table.imports[(module, name)] = fqn
    raw_internal_roots = deps.sequence_or_none_fn(payload.get("internal_roots"))
    if raw_internal_roots is not None:
        for entry in raw_internal_roots:
            deps.check_deadline_fn()
            if type(entry) is str:
                table.internal_roots.add(entry)
    raw_star_imports = deps.mapping_or_none_fn(payload.get("star_imports"))
    if raw_star_imports is not None:
        for module, raw_names in raw_star_imports.items():
            deps.check_deadline_fn()
            if type(module) is str:
                names = deps.str_set_from_sequence_fn(raw_names)
                table.star_imports[module] = names
    raw_module_exports = deps.mapping_or_none_fn(payload.get("module_exports"))
    if raw_module_exports is not None:
        for module, raw_names in raw_module_exports.items():
            deps.check_deadline_fn()
            if type(module) is str:
                names = deps.str_set_from_sequence_fn(raw_names)
                table.module_exports[module] = names
    raw_module_export_map = deps.mapping_or_none_fn(payload.get("module_export_map"))
    if raw_module_export_map is not None:
        for module, raw_mapping in raw_module_export_map.items():
            deps.check_deadline_fn()
            if type(module) is str:
                mapping: dict[str, str] = {}
                mapping_payload = deps.mapping_or_empty_fn(raw_mapping)
                for name, mapped in mapping_payload.items():
                    deps.check_deadline_fn()
                    if type(name) is str and type(mapped) is str:
                        mapping[name] = mapped
                table.module_export_map[module] = mapping
    return table
