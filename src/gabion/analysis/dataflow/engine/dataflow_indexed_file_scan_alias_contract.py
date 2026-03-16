# gabion:ambiguity_boundary_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_indexed_file_scan_alias_contract
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AliasBindingSpec:
    source_name: str
    export_name: str


@dataclass(frozen=True)
class ModuleAliasSpec:
    module_path: str
    bindings: tuple[AliasBindingSpec, ...]


@dataclass(frozen=True)
class AliasGroupSpec:
    group_id: str
    label: str
    module_specs: tuple[ModuleAliasSpec, ...]


@dataclass(frozen=True)
class AliasSurfaceMaterialization:
    exports: dict[str, object]
    inventory: dict[str, object]
    telemetry: dict[str, object]


BindingLike = str | tuple[str, str]


def alias_binding(binding: BindingLike) -> AliasBindingSpec:
    if isinstance(binding, str):
        return AliasBindingSpec(source_name=binding, export_name=binding)
    source_name, export_name = binding
    return AliasBindingSpec(source_name=source_name, export_name=export_name)


def module_alias(module_path: str, *bindings: BindingLike) -> ModuleAliasSpec:
    return ModuleAliasSpec(
        module_path=module_path,
        bindings=tuple(alias_binding(binding) for binding in bindings),
    )


def alias_group(
    group_id: str,
    label: str,
    *module_specs: ModuleAliasSpec,
) -> AliasGroupSpec:
    return AliasGroupSpec(
        group_id=group_id,
        label=label,
        module_specs=tuple(module_specs),
    )
