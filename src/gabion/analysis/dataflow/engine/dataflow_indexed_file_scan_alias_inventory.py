# gabion:ambiguity_boundary_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_indexed_file_scan_alias_surface
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
from typing import Iterable, Iterator, Literal, Mapping, Sequence

from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_analysis import (
    ANALYSIS_ALIAS_GROUPS,
)
from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_compatibility import (
    COMPATIBILITY_ALIAS_GROUPS,
)
from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_decision import (
    DECISION_ALIAS_GROUPS,
)
from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_projection import (
    PROJECTION_ALIAS_GROUPS,
)
from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_runtime import (
    RUNTIME_ALIAS_GROUPS,
)
from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_contract import (
    AliasGroupSpec,
    AliasSurfaceMaterialization,
)


BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 hard-cut completed; retain monolith alias surface while external importers migrate",
    "scope": "dataflow_indexed_file_scan.alias_surface",
    "start": "2026-03-05",
    "expiry": "WS-5 compatibility-shim retirement",
    "rollback_condition": "no external consumers require monolith path aliases",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}


_LOCAL_SUPPORT_BINDINGS: tuple[tuple[str, object], ...] = (
    ("argparse", argparse),
    ("ast", ast),
    ("sys", sys),
    ("ExitStack", ExitStack),
    ("Path", Path),
    ("Iterable", Iterable),
    ("Iterator", Iterator),
    ("Literal", Literal),
    ("Mapping", Mapping),
    ("Sequence", Sequence),
)

_LOCAL_TYPE_ALIAS_NAMES: tuple[str, ...] = (
    "FunctionNode",
    "OptionalIgnoredParams",
    "ParamAnnotationMap",
    "ReturnAliasMap",
    "OptionalReturnAliasMap",
    "OptionalClassName",
    "Span4",
    "OptionalSpan4",
    "OptionalString",
    "OptionalFloat",
    "OptionalPath",
    "OptionalStringSet",
    "OptionalPrimeRegistry",
    "OptionalTypeConstructorRegistry",
    "OptionalSynthRegistry",
    "OptionalJsonObject",
    "OptionalForestSpec",
    "OptionalDeprecatedExtractionArtifacts",
    "OptionalAstCall",
    "NodeIdOrNone",
    "ParseCacheValue",
    "ReportProjectionPhase",
)

_OWNER_DOMAIN_ADAPTERS: tuple[tuple[str, tuple[AliasGroupSpec, ...]], ...] = (
    (
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_compatibility",
        COMPATIBILITY_ALIAS_GROUPS,
    ),
    (
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_decision",
        DECISION_ALIAS_GROUPS,
    ),
    (
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_runtime",
        RUNTIME_ALIAS_GROUPS,
    ),
    (
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_analysis",
        ANALYSIS_ALIAS_GROUPS,
    ),
    (
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_projection",
        PROJECTION_ALIAS_GROUPS,
    ),
)

ALIAS_GROUP_SPECS: tuple[AliasGroupSpec, ...] = tuple(
    group
    for _, group_set in _OWNER_DOMAIN_ADAPTERS
    for group in group_set
)


def _materialize_module_aliases() -> tuple[
    dict[str, object], tuple[dict[str, object], ...], Counter[str]
]:
    exports: dict[str, object] = {name: value for name, value in _LOCAL_SUPPORT_BINDINGS}
    group_inventory: list[dict[str, object]] = []
    module_alias_counts: Counter[str] = Counter()

    for group in ALIAS_GROUP_SPECS:
        group_export_names: list[str] = []
        module_inventory: list[dict[str, object]] = []
        for module_spec in group.module_specs:
            loaded = importlib.import_module(module_spec.module_path)
            module_binding_inventory: list[dict[str, str]] = []
            for binding in module_spec.bindings:
                exports[binding.export_name] = getattr(loaded, binding.source_name)
                module_alias_counts[module_spec.module_path] += 1
                group_export_names.append(binding.export_name)
                module_binding_inventory.append(
                    {
                        "source_name": binding.source_name,
                        "export_name": binding.export_name,
                    }
                )
            module_inventory.append(
                {
                    "module_path": module_spec.module_path,
                    "alias_count": len(module_spec.bindings),
                    "bindings": tuple(module_binding_inventory),
                }
            )
        group_inventory.append(
            {
                "group_id": group.group_id,
                "label": group.label,
                "alias_count": len(group_export_names),
                "export_names": tuple(group_export_names),
                "modules": tuple(module_inventory),
            }
        )

    return exports, tuple(group_inventory), module_alias_counts


def _add_local_type_aliases(surface: dict[str, object]) -> None:
    # Keep historically exposed boundary aliases available on the compatibility surface.
    surface["FunctionNode"] = ast.FunctionDef | ast.AsyncFunctionDef
    surface["OptionalIgnoredParams"] = set[str] | None
    surface["ParamAnnotationMap"] = dict[str, str | None]
    surface["ReturnAliasMap"] = dict[str, tuple[list[str], list[str]]]
    surface["OptionalReturnAliasMap"] = surface["ReturnAliasMap"] | None
    surface["OptionalClassName"] = str | None
    surface["Span4"] = tuple[int, int, int, int]
    surface["OptionalSpan4"] = surface["Span4"] | None
    surface["OptionalString"] = str | None
    surface["OptionalFloat"] = float | None
    surface["OptionalPath"] = Path | None
    surface["OptionalStringSet"] = set[str] | None
    surface["OptionalPrimeRegistry"] = surface["PrimeRegistry"] | None
    surface["OptionalTypeConstructorRegistry"] = (
        surface["TypeConstructorRegistry"] | None
    )
    surface["OptionalSynthRegistry"] = surface["SynthRegistry"] | None
    surface["OptionalJsonObject"] = surface["JSONObject"] | None
    surface["OptionalForestSpec"] = surface["ForestSpec"] | None
    surface["OptionalDeprecatedExtractionArtifacts"] = (
        surface["DeprecatedExtractionArtifacts"] | None
    )
    surface["OptionalAstCall"] = ast.Call | None
    surface["NodeIdOrNone"] = surface["NodeId"] | None
    surface["ParseCacheValue"] = ast.Module | BaseException
    surface["ReportProjectionPhase"] = Literal["collection", "forest", "edge", "post"]


def materialize_alias_boundary_surface() -> AliasSurfaceMaterialization:
    exports, group_inventory, module_alias_counts = _materialize_module_aliases()
    _add_local_type_aliases(exports)

    exported_names = tuple(exports)
    star_export_names = tuple(name for name in exported_names if not name.startswith("_"))
    remaining_hot_spots = tuple(
        {
            "module_path": module_path,
            "alias_count": alias_count,
        }
        for module_path, alias_count in module_alias_counts.most_common()
        if alias_count >= 10
    )
    owner_adapter_modules = tuple(module_path for module_path, _ in _OWNER_DOMAIN_ADAPTERS)

    inventory = {
        "surface_module": "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        "compatibility_scope": BOUNDARY_ADAPTER_LIFECYCLE["scope"],
        "local_support_names": tuple(name for name, _ in _LOCAL_SUPPORT_BINDINGS),
        "local_type_alias_names": _LOCAL_TYPE_ALIAS_NAMES,
        "owner_adapter_modules": owner_adapter_modules,
        "owner_adapter_count": len(_OWNER_DOMAIN_ADAPTERS),
        "group_count": len(group_inventory),
        "module_groups": group_inventory,
        "exported_names": exported_names,
        "star_export_names": star_export_names,
    }
    telemetry = {
        "surface_module": "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        "compatibility_scope": BOUNDARY_ADAPTER_LIFECYCLE["scope"],
        "exported_alias_count": len(exported_names),
        "star_export_count": len(star_export_names),
        "local_support_count": len(_LOCAL_SUPPORT_BINDINGS),
        "local_type_alias_count": len(_LOCAL_TYPE_ALIAS_NAMES),
        "owner_module_spread": len(module_alias_counts),
        "owner_adapter_count": len(_OWNER_DOMAIN_ADAPTERS),
        "owner_adapter_modules": owner_adapter_modules,
        "remaining_hot_spots": remaining_hot_spots,
        "compatibility_policy_surfaces": {
            "private_symbol_allowlist_path": "docs/policy/private_symbol_import_allowlist.txt",
            "debt_ledger_path": "docs/audits/dataflow_runtime_debt_ledger.md",
            "retirement_ledger_path": "docs/audits/dataflow_runtime_retirement_ledger.md",
            "decomposition_ledger_path": "docs/ws5_decomposition_ledger.md",
        },
    }
    return AliasSurfaceMaterialization(
        exports=exports,
        inventory=inventory,
        telemetry=telemetry,
    )
