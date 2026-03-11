from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from gabion.analysis.foundation.json_types import JSONObject
from gabion.invariants import never

_PARSE_MODULE_ERROR_TYPES = (
    OSError,
    UnicodeError,
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)


class _ParseModuleStage(StrEnum):
    PARAM_ANNOTATIONS = "param_annotations"
    DEADLINE_FUNCTION_FACTS = "deadline_function_facts"
    CALL_NODES = "call_nodes"
    SUITE_CONTAINMENT = "suite_containment"
    SYMBOL_TABLE = "symbol_table"
    CLASS_INDEX = "class_index"
    FUNCTION_INDEX = "function_index"
    CONFIG_FIELDS = "config_fields"
    DATACLASS_REGISTRY = "dataclass_registry"
    DATACLASS_CALL_BUNDLES = "dataclass_call_bundles"
    RAW_SORTED_AUDIT = "raw_sorted_audit"


@dataclass(frozen=True)
class _ParseModuleSuccess:
    kind: Literal["parsed"]
    tree: ast.Module


@dataclass(frozen=True)
class _ParseModuleFailure:
    kind: Literal["parse_failure"]
    witness: JSONObject


ParseModuleOutcome = _ParseModuleSuccess | _ParseModuleFailure


def _parse_failure_witness(
    *, path: Path, stage: _ParseModuleStage, error: Exception
) -> JSONObject:
    return {
        "path": str(path),
        "stage": stage.value,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _parse_module_tree(
    path: Path,
    *,
    stage: _ParseModuleStage,
    parse_failure_witnesses: list[JSONObject],
) -> ParseModuleOutcome:
    try:
        return _ParseModuleSuccess(
            kind="parsed",
            tree=ast.parse(path.read_text(encoding="utf-8")),
        )
    except _PARSE_MODULE_ERROR_TYPES as exc:
        witness = _parse_failure_witness(path=path, stage=stage, error=exc)
        parse_failure_witnesses.append(witness)
        return _ParseModuleFailure(kind="parse_failure", witness=witness)


def _parse_module_tree_optional(
    path: Path,
    *,
    stage: _ParseModuleStage,
    parse_failure_witnesses: list[JSONObject],
):
    outcome = _parse_module_tree(
        path,
        stage=stage,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    match outcome:
        case _ParseModuleSuccess(tree=tree):
            return tree
        case _:
            return None


            never("unreachable wildcard match fall-through")
def _forbid_adhoc_bundle_discovery(reason: str) -> None:
    if os.environ.get("GABION_FORBID_ADHOC_BUNDLES") == "1":
        raise AssertionError(
            f"Ad-hoc bundle discovery invoked while forest-only invariant active: {reason}"
        )
