#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.event_algebra import (
    CanonicalRunContext,
    GlobalEventSequencer,
    derive_identity_projection_from_tokens,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from gabion.tooling.policy_rules.fiber_diagnostics import (
    FiberApplicabilityBounds,
    FiberCounterfactualBoundary,
    FiberTraceEvent,
)
from gabion.tooling.runtime.deadline_runtime import DeadlineBudget, deadline_scope_from_ticks

TARGET_GLOB = "src/gabion/**/*.py"
BOUNDARY_MARKER = "gabion:boundary_normalization_module"
_TAINT_RE = re.compile(
    r"gabion:(taint_intro|taint_erase)\s+input=([^\s]+)\s+class=(parse|validate|narrow)"
)
_DEFAULT_POLICY_TIMEOUT_BUDGET = DeadlineBudget(ticks=120_000, tick_ns=1_000_000)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    normalization_class: str
    input_slot: str
    flow_identity: str
    structured_hash: str
    fiber_trace: tuple[FiberTraceEvent, ...] = ()
    applicability_bounds: FiberApplicabilityBounds | None = None
    counterfactual_boundary: FiberCounterfactualBoundary | None = None

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.kind}:{self.structured_hash}"

    @property
    def legacy_key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.line}:{self.kind}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class _NormalizationEvent:
    line: int
    column: int
    normalization_class: str
    input_slot: str
    kind: str
    phase_hint: str
    pre_core: bool


def collect_violations(*, root: Path, files: Sequence[Path] | None = None) -> list[Violation]:
    with deadline_scope_from_ticks(_DEFAULT_POLICY_TIMEOUT_BUDGET):
        candidates = files if files is not None else tuple(sorted(root.glob(TARGET_GLOB)))
        violations: list[Violation] = []
        run_context = CanonicalRunContext(
            run_id="policy:fiber_normalization_contract",
            sequencer=GlobalEventSequencer(),
            identity_space=GlobalIdentitySpace(
                allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
            ),
        )

        for path in candidates:
            path_is_scannable = path.is_file() and not any(
                part == "__pycache__" for part in path.parts
            )
            if path_is_scannable:
                rel_path = path.relative_to(root).as_posix()
                source = _read_source(path)
                if source is not None:
                    lines = source.splitlines()
                    if _module_has_boundary_marker(lines):
                        tree = _parse_tree(source)
                        if tree is not None:
                            annotation_events = _annotation_events(lines)
                            for node in tree.body:
                                match node:
                                    case ast.FunctionDef() | ast.AsyncFunctionDef():
                                        violations.extend(
                                            _function_violations(
                                                rel_path=rel_path,
                                                node=node,
                                                annotations=annotation_events,
                                                run_context=run_context,
                                            )
                                        )
                                    case _:
                                        pass
        return violations


def _read_source(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_tree(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _module_has_boundary_marker(lines: list[str]) -> bool:
    for raw in lines[:100]:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and BOUNDARY_MARKER in stripped:
            return True
    return False


def _annotation_events(lines: list[str]) -> tuple[_NormalizationEvent, ...]:
    events: list[_NormalizationEvent] = []
    for idx, raw in enumerate(lines, start=1):
        match = _TAINT_RE.search(raw)
        if not match:
            continue
        kind, input_slot, klass = match.groups()
        events.append(
            _NormalizationEvent(
                line=idx,
                column=1,
                normalization_class=klass,
                input_slot=input_slot,
                kind=kind,
                phase_hint="annotation",
                pre_core=False,
            )
        )
    return tuple(events)


def _function_violations(
    *,
    rel_path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    annotations: tuple[_NormalizationEvent, ...],
    run_context: CanonicalRunContext,
) -> list[Violation]:
    events: list[_NormalizationEvent] = []
    first_core_line = _first_core_call_line(node)

    for ann in annotations:
        ann_pre_core = first_core_line is None or ann.line < first_core_line
        ann_in_scope = int(node.lineno or 1) <= ann.line <= int(
            getattr(node, "end_lineno", node.lineno) or node.lineno
        )
        if ann_pre_core and ann_in_scope:
            events.append(
                _NormalizationEvent(
                    line=ann.line,
                    column=ann.column,
                    normalization_class=ann.normalization_class,
                    input_slot=ann.input_slot,
                    kind=ann.kind,
                    phase_hint=ann.phase_hint,
                    pre_core=True,
                )
            )

    for child in ast.walk(node):
        match child:
            case ast.Call():
                line = int(getattr(child, "lineno", 1) or 1)
                pre_core = first_core_line is None or line < first_core_line
                event = _syntax_event_from_call(child, pre_core=pre_core)
                if event is not None:
                    events.append(event)
            case _:
                pass

    if not events:
        return []

    flow_identity = _derive_flow_identity(
        run_context=run_context,
        rel_path=rel_path,
        qualname=node.name,
    )

    seen: set[tuple[str, str]] = set()
    first_seen_global_ordinal: dict[tuple[str, str], int] = {}
    violations: list[Violation] = []
    ordered_events = sorted(events, key=lambda item: (item.line, item.column, item.kind))
    indexed_events = list(enumerate(ordered_events, start=1))
    for global_ordinal, event in indexed_events:
        if not event.pre_core:
            continue
        key = (event.input_slot, event.normalization_class)
        if key in seen:
            first_global_ordinal = first_seen_global_ordinal[key]
            trace_rows = [
                item for item in indexed_events if item[1].input_slot == event.input_slot
            ]
            trace_by_ordinal = {
                row_ordinal: local_ordinal
                for local_ordinal, (row_ordinal, _row_event) in enumerate(trace_rows, start=1)
            }
            local_duplicate_ordinal = trace_by_ordinal[global_ordinal]
            local_first_ordinal = trace_by_ordinal[first_global_ordinal]
            local_boundary_before_ordinal = (
                sum(1 for _global, row in trace_rows if row.pre_core) + 1
            )
            boundary_domain_max_before_ordinal = len(trace_rows) + 1
            msg = (
                f"normalization class '{event.normalization_class}' was applied more than once "
                f"to input '{event.input_slot}' before core entry"
            )
            violations.append(
                Violation(
                    path=rel_path,
                    line=event.line,
                    column=event.column,
                    qualname=node.name,
                    kind="duplicate_normalization_before_core",
                    message=msg,
                    normalization_class=event.normalization_class,
                    input_slot=event.input_slot,
                    flow_identity=flow_identity,
                    structured_hash=_structured_hash(
                        rel_path,
                        node.name,
                        event.input_slot,
                        event.normalization_class,
                        flow_identity,
                        str(event.column),
                    ),
                    fiber_trace=tuple(
                        FiberTraceEvent(
                            ordinal=local_ordinal,
                            line=row_event.line,
                            column=row_event.column,
                            event_kind=row_event.kind,
                            normalization_class=row_event.normalization_class,
                            input_slot=row_event.input_slot,
                            phase_hint=row_event.phase_hint,
                            pre_core=row_event.pre_core,
                        )
                        for local_ordinal, (_row_ordinal, row_event) in enumerate(
                            trace_rows, start=1
                        )
                    ),
                    applicability_bounds=FiberApplicabilityBounds(
                        current_boundary_before_ordinal=local_boundary_before_ordinal,
                        violation_applies_when_boundary_before_ordinal_gt=local_duplicate_ordinal,
                        violation_clears_when_boundary_before_ordinal_lte=local_duplicate_ordinal,
                        boundary_domain_max_before_ordinal=boundary_domain_max_before_ordinal,
                        core_entry_before_ordinal=(
                            local_boundary_before_ordinal
                            if first_core_line is not None
                            else None
                        ),
                    ),
                    counterfactual_boundary=FiberCounterfactualBoundary(
                        suggested_boundary_before_ordinal=local_duplicate_ordinal,
                        boundary_event_kind=event.kind,
                        boundary_line=event.line,
                        boundary_column=event.column,
                        eliminates_violation_without_other_changes=True,
                        preserves_prior_normalization=(
                            local_first_ordinal < local_duplicate_ordinal
                        ),
                        rationale=(
                            "Move boundary to immediately before the duplicate normalization event."
                        ),
                    ),
                )
            )
        else:
            seen.add(key)
            first_seen_global_ordinal[key] = global_ordinal
    return violations


def _first_core_call_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    best: int | None = None
    for child in ast.walk(node):
        match child:
            case ast.Call():
                dotted = _dotted_name(child.func)
                if dotted is not None and (
                    dotted.endswith("_core") or "_core." in dotted
                ):
                    line = int(getattr(child, "lineno", 1) or 1)
                    if best is None or line < best:
                        best = line
            case _:
                pass
    return best


def _syntax_event_from_call(
    node: ast.Call,
    *,
    pre_core: bool,
) -> _NormalizationEvent | None:
    line = int(getattr(node, "lineno", 1) or 1)
    col = int(getattr(node, "col_offset", 0) or 0) + 1
    dotted = _dotted_name(node.func)
    event: _NormalizationEvent | None = None
    if dotted is not None:
        if dotted == "isinstance" and len(node.args) >= 1:
            slot = _expr_slot(node.args[0])
            if slot:
                event = _NormalizationEvent(
                    line=line,
                    column=col,
                    normalization_class="narrow",
                    input_slot=slot,
                    kind="syntax:isinstance",
                    phase_hint="syntax",
                    pre_core=pre_core,
                )
        elif dotted.endswith("cast") and len(node.args) >= 2:
            slot = _expr_slot(node.args[1])
            if slot:
                event = _NormalizationEvent(
                    line=line,
                    column=col,
                    normalization_class="narrow",
                    input_slot=slot,
                    kind="syntax:cast",
                    phase_hint="syntax",
                    pre_core=pre_core,
                )
        elif dotted == "json.loads" and len(node.args) >= 1:
            slot = _expr_slot(node.args[0])
            if slot:
                event = _NormalizationEvent(
                    line=line,
                    column=col,
                    normalization_class="parse",
                    input_slot=slot,
                    kind="syntax:json_loads",
                    phase_hint="syntax",
                    pre_core=pre_core,
                )
        elif dotted.endswith("model_validate") and len(node.args) >= 1:
            slot = _expr_slot(node.args[0])
            if slot:
                event = _NormalizationEvent(
                    line=line,
                    column=col,
                    normalization_class="validate",
                    input_slot=slot,
                    kind="syntax:model_validate",
                    phase_hint="syntax",
                    pre_core=pre_core,
                )
        elif dotted.endswith("parse_args"):
            event = _NormalizationEvent(
                line=line,
                column=col,
                normalization_class="parse",
                input_slot="argv",
                kind="syntax:parse_args",
                phase_hint="syntax",
                pre_core=pre_core,
            )
    return event


def _expr_slot(expr: ast.AST) -> str | None:
    match expr:
        case ast.Name(id=identifier):
            return identifier
        case ast.Attribute(value=value, attr=attr):
            base = _expr_slot(value)
            if base:
                return f"{base}.{attr}"
        case ast.Subscript(value=value):
            return _expr_slot(value)
        case _:
            pass
    return None


def _derive_flow_identity(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    qualname: str,
) -> str:
    projection = derive_identity_projection_from_tokens(
        run_context=run_context,
        tokens=(
            "fiber.normalization_flow",
            f"path:{rel_path}",
            f"qualname:{qualname}",
        ),
    )
    atoms = ".".join(str(atom) for atom in projection.basis_path.atoms)
    return f"{projection.basis_path.namespace}:{atoms}"


def _dotted_name(node: ast.AST) -> str | None:
    match node:
        case ast.Name(id=identifier):
            return identifier
        case ast.Attribute(value=value, attr=attr):
            parent = _dotted_name(value)
            if parent is not None:
                return f"{parent}.{attr}"
        case _:
            pass
    return None


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


__all__ = [
    "Violation",
    "collect_violations",
]
