#!/usr/bin/env python3
# gabion:decision_protocol_module
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
            if not path.is_file() or any(part == "__pycache__" for part in path.parts):
                continue
            rel_path = path.relative_to(root).as_posix()
            source = _read_source(path)
            if source is None:
                continue
            lines = source.splitlines()
            if not _module_has_boundary_marker(lines):
                continue
            tree = _parse_tree(source)
            if tree is None:
                continue

            annotation_events = _annotation_events(lines)
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                violations.extend(
                    _function_violations(
                        rel_path=rel_path,
                        node=node,
                        annotations=annotation_events,
                        run_context=run_context,
                    )
                )
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
        if first_core_line is not None and ann.line >= first_core_line:
            continue
        if ann.line < int(node.lineno or 1) or ann.line > int(getattr(node, "end_lineno", node.lineno) or node.lineno):
            continue
        events.append(ann)

    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        line = int(getattr(child, "lineno", 1) or 1)
        if first_core_line is not None and line >= first_core_line:
            continue
        event = _syntax_event_from_call(child)
        if event is None:
            continue
        events.append(event)

    if not events:
        return []

    flow_identity = _derive_flow_identity(
        run_context=run_context,
        rel_path=rel_path,
        qualname=node.name,
    )

    seen: set[tuple[str, str]] = set()
    violations: list[Violation] = []
    for event in sorted(events, key=lambda item: (item.line, item.column, item.kind)):
        key = (event.input_slot, event.normalization_class)
        if key in seen:
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
                )
            )
            continue
        seen.add(key)
    return violations


def _first_core_call_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    best: int | None = None
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        dotted = _dotted_name(func)
        if dotted is None:
            continue
        if dotted.endswith("_core") or "_core." in dotted:
            line = int(getattr(child, "lineno", 1) or 1)
            if best is None or line < best:
                best = line
    return best


def _syntax_event_from_call(node: ast.Call) -> _NormalizationEvent | None:
    line = int(getattr(node, "lineno", 1) or 1)
    col = int(getattr(node, "col_offset", 0) or 0) + 1
    dotted = _dotted_name(node.func)
    if dotted is None:
        return None

    if dotted == "isinstance" and len(node.args) >= 1:
        slot = _expr_slot(node.args[0])
        if slot:
            return _NormalizationEvent(
                line=line,
                column=col,
                normalization_class="narrow",
                input_slot=slot,
                kind="syntax:isinstance",
            )

    if dotted.endswith("cast") and len(node.args) >= 2:
        slot = _expr_slot(node.args[1])
        if slot:
            return _NormalizationEvent(
                line=line,
                column=col,
                normalization_class="narrow",
                input_slot=slot,
                kind="syntax:cast",
            )

    if dotted == "json.loads" and len(node.args) >= 1:
        slot = _expr_slot(node.args[0])
        if slot:
            return _NormalizationEvent(
                line=line,
                column=col,
                normalization_class="parse",
                input_slot=slot,
                kind="syntax:json_loads",
            )

    if dotted.endswith("model_validate") and len(node.args) >= 1:
        slot = _expr_slot(node.args[0])
        if slot:
            return _NormalizationEvent(
                line=line,
                column=col,
                normalization_class="validate",
                input_slot=slot,
                kind="syntax:model_validate",
            )

    if dotted.endswith("parse_args"):
        return _NormalizationEvent(
            line=line,
            column=col,
            normalization_class="parse",
            input_slot="argv",
            kind="syntax:parse_args",
        )

    return None


def _expr_slot(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _expr_slot(expr.value)
        if base:
            return f"{base}.{expr.attr}"
    if isinstance(expr, ast.Subscript):
        return _expr_slot(expr.value)
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
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
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
