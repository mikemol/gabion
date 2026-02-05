#!/usr/bin/env python3
"""Infer forwarding-based parameter bundles and propagate them across calls.

This script performs a two-stage analysis:
  1) Local grouping: within a function, parameters used *only* as direct
     call arguments are grouped by identical forwarding signatures.
  2) Propagation: if a function f calls g, and g has local bundles, then
     f's parameters passed into g's bundled positions are linked as a
     candidate bundle. This is iterated to a fixed point.

The goal is to surface "dataflow grammar" candidates for config dataclasses.

It can also emit a DOT graph (see --dot) so downstream tooling can render
bundle candidates as a dependency graph.
"""
from __future__ import annotations

import argparse
import ast
import json
import hashlib
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator
import re

from gabion.analysis.visitors import ImportVisitor, ParentAnnotator, UseVisitor
from gabion.analysis.evidence import (
    Site,
    exception_obligation_summary_for_site,
    normalize_bundle_key,
)
from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.schema_audit import find_anonymous_schema_surfaces
from gabion.config import (
    dataflow_defaults,
    decision_defaults,
    decision_tier_map,
    exception_defaults,
    exception_never_list,
    fingerprint_defaults,
    merge_payload,
    synthesis_defaults,
)
from gabion.analysis.type_fingerprints import (
    Fingerprint,
    FingerprintDimension,
    PrimeRegistry,
    TypeConstructorRegistry,
    SynthRegistry,
    build_synth_registry,
    build_fingerprint_registry,
    build_synth_registry_from_payload,
    bundle_fingerprint_dimensional,
    format_fingerprint,
    fingerprint_carrier_soundness,
    fingerprint_to_type_keys_with_remainder,
    synth_registry_payload,
)
from gabion.schema import SynthesisResponse
from gabion.synthesis import NamingContext, SynthesisConfig, Synthesizer
from gabion.synthesis.merge import merge_bundles
from gabion.synthesis.schedule import topological_schedule

@dataclass
class ParamUse:
    direct_forward: set[tuple[str, str]]
    non_forward: bool
    current_aliases: set[str]
    forward_sites: dict[tuple[str, str], set[tuple[int, int, int, int]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class CallArgs:
    callee: str
    pos_map: dict[str, str]
    kw_map: dict[str, str]
    const_pos: dict[str, str]
    const_kw: dict[str, str]
    non_const_pos: set[str]
    non_const_kw: set[str]
    star_pos: list[tuple[int, str]]
    star_kw: list[str]
    is_test: bool
    span: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class InvariantProposition:
    form: str
    terms: tuple[str, ...]
    scope: str | None = None
    source: str | None = None

    def as_dict(self) -> JSONObject:
        payload: JSONObject = {
            "form": self.form,
            "terms": list(self.terms),
        }
        if self.scope is not None:
            payload["scope"] = self.scope
        if self.source is not None:
            payload["source"] = self.source
        return payload

@dataclass
class SymbolTable:
    imports: dict[tuple[str, str], str] = field(default_factory=dict)
    internal_roots: set[str] = field(default_factory=set)
    external_filter: bool = True
    star_imports: dict[str, set[str]] = field(default_factory=dict)
    module_exports: dict[str, set[str]] = field(default_factory=dict)
    module_export_map: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolve(self, current_module: str, name: str) -> str | None:
        if (current_module, name) in self.imports:
            fqn = self.imports[(current_module, name)]
            if self.external_filter:
                root = fqn.split(".")[0]
                if root not in self.internal_roots:
                    return None
            return fqn
        return f"{current_module}.{name}"

    def resolve_star(self, current_module: str, name: str) -> str | None:
        candidates = self.star_imports.get(current_module, set())
        if not candidates:
            return None
        for module in sorted(candidates):
            exports = self.module_exports.get(module)
            if exports is None or name not in exports:
                continue
            export_map = self.module_export_map.get(module, {})
            mapped = export_map.get(name)
            if mapped:
                if self.external_filter and mapped:
                    root = mapped.split(".")[0]
                    if root not in self.internal_roots:
                        continue
                return mapped
            if self.external_filter and module:
                root = module.split(".")[0]
                if root not in self.internal_roots:
                    continue
            if module:
                return f"{module}.{name}"
            return name
        return None


@dataclass
class AuditConfig:
    project_root: Path | None = None
    exclude_dirs: set[str] = field(default_factory=set)
    ignore_params: set[str] = field(default_factory=set)
    external_filter: bool = True
    strictness: str = "high"
    transparent_decorators: set[str] | None = None
    decision_tiers: dict[str, int] = field(default_factory=dict)
    never_exceptions: set[str] = field(default_factory=set)
    fingerprint_registry: PrimeRegistry | None = None
    fingerprint_index: dict[Fingerprint, set[str]] = field(default_factory=dict)
    constructor_registry: TypeConstructorRegistry | None = None
    fingerprint_synth_min_occurrences: int = 0
    fingerprint_synth_version: str = "synth@1"
    fingerprint_synth_registry: SynthRegistry | None = None
    invariant_emitters: tuple[
        Callable[[ast.FunctionDef], Iterable[InvariantProposition]],
        ...,
    ] = field(default_factory=tuple)

    def is_ignored_path(self, path: Path) -> bool:
        parts = set(path.parts)
        return bool(self.exclude_dirs & parts)


def _call_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> tuple[ast.Call | None, bool]:
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, ast.Call):
            if child in parent.args:
                return parent, True
            for kw in parent.keywords:
                if child is kw or child is kw.value:
                    return parent, True
            return parent, False
        child = parent
        parent = parents.get(child)
    return None, False


@dataclass
class AnalysisResult:
    groups_by_path: dict[Path, dict[str, list[set[str]]]]
    param_spans_by_path: dict[Path, dict[str, dict[str, tuple[int, int, int, int]]]]
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]]
    type_suggestions: list[str]
    type_ambiguities: list[str]
    type_callsite_evidence: list[str]
    constant_smells: list[str]
    unused_arg_smells: list[str]
    lint_lines: list[str] = field(default_factory=list)
    deadness_witnesses: list[JSONObject] = field(default_factory=list)
    coherence_witnesses: list[JSONObject] = field(default_factory=list)
    rewrite_plans: list[JSONObject] = field(default_factory=list)
    exception_obligations: list[JSONObject] = field(default_factory=list)
    handledness_witnesses: list[JSONObject] = field(default_factory=list)
    decision_surfaces: list[str] = field(default_factory=list)
    value_decision_surfaces: list[str] = field(default_factory=list)
    decision_warnings: list[str] = field(default_factory=list)
    fingerprint_warnings: list[str] = field(default_factory=list)
    fingerprint_matches: list[str] = field(default_factory=list)
    fingerprint_synth: list[str] = field(default_factory=list)
    fingerprint_synth_registry: JSONObject | None = None
    fingerprint_provenance: list[JSONObject] = field(default_factory=list)
    context_suggestions: list[str] = field(default_factory=list)
    invariant_propositions: list[InvariantProposition] = field(default_factory=list)
    value_decision_rewrites: list[str] = field(default_factory=list)


def _callee_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except Exception:
        return "<call>"


def _normalize_callee(name: str, class_name: str | None) -> str:
    if not class_name:
        return name
    if name.startswith("self.") or name.startswith("cls."):
        parts = name.split(".")
        if len(parts) == 2:
            return f"{class_name}.{parts[1]}"
    return name


def _iter_paths(paths: Iterable[str], config: AuditConfig) -> list[Path]:
    """Expand input paths to python files, pruning ignored directories early."""
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for root, dirnames, filenames in os.walk(path, topdown=True):
                if config.exclude_dirs:
                    # Prune excluded dirs before descending to avoid scanning
                    # large env/vendor trees like `.venv/`.
                    dirnames[:] = [d for d in dirnames if d not in config.exclude_dirs]
                dirnames.sort()
                for filename in sorted(filenames):
                    if not filename.endswith(".py"):
                        continue
                    candidate = Path(root) / filename
                    if config.is_ignored_path(candidate):
                        continue
                    out.append(candidate)
        else:
            if config.is_ignored_path(path):
                continue
            out.append(path)
    return sorted(out)


def _collect_functions(tree: ast.AST):
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return funcs


def _invariant_term(expr: ast.AST, params: set[str]) -> str | None:
    if isinstance(expr, ast.Name) and expr.id in params:
        return expr.id
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "len"
        and len(expr.args) == 1
    ):
        arg = expr.args[0]
        if isinstance(arg, ast.Name) and arg.id in params:
            return f"{arg.id}.length"
    return None


def _extract_invariant_from_expr(
    expr: ast.AST,
    params: set[str],
    *,
    scope: str,
    source: str = "assert",
) -> InvariantProposition | None:
    if not isinstance(expr, ast.Compare):
        return None
    if len(expr.ops) != 1 or len(expr.comparators) != 1:
        return None
    if not isinstance(expr.ops[0], ast.Eq):
        return None
    left = _invariant_term(expr.left, params)
    right = _invariant_term(expr.comparators[0], params)
    if left is None or right is None:
        return None
    return InvariantProposition(
        form="Equal",
        terms=(left, right),
        scope=scope,
        source=source,
    )


class _InvariantCollector(ast.NodeVisitor):
    # dataflow-bundle: params, scope
    def __init__(self, params: set[str], scope: str) -> None:
        self._params = params
        self._scope = scope
        self.propositions: list[InvariantProposition] = []
        self._seen: set[tuple[str, tuple[str, ...], str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Assert(self, node: ast.Assert) -> None:
        prop = _extract_invariant_from_expr(
            node.test,
            self._params,
            scope=self._scope,
        )
        if prop is not None:
            key = (prop.form, prop.terms, prop.scope or "")
            if key not in self._seen:
                self._seen.add(key)
                self.propositions.append(prop)
        self.generic_visit(node)


def _scope_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def _collect_invariant_propositions(
    path: Path,
    *,
    ignore_params: set[str],
    project_root: Path | None,
    emitters: Iterable[
        Callable[[ast.FunctionDef], Iterable[InvariantProposition]]
    ] = (),
) -> list[InvariantProposition]:
    tree = ast.parse(path.read_text())
    propositions: list[InvariantProposition] = []
    for fn in _collect_functions(tree):
        params = set(_param_names(fn, ignore_params))
        if not params:
            continue
        scope = f"{_scope_path(path, project_root)}:{fn.name}"
        collector = _InvariantCollector(params, scope)
        for stmt in fn.body:
            collector.visit(stmt)
        propositions.extend(collector.propositions)
        for emitter in emitters:
            emitted = emitter(fn)
            for prop in emitted:
                if not isinstance(prop, InvariantProposition):
                    raise TypeError(
                        "Invariant emitters must yield InvariantProposition instances."
                    )
                normalized = InvariantProposition(
                    form=prop.form,
                    terms=prop.terms,
                    scope=prop.scope or scope,
                    source=prop.source or "emitter",
                )
                propositions.append(normalized)
    return propositions


def _format_invariant_proposition(prop: InvariantProposition) -> str:
    if prop.form == "Equal" and len(prop.terms) == 2:
        rendered = f"{prop.terms[0]} == {prop.terms[1]}"
    else:
        rendered = f"{prop.form}({', '.join(prop.terms)})"
    prefix = f"{prop.scope}: " if prop.scope else ""
    suffix = f" [{prop.source}]" if prop.source else ""
    return f"{prefix}{rendered}{suffix}"


def _format_invariant_propositions(
    props: list[InvariantProposition],
) -> list[str]:
    return [
        _format_invariant_proposition(prop)
        for prop in props
    ]


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return None
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _decorator_matches(name: str, allowlist: set[str]) -> bool:
    if name in allowlist:
        return True
    if "." in name and name.split(".")[-1] in allowlist:
        return True
    return False


def _decorators_transparent(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    transparent_decorators: set[str] | None,
) -> bool:
    if not fn.decorator_list:
        return True
    if not transparent_decorators:
        return True
    for deco in fn.decorator_list:
        name = _decorator_name(deco)
        if not name:
            return False
        if not _decorator_matches(name, transparent_decorators):
            return False
    return True


def _collect_local_class_bases(
    tree: ast.AST, parents: dict[ast.AST, ast.AST]
) -> dict[str, list[str]]:
    class_bases: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        scopes = _enclosing_class_scopes(node, parents)
        qual_parts = list(scopes)
        qual_parts.append(node.name)
        qual = ".".join(qual_parts)
        bases: list[str] = []
        for base in node.bases:
            base_name = _base_identifier(base)
            if base_name:
                bases.append(base_name)
        class_bases[qual] = bases
    return class_bases


def _local_class_name(base: str, class_bases: dict[str, list[str]]) -> str | None:
    if base in class_bases:
        return base
    if "." in base:
        tail = base.split(".")[-1]
        if tail in class_bases:
            return tail
    return None


def _resolve_local_method_in_hierarchy(
    class_name: str,
    method: str,
    *,
    class_bases: dict[str, list[str]],
    local_functions: set[str],
    seen: set[str],
) -> str | None:
    if class_name in seen:
        return None
    seen.add(class_name)
    candidate = f"{class_name}.{method}"
    if candidate in local_functions:
        return candidate
    for base in class_bases.get(class_name, []):
        base_name = _local_class_name(base, class_bases)
        if base_name is None:
            continue
        resolved = _resolve_local_method_in_hierarchy(
            base_name,
            method,
            class_bases=class_bases,
            local_functions=local_functions,
            seen=seen,
        )
        if resolved is not None:
            return resolved
    return None


def _param_names(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ignore_params: set[str] | None = None,
) -> list[str]:
    args = (
        fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    )
    names = [a.arg for a in args]
    if fn.args.vararg:
        names.append(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.append(fn.args.kwarg.arg)
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    if ignore_params:
        names = [name for name in names if name not in ignore_params]
    return names


def _decision_root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _decision_surface_params(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ignore_params: set[str] | None = None,
) -> set[str]:
    params = set(_param_names(fn, ignore_params))
    if not params:
        return set()

    def _mark(expr: ast.AST, out: set[str]) -> None:
        for node in ast.walk(expr):
            if isinstance(node, ast.Name) and node.id in params:
                out.add(node.id)
                continue
            if isinstance(node, (ast.Attribute, ast.Subscript)):
                root = _decision_root_name(node)
                if root in params:
                    out.add(root)

    decision_params: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            _mark(node.test, decision_params)
        elif isinstance(node, ast.While):
            _mark(node.test, decision_params)
        elif isinstance(node, ast.Assert):
            _mark(node.test, decision_params)
        elif isinstance(node, ast.IfExp):
            _mark(node.test, decision_params)
        elif isinstance(node, ast.Match):
            _mark(node.subject, decision_params)
            for case in node.cases:
                if case.guard is not None:
                    _mark(case.guard, decision_params)
    return decision_params


def _mark_param_roots(expr: ast.AST, params: set[str], out: set[str]) -> None:
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in params:
            out.add(node.id)
            continue
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            root = _decision_root_name(node)
            if root in params:
                out.add(root)


def _collect_param_roots(expr: ast.AST, params: set[str]) -> set[str]:
    found: set[str] = set()
    _mark_param_roots(expr, params, found)
    return found


def _contains_boolish(expr: ast.AST) -> bool:
    for node in ast.walk(expr):
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True
    return False


def _value_encoded_decision_params(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ignore_params: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    params = set(_param_names(fn, ignore_params))
    if not params:
        return set(), set()
    flagged: set[str] = set()
    reasons: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"min", "max"}:
                reasons.add("min/max")
                _mark_param_roots(node, params, flagged)
            elif isinstance(func, ast.Attribute) and func.attr in {"min", "max"}:
                reasons.add("min/max")
                _mark_param_roots(node, params, flagged)
        elif isinstance(node, ast.BinOp):
            op = node.op
            left_bool = _contains_boolish(node.left)
            right_bool = _contains_boolish(node.right)
            if isinstance(
                op,
                (
                    ast.Mult,
                    ast.Add,
                    ast.Sub,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                ),
            ):
                if left_bool or right_bool:
                    reasons.add("boolean arithmetic")
                    if left_bool:
                        flagged |= _collect_param_roots(node.left, params)
                    if right_bool:
                        flagged |= _collect_param_roots(node.right, params)
                if isinstance(
                    op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
                ) and not (left_bool or right_bool):
                    left_roots = _collect_param_roots(node.left, params)
                    right_roots = _collect_param_roots(node.right, params)
                    if left_roots or right_roots:
                        reasons.add("bitmask")
                        flagged |= left_roots | right_roots
    return flagged, reasons


def analyze_decision_surfaces_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
    decision_tiers: dict[str, int] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    by_name, by_qual = _build_function_index(
        paths,
        project_root,
        ignore_params,
        strictness,
        transparent_decorators,
    )
    symbol_table = _build_symbol_table(
        paths, project_root, external_filter=external_filter
    )
    class_index = _collect_class_index(paths, project_root)
    callers_by_qual: dict[str, set[str]] = defaultdict(set)
    for infos in by_name.values():
        for info in infos:
            for call in info.calls:
                if call.is_test:
                    continue
                callee = _resolve_callee(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table,
                    project_root,
                    class_index,
                )
                if callee is None:
                    continue
                callers_by_qual[callee.qual].add(info.qual)
    transitive_callers = _collect_transitive_callers(callers_by_qual, by_qual)

    surfaces: list[str] = []
    warnings: list[str] = []
    lint_lines: list[str] = []
    tier_map = decision_tiers or {}
    for info in by_qual.values():
        if not info.decision_params:
            continue
        caller_count = len(transitive_callers.get(info.qual, set()))
        boundary = "boundary" if caller_count == 0 else f"internal callers: {caller_count}"
        params = sorted(info.decision_params)
        surfaces.append(
            f"{info.path.name}:{info.qual} decision surface params: "
            + ", ".join(params)
            + f" ({boundary})"
        )
        for param in params:
            tier = _decision_tier_for(
                info,
                param,
                tier_map=tier_map,
                project_root=project_root,
            )
            if caller_count == 0 and tier is None:
                lint = _decision_param_lint_line(
                    info,
                    param,
                    project_root=project_root,
                    code="GABION_DECISION_SURFACE",
                    message=f"decision surface param '{param}' ({boundary})",
                )
                if lint is not None:
                    lint_lines.append(lint)
            if not tier_map:
                continue
            if tier is None:
                message = f"decision param '{param}' missing decision tier metadata"
                warnings.append(f"{info.path.name}:{info.qual} {message}")
                lint = _decision_param_lint_line(
                    info,
                    param,
                    project_root=project_root,
                    code="GABION_DECISION_TIER",
                    message=message,
                )
                if lint is not None:
                    lint_lines.append(lint)
            elif tier in {2, 3} and caller_count > 0:
                message = (
                    f"tier-{tier} decision param '{param}' used below boundary ({boundary})"
                )
                warnings.append(f"{info.path.name}:{info.qual} {message}")
                lint = _decision_param_lint_line(
                    info,
                    param,
                    project_root=project_root,
                    code="GABION_DECISION_TIER",
                    message=message,
                )
                if lint is not None:
                    lint_lines.append(lint)
    return sorted(surfaces), sorted(set(warnings)), sorted(set(lint_lines))


def analyze_value_encoded_decisions_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
    decision_tiers: dict[str, int] | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    by_name, by_qual = _build_function_index(
        paths,
        project_root,
        ignore_params,
        strictness,
        transparent_decorators,
    )
    symbol_table = _build_symbol_table(
        paths, project_root, external_filter=external_filter
    )
    class_index = _collect_class_index(paths, project_root)
    callers_by_qual: dict[str, set[str]] = defaultdict(set)
    for infos in by_name.values():
        for info in infos:
            for call in info.calls:
                if call.is_test:
                    continue
                callee = _resolve_callee(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table,
                    project_root,
                    class_index,
                )
                if callee is None:
                    continue
                callers_by_qual[callee.qual].add(info.qual)
    transitive_callers = _collect_transitive_callers(callers_by_qual, by_qual)
    surfaces: list[str] = []
    warnings: list[str] = []
    rewrites: list[str] = []
    lint_lines: list[str] = []
    tier_map = decision_tiers or {}
    for info in by_qual.values():
        if not info.value_decision_params:
            continue
        reasons = ", ".join(sorted(info.value_decision_reasons)) or "heuristic"
        caller_count = len(transitive_callers.get(info.qual, set()))
        boundary = "boundary" if caller_count == 0 else f"internal callers: {caller_count}"
        params = sorted(info.value_decision_params)
        surfaces.append(
            f"{info.path.name}:{info.qual} value-encoded decision params: "
            + ", ".join(params)
            + f" ({reasons})"
        )
        rewrites.append(
            f"{info.path.name}:{info.qual} consider rebranching value-encoded decision params: "
            + ", ".join(params)
            + f" ({reasons})"
        )
        for param in params:
            tier = _decision_tier_for(
                info,
                param,
                tier_map=tier_map,
                project_root=project_root,
            )
            if tier is None:
                lint = _decision_param_lint_line(
                    info,
                    param,
                    project_root=project_root,
                    code="GABION_VALUE_DECISION_SURFACE",
                    message=f"value-encoded decision param '{param}' ({boundary}; {reasons})",
                )
                if lint is not None:
                    lint_lines.append(lint)
            if not tier_map:
                continue
            if tier is None:
                message = (
                    f"value-encoded decision param '{param}' missing decision tier metadata ({reasons})"
                )
                warnings.append(f"{info.path.name}:{info.qual} {message}")
                lint = _decision_param_lint_line(
                    info,
                    param,
                    project_root=project_root,
                    code="GABION_VALUE_DECISION_TIER",
                    message=message,
                )
                if lint is not None:
                    lint_lines.append(lint)
            elif tier in {2, 3} and caller_count > 0:
                message = (
                    f"tier-{tier} value-encoded decision param '{param}' used below boundary ({boundary}; {reasons})"
                )
                warnings.append(f"{info.path.name}:{info.qual} {message}")
                lint = _decision_param_lint_line(
                    info,
                    param,
                    project_root=project_root,
                    code="GABION_VALUE_DECISION_TIER",
                    message=message,
                )
                if lint is not None:
                    lint_lines.append(lint)
    return (
        sorted(surfaces),
        sorted(set(warnings)),
        sorted(set(rewrites)),
        sorted(set(lint_lines)),
    )


def _node_span(node: ast.AST) -> tuple[int, int, int, int] | None:
    if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
        return None
    start_line = max(getattr(node, "lineno", 1) - 1, 0)
    start_col = max(getattr(node, "col_offset", 0), 0)
    end_line = max(getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1, 0)
    end_col = getattr(node, "end_col_offset", start_col + 1)
    if end_line == start_line and end_col <= start_col:
        end_col = start_col + 1
    return (start_line, start_col, end_line, end_col)


def _param_spans(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ignore_params: set[str] | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    spans: dict[str, tuple[int, int, int, int]] = {}
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    names = [a.arg for a in args]
    if names and names[0] in {"self", "cls"}:
        args = args[1:]
        names = names[1:]
    for arg in args:
        if ignore_params and arg.arg in ignore_params:
            continue
        span = _node_span(arg)
        if span is not None:
            spans[arg.arg] = span
    if fn.args.vararg:
        name = fn.args.vararg.arg
        if not ignore_params or name not in ignore_params:
            span = _node_span(fn.args.vararg)
            if span is not None:
                spans[name] = span
    if fn.args.kwarg:
        name = fn.args.kwarg.arg
        if not ignore_params or name not in ignore_params:
            span = _node_span(fn.args.kwarg)
            if span is not None:
                spans[name] = span
    return spans


def _function_key(scope: Iterable[str], name: str) -> str:
    parts = list(scope)
    parts.append(name)
    return ".".join(parts)


def _enclosing_class(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> str | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.ClassDef):
            return current.name
        current = parents.get(current)
    return None


def _enclosing_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.ClassDef):
            scopes.append(current.name)
        elif isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(current.name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_class_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.ClassDef):
            scopes.append(current.name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_function_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(current.name)
        current = parents.get(current)
    return list(reversed(scopes))


def _param_annotations(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ignore_params: set[str] | None = None,
) -> dict[str, str | None]:
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    names = [a.arg for a in args]
    annots: dict[str, str | None] = {}
    for name, arg in zip(names, args):
        if arg.annotation is None:
            annots[name] = None
        else:
            try:
                annots[name] = ast.unparse(arg.annotation)
            except Exception:
                annots[name] = None
    if fn.args.vararg:
        vararg = fn.args.vararg
        if vararg.annotation is None:
            annots[vararg.arg] = None
        else:
            try:
                annots[vararg.arg] = ast.unparse(vararg.annotation)
            except Exception:  # pragma: no cover - defensive against malformed AST nodes
                annots[vararg.arg] = None  # pragma: no cover
    if fn.args.kwarg:
        kwarg = fn.args.kwarg
        if kwarg.annotation is None:
            annots[kwarg.arg] = None
        else:
            try:
                annots[kwarg.arg] = ast.unparse(kwarg.annotation)
            except Exception:  # pragma: no cover - defensive against malformed AST nodes
                annots[kwarg.arg] = None  # pragma: no cover
    if names and names[0] in {"self", "cls"}:
        annots.pop(names[0], None)
    if ignore_params:
        for name in list(annots.keys()):
            if name in ignore_params:
                annots.pop(name, None)
    return annots


def _param_defaults(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ignore_params: set[str] | None = None,
) -> set[str]:
    defaults: set[str] = set()
    args = fn.args.posonlyargs + fn.args.args
    names = [a.arg for a in args]
    if fn.args.defaults:
        defaulted = names[-len(fn.args.defaults) :]
        defaults.update(defaulted)
    for kw_arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if default is not None:
            defaults.add(kw_arg.arg)
    if names and names[0] in {"self", "cls"}:
        defaults.discard(names[0])
    if ignore_params:
        defaults = {name for name in defaults if name not in ignore_params}
    return defaults


def _param_annotations_by_path(
    paths: list[Path],
    *,
    ignore_params: set[str],
) -> dict[Path, dict[str, dict[str, str | None]]]:
    annotations: dict[Path, dict[str, dict[str, str | None]]] = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        parent = ParentAnnotator()
        parent.visit(tree)
        parents = parent.parents
        by_fn: dict[str, dict[str, str | None]] = {}
        for fn in _collect_functions(tree):
            scopes = _enclosing_scopes(fn, parents)
            fn_key = _function_key(scopes, fn.name)
            by_fn[fn_key] = _param_annotations(fn, ignore_params)
        annotations[path] = by_fn
    return annotations


def _compute_fingerprint_warnings(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, str | None]]],
    *,
    registry: PrimeRegistry,
    index: dict[Fingerprint, set[str]],
    ctor_registry: TypeConstructorRegistry | None = None,
) -> list[str]:
    warnings: list[str] = []
    if not index:
        return warnings
    for path, groups in groups_by_path.items():
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                missing = [param for param in bundle if not fn_annots.get(param)]
                if missing:
                    warnings.append(
                        f"{path.name}:{fn_name} bundle {sorted(bundle)} missing type annotations: "
                        + ", ".join(sorted(missing))
                    )
                    continue
                types = [fn_annots[param] for param in sorted(bundle)]
                if any(t is None for t in types):
                    continue
                hint_list = [t for t in types if t is not None]
                fingerprint = bundle_fingerprint_dimensional(
                    hint_list,
                    registry,
                    ctor_registry,
                )
                soundness_issues = _fingerprint_soundness_issues(fingerprint)
                names = index.get(fingerprint)
                if not soundness_issues and names:
                    continue

                base_keys, base_remaining = fingerprint_to_type_keys_with_remainder(
                    fingerprint.base.product, registry
                )
                ctor_keys, ctor_remaining = fingerprint_to_type_keys_with_remainder(
                    fingerprint.ctor.product, registry
                )
                ctor_keys = [
                    key[len("ctor:") :] if key.startswith("ctor:") else key
                    for key in ctor_keys
                ]
                details = f" base={sorted(base_keys)}"
                if ctor_keys:
                    details += f" ctor={sorted(ctor_keys)}"
                if base_remaining not in (0, 1) or ctor_remaining not in (0, 1):
                    details += f" remainder=({base_remaining},{ctor_remaining})"
                if soundness_issues:
                    warnings.append(
                        f"{path.name}:{fn_name} bundle {sorted(bundle)} fingerprint carrier soundness failed for "
                        + ", ".join(soundness_issues)
                        + details
                    )
                if not names:
                    warnings.append(
                        f"{path.name}:{fn_name} bundle {sorted(bundle)} fingerprint missing glossary match{details}"
                    )
    return sorted(set(warnings))


def _compute_fingerprint_matches(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, str | None]]],
    *,
    registry: PrimeRegistry,
    index: dict[Fingerprint, set[str]],
    ctor_registry: TypeConstructorRegistry | None = None,
) -> list[str]:
    matches: list[str] = []
    if not index:
        return matches
    for path, groups in groups_by_path.items():
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                missing = [param for param in bundle if param not in fn_annots]
                if missing:
                    continue
                types = [fn_annots[param] for param in sorted(bundle)]
                if any(t is None for t in types):
                    continue
                hint_list = [t for t in types if t is not None]
                fingerprint = bundle_fingerprint_dimensional(
                    hint_list,
                    registry,
                    ctor_registry,
                )
                names = index.get(fingerprint)
                if not names:
                    continue
                base_keys, base_remaining = fingerprint_to_type_keys_with_remainder(
                    fingerprint.base.product, registry
                )
                ctor_keys, ctor_remaining = fingerprint_to_type_keys_with_remainder(
                    fingerprint.ctor.product, registry
                )
                ctor_keys = [
                    key[len("ctor:") :] if key.startswith("ctor:") else key
                    for key in ctor_keys
                ]
                details = f" base={sorted(base_keys)}"
                if ctor_keys:
                    details += f" ctor={sorted(ctor_keys)}"
                if base_remaining not in (0, 1) or ctor_remaining not in (0, 1):
                    details += f" remainder=({base_remaining},{ctor_remaining})"
                matches.append(
                    f"{path.name}:{fn_name} bundle {sorted(bundle)} fingerprint {format_fingerprint(fingerprint)} matches: "
                    + ", ".join(sorted(names))
                    + details
                )
    return sorted(set(matches))


def _fingerprint_soundness_issues(
    fingerprint: Fingerprint,
) -> list[str]:
    def _is_empty(dim: FingerprintDimension) -> bool:
        return dim.product in (0, 1) and dim.mask == 0

    pairs = [
        ("base/ctor", fingerprint.base, fingerprint.ctor),
        ("base/provenance", fingerprint.base, fingerprint.provenance),
        ("base/synth", fingerprint.base, fingerprint.synth),
        ("ctor/provenance", fingerprint.ctor, fingerprint.provenance),
        ("ctor/synth", fingerprint.ctor, fingerprint.synth),
        ("provenance/synth", fingerprint.provenance, fingerprint.synth),
    ]
    issues: list[str] = []
    for label, left, right in pairs:
        if _is_empty(left) or _is_empty(right):
            continue
        if not fingerprint_carrier_soundness(left, right):
            issues.append(label)
    return issues


def _compute_fingerprint_provenance(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, str | None]]],
    *,
    registry: PrimeRegistry,
    project_root: Path | None = None,
    index: dict[Fingerprint, set[str]] | None = None,
    ctor_registry: TypeConstructorRegistry | None = None,
) -> list[JSONObject]:
    entries: list[JSONObject] = []
    for path, groups in groups_by_path.items():
        path_value = _normalize_snapshot_path(path, project_root)
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                missing = [param for param in bundle if param not in fn_annots]
                if missing:
                    continue
                types = [fn_annots[param] for param in sorted(bundle)]
                if any(t is None for t in types):
                    continue
                hint_list = [t for t in types if t is not None]
                fingerprint = bundle_fingerprint_dimensional(
                    hint_list,
                    registry,
                    ctor_registry,
                )
                soundness_issues = _fingerprint_soundness_issues(fingerprint)
                base_keys, base_remaining = fingerprint_to_type_keys_with_remainder(
                    fingerprint.base.product, registry
                )
                ctor_keys, ctor_remaining = fingerprint_to_type_keys_with_remainder(
                    fingerprint.ctor.product, registry
                )
                ctor_keys = [
                    key[len("ctor:") :] if key.startswith("ctor:") else key
                    for key in ctor_keys
                ]
                matches = []
                if index:
                    matches = sorted(index.get(fingerprint, set()))
                bundle_key = ",".join(sorted(bundle))
                entries.append(
                    {
                        "provenance_id": f"{path_value}:{fn_name}:{bundle_key}",
                        "path": path_value,
                        "function": fn_name,
                        "bundle": sorted(bundle),
                        "fingerprint": {
                            "base": {
                                "product": fingerprint.base.product,
                                "mask": fingerprint.base.mask,
                            },
                            "ctor": {
                                "product": fingerprint.ctor.product,
                                "mask": fingerprint.ctor.mask,
                            },
                            "provenance": {
                                "product": fingerprint.provenance.product,
                                "mask": fingerprint.provenance.mask,
                            },
                            "synth": {
                                "product": fingerprint.synth.product,
                                "mask": fingerprint.synth.mask,
                            },
                        },
                        "base_keys": sorted(base_keys),
                        "ctor_keys": sorted(ctor_keys),
                        "remainder": {
                            "base": base_remaining,
                            "ctor": ctor_remaining,
                        },
                        "soundness_issues": soundness_issues,
                        "glossary_matches": matches,
                    }
                )
    return entries


def _summarize_fingerprint_provenance(
    entries: list[JSONObject],
    *,
    max_groups: int = 20,
    max_examples: int = 3,
) -> list[str]:
    if not entries:
        return []
    grouped: dict[tuple[object, ...], list[JSONObject]] = {}
    for entry in entries:
        matches = entry.get("glossary_matches") or []
        if isinstance(matches, list) and matches:
            key = ("glossary", tuple(matches))
        else:
            base_keys = tuple(entry.get("base_keys") or [])
            ctor_keys = tuple(entry.get("ctor_keys") or [])
            key = ("types", base_keys, ctor_keys)
        grouped.setdefault(key, []).append(entry)
    lines: list[str] = []
    for key, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))[
        :max_groups
    ]:
        label = ""
        if key and key[0] == "glossary":
            label = "glossary=" + ", ".join(key[1])
        elif key and key[0] == "types":
            base_keys = list(key[1])
            ctor_keys = list(key[2])
            label = f"base={base_keys}"
            if ctor_keys:
                label += f" ctor={ctor_keys}"
        lines.append(f"- {label} occurrences={len(group)}")
        for entry in group[:max_examples]:
            path = entry.get("path")
            fn_name = entry.get("function")
            bundle = entry.get("bundle")
            lines.append(f"  - {path}:{fn_name} bundle={bundle}")
        if len(group) > max_examples:
            lines.append(f"  - ... ({len(group) - max_examples} more)")
    return lines


def _summarize_deadness_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        path = entry.get("path", "?")
        function = entry.get("function", "?")
        bundle = entry.get("bundle", [])
        predicate = entry.get("predicate", "")
        environment = entry.get("environment", {})
        result = entry.get("result", "UNKNOWN")
        core = entry.get("core", [])
        core_count = len(core) if isinstance(core, list) else 0
        lines.append(
            f"{path}:{function} bundle {bundle} result={result} "
            f"predicate={predicate} env={environment} core={core_count}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def _compute_fingerprint_coherence(
    entries: list[JSONObject],
    *,
    synth_version: str,
) -> list[JSONObject]:
    witnesses: list[JSONObject] = []
    for entry in entries:
        matches = entry.get("glossary_matches") or []
        if not isinstance(matches, list) or len(matches) < 2:
            continue
        path = entry.get("path")
        function = entry.get("function")
        bundle = entry.get("bundle")
        provenance_id = entry.get("provenance_id")
        base_keys = entry.get("base_keys") or []
        ctor_keys = entry.get("ctor_keys") or []
        bundle_key = ",".join(bundle or [])
        witnesses.append(
            {
                "coherence_id": f"{path}:{function}:{bundle_key}:glossary-ambiguity",
                "site": {
                    "path": path,
                    "function": function,
                    "bundle": bundle,
                },
                "boundary": {
                    "base_keys": base_keys,
                    "ctor_keys": ctor_keys,
                    "synth_version": synth_version,
                },
                "alternatives": sorted(set(str(m) for m in matches)),
                "fork_signature": "glossary-ambiguity",
                "frack_path": ["provenance", "glossary"],
                "result": "UNKNOWN",
                "remainder": {"glossary_matches": matches},
                "provenance_id": provenance_id,
            }
        )
    return sorted(
        witnesses,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("fork_signature", "")),
        ),
    )


def _summarize_coherence_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        site = entry.get("site", {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        result = entry.get("result", "UNKNOWN")
        fork_signature = entry.get("fork_signature", "")
        alternatives = entry.get("alternatives", [])
        lines.append(
            f"{path}:{function} bundle {bundle} result={result} "
            f"fork={fork_signature} alternatives={alternatives}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def _compute_fingerprint_rewrite_plans(
    provenance: list[JSONObject],
    coherence: list[JSONObject],
    *,
    synth_version: str,
    exception_obligations: list[JSONObject] | None = None,
) -> list[JSONObject]:
    coherence_map: dict[tuple[str, str, str], JSONObject] = {}
    for entry in coherence:
        raw_site = entry.get("site", {}) or {}
        site = Site.from_payload(raw_site)
        if site is None:
            continue
        coherence_map[site.key()] = entry

    include_exception_predicates = exception_obligations is not None
    exception_summary_map: dict[tuple[str, str, str], dict[str, int]] = {}
    if exception_obligations is not None:
        for entry in exception_obligations:
            raw_site = entry.get("site", {}) or {}
            site = Site.from_payload(raw_site)
            if site is None:
                continue
            if not site.path or not site.function:
                continue
            summary = exception_summary_map.setdefault(
                site.key(),
                {"UNKNOWN": 0, "DEAD": 0, "HANDLED": 0, "total": 0},
            )
            status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
            if status not in {"UNKNOWN", "DEAD", "HANDLED"}:
                status = "UNKNOWN"
            summary[status] += 1
            summary["total"] += 1

    plans: list[JSONObject] = []
    for entry in provenance:
        matches = entry.get("glossary_matches") or []
        if not isinstance(matches, list) or len(matches) < 2:
            continue
        site = Site.from_payload(entry)
        if site is None or not site.path or not site.function:
            continue
        bundle_key = site.bundle_key()
        coherence_entry = coherence_map.get(site.key())
        coherence_id = None
        if coherence_entry:
            coherence_id = coherence_entry.get("coherence_id")
        plan_id = f"rewrite:{site.path}:{site.function}:{bundle_key}:glossary-ambiguity"
        candidates = sorted(set(str(m) for m in matches))
        pre_exception_summary: dict[str, int] | None = None
        if include_exception_predicates:
            pre_exception_summary = exception_summary_map.get(
                site.key(),
                {"UNKNOWN": 0, "DEAD": 0, "HANDLED": 0, "total": 0},
            )
        plans.append(
            {
                "plan_id": plan_id,
                "status": "UNVERIFIED",
                "site": {
                    "path": site.path,
                    "function": site.function,
                    "bundle": list(site.bundle),
                },
                "pre": {
                    "base_keys": entry.get("base_keys") or [],
                    "ctor_keys": entry.get("ctor_keys") or [],
                    "glossary_matches": matches,
                    "remainder": entry.get("remainder") or {},
                    "synth_version": synth_version,
                    **(
                        {"exception_obligations_summary": pre_exception_summary}
                        if pre_exception_summary is not None
                        else {}
                    ),
                },
                "rewrite": {
                    "kind": "BUNDLE_ALIGN",
                    "selector": {"bundle": list(site.bundle)},
                    "parameters": {"candidates": candidates},
                },
                "evidence": {
                    "provenance_id": entry.get("provenance_id"),
                    "coherence_id": coherence_id,
                },
                "post_expectation": {
                    "match_strata": "exact",
                    "base_conservation": True,
                    "ctor_coherence": True,
                },
                "verification": {
                    "mode": "re-audit",
                    "status": "UNVERIFIED",
                    # Minimal executable predicate set (see in/in-26.md §6).
                    # The evaluator (`verify_rewrite_plan`) intentionally treats transport
                    # details as erased; only the semantic payloads matter.
                    "predicates": [
                        {
                            "kind": "base_conservation",
                            "expect": True,
                        },
                        {
                            "kind": "ctor_coherence",
                            "expect": True,
                        },
                        {
                            "kind": "match_strata",
                            "expect": "exact",
                            "candidates": candidates,
                        },
                        {
                            "kind": "remainder_non_regression",
                            "expect": "no-new-remainder",
                        },
                        *(
                            [
                                {
                                    "kind": "exception_obligation_non_regression",
                                    "expect": "XV1",
                                }
                            ]
                            if include_exception_predicates
                            else []
                        ),
                    ],
                },
            }
        )
    return sorted(
        plans,
        key=lambda plan: (
            str(plan.get("site", {}).get("path", "")),
            str(plan.get("site", {}).get("function", "")),
            ",".join(plan.get("site", {}).get("bundle", []) or []),
            str(plan.get("plan_id", "")),
        ),
    )


def _glossary_match_strata(matches: object) -> str:
    if not isinstance(matches, list) or not matches:
        return "none"
    if len(matches) == 1:
        return "exact"
    return "ambiguous"


def _find_provenance_entry_for_site(
    provenance: list[JSONObject],
    *,
    site: Site,
) -> JSONObject | None:
    target_key = site.key()
    for entry in provenance:
        entry_site = Site.from_payload(entry)
        if entry_site is None:
            continue
        if entry_site.key() == target_key:
            return entry
    return None


def _exception_obligation_summary_for_site(
    obligations: list[JSONObject],
    *,
    site: Site,
) -> dict[str, int]:
    return exception_obligation_summary_for_site(obligations, site=site)


def verify_rewrite_plan(
    plan: JSONObject,
    *,
    post_provenance: list[JSONObject],
    post_exception_obligations: list[JSONObject] | None = None,
) -> JSONObject:
    """Verify a single rewrite plan using a post-state provenance artifact.

    The pre-state is taken from the plan's embedded boundary evidence; the
    evaluator only needs the post provenance entry for the plan's site.
    """
    plan_id = str(plan.get("plan_id", ""))
    raw_site = plan.get("site", {}) or {}
    site = Site.from_payload(raw_site)
    if site is None or not site.path or not site.function:
        return {
            "plan_id": plan_id,
            "accepted": False,
            "issues": ["missing or invalid plan site"],
            "predicate_results": [],
        }
    path = site.path
    function = site.function
    bundle = list(site.bundle)

    issues: list[str] = []
    post_entry = _find_provenance_entry_for_site(
        post_provenance,
        site=site,
    )
    if post_entry is None:
        issues.append("missing post provenance entry for site")
        return {
            "plan_id": plan_id,
            "accepted": False,
            "issues": issues,
            "predicate_results": [],
        }

    pre = plan.get("pre") or {}
    if not isinstance(pre, dict):
        pre = {}
    expected_base = list(pre.get("base_keys") or [])
    expected_ctor = list(pre.get("ctor_keys") or [])
    expected_remainder = pre.get("remainder") or {}
    if not isinstance(expected_remainder, dict):
        expected_remainder = {}
    post_expectation = plan.get("post_expectation") or {}
    if not isinstance(post_expectation, dict):
        post_expectation = {}
    expected_strata = str(post_expectation.get("match_strata", ""))

    post_base = list(post_entry.get("base_keys") or [])
    post_ctor = list(post_entry.get("ctor_keys") or [])
    post_remainder = post_entry.get("remainder") or {}
    if not isinstance(post_remainder, dict):
        post_remainder = {}
    post_matches = post_entry.get("glossary_matches") or []
    post_strata = _glossary_match_strata(post_matches)

    predicate_results: list[JSONObject] = []

    expected_candidates: list[str] = []
    rewrite = plan.get("rewrite") or {}
    if not isinstance(rewrite, dict):
        rewrite = {}
    params = rewrite.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    expected_candidates = [str(v) for v in (params.get("candidates") or []) if v]

    requested_predicates: list[JSONObject] = []
    verification = plan.get("verification") or {}
    if isinstance(verification, dict):
        predicates = verification.get("predicates")
        if isinstance(predicates, list):
            requested_predicates = [
                p for p in predicates if isinstance(p, dict) and p.get("kind")
            ]
    if not requested_predicates:
        requested_predicates = [
            {"kind": "base_conservation", "expect": True},
            {"kind": "ctor_coherence", "expect": True},
            {
                "kind": "match_strata",
                "expect": expected_strata,
                "candidates": expected_candidates,
            },
            {"kind": "remainder_non_regression", "expect": "no-new-remainder"},
        ]

    def _clean(value: int) -> bool:
        return value in (0, 1)

    for predicate in requested_predicates:
        kind = str(predicate.get("kind", ""))
        if kind == "base_conservation":
            base_ok = post_base == expected_base
            predicate_results.append(
                {
                    "kind": kind,
                    "passed": base_ok,
                    "expected": expected_base,
                    "observed": post_base,
                }
            )
            continue
        if kind == "ctor_coherence":
            ctor_ok = post_ctor == expected_ctor
            predicate_results.append(
                {
                    "kind": kind,
                    "passed": ctor_ok,
                    "expected": expected_ctor,
                    "observed": post_ctor,
                }
            )
            continue
        if kind == "match_strata":
            strata_expect = str(predicate.get("expect", expected_strata) or "")
            candidates = [
                str(item)
                for item in (predicate.get("candidates") or expected_candidates)
                if item
            ]
            strata_ok = True
            if strata_expect:
                strata_ok = post_strata == strata_expect
            if strata_expect == "exact" and isinstance(post_matches, list) and len(post_matches) == 1:
                strata_ok = strata_ok and (str(post_matches[0]) in set(candidates))
            predicate_results.append(
                {
                    "kind": kind,
                    "passed": strata_ok,
                    "expected": strata_expect,
                    "observed": post_strata,
                    "candidates": candidates,
                    "observed_matches": post_matches,
                }
            )
            continue
        if kind == "remainder_non_regression":
            pre_base_rem = int(expected_remainder.get("base", 1) or 1)
            pre_ctor_rem = int(expected_remainder.get("ctor", 1) or 1)
            post_base_rem = int(post_remainder.get("base", 1) or 1)
            post_ctor_rem = int(post_remainder.get("ctor", 1) or 1)
            rem_ok = True
            if _clean(pre_base_rem):
                rem_ok = rem_ok and _clean(post_base_rem)
            if _clean(pre_ctor_rem):
                rem_ok = rem_ok and _clean(post_ctor_rem)
            predicate_results.append(
                {
                    "kind": kind,
                    "passed": rem_ok,
                    "expected": {"base": pre_base_rem, "ctor": pre_ctor_rem},
                    "observed": {"base": post_base_rem, "ctor": post_ctor_rem},
                }
            )
            continue
        if kind == "exception_obligation_non_regression":
            pre_summary = pre.get("exception_obligations_summary")
            if not isinstance(pre_summary, dict):
                pre_summary = None
            if post_exception_obligations is None:
                predicate_results.append(
                    {
                        "kind": kind,
                        "passed": False,
                        "expected": pre_summary,
                        "observed": None,
                        "issue": "missing post exception obligations",
                    }
                )
                continue
            if pre_summary is None:
                predicate_results.append(
                    {
                        "kind": kind,
                        "passed": False,
                        "expected": None,
                        "observed": None,
                        "issue": "missing pre exception obligations summary",
                    }
                )
                continue
            post_summary = _exception_obligation_summary_for_site(
                post_exception_obligations,
                site=site,
            )
            try:
                pre_unknown = int(pre_summary.get("UNKNOWN", 0) or 0)
                pre_discharged = int(pre_summary.get("DEAD", 0) or 0) + int(
                    pre_summary.get("HANDLED", 0) or 0
                )
            except (TypeError, ValueError):
                pre_unknown = 0
                pre_discharged = 0
            post_unknown = int(post_summary.get("UNKNOWN", 0) or 0)
            post_discharged = int(post_summary.get("DEAD", 0) or 0) + int(
                post_summary.get("HANDLED", 0) or 0
            )
            exc_ok = (post_unknown <= pre_unknown) and (post_discharged >= pre_discharged)
            predicate_results.append(
                {
                    "kind": kind,
                    "passed": exc_ok,
                    "expected": {"UNKNOWN": pre_unknown, "DISCHARGED": pre_discharged},
                    "observed": {"UNKNOWN": post_unknown, "DISCHARGED": post_discharged},
                    "pre_summary": pre_summary,
                    "post_summary": post_summary,
                }
            )
            continue
        predicate_results.append(
            {
                "kind": kind,
                "passed": False,
                "expected": predicate.get("expect"),
                "observed": None,
                "issue": "unknown predicate kind",
            }
        )

    accepted = all(bool(result.get("passed")) for result in predicate_results)
    if not accepted:
        issues.append("verification predicates failed")
    return {
        "plan_id": plan_id,
        "accepted": accepted,
        "issues": issues,
        "predicate_results": predicate_results,
    }


def verify_rewrite_plans(
    plans: list[JSONObject],
    *,
    post_provenance: list[JSONObject],
    post_exception_obligations: list[JSONObject] | None = None,
) -> list[JSONObject]:
    return [
        verify_rewrite_plan(
            plan,
            post_provenance=post_provenance,
            post_exception_obligations=post_exception_obligations,
        )
        for plan in plans
    ]


def _summarize_rewrite_plans(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        plan_id = entry.get("plan_id", "?")
        site = entry.get("site", {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        kind = entry.get("rewrite", {}).get("kind", "?")
        status = entry.get("status", "UNVERIFIED")
        lines.append(
            f"{plan_id} {path}:{function} bundle={bundle} kind={kind} status={status}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def _enclosing_function_node(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _exception_param_names(expr: ast.AST | None, params: set[str]) -> list[str]:
    if expr is None:
        return []
    names: set[str] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in params:
            names.add(node.id)
    return sorted(names)


def _exception_type_name(expr: ast.AST | None) -> str | None:
    if expr is None:
        return None
    if isinstance(expr, ast.Call):
        return _decorator_name(expr.func)
    return _decorator_name(expr)


def _exception_path_id(
    *,
    path: str,
    function: str,
    source_kind: str,
    lineno: int,
    col: int,
    kind: str,
) -> str:
    return f"{path}:{function}:{source_kind}:{lineno}:{col}:{kind}"


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in {"Exception", "BaseException"}
    if isinstance(handler.type, ast.Attribute):
        return handler.type.attr in {"Exception", "BaseException"}
    return False


def _handler_label(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return "except:"
    try:
        return f"except {ast.unparse(handler.type)}"
    except Exception:
        return "except <unknown>"


def _node_in_try_body(node: ast.AST, try_node: ast.Try) -> bool:
    for stmt in try_node.body:
        if node is stmt:
            return True
        for child in ast.walk(stmt):
            if node is child:
                return True
    return False


def _find_handling_try(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.Try | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.Try) and _node_in_try_body(node, current):
            return current
        current = parents.get(current)
    return None


def _node_in_block(node: ast.AST, block: list[ast.stmt]) -> bool:
    for stmt in block:
        if node is stmt:
            return True
        for child in ast.walk(stmt):
            if node is child:
                return True
    return False


def _names_in_expr(expr: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _eval_value_expr(expr: ast.AST, env: dict[str, JSONValue]) -> JSONValue | None:
    if isinstance(expr, ast.Constant):
        value = expr.value
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return None
    if isinstance(expr, ast.Name):
        if expr.id in env:
            return env[expr.id]
        return None
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.USub, ast.UAdd)):
        operand = _eval_value_expr(expr.operand, env)
        if isinstance(operand, (int, float)):
            return -operand if isinstance(expr.op, ast.USub) else operand
    return None


def _eval_bool_expr(expr: ast.AST, env: dict[str, JSONValue]) -> bool | None:
    if isinstance(expr, ast.Constant):
        return bool(expr.value)
    if isinstance(expr, ast.Name):
        if expr.id not in env:
            return None
        return bool(env[expr.id])
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        inner = _eval_bool_expr(expr.operand, env)
        if inner is None:
            return None
        return not inner
    if isinstance(expr, ast.BoolOp):
        if isinstance(expr.op, ast.And):
            any_unknown = False
            for value in expr.values:
                result = _eval_bool_expr(value, env)
                if result is False:
                    return False
                if result is None:
                    any_unknown = True
            return None if any_unknown else True
        if isinstance(expr.op, ast.Or):
            any_unknown = False
            for value in expr.values:
                result = _eval_bool_expr(value, env)
                if result is True:
                    return True
                if result is None:
                    any_unknown = True
            return None if any_unknown else False
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
        left = _eval_value_expr(expr.left, env)
        right = _eval_value_expr(expr.comparators[0], env)
        if left is None or right is None:
            return None
        op = expr.ops[0]
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left < right
        if isinstance(op, ast.LtE) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left <= right
        if isinstance(op, ast.Gt) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left > right
        if isinstance(op, ast.GtE) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left >= right
    return None


def _branch_reachability_under_env(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    env: dict[str, JSONValue],
) -> bool | None:
    """Conservatively evaluate nested-if constraints for `node` under `env`."""
    constraints: list[tuple[ast.AST, bool]] = []
    current_node: ast.AST = node
    current = parents.get(current_node)
    while current is not None:
        if isinstance(current, ast.If):
            if _node_in_block(current_node, current.body):
                constraints.append((current.test, True))
            elif _node_in_block(current_node, current.orelse):
                constraints.append((current.test, False))
        current_node = current
        current = parents.get(current_node)
    if not constraints:
        return None
    any_unknown = False
    for test, want_true in constraints:
        result = _eval_bool_expr(test, env)
        if result is None:
            any_unknown = True
            continue
        if result != want_true:
            return False
    return None if any_unknown else True


def _collect_handledness_witnesses(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
) -> list[JSONObject]:
    witnesses: list[JSONObject] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        parent = ParentAnnotator()
        parent.visit(tree)
        parents = parent.parents
        params_by_fn: dict[ast.AST, set[str]] = {}
        for fn in _collect_functions(tree):
            params_by_fn[fn] = set(_param_names(fn, ignore_params))
        path_value = _normalize_snapshot_path(path, project_root)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Raise, ast.Assert)):
                continue
            try_node = _find_handling_try(node, parents)
            source_kind = "E0"
            kind = "raise" if isinstance(node, ast.Raise) else "assert"
            fn_node = _enclosing_function_node(node, parents)
            if fn_node is None:
                function = "<module>"
                params = set()
            else:
                scopes = _enclosing_scopes(fn_node, parents)
                function = _function_key(scopes, fn_node.name)
                params = params_by_fn.get(fn_node, set())
            expr = node.exc if isinstance(node, ast.Raise) else node.test
            exception_name = _exception_type_name(expr)
            bundle = _exception_param_names(expr, params)
            lineno = getattr(node, "lineno", 0)
            col = getattr(node, "col_offset", 0)
            exception_id = _exception_path_id(
                path=path_value,
                function=function,
                source_kind=source_kind,
                lineno=lineno,
                col=col,
                kind=kind,
            )
            handledness_id = f"handled:{exception_id}"
            handler_kind = None
            handler_boundary = None
            if try_node is not None:
                handler = next(
                    (h for h in try_node.handlers if _handler_is_broad(h)), None
                )
                if handler is not None:
                    handler_kind = "catch"
                    handler_boundary = _handler_label(handler)
            if handler_kind is None and exception_name == "SystemExit":
                handler_kind = "convert"
                handler_boundary = "process exit"
            if handler_kind is None:
                continue
            witnesses.append(
                {
                    "handledness_id": handledness_id,
                    "exception_path_id": exception_id,
                    "site": {
                        "path": path_value,
                        "function": function,
                        "bundle": bundle,
                    },
                    "handler_kind": handler_kind,
                    "handler_boundary": handler_boundary,
                    "environment": {},
                    "core": (
                        [f"enclosed by {handler_boundary}"]
                        if handler_kind == "catch"
                        else ["converted to process exit"]
                    ),
                    "result": "HANDLED",
                }
            )
    return sorted(
        witnesses,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("exception_path_id", "")),
        ),
    )


def _collect_exception_obligations(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    handledness_witnesses: list[JSONObject] | None = None,
    deadness_witnesses: list[JSONObject] | None = None,
    never_exceptions: set[str] | None = None,
) -> list[JSONObject]:
    obligations: list[JSONObject] = []
    never_exceptions_set = set(never_exceptions or [])
    handled_map: dict[str, JSONObject] = {}
    if handledness_witnesses:
        for entry in handledness_witnesses:
            exception_id = str(entry.get("exception_path_id", ""))
            if exception_id:
                handled_map[exception_id] = entry
    dead_env_map: dict[tuple[str, str], dict[str, tuple[JSONValue, JSONObject]]] = {}
    if deadness_witnesses:
        for entry in deadness_witnesses:
            path_value = str(entry.get("path", ""))
            function_value = str(entry.get("function", ""))
            bundle = entry.get("bundle", []) or []
            if not isinstance(bundle, list) or not bundle:
                continue
            param = str(bundle[0])
            environment = entry.get("environment", {})
            if not isinstance(environment, dict):
                continue
            value_str = environment.get(param)
            if not isinstance(value_str, str):
                continue
            try:
                literal_value = ast.literal_eval(value_str)
            except Exception:
                continue
            dead_env_map.setdefault((path_value, function_value), {})[param] = (
                literal_value,
                entry,
            )
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        parent = ParentAnnotator()
        parent.visit(tree)
        parents = parent.parents
        params_by_fn: dict[ast.AST, set[str]] = {}
        for fn in _collect_functions(tree):
            params_by_fn[fn] = set(_param_names(fn, ignore_params))
        path_value = _normalize_snapshot_path(path, project_root)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Raise, ast.Assert)):
                continue
            source_kind = "E0"
            kind = "raise" if isinstance(node, ast.Raise) else "assert"
            fn_node = _enclosing_function_node(node, parents)
            if fn_node is None:
                function = "<module>"
                params = set()
            else:
                scopes = _enclosing_scopes(fn_node, parents)
                function = _function_key(scopes, fn_node.name)
                params = params_by_fn.get(fn_node, set())
            expr = node.exc if isinstance(node, ast.Raise) else node.test
            exception_name = _exception_type_name(expr)
            protocol: str | None = None
            if (
                exception_name
                and never_exceptions_set
                and _decorator_matches(exception_name, never_exceptions_set)
            ):
                protocol = "never"
            bundle = _exception_param_names(expr, params)
            lineno = getattr(node, "lineno", 0)
            col = getattr(node, "col_offset", 0)
            exception_id = _exception_path_id(
                path=path_value,
                function=function,
                source_kind=source_kind,
                lineno=lineno,
                col=col,
                kind=kind,
            )
            handled = handled_map.get(exception_id)
            status = "UNKNOWN"
            witness_ref = None
            remainder: JSONObject | None = {"exception_kind": kind}
            environment_ref: JSONObject | None = None
            if handled:
                status = "HANDLED"
                witness_ref = handled.get("handledness_id")
                remainder = {}
                environment_ref = handled.get("environment") or {}
            else:
                env_entries = dead_env_map.get((path_value, function), {})
                if env_entries:
                    env = {name: value for name, (value, _) in env_entries.items()}
                    reachability = _branch_reachability_under_env(node, parents, env)
                    if reachability is False:
                        names: set[str] = set()
                        current = parents.get(node)
                        while current is not None:
                            if isinstance(current, ast.If):
                                names.update(_names_in_expr(current.test))
                            current = parents.get(current)
                        for name in sorted(names):
                            if name not in env_entries:
                                continue
                            _, witness = env_entries[name]
                            status = "DEAD"
                            witness_ref = witness.get("deadness_id")
                            remainder = {}
                            environment_ref = witness.get("environment") or {}
                            break
            if protocol == "never" and status != "DEAD":
                status = "FORBIDDEN"
            obligations.append(
                {
                    "exception_path_id": exception_id,
                    "site": {
                        "path": path_value,
                        "function": function,
                        "bundle": bundle,
                    },
                    "source_kind": source_kind,
                    "status": status,
                    "witness_ref": witness_ref,
                    "remainder": remainder,
                    "environment_ref": environment_ref,
                    "exception_name": exception_name,
                    "protocol": protocol,
                }
            )
    return sorted(
        obligations,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("source_kind", "")),
            str(entry.get("exception_path_id", "")),
        ),
    )


def _summarize_exception_obligations(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        site = entry.get("site", {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        status = entry.get("status", "UNKNOWN")
        source = entry.get("source_kind", "?")
        exception_name = entry.get("exception_name")
        protocol = entry.get("protocol")
        suffix = ""
        if exception_name:
            suffix += f" exception={exception_name}"
        if protocol:
            suffix += f" protocol={protocol}"
        lines.append(
            f"{path}:{function} bundle={bundle} source={source} status={status}{suffix}"
        )
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def _exception_protocol_warnings(entries: list[JSONObject]) -> list[str]:
    warnings: list[str] = []
    for entry in entries:
        if entry.get("protocol") != "never":
            continue
        if entry.get("status") == "DEAD":
            continue
        site = entry.get("site", {}) or {}
        path = site.get("path", "?")
        function = site.get("function", "?")
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        warnings.append(
            f"{path}:{function} raises {exception_name} (protocol=never, status={status})"
        )
    return warnings


def _exception_protocol_evidence(entries: list[JSONObject]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        if entry.get("protocol") != "never":
            continue
        exception_id = entry.get("exception_path_id", "?")
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        lines.append(
            f"{exception_id} exception={exception_name} protocol=never status={status}"
        )
    return lines


def _parse_lint_location(line: str) -> tuple[str, int, int, str] | None:
    match = re.match(r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+)", line)
    if not match:
        return None
    path = match.group("path")
    try:
        lineno = int(match.group("line"))
        col = int(match.group("col"))
    except ValueError:
        return None
    remainder = line[match.end() :].lstrip(": ").strip()
    if remainder.startswith("-"):
        trimmed = remainder[1:]
        range_match = re.match(r"^(\d+):(\d+)(:)?\s*", trimmed)
        if range_match:
            remainder = trimmed[range_match.end() :].strip()
    return path, lineno, col, remainder


def _lint_line(path: str, line: int, col: int, code: str, message: str) -> str:
    return f"{path}:{line}:{col}: {code} {message}".strip()


def _decision_param_lint_line(
    info: "FunctionInfo",
    param: str,
    *,
    project_root: Path | None,
    code: str,
    message: str,
) -> str | None:
    span = info.param_spans.get(param)
    if span is None:
        return None
    path = _normalize_snapshot_path(info.path, project_root)
    line, col, _, _ = span
    return _lint_line(path, line + 1, col + 1, code, message)


def _decision_tier_for(
    info: "FunctionInfo",
    param: str,
    *,
    tier_map: dict[str, int],
    project_root: Path | None,
) -> int | None:
    if not tier_map:
        return None
    span = info.param_spans.get(param)
    if span is not None:
        path = _normalize_snapshot_path(info.path, project_root)
        line, col, _, _ = span
        location = f"{path}:{line + 1}:{col + 1}"
        for key in (location, f"{location}:{param}"):
            if key in tier_map:
                return tier_map[key]
    for key in (f"{info.qual}:{param}", f"{info.qual}.{param}", param):
        if key in tier_map:
            return tier_map[key]
    return None


def _collect_transitive_callers(
    callers_by_qual: dict[str, set[str]],
    by_qual: dict[str, FunctionInfo],
) -> dict[str, set[str]]:
    transitive: dict[str, set[str]] = {}
    for qual in by_qual:
        seen: set[str] = set()
        stack = list(callers_by_qual.get(qual, set()))
        while stack:
            caller = stack.pop()
            if caller in seen:
                continue
            seen.add(caller)
            stack.extend(callers_by_qual.get(caller, set()))
        transitive[qual] = seen
    return transitive


def _lint_lines_from_bundle_evidence(evidence: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for entry in evidence:
        parsed = _parse_lint_location(entry)
        if not parsed:
            continue
        path, lineno, col, remainder = parsed
        message = remainder or "undocumented bundle"
        lines.append(_lint_line(path, lineno, col, "GABION_BUNDLE_UNDOC", message))
    return lines


def _lint_lines_from_type_evidence(evidence: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for entry in evidence:
        parsed = _parse_lint_location(entry)
        if not parsed:
            continue
        path, lineno, col, remainder = parsed
        message = remainder or "type-flow evidence"
        lines.append(_lint_line(path, lineno, col, "GABION_TYPE_FLOW", message))
    return lines


def _lint_lines_from_unused_arg_smells(smells: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for entry in smells:
        parsed = _parse_lint_location(entry)
        if not parsed:
            continue
        path, lineno, col, remainder = parsed
        message = remainder or "unused argument flow"
        lines.append(_lint_line(path, lineno, col, "GABION_UNUSED_ARG", message))
    return lines


def _extract_smell_sample(entry: str) -> str | None:
    match = re.search(r"\(e\.g\.\s*([^)]+)\)", entry)
    if not match:
        return None
    return match.group(1).strip()


def _lint_lines_from_constant_smells(smells: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for entry in smells:
        parsed = _parse_lint_location(entry)
        if not parsed:
            sample = _extract_smell_sample(entry)
            if sample:
                parsed = _parse_lint_location(sample)
        if not parsed:
            continue
        path, lineno, col, _ = parsed
        lines.append(_lint_line(path, lineno, col, "GABION_CONST_FLOW", entry))
    return lines


def _parse_exception_path_id(value: str) -> tuple[str, int, int] | None:
    parts = value.split(":", 5)
    if len(parts) != 6:
        return None
    path = parts[0]
    try:
        lineno = int(parts[3])
        col = int(parts[4])
    except ValueError:
        return None
    return path, lineno, col


def _exception_protocol_lint_lines(entries: list[JSONObject]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        if entry.get("protocol") != "never":
            continue
        if entry.get("status") == "DEAD":
            continue
        exception_id = str(entry.get("exception_path_id", ""))
        parsed = _parse_exception_path_id(exception_id)
        if not parsed:
            continue
        path, lineno, col = parsed
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        message = f"never-throw exception {exception_name} (status={status})"
        lines.append(_lint_line(path, lineno, col, "GABION_EXC_NEVER", message))
    return lines


def _collect_bundle_evidence_lines(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]],
) -> list[str]:
    if not groups_by_path:
        return []
    nodes, adj, bundle_map = _component_graph(groups_by_path)
    components = _connected_components(nodes, adj)
    common = os.path.commonpath([str(p) for p in groups_by_path])
    root = Path(common)
    file_paths = sorted(groups_by_path)
    config_bundles_by_path = _collect_config_bundles(file_paths)
    declared_global: set[tuple[str, ...]] = set()
    for bundles in config_bundles_by_path.values():
        for fields in bundles.values():
            declared_global.add(tuple(sorted(fields)))
    symbol_table = _build_symbol_table(
        file_paths,
        root,
        external_filter=True,
    )
    dataclass_registry = _collect_dataclass_registry(
        file_paths,
        project_root=root,
    )
    documented_bundles_by_path: dict[Path, set[tuple[str, ...]]] = {}
    for path in file_paths:
        documented = _iter_documented_bundles(path)
        promoted = _iter_dataclass_call_bundles(
            path,
            project_root=root,
            symbol_table=symbol_table,
            dataclass_registry=dataclass_registry,
        )
        documented_bundles_by_path[path] = documented | promoted
    evidence_lines: list[str] = []
    for comp in components:
        evidence = _render_component_callsite_evidence(
            component=comp,
            nodes=nodes,
            bundle_map=bundle_map,
            documented_bundles_by_path=documented_bundles_by_path,
            declared_global=declared_global,
            bundle_sites_by_path=bundle_sites_by_path,
            root=root,
        )
        if evidence:
            evidence_lines.extend(evidence)
    return evidence_lines


def _compute_lint_lines(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]],
    type_callsite_evidence: list[str],
    exception_obligations: list[JSONObject],
    decision_lint_lines: list[str],
    constant_smells: list[str],
    unused_arg_smells: list[str],
) -> list[str]:
    lint_lines: list[str] = []
    bundle_evidence = _collect_bundle_evidence_lines(
        groups_by_path=groups_by_path,
        bundle_sites_by_path=bundle_sites_by_path,
    )
    lint_lines.extend(_lint_lines_from_bundle_evidence(bundle_evidence))
    lint_lines.extend(_lint_lines_from_type_evidence(type_callsite_evidence))
    lint_lines.extend(_exception_protocol_lint_lines(exception_obligations))
    lint_lines.extend(decision_lint_lines)
    lint_lines.extend(_lint_lines_from_constant_smells(constant_smells))
    lint_lines.extend(_lint_lines_from_unused_arg_smells(unused_arg_smells))
    return sorted(set(lint_lines))


def _summarize_handledness_witnesses(
    entries: list[JSONObject],
    *,
    max_entries: int = 10,
) -> list[str]:
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:max_entries]:
        site = entry.get("site", {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        handler = entry.get("handler_boundary", "?")
        lines.append(f"{path}:{function} bundle={bundle} handler={handler}")
    if len(entries) > max_entries:
        lines.append(f"... {len(entries) - max_entries} more")
    return lines


def _compute_fingerprint_synth(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    annotations_by_path: dict[Path, dict[str, dict[str, str | None]]],
    *,
    registry: PrimeRegistry,
    ctor_registry: TypeConstructorRegistry | None,
    min_occurrences: int,
    version: str,
    existing: SynthRegistry | None = None,
) -> tuple[list[str], JSONObject | None]:
    if min_occurrences < 2 and existing is None:
        return [], None
    fingerprints: list[Fingerprint] = []
    for path, groups in groups_by_path.items():
        annots_by_fn = annotations_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            fn_annots = annots_by_fn.get(fn_name, {})
            for bundle in bundles:
                if any(param not in fn_annots for param in bundle):
                    continue
                types = [fn_annots[param] for param in sorted(bundle)]
                if any(t is None for t in types):
                    continue
                hint_list = [t for t in types if t is not None]
                fingerprint = bundle_fingerprint_dimensional(
                    hint_list,
                    registry,
                    ctor_registry,
                )
                fingerprints.append(fingerprint)
    if not fingerprints and existing is None:
        return [], None
    if existing is not None:
        synth_registry = existing
        payload = synth_registry_payload(
            synth_registry,
            registry,
            min_occurrences=min_occurrences,
        )
    else:
        synth_registry = build_synth_registry(
            fingerprints,
            registry,
            min_occurrences=min_occurrences,
            version=version,
        )
        if not synth_registry.tails:
            return [], None
        payload = synth_registry_payload(
            synth_registry,
            registry,
            min_occurrences=min_occurrences,
        )
    lines: list[str] = [f"synth registry {synth_registry.version}:"]
    for entry in payload.get("entries", []):
        tail = entry.get("tail", {})
        base_keys = entry.get("base_keys", [])
        ctor_keys = entry.get("ctor_keys", [])
        remainder = entry.get("remainder", {})
        details = f"base={base_keys}"
        if ctor_keys:
            details += f" ctor={ctor_keys}"
        if remainder.get("base") not in (0, 1) or remainder.get("ctor") not in (0, 1):
            details += f" remainder=({remainder.get('base')},{remainder.get('ctor')})"
        lines.append(
            f"- synth_prime={entry.get('prime')} tail="
            f"{{base={tail.get('base', {}).get('product')}, "
            f"ctor={tail.get('ctor', {}).get('product')}}} "
            f"{details}"
        )
    return lines, payload


def _build_synth_registry_payload(
    synth_registry: "SynthRegistry",
    registry: PrimeRegistry,
    *,
    min_occurrences: int,
) -> JSONObject:
    entries: list[JSONObject] = []
    for prime, tail in sorted(synth_registry.tails.items()):
        base_keys, base_remaining = fingerprint_to_type_keys_with_remainder(
            tail.base.product, registry
        )
        ctor_keys, ctor_remaining = fingerprint_to_type_keys_with_remainder(
            tail.ctor.product, registry
        )
        ctor_keys = [
            key[len("ctor:") :] if key.startswith("ctor:") else key
            for key in ctor_keys
        ]
        entries.append(
            {
                "prime": prime,
                "tail": {
                    "base": {
                        "product": tail.base.product,
                        "mask": tail.base.mask,
                    },
                    "ctor": {
                        "product": tail.ctor.product,
                        "mask": tail.ctor.mask,
                    },
                    "provenance": {
                        "product": tail.provenance.product,
                        "mask": tail.provenance.mask,
                    },
                    "synth": {
                        "product": tail.synth.product,
                        "mask": tail.synth.mask,
                    },
                },
                "base_keys": sorted(base_keys),
                "ctor_keys": sorted(ctor_keys),
                "remainder": {
                    "base": base_remaining,
                    "ctor": ctor_remaining,
                },
            }
        )
    return {
        "version": synth_registry.version,
        "min_occurrences": min_occurrences,
        "entries": entries,
    }


class _ReturnAliasCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.returns: list[ast.AST | None] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node.value)


def _return_aliases(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ignore_params: set[str] | None = None,
) -> list[str] | None:
    params = _param_names(fn, ignore_params)
    if not params:
        return None
    param_set = set(params)
    collector = _ReturnAliasCollector()
    for stmt in fn.body:
        collector.visit(stmt)
    if not collector.returns:
        return None
    alias: list[str] | None = None

    def _alias_from_expr(expr: ast.AST | None) -> list[str] | None:
        if expr is None:
            return None
        if isinstance(expr, ast.Name) and expr.id in param_set:
            return [expr.id]
        if isinstance(expr, (ast.Tuple, ast.List)):
            names: list[str] = []
            for elt in expr.elts:
                if isinstance(elt, ast.Name) and elt.id in param_set:
                    names.append(elt.id)
                else:
                    return None
            return names
        return None

    for expr in collector.returns:
        candidate = _alias_from_expr(expr)
        if candidate is None:
            return None
        if alias is None:
            alias = candidate
            continue
        if alias != candidate:
            return None
    return alias


def _collect_return_aliases(
    funcs: list[ast.FunctionDef | ast.AsyncFunctionDef],
    parents: dict[ast.AST, ast.AST],
    *,
    ignore_params: set[str] | None,
) -> dict[str, tuple[list[str], list[str]]]:
    aliases: dict[str, tuple[list[str], list[str]]] = {}
    conflicts: set[str] = set()
    for fn in funcs:
        alias = _return_aliases(fn, ignore_params)
        if not alias:
            continue
        params = _param_names(fn, ignore_params)
        class_name = _enclosing_class(fn, parents)
        scopes = _enclosing_scopes(fn, parents)
        keys = {fn.name}
        if class_name:
            keys.add(f"{class_name}.{fn.name}")
        if scopes:
            keys.add(_function_key(scopes, fn.name))
        info = (params, alias)
        for key in keys:
            if key in conflicts:
                continue
            if key in aliases:
                aliases.pop(key, None)
                conflicts.add(key)
                continue
            aliases[key] = info
    return aliases


def _const_repr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.USub, ast.UAdd)
    ) and isinstance(node.operand, ast.Constant):
        try:
            return ast.unparse(node)
        except Exception:
            return None
    if isinstance(node, ast.Attribute):
        if node.attr.isupper():
            try:
                return ast.unparse(node)
            except Exception:
                return None
        return None
    return None


def _type_from_const_repr(value: str) -> str | None:
    try:
        literal = ast.literal_eval(value)
    except Exception:
        return None
    if literal is None:
        return "None"
    if isinstance(literal, bool):
        return "bool"
    if isinstance(literal, int):
        return "int"
    if isinstance(literal, float):
        return "float"
    if isinstance(literal, complex):
        return "complex"
    if isinstance(literal, str):
        return "str"
    if isinstance(literal, bytes):
        return "bytes"
    if isinstance(literal, list):
        return "list"
    if isinstance(literal, tuple):
        return "tuple"
    if isinstance(literal, set):
        return "set"
    if isinstance(literal, dict):
        return "dict"
    return None


def _is_test_path(path: Path) -> bool:
    if "tests" in path.parts:
        return True
    return path.name.startswith("test_")


def _analyze_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
    *,
    is_test: bool,
    ignore_params: set[str] | None = None,
    strictness: str = "high",
    class_name: str | None = None,
    return_aliases: dict[str, tuple[list[str], list[str]]] | None = None,
) -> tuple[dict[str, ParamUse], list[CallArgs]]:
    params = _param_names(fn, ignore_params)
    use_map = {p: ParamUse(set(), False, {p}) for p in params}
    alias_to_param: dict[str, str] = {p: p for p in params}
    call_args: list[CallArgs] = []

    visitor = UseVisitor(
        parents=parents,
        use_map=use_map,
        call_args=call_args,
        alias_to_param=alias_to_param,
        is_test=is_test,
        strictness=strictness,
        const_repr=_const_repr,
        callee_name=lambda call: _normalize_callee(_callee_name(call), class_name),
        call_args_factory=CallArgs,
        call_context=_call_context,
        return_aliases=return_aliases,
    )
    visitor.visit(fn)
    return use_map, call_args


def _unused_params(use_map: dict[str, ParamUse]) -> set[str]:
    unused: set[str] = set()
    for name, info in use_map.items():
        if info.non_forward:
            continue
        if info.direct_forward:
            continue
        unused.add(name)
    return unused


def _group_by_signature(use_map: dict[str, ParamUse]) -> list[set[str]]:
    sig_map: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    for name, info in use_map.items():
        if info.non_forward:
            continue
        sig = tuple(sorted(info.direct_forward))
        # Empty forwarding signatures are usually just unused params; treating them as
        # bundles creates noisy Tier-3 violations and unstable fingerprint baselines.
        if not sig:
            continue
        sig_map[sig].append(name)
    groups = [set(names) for names in sig_map.values() if len(names) > 1]
    return groups


def _union_groups(groups: list[set[str]]) -> list[set[str]]:
    changed = True
    while changed:
        changed = False
        out = []
        while groups:
            base = groups.pop()
            merged = True
            while merged:
                merged = False
                for i, other in enumerate(groups):
                    if base & other:
                        base |= other
                        groups.pop(i)
                        merged = True
                        changed = True
                        break
            out.append(base)
        groups = out
    return groups


def _propagate_groups(
    call_args: list[CallArgs],
    callee_groups: dict[str, list[set[str]]],
    callee_param_orders: dict[str, list[str]],
    strictness: str,
    opaque_callees: set[str] | None = None,
) -> list[set[str]]:
    groups: list[set[str]] = []
    for call in call_args:
        if opaque_callees and call.callee in opaque_callees:
            continue
        if call.callee not in callee_groups:
            continue
        callee_params = callee_param_orders[call.callee]
        # Build mapping from callee param to caller param.
        callee_to_caller: dict[str, str] = {}
        for idx, pname in enumerate(callee_params):
            key = str(idx)
            if key in call.pos_map:
                callee_to_caller[pname] = call.pos_map[key]
        for kw, caller_name in call.kw_map.items():
            callee_to_caller[kw] = caller_name
        if strictness == "low":
            mapped = set(callee_to_caller.keys())
            remaining = [p for p in callee_params if p not in mapped]
            if len(call.star_pos) == 1:
                _, star_param = call.star_pos[0]
                for param in remaining:
                    callee_to_caller.setdefault(param, star_param)
            if len(call.star_kw) == 1:
                star_param = call.star_kw[0]
                for param in remaining:
                    callee_to_caller.setdefault(param, star_param)
        for group in callee_groups[call.callee]:
            mapped = {callee_to_caller.get(p) for p in group}
            mapped.discard(None)
            if len(mapped) > 1:
                groups.append(set(mapped))
    return groups


def _callsite_evidence_for_bundle(
    calls: list[CallArgs],
    bundle: set[str],
    *,
    limit: int = 12,
) -> list[JSONObject]:
    """Collect callsite evidence for where bundle params are forwarded.

    A bundle can be induced either by co-forwarding in a single callsite or by
    repeated forwarding to identical callee/slot pairs across distinct callsites.
    """
    out: list[JSONObject] = []
    seen: set[tuple[tuple[int, int, int, int], str, tuple[str, ...], tuple[str, ...]]] = set()
    for call in calls:
        if call.span is None:
            continue
        params_in_call: list[str] = []
        slots: list[str] = []
        for idx_str, param in call.pos_map.items():
            if param in bundle:
                params_in_call.append(param)
                slots.append(f"arg[{idx_str}]")
        for name, param in call.kw_map.items():
            if param in bundle:
                params_in_call.append(param)
                slots.append(f"kw[{name}]")
        for idx, param in call.star_pos:
            if param in bundle:
                params_in_call.append(param)
                slots.append(f"arg[{idx}]*")
        for param in call.star_kw:
            if param in bundle:
                params_in_call.append(param)
                slots.append("kw[**]")
        distinct = tuple(sorted(set(params_in_call)))
        if not distinct:
            continue
        slot_list = tuple(sorted(set(slots)))
        key = (call.span, call.callee, distinct, slot_list)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "callee": call.callee,
                "span": list(call.span),
                "params": list(distinct),
                "slots": list(slot_list),
            }
        )
    out.sort(
        key=lambda entry: (
            -len(entry.get("params") or []),
            tuple(entry.get("span") or []),
            str(entry.get("callee") or ""),
            tuple(entry.get("params") or []),
        )
    )
    return out[:limit]


def _analyze_file_internal(
    path: Path,
    recursive: bool = True,
    *,
    config: AuditConfig | None = None,
) -> tuple[
    dict[str, list[set[str]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, list[list[JSONObject]]],
]:
    if config is None:
        config = AuditConfig()
    tree = ast.parse(path.read_text())
    parent = ParentAnnotator()
    parent.visit(tree)
    parents = parent.parents
    is_test = _is_test_path(path)

    funcs = _collect_functions(tree)
    return_aliases = _collect_return_aliases(
        funcs, parents, ignore_params=config.ignore_params
    )
    fn_param_orders: dict[str, list[str]] = {}
    fn_param_spans: dict[str, dict[str, tuple[int, int, int, int]]] = {}
    fn_use = {}
    fn_calls = {}
    fn_names: dict[str, str] = {}
    fn_lexical_scopes: dict[str, tuple[str, ...]] = {}
    fn_class_names: dict[str, str | None] = {}
    opaque_callees: set[str] = set()
    for f in funcs:
        class_name = _enclosing_class(f, parents)
        scopes = _enclosing_scopes(f, parents)
        lexical_scopes = _enclosing_function_scopes(f, parents)
        fn_key = _function_key(scopes, f.name)
        if not _decorators_transparent(f, config.transparent_decorators):
            opaque_callees.add(fn_key)
        use_map, call_args = _analyze_function(
            f,
            parents,
            is_test=is_test,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            class_name=class_name,
            return_aliases=return_aliases,
        )
        fn_use[fn_key] = use_map
        fn_calls[fn_key] = call_args
        fn_param_orders[fn_key] = _param_names(f, config.ignore_params)
        fn_param_spans[fn_key] = _param_spans(f, config.ignore_params)
        fn_names[fn_key] = f.name
        fn_lexical_scopes[fn_key] = tuple(lexical_scopes)
        fn_class_names[fn_key] = class_name

    local_by_name: dict[str, list[str]] = defaultdict(list)
    for key, name in fn_names.items():
        local_by_name[name].append(key)

    def _resolve_local_callee(callee: str, caller_key: str) -> str | None:
        if "." in callee:
            return None
        candidates = local_by_name.get(callee, [])
        if not candidates:
            return None
        effective_scope = list(fn_lexical_scopes.get(caller_key, ())) + [fn_names[caller_key]]
        while True:
            scoped = [
                key
                for key in candidates
                if fn_lexical_scopes.get(key, ()) == tuple(effective_scope)
                and not (fn_class_names.get(key) and not fn_lexical_scopes.get(key))
            ]
            if len(scoped) == 1:
                return scoped[0]
            if len(scoped) > 1:
                return None
            if not effective_scope:
                break
            effective_scope = effective_scope[:-1]
        return None

    for caller_key, calls in list(fn_calls.items()):
        resolved_calls: list[CallArgs] = []
        for call in calls:
            resolved = _resolve_local_callee(call.callee, caller_key)
            if resolved:
                resolved_calls.append(replace(call, callee=resolved))
            else:
                resolved_calls.append(call)
        fn_calls[caller_key] = resolved_calls

    class_bases = _collect_local_class_bases(tree, parents)
    if class_bases:
        local_functions = set(fn_use.keys())

        def _resolve_local_method(callee: str) -> str | None:
            class_part, method = callee.rsplit(".", 1)
            return _resolve_local_method_in_hierarchy(
                class_part,
                method,
                class_bases=class_bases,
                local_functions=local_functions,
                seen=set(),
            )

        for caller_key, calls in list(fn_calls.items()):
            resolved_calls = []
            for call in calls:
                if "." in call.callee:
                    resolved = _resolve_local_method(call.callee)
                    if resolved and resolved != call.callee:
                        resolved_calls.append(replace(call, callee=resolved))
                        continue
                resolved_calls.append(call)
            fn_calls[caller_key] = resolved_calls

    groups_by_fn = {fn: _group_by_signature(use_map) for fn, use_map in fn_use.items()}

    if not recursive:
        bundle_sites_by_fn: dict[str, list[list[JSONObject]]] = {}
        for fn_key, bundles in groups_by_fn.items():
            calls = fn_calls.get(fn_key, [])
            bundle_sites_by_fn[fn_key] = [
                _callsite_evidence_for_bundle(calls, bundle) for bundle in bundles
            ]
        return groups_by_fn, fn_param_spans, bundle_sites_by_fn

    changed = True
    while changed:
        changed = False
        for fn in fn_use:
            propagated = _propagate_groups(
                fn_calls[fn],
                groups_by_fn,
                fn_param_orders,
                config.strictness,
                opaque_callees,
            )
            if not propagated:
                continue
            combined = _union_groups(groups_by_fn.get(fn, []) + propagated)
            if combined != groups_by_fn.get(fn, []):
                groups_by_fn[fn] = combined
                changed = True
    bundle_sites_by_fn: dict[str, list[list[JSONObject]]] = {}
    for fn_key, bundles in groups_by_fn.items():
        calls = fn_calls.get(fn_key, [])
        bundle_sites_by_fn[fn_key] = [
            _callsite_evidence_for_bundle(calls, bundle) for bundle in bundles
        ]
    return groups_by_fn, fn_param_spans, bundle_sites_by_fn


def analyze_file(
    path: Path,
    recursive: bool = True,
    *,
    config: AuditConfig | None = None,
) -> tuple[dict[str, list[set[str]]], dict[str, dict[str, tuple[int, int, int, int]]]]:
    groups, spans, _ = _analyze_file_internal(path, recursive=recursive, config=config)
    return groups, spans


def _callee_key(name: str) -> str:
    if not name:
        return name
    return name.split(".")[-1]


def _is_broad_type(annot: str | None) -> bool:
    if annot is None:
        return True
    base = annot.replace("typing.", "")
    return base in {"Any", "object"}


_NONE_TYPES = {"None", "NoneType", "type(None)"}


def _split_top_level(value: str, sep: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in value:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(depth - 1, 0)
        if ch == sep and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _expand_type_hint(hint: str) -> set[str]:
    hint = hint.strip()
    if not hint:
        return set()
    if hint.startswith("Optional[") and hint.endswith("]"):
        inner = hint[len("Optional[") : -1]
        return {_strip_type(t) for t in _split_top_level(inner, ",")} | {"None"}
    if hint.startswith("Union[") and hint.endswith("]"):
        inner = hint[len("Union[") : -1]
        return {_strip_type(t) for t in _split_top_level(inner, ",")}
    if "|" in hint:
        return {_strip_type(t) for t in _split_top_level(hint, "|")}
    return {hint}


def _strip_type(value: str) -> str:
    return value.strip()


def _combine_type_hints(types: set[str]) -> tuple[str, bool]:
    normalized_sets = []
    for hint in types:
        expanded = _expand_type_hint(hint)
        normalized_sets.append(
            tuple(sorted(t for t in expanded if t not in _NONE_TYPES))
        )
    unique_normalized = {norm for norm in normalized_sets if norm}
    expanded: set[str] = set()
    for hint in types:
        expanded.update(_expand_type_hint(hint))
    none_types = {t for t in expanded if t in _NONE_TYPES}
    expanded -= none_types
    if not expanded:
        return "Any", bool(types)
    sorted_types = sorted(expanded)
    if len(sorted_types) == 1:
        base = sorted_types[0]
        if none_types:
            conflicted = len(unique_normalized) > 1
            return f"Optional[{base}]", conflicted
        return base, len(unique_normalized) > 1
    union = f"Union[{', '.join(sorted_types)}]"
    if none_types:
        return f"Optional[{union}]", len(unique_normalized) > 1
    return union, len(unique_normalized) > 1


@dataclass
class FunctionInfo:
    name: str
    qual: str
    path: Path
    params: list[str]
    annots: dict[str, str | None]
    calls: list[CallArgs]
    unused_params: set[str]
    defaults: set[str] = field(default_factory=set)
    transparent: bool = True
    class_name: str | None = None
    scope: tuple[str, ...] = ()
    lexical_scope: tuple[str, ...] = ()
    decision_params: set[str] = field(default_factory=set)
    value_decision_params: set[str] = field(default_factory=set)
    value_decision_reasons: set[str] = field(default_factory=set)
    positional_params: tuple[str, ...] = ()
    kwonly_params: tuple[str, ...] = ()
    vararg: str | None = None
    kwarg: str | None = None
    param_spans: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)


@dataclass
class ClassInfo:
    qual: str
    module: str
    bases: list[str]
    methods: set[str]


def _module_name(path: Path, project_root: Path | None = None) -> str:
    rel = path.with_suffix("")
    if project_root is not None:
        try:
            rel = rel.relative_to(project_root)
        except ValueError:
            pass
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _string_list(node: ast.AST) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        values: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.append(elt.value)
            else:
                return None
        return values
    return None


def _base_identifier(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        try:
            return ast.unparse(node)
        except Exception:
            return None
    if isinstance(node, ast.Subscript):
        return _base_identifier(node.value)
    if isinstance(node, ast.Call):
        return _base_identifier(node.func)
    return None


def _collect_module_exports(
    tree: ast.AST,
    *,
    module_name: str,
    import_map: dict[str, str],
) -> tuple[set[str], dict[str, str]]:
    explicit_all: list[str] | None = None
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                values = _string_list(stmt.value)
                if values is not None:
                    explicit_all = list(values)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__":
                values = _string_list(stmt.value) if stmt.value is not None else None
                if values is not None:
                    explicit_all = list(values)
        elif isinstance(stmt, ast.AugAssign):
            if (
                isinstance(stmt.target, ast.Name)
                and stmt.target.id == "__all__"
                and isinstance(stmt.op, ast.Add)
            ):
                values = _string_list(stmt.value)
                if values is not None:
                    if explicit_all is None:
                        explicit_all = []
                    explicit_all.extend(values)

    local_defs: set[str] = set()
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not stmt.name.startswith("_"):
                local_defs.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    local_defs.add(target.id)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and not stmt.target.id.startswith("_"):
                local_defs.add(stmt.target.id)

    if explicit_all is not None:
        export_names = set(explicit_all)
    else:
        export_names = set(local_defs) | {
            name for name in import_map.keys() if not name.startswith("_")
        }
        export_names = {name for name in export_names if not name.startswith("_")}

    export_map: dict[str, str] = {}
    for name in export_names:
        if name in import_map:
            export_map[name] = import_map[name]
        elif name in local_defs:
            export_map[name] = f"{module_name}.{name}" if module_name else name
    return export_names, export_map

def _build_symbol_table(
    paths: list[Path],
    project_root: Path | None,
    *,
    external_filter: bool,
) -> SymbolTable:
    table = SymbolTable(external_filter=external_filter)
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        module = _module_name(path, project_root)
        if module:
            table.internal_roots.add(module.split(".")[0])
        visitor = ImportVisitor(module, table)
        visitor.visit(tree)
        if module:
            import_map = {
                local: fqn
                for (mod, local), fqn in table.imports.items()
                if mod == module
            }
            exports, export_map = _collect_module_exports(
                tree,
                module_name=module,
                import_map=import_map,
            )
            table.module_exports[module] = exports
            table.module_export_map[module] = export_map
    return table


def _collect_class_index(
    paths: list[Path],
    project_root: Path | None,
) -> dict[str, ClassInfo]:
    class_index: dict[str, ClassInfo] = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        parents = ParentAnnotator()
        parents.visit(tree)
        module = _module_name(path, project_root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            scopes = _enclosing_class_scopes(node, parents.parents)
            qual_parts = [module] if module else []
            qual_parts.extend(scopes)
            qual_parts.append(node.name)
            qual = ".".join(qual_parts)
            bases: list[str] = []
            for base in node.bases:
                base_name = _base_identifier(base)
                if base_name:
                    bases.append(base_name)
            methods: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(stmt.name)
            class_index[qual] = ClassInfo(
                qual=qual,
                module=module,
                bases=bases,
                methods=methods,
            )
    return class_index


def _resolve_class_candidates(
    base: str,
    *,
    module: str,
    symbol_table: SymbolTable | None,
    class_index: dict[str, ClassInfo],
) -> list[str]:
    if not base:
        return []
    candidates: list[str] = []
    if "." in base:
        parts = base.split(".")
        head = parts[0]
        tail = ".".join(parts[1:])
        if symbol_table is not None:
            resolved_head = symbol_table.resolve(module, head)
            if resolved_head:
                candidates.append(f"{resolved_head}.{tail}")
        if module:
            candidates.append(f"{module}.{base}")
        candidates.append(base)
    else:
        if symbol_table is not None:
            resolved = symbol_table.resolve(module, base)
            if resolved:
                candidates.append(resolved)
            resolved_star = symbol_table.resolve_star(module, base)
            if resolved_star:
                candidates.append(resolved_star)
        if module:
            candidates.append(f"{module}.{base}")
        candidates.append(base)
    seen: set[str] = set()
    resolved: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in class_index:
            resolved.append(candidate)
    return resolved


def _resolve_method_in_hierarchy(
    class_qual: str,
    method: str,
    *,
    class_index: dict[str, ClassInfo],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable | None,
    seen: set[str],
) -> FunctionInfo | None:
    if class_qual in seen:
        return None
    seen.add(class_qual)
    candidate = f"{class_qual}.{method}"
    if candidate in by_qual:
        return by_qual[candidate]
    info = class_index.get(class_qual)
    if info is None:
        return None
    for base in info.bases:
        for base_qual in _resolve_class_candidates(
            base,
            module=info.module,
            symbol_table=symbol_table,
            class_index=class_index,
        ):
            resolved = _resolve_method_in_hierarchy(
                base_qual,
                method,
                class_index=class_index,
                by_qual=by_qual,
                symbol_table=symbol_table,
                seen=seen,
            )
            if resolved is not None:
                return resolved
    return None


def _build_function_index(
    paths: list[Path],
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators: set[str] | None = None,
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]]:
    by_name: dict[str, list[FunctionInfo]] = defaultdict(list)
    by_qual: dict[str, FunctionInfo] = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        funcs = _collect_functions(tree)
        if not funcs:
            continue
        parents = ParentAnnotator()
        parents.visit(tree)
        parent_map = parents.parents
        module = _module_name(path, project_root)
        return_aliases = _collect_return_aliases(
            funcs, parent_map, ignore_params=ignore_params
        )
        for fn in funcs:
            class_name = _enclosing_class(fn, parent_map)
            scopes = _enclosing_scopes(fn, parent_map)
            lexical_scopes = _enclosing_function_scopes(fn, parent_map)
            use_map, call_args = _analyze_function(
                fn,
                parent_map,
                is_test=_is_test_path(path),
                ignore_params=ignore_params,
                strictness=strictness,
                class_name=class_name,
                return_aliases=return_aliases,
            )
            unused_params = _unused_params(use_map)
            value_params, value_reasons = _value_encoded_decision_params(
                fn, ignore_params
            )
            pos_args = [a.arg for a in (fn.args.posonlyargs + fn.args.args)]
            kwonly_args = [a.arg for a in fn.args.kwonlyargs]
            if pos_args and pos_args[0] in {"self", "cls"}:
                pos_args = pos_args[1:]
            if ignore_params:
                pos_args = [name for name in pos_args if name not in ignore_params]
                kwonly_args = [
                    name for name in kwonly_args if name not in ignore_params
                ]
            vararg = None
            if fn.args.vararg is not None:
                candidate = fn.args.vararg.arg
                if not ignore_params or candidate not in ignore_params:
                    vararg = candidate
            kwarg = None
            if fn.args.kwarg is not None:
                candidate = fn.args.kwarg.arg
                if not ignore_params or candidate not in ignore_params:
                    kwarg = candidate
            qual_parts = [module] if module else []
            if scopes:
                qual_parts.extend(scopes)
            qual_parts.append(fn.name)
            qual = ".".join(qual_parts)
            info = FunctionInfo(
                name=fn.name,
                qual=qual,
                path=path,
                params=_param_names(fn, ignore_params),
                annots=_param_annotations(fn, ignore_params),
                defaults=_param_defaults(fn, ignore_params),
                calls=call_args,
                unused_params=unused_params,
                transparent=_decorators_transparent(fn, transparent_decorators),
                class_name=class_name,
                scope=tuple(scopes),
                lexical_scope=tuple(lexical_scopes),
                decision_params=_decision_surface_params(fn, ignore_params),
                value_decision_params=value_params,
                value_decision_reasons=value_reasons,
                positional_params=tuple(pos_args),
                kwonly_params=tuple(kwonly_args),
                vararg=vararg,
                kwarg=kwarg,
                param_spans=_param_spans(fn, ignore_params),
            )
            by_name[fn.name].append(info)
            by_qual[info.qual] = info
    return by_name, by_qual


def _resolve_callee(
    callee_key: str,
    caller: FunctionInfo,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable | None = None,
    project_root: Path | None = None,
    class_index: dict[str, ClassInfo] | None = None,
) -> FunctionInfo | None:
    # dataflow-bundle: by_name, caller
    if not callee_key:
        return None
    caller_module = _module_name(caller.path, project_root=project_root)
    candidates = by_name.get(_callee_key(callee_key), [])
    if "." not in callee_key:
        ambiguous = False
        effective_scope = list(caller.lexical_scope) + [caller.name]
        while True:
            scoped = [
                info
                for info in candidates
                if list(info.lexical_scope) == effective_scope
                and not (info.class_name and not info.lexical_scope)
            ]
            if len(scoped) == 1:
                return scoped[0]
            if len(scoped) > 1:
                ambiguous = True
                break
            if not effective_scope:
                break
            effective_scope = effective_scope[:-1]
        if ambiguous:
            pass
        globals_only = [
            info
            for info in candidates
            if not info.lexical_scope
            and not (info.class_name and not info.lexical_scope)
            and info.path == caller.path
        ]
        if len(globals_only) == 1:
            return globals_only[0]
    if symbol_table is not None:
        if "." not in callee_key:
            if (caller_module, callee_key) in symbol_table.imports:
                fqn = symbol_table.resolve(caller_module, callee_key)
                if fqn is None:
                    return None
                if fqn in by_qual:
                    return by_qual[fqn]
            resolved = symbol_table.resolve_star(caller_module, callee_key)
            if resolved is not None and resolved in by_qual:
                return by_qual[resolved]
        else:
            parts = callee_key.split(".")
            base = parts[0]
            if base in ("self", "cls"):
                method = parts[-1]
                if caller.class_name:
                    candidate = f"{caller_module}.{caller.class_name}.{method}"
                    if candidate in by_qual:
                        return by_qual[candidate]
            elif len(parts) == 2:
                candidate = f"{caller_module}.{base}.{parts[1]}"
                if candidate in by_qual:
                    return by_qual[candidate]
            if (caller_module, base) in symbol_table.imports:
                base_fqn = symbol_table.resolve(caller_module, base)
                if base_fqn is None:
                    return None
                candidate = base_fqn + "." + ".".join(parts[1:])
                if candidate in by_qual:
                    return by_qual[candidate]
    # Exact qualified name match.
    if callee_key in by_qual:
        return by_qual[callee_key]
    if class_index is not None and "." in callee_key:
        parts = callee_key.split(".")
        if len(parts) >= 2:
            method = parts[-1]
            class_part = ".".join(parts[:-1])
            if class_part in {"self", "cls"} and caller.class_name:
                class_candidates = _resolve_class_candidates(
                    caller.class_name,
                    module=caller_module,
                    symbol_table=symbol_table,
                    class_index=class_index,
                )
            else:
                class_candidates = _resolve_class_candidates(
                    class_part,
                    module=caller_module,
                    symbol_table=symbol_table,
                    class_index=class_index,
                )
            for class_qual in class_candidates:
                resolved = _resolve_method_in_hierarchy(
                    class_qual,
                    method,
                    class_index=class_index,
                    by_qual=by_qual,
                    symbol_table=symbol_table,
                    seen=set(),
                )
                if resolved is not None:
                    return resolved
    return None


def _format_type_flow_site(
    *,
    caller: FunctionInfo,
    call: CallArgs,
    callee_info: FunctionInfo,
    caller_param: str,
    callee_param: str,
    annot: str,
    project_root: Path | None,
) -> str:
    """Format a stable, machine-actionable callsite for type-flow evidence."""
    caller_name = _function_key(caller.scope, caller.name)
    caller_path = _normalize_snapshot_path(caller.path, project_root)
    if call.span is None:
        loc = f"{caller_path}:{caller_name}"
    else:
        line, col, _, _ = call.span
        loc = f"{caller_path}:{line + 1}:{col + 1}"
    return (
        f"{loc}: {caller_name}.{caller_param} -> {callee_info.qual}.{callee_param} expects {annot}"
    )


def _infer_type_flow(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
    max_sites_per_param: int = 3,
) -> tuple[dict[str, dict[str, str | None]], list[str], list[str], list[str]]:
    """Repo-wide fixed-point pass for downstream type tightening + evidence."""
    by_name, by_qual = _build_function_index(
        paths,
        project_root,
        ignore_params,
        strictness,
        transparent_decorators,
    )
    symbol_table = _build_symbol_table(
        paths, project_root, external_filter=external_filter
    )
    class_index = _collect_class_index(paths, project_root)
    inferred: dict[str, dict[str, str | None]] = {}
    for infos in by_name.values():
        for info in infos:
            inferred[info.qual] = dict(info.annots)

    def _get_annot(info: FunctionInfo, param: str) -> str | None:
        return inferred.get(info.qual, {}).get(param)

    def _downstream_for(info: FunctionInfo) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
        downstream: dict[str, set[str]] = defaultdict(set)
        sites: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for call in info.calls:
            callee = _resolve_callee(
                call.callee,
                info,
                by_name,
                by_qual,
                symbol_table,
                project_root,
                class_index,
            )
            if callee is None or not callee.transparent:
                continue
            pos_params = (
                list(callee.positional_params)
                if callee.positional_params
                else list(callee.params)
            )
            kwonly_params = set(callee.kwonly_params or ())
            named_params = set(pos_params) | kwonly_params
            mapped_params: set[str] = set()
            callee_to_caller: dict[str, set[str]] = defaultdict(set)
            for pos_idx, caller_param in call.pos_map.items():
                idx = int(pos_idx)
                if idx < len(pos_params):
                    callee_param = pos_params[idx]
                elif callee.vararg is not None:
                    callee_param = callee.vararg
                else:
                    continue
                mapped_params.add(callee_param)
                callee_to_caller[callee_param].add(caller_param)
            for kw_name, caller_param in call.kw_map.items():
                if kw_name in named_params:
                    mapped_params.add(kw_name)
                    callee_to_caller[kw_name].add(caller_param)
                elif callee.kwarg is not None:
                    mapped_params.add(callee.kwarg)
                    callee_to_caller[callee.kwarg].add(caller_param)
            if strictness == "low":
                remaining = [p for p in sorted(named_params) if p not in mapped_params]
                if callee.vararg is not None and callee.vararg not in mapped_params:
                    remaining.append(callee.vararg)
                if callee.kwarg is not None and callee.kwarg not in mapped_params:
                    remaining.append(callee.kwarg)
                if len(call.star_pos) == 1:
                    _, star_param = call.star_pos[0]
                    for param in remaining:
                        callee_to_caller[param].add(star_param)
                if len(call.star_kw) == 1:
                    star_param = call.star_kw[0]
                    for param in remaining:
                        callee_to_caller[param].add(star_param)
            for callee_param, callers in callee_to_caller.items():
                annot = _get_annot(callee, callee_param)
                if not annot:
                    continue
                for caller_param in callers:
                    downstream[caller_param].add(annot)
                    sites[caller_param][annot].add(
                        _format_type_flow_site(
                            caller=info,
                            call=call,
                            callee_info=callee,
                            caller_param=caller_param,
                            callee_param=callee_param,
                            annot=annot,
                            project_root=project_root,
                        )
                    )
        return downstream, sites

    # Fixed-point inference pass.
    changed = True
    while changed:
        changed = False
        for infos in by_name.values():
            for info in infos:
                if _is_test_path(info.path):
                    continue
                downstream, _ = _downstream_for(info)
                for param, annots in downstream.items():
                    if len(annots) != 1:
                        continue
                    downstream_annot = next(iter(annots))
                    current = _get_annot(info, param)
                    if _is_broad_type(current) and downstream_annot:
                        if inferred[info.qual].get(param) != downstream_annot:
                            inferred[info.qual][param] = downstream_annot
                            changed = True

    suggestions: set[str] = set()
    ambiguities: set[str] = set()
    evidence_lines: set[str] = set()
    for infos in by_name.values():
        for info in infos:
            if _is_test_path(info.path):
                continue
            downstream, sites = _downstream_for(info)
            fn_key = _function_key(info.scope, info.name)
            path_key = _normalize_snapshot_path(info.path, project_root)
            for param, annots in downstream.items():
                if len(annots) > 1:
                    ambiguities.add(
                        f"{path_key}:{fn_key}.{param} downstream types conflict: {sorted(annots)}"
                    )
                    for annot in sorted(annots):
                        for site in sorted(sites.get(param, {}).get(annot, set()))[
                            :max_sites_per_param
                        ]:
                            evidence_lines.add(site)
                    continue
                downstream_annot = next(iter(annots))
                original = info.annots.get(param)
                final = inferred.get(info.qual, {}).get(param)
                if _is_broad_type(original) and final == downstream_annot and downstream_annot:
                    suggestions.add(
                        f"{path_key}:{fn_key}.{param} can tighten to {downstream_annot}"
                    )
                    for site in sorted(
                        sites.get(param, {}).get(downstream_annot, set())
                    )[:max_sites_per_param]:
                        evidence_lines.add(site)
    return inferred, sorted(suggestions), sorted(ambiguities), sorted(evidence_lines)


def analyze_type_flow_repo_with_map(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
) -> tuple[dict[str, dict[str, str | None]], list[str], list[str]]:
    """Repo-wide fixed-point pass for downstream type tightening."""
    inferred, suggestions, ambiguities, _ = _infer_type_flow(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
    )
    return inferred, suggestions, ambiguities


def analyze_type_flow_repo_with_evidence(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
    max_sites_per_param: int = 3,
) -> tuple[list[str], list[str], list[str]]:
    _, suggestions, ambiguities, evidence = _infer_type_flow(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        max_sites_per_param=max_sites_per_param,
    )
    return suggestions, ambiguities, evidence


def analyze_type_flow_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    inferred, suggestions, ambiguities = analyze_type_flow_repo_with_map(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
    )
    return suggestions, ambiguities


def analyze_constant_flow_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
) -> list[str]:
    """Detect parameters that only receive a single constant value (non-test)."""
    details = _collect_constant_flow_details(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
    )
    smells: list[str] = []
    for detail in details:
        path_name = detail.path.name if isinstance(detail.path, Path) else str(detail.path)
        site_suffix = ""
        if detail.sites:
            sample = ", ".join(detail.sites[:3])
            site_suffix = f" (e.g. {sample})"
        smells.append(
            f"{path_name}:{detail.name}.{detail.param} only observed constant {detail.value} across {detail.count} non-test call(s){site_suffix}"
        )
    return sorted(smells)


@dataclass(frozen=True)
class ConstantFlowDetail:
    path: Path
    qual: str
    name: str
    param: str
    value: str
    count: int
    sites: tuple[str, ...] = ()


def _format_call_site(caller: FunctionInfo, call: CallArgs) -> str:
    """Render a stable, human-friendly call site identifier.

    Spans are stored 0-based; we report 1-based line/col for readability.
    """
    caller_name = _function_key(caller.scope, caller.name)
    span = call.span
    if span is None:
        return f"{caller.path.name}:{caller_name}"
    line, col, _, _ = span
    return f"{caller.path.name}:{line + 1}:{col + 1}:{caller_name}"


def _collect_constant_flow_details(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
) -> list[ConstantFlowDetail]:
    by_name, by_qual = _build_function_index(
        paths,
        project_root,
        ignore_params,
        strictness,
        transparent_decorators,
    )
    symbol_table = _build_symbol_table(
        paths, project_root, external_filter=external_filter
    )
    class_index = _collect_class_index(paths, project_root)
    const_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    non_const: dict[tuple[str, str], bool] = defaultdict(bool)
    call_counts: dict[tuple[str, str], int] = defaultdict(int)
    call_sites: dict[tuple[str, str], set[str]] = defaultdict(set)

    def _record_site(key: tuple[str, str], caller: FunctionInfo, call: CallArgs) -> None:
        call_sites[key].add(_format_call_site(caller, call))

    for infos in by_name.values():
        for info in infos:
            for call in info.calls:
                if call.is_test:
                    continue
                callee = _resolve_callee(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table,
                    project_root,
                    class_index,
                )
                if callee is None:
                    continue
                if not callee.transparent:
                    continue
                pos_params = (
                    list(callee.positional_params)
                    if callee.positional_params
                    else list(callee.params)
                )
                kwonly_params = set(callee.kwonly_params or ())
                named_params = set(pos_params) | kwonly_params
                mapped_params = set()
                for idx_str in call.pos_map:
                    idx = int(idx_str)
                    if idx < len(pos_params):
                        mapped_params.add(pos_params[idx])
                    elif callee.vararg is not None:
                        mapped_params.add(callee.vararg)
                for kw in call.kw_map:
                    if kw in named_params:
                        mapped_params.add(kw)
                remaining = [p for p in named_params if p not in mapped_params]

                for idx_str, value in call.const_pos.items():
                    idx = int(idx_str)
                    if idx < len(pos_params):
                        key = (callee.qual, pos_params[idx])
                        const_values[key].add(value)
                        call_counts[key] += 1
                        _record_site(key, info, call)
                    elif callee.vararg is not None:
                        non_const[(callee.qual, callee.vararg)] = True
                for idx_str in call.pos_map:
                    idx = int(idx_str)
                    if idx < len(pos_params):
                        key = (callee.qual, pos_params[idx])
                        non_const[key] = True
                        call_counts[key] += 1
                    elif callee.vararg is not None:
                        non_const[(callee.qual, callee.vararg)] = True
                for idx_str in call.non_const_pos:
                    idx = int(idx_str)
                    if idx < len(pos_params):
                        key = (callee.qual, pos_params[idx])
                        non_const[key] = True
                        call_counts[key] += 1
                    elif callee.vararg is not None:
                        non_const[(callee.qual, callee.vararg)] = True
                if strictness == "low":
                    if len(call.star_pos) == 1:
                        for param in remaining:
                            key = (callee.qual, param)
                            non_const[key] = True
                            call_counts[key] += 1
                        if callee.vararg is not None:
                            non_const[(callee.qual, callee.vararg)] = True

                for kw, value in call.const_kw.items():
                    if kw not in named_params:
                        continue
                    key = (callee.qual, kw)
                    const_values[key].add(value)
                    call_counts[key] += 1
                    _record_site(key, info, call)
                for kw in call.kw_map:
                    if kw not in named_params:
                        continue
                    key = (callee.qual, kw)
                    non_const[key] = True
                    call_counts[key] += 1
                for kw in call.non_const_kw:
                    if kw not in named_params:
                        continue
                    key = (callee.qual, kw)
                    non_const[key] = True
                    call_counts[key] += 1
                if strictness == "low":
                    if len(call.star_kw) == 1:
                        for param in remaining:
                            key = (callee.qual, param)
                            non_const[key] = True
                            call_counts[key] += 1
                        if callee.kwarg is not None:
                            non_const[(callee.qual, callee.kwarg)] = True

    details: list[ConstantFlowDetail] = []
    for key, values in const_values.items():
        if non_const.get(key):
            continue
        if len(values) != 1:
            continue
        qual, param = key
        info = by_qual.get(qual)
        path = info.path if info is not None else Path(qual)
        # Use the same scope-aware function key used elsewhere in the audit so
        # cross-artifact joins (e.g., deadness ↔ exception obligations) work.
        name = (
            _function_key(info.scope, info.name)
            if info is not None
            else qual.split(".")[-1]
        )
        count = call_counts.get(key, 0)
        details.append(
            ConstantFlowDetail(
                path=path,
                qual=qual,
                name=name,
                param=param,
                value=next(iter(values)),
                count=count,
                sites=tuple(sorted(call_sites.get(key, set()))),
            )
        )
    return sorted(details, key=lambda entry: (str(entry.path), entry.name, entry.param))


def analyze_deadness_flow_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
) -> list[JSONObject]:
    """Emit deadness witnesses based on constant-only parameter flows."""
    details = _collect_constant_flow_details(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
    )
    witnesses: list[JSONObject] = []
    for detail in details:
        path_value = _normalize_snapshot_path(detail.path, project_root)
        predicate = f"{detail.param} != {detail.value}"
        core = [
            f"observed constant {detail.value} across {detail.count} non-test call(s)"
        ]
        deadness_id = f"deadness:{path_value}:{detail.name}:{detail.param}:{detail.value}"
        witnesses.append(
            {
                "deadness_id": deadness_id,
                "path": path_value,
                "function": detail.name,
                "bundle": [detail.param],
                "environment": {detail.param: detail.value},
                "predicate": predicate,
                "core": core,
                "result": "UNREACHABLE",
                "call_sites": list(detail.sites[:10]),
                "projection": (
                    f"{detail.name}.{detail.param} constant {detail.value} across "
                    f"{detail.count} non-test call(s)"
                ),
            }
        )
    return sorted(
        witnesses,
        key=lambda entry: (
            str(entry.get("path", "")),
            str(entry.get("function", "")),
            ",".join(entry.get("bundle", [])),
            str(entry.get("predicate", "")),
        ),
    )


def _compute_knob_param_names(
    *,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable,
    project_root: Path | None,
    class_index: dict[str, ClassInfo],
    strictness: str,
) -> set[str]:
    const_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    non_const: dict[tuple[str, str], bool] = defaultdict(bool)
    explicit_passed: dict[tuple[str, str], bool] = defaultdict(bool)
    call_counts: dict[str, int] = defaultdict(int)
    for infos in by_name.values():
        for info in infos:
            for call in info.calls:
                if call.is_test:
                    continue
                callee = _resolve_callee(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table,
                    project_root,
                    class_index,
                )
                if callee is None or not callee.transparent:
                    continue
                call_counts[callee.qual] += 1
                pos_params = (
                    list(callee.positional_params)
                    if callee.positional_params
                    else list(callee.params)
                )
                kwonly_params = set(callee.kwonly_params or ())
                named_params = set(pos_params) | kwonly_params
                remaining = set(named_params)
                if callee.vararg is not None:
                    remaining.add(callee.vararg)
                if callee.kwarg is not None:
                    remaining.add(callee.kwarg)
                for idx_str, value in call.const_pos.items():
                    idx = int(idx_str)
                    if idx < len(pos_params):
                        param = pos_params[idx]
                        const_values[(callee.qual, param)].add(value)
                        explicit_passed[(callee.qual, param)] = True
                        remaining.discard(param)
                    elif callee.vararg is not None:
                        non_const[(callee.qual, callee.vararg)] = True
                        explicit_passed[(callee.qual, callee.vararg)] = True
                        remaining.discard(callee.vararg)
                for idx_str in call.pos_map:
                    idx = int(idx_str)
                    if idx < len(pos_params):
                        param = pos_params[idx]
                        non_const[(callee.qual, param)] = True
                        explicit_passed[(callee.qual, param)] = True
                        remaining.discard(param)
                    elif callee.vararg is not None:
                        non_const[(callee.qual, callee.vararg)] = True
                        explicit_passed[(callee.qual, callee.vararg)] = True
                        remaining.discard(callee.vararg)
                for idx_str in call.non_const_pos:
                    idx = int(idx_str)
                    if idx < len(pos_params):
                        param = pos_params[idx]
                        non_const[(callee.qual, param)] = True
                        explicit_passed[(callee.qual, param)] = True
                        remaining.discard(param)
                    elif callee.vararg is not None:
                        non_const[(callee.qual, callee.vararg)] = True
                        explicit_passed[(callee.qual, callee.vararg)] = True
                        remaining.discard(callee.vararg)
                for kw, value in call.const_kw.items():
                    if kw in named_params:
                        const_values[(callee.qual, kw)].add(value)
                        explicit_passed[(callee.qual, kw)] = True
                        remaining.discard(kw)
                    elif callee.kwarg is not None:
                        non_const[(callee.qual, callee.kwarg)] = True
                        explicit_passed[(callee.qual, callee.kwarg)] = True
                        remaining.discard(callee.kwarg)
                for kw in call.kw_map:
                    if kw in named_params:
                        non_const[(callee.qual, kw)] = True
                        explicit_passed[(callee.qual, kw)] = True
                        remaining.discard(kw)
                    elif callee.kwarg is not None:
                        non_const[(callee.qual, callee.kwarg)] = True
                        explicit_passed[(callee.qual, callee.kwarg)] = True
                        remaining.discard(callee.kwarg)
                for kw in call.non_const_kw:
                    if kw in named_params:
                        non_const[(callee.qual, kw)] = True
                        explicit_passed[(callee.qual, kw)] = True
                        remaining.discard(kw)
                    elif callee.kwarg is not None:
                        non_const[(callee.qual, callee.kwarg)] = True
                        explicit_passed[(callee.qual, callee.kwarg)] = True
                        remaining.discard(callee.kwarg)
                if strictness == "low":
                    if len(call.star_pos) == 1:
                        for param in remaining:
                            non_const[(callee.qual, param)] = True
                            explicit_passed[(callee.qual, param)] = True
                    if len(call.star_kw) == 1:
                        for param in remaining:
                            non_const[(callee.qual, param)] = True
                            explicit_passed[(callee.qual, param)] = True
    knob_names: set[str] = set()
    for key, values in const_values.items():
        if non_const.get(key):
            continue
        if len(values) == 1:
            knob_names.add(key[1])
    for qual, info in by_qual.items():
        if call_counts.get(qual, 0) == 0:
            continue
        for param in info.defaults:
            if not explicit_passed.get((qual, param), False):
                knob_names.add(param)
    return knob_names


def analyze_unused_arg_flow_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: set[str] | None = None,
) -> list[str]:
    """Detect non-constant arguments passed into unused callee parameters."""
    by_name, by_qual = _build_function_index(
        paths,
        project_root,
        ignore_params,
        strictness,
        transparent_decorators,
    )
    symbol_table = _build_symbol_table(
        paths, project_root, external_filter=external_filter
    )
    class_index = _collect_class_index(paths, project_root)
    smells: set[str] = set()

    def _format(
        caller: FunctionInfo,
        callee_info: FunctionInfo,
        callee_param: str,
        arg_desc: str,
        *,
        call: CallArgs | None = None,
    ) -> str:
        # dataflow-bundle: callee_info, caller
        prefix = f"{caller.path.name}:{caller.name}"
        if call is not None and call.span is not None:
            line, col, _, _ = call.span
            prefix = f"{caller.path.name}:{line + 1}:{col + 1}:{caller.name}"
        return (
            f"{prefix} passes {arg_desc} "
            f"to unused {callee_info.path.name}:{callee_info.name}.{callee_param}"
        )

    for infos in by_name.values():
        for info in infos:
            for call in info.calls:
                if call.is_test:
                    continue
                callee = _resolve_callee(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table,
                    project_root,
                    class_index,
                )
                if callee is None:
                    continue
                if not callee.transparent:
                    continue
                if not callee.unused_params:
                    continue
                callee_params = callee.params
                mapped_params = set()
                for idx_str in call.pos_map:
                    idx = int(idx_str)
                    if idx >= len(callee_params):
                        continue
                    mapped_params.add(callee_params[idx])
                for kw in call.kw_map:
                    if kw in callee_params:
                        mapped_params.add(kw)
                remaining = [
                    (idx, name)
                    for idx, name in enumerate(callee_params)
                    if name not in mapped_params
                ]

                for idx_str, caller_param in call.pos_map.items():
                    idx = int(idx_str)
                    if idx >= len(callee_params):
                        continue
                    callee_param = callee_params[idx]
                    if callee_param in callee.unused_params:
                        smells.add(
                            _format(
                                info,
                                callee,
                                callee_param,
                                f"param {caller_param}",
                                call=call,
                            )
                        )
                for idx_str in call.non_const_pos:
                    idx = int(idx_str)
                    if idx >= len(callee_params):
                        continue
                    callee_param = callee_params[idx]
                    if callee_param in callee.unused_params:
                        smells.add(
                            _format(
                                info,
                                callee,
                                callee_param,
                                f"non-constant arg at position {idx}",
                                call=call,
                            )
                        )
                for kw, caller_param in call.kw_map.items():
                    if kw not in callee_params:
                        continue
                    if kw in callee.unused_params:
                        smells.add(
                            _format(
                                info,
                                callee,
                                kw,
                                f"param {caller_param}",
                                call=call,
                            )
                        )
                for kw in call.non_const_kw:
                    if kw not in callee_params:
                        continue
                    if kw in callee.unused_params:
                        smells.add(
                            _format(
                                info,
                                callee,
                                kw,
                                f"non-constant kw '{kw}'",
                                call=call,
                            )
                        )
                if strictness == "low":
                    if len(call.star_pos) == 1:
                        for idx, param in remaining:
                            if param in callee.unused_params:
                                smells.add(
                                    _format(
                                        info,
                                        callee,
                                        param,
                                        f"non-constant arg at position {idx}",
                                        call=call,
                                    )
                                )
                    if len(call.star_kw) == 1:
                        for _, param in remaining:
                            if param in callee.unused_params:
                                smells.add(
                                    _format(
                                        info,
                                        callee,
                                        param,
                                        f"non-constant kw '{param}'",
                                        call=call,
                                    )
                                )
    return sorted(smells)


def _iter_config_fields(path: Path) -> dict[str, set[str]]:
    """Best-effort extraction of config bundles from dataclasses."""
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return {}
    bundles: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {getattr(d, "id", None) for d in node.decorator_list}
        is_dataclass = "dataclass" in decorators
        is_config = node.name.endswith("Config")
        if not is_dataclass and not is_config:
            continue
        fields: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                if is_config or name.endswith("_fn"):
                    fields.add(name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if is_config or target.id.endswith("_fn"):
                            fields.add(target.id)
        if fields:
            bundles[node.name] = fields
    return bundles


def _collect_config_bundles(paths: list[Path]) -> dict[Path, dict[str, set[str]]]:
    bundles_by_path: dict[Path, dict[str, set[str]]] = {}
    for path in paths:
        bundles = _iter_config_fields(path)
        if bundles:
            bundles_by_path[path] = bundles
    return bundles_by_path


_BUNDLE_MARKER = re.compile(r"dataflow-bundle:\s*(.*)")


def _iter_documented_bundles(path: Path) -> set[tuple[str, ...]]:
    """Return bundles documented via '# dataflow-bundle: a, b' markers."""
    bundles: set[tuple[str, ...]] = set()
    try:
        text = path.read_text()
    except Exception:
        return bundles
    for line in text.splitlines():
        match = _BUNDLE_MARKER.search(line)
        if not match:
            continue
        payload = match.group(1)
        if not payload:
            continue
        parts = [p.strip() for p in re.split(r"[,\s]+", payload) if p.strip()]
        if len(parts) < 2:
            continue
        bundles.add(tuple(sorted(parts)))
    return bundles


def _collect_dataclass_registry(
    paths: list[Path],
    *,
    project_root: Path | None,
) -> dict[str, list[str]]:
    registry: dict[str, list[str]] = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        module = _module_name(path, project_root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            decorators = {
                ast.unparse(dec) if hasattr(ast, "unparse") else ""
                for dec in node.decorator_list
            }
            if not any("dataclass" in dec for dec in decorators):
                continue
            fields: list[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            fields.append(target.id)
            if not fields:
                continue
            if module:
                registry[f"{module}.{node.name}"] = fields
            else:  # pragma: no cover - module name is always non-empty for file paths
                registry[node.name] = fields
    return registry


def _bundle_name_registry(root: Path) -> dict[tuple[str, ...], set[str]]:
    file_paths = sorted(root.rglob("*.py"))
    config_bundles_by_path = _collect_config_bundles(file_paths)
    dataclass_registry = _collect_dataclass_registry(
        file_paths,
        project_root=root,
    )
    name_map: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for bundles in config_bundles_by_path.values():
        for name, fields in bundles.items():
            key = tuple(sorted(fields))
            if key:
                name_map[key].add(name)
    for qual_name, fields in dataclass_registry.items():
        key = tuple(sorted(fields))
        if key:
            name_map[key].add(qual_name.split(".")[-1])
    return name_map


def _iter_dataclass_call_bundles(
    path: Path,
    *,
    project_root: Path | None = None,
    symbol_table: SymbolTable | None = None,
    dataclass_registry: dict[str, list[str]] | None = None,
) -> set[tuple[str, ...]]:
    """Return bundles promoted via @dataclass constructor calls."""
    bundles: set[tuple[str, ...]] = set()
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return bundles
    module = _module_name(path, project_root)
    local_dataclasses: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {
            ast.unparse(dec) if hasattr(ast, "unparse") else ""
            for dec in node.decorator_list
        }
        if any("dataclass" in dec for dec in decorators):
            fields: list[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            fields.append(target.id)
            if fields:
                local_dataclasses[node.name] = fields
    if dataclass_registry is None:
        dataclass_registry = {}
        for name, fields in local_dataclasses.items():
            if module:
                dataclass_registry[f"{module}.{name}"] = fields
            else:  # pragma: no cover - module name is always non-empty for file paths
                dataclass_registry[name] = fields

    def _resolve_fields(call: ast.Call) -> list[str] | None:
        if isinstance(call.func, ast.Name):
            name = call.func.id
            if name in local_dataclasses:
                return local_dataclasses[name]
            if module:
                candidate = f"{module}.{name}"
                if candidate in dataclass_registry:
                    return dataclass_registry[candidate]
            if symbol_table is not None and module:
                resolved = symbol_table.resolve(module, name)
                if resolved in dataclass_registry:
                    return dataclass_registry[resolved]
                resolved_star = symbol_table.resolve_star(module, name)
                if resolved_star in dataclass_registry:
                    return dataclass_registry[resolved_star]
            if name in dataclass_registry:
                return dataclass_registry[name]
        if isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Name):
                base = call.func.value.id
                attr = call.func.attr
                if symbol_table is not None and module:
                    base_fqn = symbol_table.resolve(module, base)
                    if base_fqn:
                        candidate = f"{base_fqn}.{attr}"
                        if candidate in dataclass_registry:
                            return dataclass_registry[candidate]
                    base_star = symbol_table.resolve_star(module, base)
                    if base_star:
                        candidate = f"{base_star}.{attr}"
                        if candidate in dataclass_registry:
                            return dataclass_registry[candidate]
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fields = _resolve_fields(node)
        if not fields:
            continue
        names: list[str] = []
        ok = True
        for idx, arg in enumerate(node.args):
            if isinstance(arg, ast.Starred):
                ok = False
                break
            if idx < len(fields):
                names.append(fields[idx])
            else:
                ok = False
                break
        if not ok:
            continue
        for kw in node.keywords:
            if kw.arg is None:
                ok = False
                break
            names.append(kw.arg)
        if not ok or len(names) < 2:
            continue
        bundles.add(tuple(sorted(names)))
    return bundles


def _emit_dot(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> str:
    lines = [
        "digraph dataflow_grammar {",
        "  rankdir=LR;",
        "  node [fontsize=10];",
    ]
    for path, groups in groups_by_path.items():
        file_id = str(path).replace("/", "_").replace(".", "_")
        lines.append(f"  subgraph cluster_{file_id} {{")
        lines.append(f"    label=\"{path}\";")
        for fn, bundles in groups.items():
            if not bundles:
                continue
            fn_id = f"fn_{file_id}_{fn}"
            lines.append(f"    {fn_id} [shape=box,label=\"{fn}\"];")
            for idx, bundle in enumerate(bundles):
                bundle_id = f"b_{file_id}_{fn}_{idx}"
                label = ", ".join(sorted(bundle))
                lines.append(
                    f"    {bundle_id} [shape=ellipse,label=\"{label}\"];"
                )
                lines.append(f"    {fn_id} -> {bundle_id};")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _component_graph(groups_by_path: dict[Path, dict[str, list[set[str]]]]):
    nodes: dict[str, dict[str, str]] = {}
    adj: dict[str, set[str]] = defaultdict(set)
    bundle_map: dict[str, set[str]] = {}
    for path, groups in groups_by_path.items():
        file_id = str(path)
        for fn, bundles in groups.items():
            if not bundles:
                continue
            fn_id = f"fn::{file_id}::{fn}"
            nodes[fn_id] = {"kind": "fn", "label": f"{path.name}:{fn}"}
            for idx, bundle in enumerate(bundles):
                bundle_id = f"b::{file_id}::{fn}::{idx}"
                nodes[bundle_id] = {
                    "kind": "bundle",
                    "label": ", ".join(sorted(bundle)),
                }
                bundle_map[bundle_id] = bundle
                adj[fn_id].add(bundle_id)
                adj[bundle_id].add(fn_id)
    return nodes, adj, bundle_map


def _connected_components(nodes: dict[str, dict[str, str]], adj: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    comps: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        q: deque[str] = deque([node])
        seen.add(node)
        comp: list[str] = []
        while q:
            curr = q.popleft()
            comp.append(curr)
            for nxt in adj.get(curr, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(sorted(comp))
    return comps


def _render_mermaid_component(
    nodes: dict[str, dict[str, str]],
    bundle_map: dict[str, set[str]],
    adj: dict[str, set[str]],
    component: list[str],
    config_bundles_by_path: dict[Path, dict[str, set[str]]],
    documented_bundles_by_path: dict[Path, set[tuple[str, ...]]],
) -> tuple[str, str]:
    # dataflow-bundle: adj, config_bundles_by_path, documented_bundles_by_path, nodes
    lines = ["```mermaid", "flowchart LR"]
    fn_nodes = [n for n in component if nodes[n]["kind"] == "fn"]
    bundle_nodes = [n for n in component if nodes[n]["kind"] == "bundle"]
    for n in fn_nodes:
        label = nodes[n]["label"].replace('"', "'")
        lines.append(f'  {abs(hash(n))}["{label}"]')
    for n in bundle_nodes:
        label = nodes[n]["label"].replace('"', "'")
        lines.append(f'  {abs(hash(n))}(({label}))')
    for n in component:
        for nxt in adj.get(n, ()):
            if nxt in component and nodes[n]["kind"] == "fn":
                lines.append(f"  {abs(hash(n))} --> {abs(hash(nxt))}")
    lines.append("  classDef fn fill:#cfe8ff,stroke:#2b6cb0,stroke-width:1px;")
    lines.append("  classDef bundle fill:#ffe9c6,stroke:#c05621,stroke-width:1px;")
    if fn_nodes:
        lines.append(
            "  class "
            + ",".join(str(abs(hash(n))) for n in fn_nodes)
            + " fn;"
        )
    if bundle_nodes:
        lines.append(
            "  class "
            + ",".join(str(abs(hash(n))) for n in bundle_nodes)
            + " bundle;"
        )
    lines.append("```")

    observed = [bundle_map[n] for n in bundle_nodes if n in bundle_map]
    bundle_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for bundle in observed:
        bundle_counts[tuple(sorted(bundle))] += 1
    component_paths: set[Path] = set()
    for n in fn_nodes:
        parts = n.split("::", 2)
        if len(parts) == 3:
            component_paths.add(Path(parts[1]))
    declared_global = set()
    for bundles in config_bundles_by_path.values():
        for fields in bundles.values():
            declared_global.add(tuple(sorted(fields)))
    declared_local = set()
    documented = set()
    for path in component_paths:
        bundles = config_bundles_by_path.get(path)
        if bundles:
            for fields in bundles.values():
                declared_local.add(tuple(sorted(fields)))
        documented |= documented_bundles_by_path.get(path, set())
    observed_norm = {tuple(sorted(b)) for b in observed}
    observed_only = (
        sorted(observed_norm - declared_global)
        if declared_global
        else sorted(observed_norm)
    )
    declared_only = sorted(declared_local - observed_norm)
    documented_only = sorted(observed_norm & documented)
    def _tier(bundle: tuple[str, ...]) -> str:
        count = bundle_counts.get(bundle, 1)
        if count > 1:
            return "tier-2"
        return "tier-3"
    summary_lines = [
        f"Functions: {len(fn_nodes)}",
        f"Observed bundles: {len(observed_norm)}",
    ]
    if not declared_local:
        summary_lines.append("Declared Config bundles: none found for this component.")
    if observed_only:
        summary_lines.append("Observed-only bundles (not declared in Configs):")
        for bundle in observed_only:
            tier = _tier(bundle)
            documented_flag = "documented" if bundle in documented else "undocumented"
            summary_lines.append(
                f"  - {', '.join(bundle)} ({tier}, {documented_flag})"
            )
    if documented_only:
        summary_lines.append(
            "Documented bundles (dataflow-bundle markers or local dataclass calls):"
        )
        summary_lines.extend(f"  - {', '.join(bundle)}" for bundle in documented_only)
    if declared_only:
        summary_lines.append("Declared Config bundles not observed in this component:")
        summary_lines.extend(f"  - {', '.join(bundle)}" for bundle in declared_only)
    summary = "\n".join(summary_lines)
    return "\n".join(lines), summary


def _render_component_callsite_evidence(
    *,
    component: list[str],
    nodes: dict[str, dict[str, str]],
    bundle_map: dict[str, set[str]],
    documented_bundles_by_path: dict[Path, set[tuple[str, ...]]],
    declared_global: set[tuple[str, ...]],
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]],
    root: Path,
    max_sites_per_bundle: int = 5,
) -> list[str]:
    """Render machine-actionable callsite evidence for undocumented bundles in a component.

    This assumes internal IDs and evidence payloads are well-formed; drift should
    fail loudly to force reification rather than silently degrade fidelity.
    """
    fn_nodes = [n for n in component if nodes[n]["kind"] == "fn"]
    bundle_nodes = [n for n in component if nodes[n]["kind"] == "bundle"]
    component_paths: set[Path] = set()
    for n in fn_nodes:
        parts = n.split("::", 2)
        if len(parts) == 3:
            component_paths.add(Path(parts[1]))
    documented: set[tuple[str, ...]] = set()
    for path in component_paths:
        documented |= documented_bundles_by_path.get(path, set())

    bundle_counts: dict[tuple[str, ...], int] = defaultdict(int)
    bundle_key_by_node: dict[str, tuple[str, ...]] = {}
    for n in bundle_nodes:
        key = tuple(sorted(bundle_map[n]))
        bundle_key_by_node[n] = key
        bundle_counts[key] += 1

    # Keep output deterministic and review-friendly.
    ordered_nodes = sorted(
        bundle_key_by_node,
        key=lambda node_id: (
            node_id.split("::", 3)[1:],
            bundle_key_by_node.get(node_id, ()),
        ),
    )

    lines: list[str] = []
    for bundle_id in ordered_nodes:
        bundle_key = bundle_key_by_node[bundle_id]
        observed_only = (not declared_global) or (bundle_key not in declared_global)
        if not observed_only or bundle_key in documented:
            continue
        tier = "tier-2" if bundle_counts.get(bundle_key, 1) > 1 else "tier-3"

        _, file_id, fn_name, idx_str = bundle_id.split("::", 3)
        bundle_idx = int(idx_str)
        path = Path(file_id)
        fn_sites = bundle_sites_by_path[path][fn_name]
        for site in fn_sites[bundle_idx][:max_sites_per_bundle]:
            start_line, start_col, end_line, end_col = site["span"]
            loc = f"{start_line + 1}:{start_col + 1}-{end_line + 1}:{end_col + 1}"
            rel = _normalize_snapshot_path(path, root)
            callee = str(site.get("callee") or "")
            params = ", ".join(site.get("params") or [])
            slots = ", ".join(site.get("slots") or [])
            bundle_label = ", ".join(bundle_key)
            lines.append(
                f"{rel}:{loc}: {fn_name} -> {callee} forwards {params} "
                f"({tier}, undocumented bundle: {bundle_label}; slots: {slots})"
            )
    return lines


def _emit_report(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]] | None = None,
    type_suggestions: list[str] | None = None,
    type_ambiguities: list[str] | None = None,
    type_callsite_evidence: list[str] | None = None,
    constant_smells: list[str] | None = None,
    unused_arg_smells: list[str] | None = None,
    deadness_witnesses: list[JSONObject] | None = None,
    coherence_witnesses: list[JSONObject] | None = None,
    rewrite_plans: list[JSONObject] | None = None,
    exception_obligations: list[JSONObject] | None = None,
    handledness_witnesses: list[JSONObject] | None = None,
    decision_surfaces: list[str] | None = None,
    value_decision_surfaces: list[str] | None = None,
    decision_warnings: list[str] | None = None,
    fingerprint_warnings: list[str] | None = None,
    fingerprint_matches: list[str] | None = None,
    fingerprint_synth: list[str] | None = None,
    fingerprint_provenance: list[JSONObject] | None = None,
    context_suggestions: list[str] | None = None,
    invariant_propositions: list[InvariantProposition] | None = None,
    value_decision_rewrites: list[str] | None = None,
) -> tuple[str, list[str]]:
    nodes, adj, bundle_map = _component_graph(groups_by_path)
    components = _connected_components(nodes, adj)
    if groups_by_path:
        common = os.path.commonpath([str(p) for p in groups_by_path])
        root = Path(common)
    else:
        root = Path(".")
    # Use the analyzed file set (not a repo-wide rglob) so reports and schema
    # audits don't accidentally ingest virtualenvs or unrelated files.
    file_paths = sorted(groups_by_path) if groups_by_path else []
    config_bundles_by_path = _collect_config_bundles(file_paths)
    declared_global: set[tuple[str, ...]] = set()
    for bundles in config_bundles_by_path.values():
        for fields in bundles.values():
            declared_global.add(tuple(sorted(fields)))
    documented_bundles_by_path = {}
    symbol_table = _build_symbol_table(
        file_paths,
        root,
        external_filter=True,
    )
    dataclass_registry = _collect_dataclass_registry(
        file_paths,
        project_root=root,
    )
    for path in file_paths:
        documented = _iter_documented_bundles(path)
        promoted = _iter_dataclass_call_bundles(
            path,
            project_root=root,
            symbol_table=symbol_table,
            dataclass_registry=dataclass_registry,
        )
        documented_bundles_by_path[path] = documented | promoted
    lines = [
        "<!-- dataflow-grammar -->",
        "Dataflow grammar audit (observed forwarding bundles).",
        "",
    ]
    violations: list[str] = []
    if not components:
        lines.append("No bundle components detected.")
    else:
        if len(components) > max_components:
            lines.append(
                f"Showing top {max_components} components of {len(components)}."
            )
        for idx, comp in enumerate(components[:max_components], start=1):
            lines.append(f"### Component {idx}")
            mermaid, summary = _render_mermaid_component(
                nodes,
                bundle_map,
                adj,
                comp,
                config_bundles_by_path,
                documented_bundles_by_path,
            )
            lines.append(mermaid)
            lines.append("")
            lines.append("Summary:")
            lines.append("```")
            lines.append(summary)
            lines.append("```")
            lines.append("")
            if bundle_sites_by_path:
                evidence = _render_component_callsite_evidence(
                    component=comp,
                    nodes=nodes,
                    bundle_map=bundle_map,
                    documented_bundles_by_path=documented_bundles_by_path,
                    declared_global=declared_global,
                    bundle_sites_by_path=bundle_sites_by_path,
                    root=root,
                )
                if evidence:
                    lines.append("Callsite evidence (undocumented bundles):")
                    lines.append("```")
                    lines.extend(evidence)
                    lines.append("```")
                    lines.append("")
            for line in summary.splitlines():
                # Violation strings are semantic objects; avoid leaking markdown
                # bullets into baseline keys.
                candidate = line.strip()
                if candidate.startswith("- "):
                    candidate = candidate[2:].strip()
                if "(tier-3, undocumented)" in candidate:
                    violations.append(candidate)
                if "(tier-1," in candidate or "(tier-2," in candidate:
                    if "undocumented" in candidate:
                        violations.append(candidate)
    if violations:
        lines.append("Violations:")
        lines.append("```")
        lines.extend(violations)
        lines.append("```")
    if type_suggestions or type_ambiguities:
        lines.append("Type-flow audit:")
        if type_suggestions or type_ambiguities:
            lines.append(_render_type_mermaid(type_suggestions or [], type_ambiguities or []))
        if type_suggestions:
            lines.append("Type tightening candidates:")
            lines.append("```")
            lines.extend(type_suggestions)
            lines.append("```")
        if type_ambiguities:
            lines.append("Type ambiguities (conflicting downstream expectations):")
            lines.append("```")
            lines.extend(type_ambiguities)
            lines.append("```")
        if type_callsite_evidence:
            lines.append("Type-flow callsite evidence:")
            lines.append("```")
            lines.extend(type_callsite_evidence)
            lines.append("```")
    if constant_smells:
        lines.append("Constant-propagation smells (non-test call sites):")
        lines.append("```")
        lines.extend(constant_smells)
        lines.append("```")
    if unused_arg_smells:
        lines.append("Unused-argument smells (non-test call sites):")
        lines.append("```")
        lines.extend(unused_arg_smells)
        lines.append("```")
    if deadness_witnesses:
        summary = _summarize_deadness_witnesses(deadness_witnesses)
        if summary:
            lines.append("Deadness evidence:")
            lines.append("```")
            lines.extend(summary)
            lines.append("```")
    if coherence_witnesses:
        summary = _summarize_coherence_witnesses(coherence_witnesses)
        if summary:
            lines.append("Coherence evidence:")
            lines.append("```")
            lines.extend(summary)
            lines.append("```")
    if rewrite_plans:
        summary = _summarize_rewrite_plans(rewrite_plans)
        if summary:
            lines.append("Rewrite plans:")
            lines.append("```")
            lines.extend(summary)
            lines.append("```")
    if exception_obligations:
        summary = _summarize_exception_obligations(exception_obligations)
        if summary:
            lines.append("Exception obligations:")
            lines.append("```")
            lines.extend(summary)
            lines.append("```")
        protocol_evidence = _exception_protocol_evidence(exception_obligations)
        if protocol_evidence:
            lines.append("Exception protocol evidence:")
            lines.append("```")
            lines.extend(protocol_evidence)
            lines.append("```")
        protocol_warnings = _exception_protocol_warnings(exception_obligations)
        if protocol_warnings:
            lines.append("Exception protocol violations:")
            lines.append("```")
            lines.extend(protocol_warnings)
            lines.append("```")
            violations.extend(protocol_warnings)
    if handledness_witnesses:
        summary = _summarize_handledness_witnesses(handledness_witnesses)
        if summary:
            lines.append("Handledness evidence:")
            lines.append("```")
            lines.extend(summary)
            lines.append("```")
    if decision_surfaces:
        lines.append("Decision surface candidates (direct param use in conditionals):")
        lines.append("```")
        lines.extend(decision_surfaces)
        lines.append("```")
    if value_decision_surfaces:
        lines.append("Value-encoded decision surface candidates (branchless control):")
        lines.append("```")
        lines.extend(value_decision_surfaces)
        lines.append("```")
    if value_decision_rewrites:
        lines.append("Value-encoded decision rebranch suggestions:")
        lines.append("```")
        lines.extend(value_decision_rewrites)
        lines.append("```")
    if decision_warnings:
        lines.append("Decision tier warnings:")
        lines.append("```")
        lines.extend(decision_warnings)
        lines.append("```")
        violations.extend(decision_warnings)
    if fingerprint_warnings:
        lines.append("Fingerprint warnings:")
        lines.append("```")
        lines.extend(fingerprint_warnings)
        lines.append("```")
        violations.extend(fingerprint_warnings)
    if fingerprint_matches:
        lines.append("Fingerprint matches:")
        lines.append("```")
        lines.extend(fingerprint_matches)
        lines.append("```")
    if fingerprint_synth:
        lines.append("Fingerprint synthesis:")
        lines.append("```")
        lines.extend(fingerprint_synth)
        lines.append("```")
    if fingerprint_provenance:
        provenance_summary = _summarize_fingerprint_provenance(fingerprint_provenance)
        if provenance_summary:
            lines.append("Packed derivation view (ASPF provenance):")
            lines.append("```")
            lines.extend(provenance_summary)
            lines.append("```")
    if invariant_propositions:
        lines.append("Invariant propositions:")
        lines.append("```")
        lines.extend(_format_invariant_propositions(invariant_propositions))
        lines.append("```")
    if context_suggestions:
        lines.append("Contextvar/ambient rewrite suggestions:")
        lines.append("```")
        lines.extend(context_suggestions)
        lines.append("```")
    schema_surfaces = find_anonymous_schema_surfaces(file_paths, project_root=root)
    if schema_surfaces:
        lines.append("Anonymous schema surfaces (dict[str, object] payloads):")
        lines.append("```")
        lines.extend(surface.format() for surface in schema_surfaces[:50])
        if len(schema_surfaces) > 50:
            lines.append(f"... {len(schema_surfaces) - 50} more")
        lines.append("```")
    return "\n".join(lines), violations


def _infer_root(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> Path:
    if groups_by_path:
        common = os.path.commonpath([str(p) for p in groups_by_path])
        return Path(common)
    return Path(".")


def _normalize_snapshot_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def load_structure_snapshot(path: Path) -> JSONObject:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid snapshot JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Snapshot must be a JSON object: {path}")
    return data


def compute_structure_metrics(
    groups_by_path: dict[Path, dict[str, list[set[str]]]]
) -> JSONObject:
    file_count = len(groups_by_path)
    function_count = sum(len(groups) for groups in groups_by_path.values())
    bundle_sizes: list[int] = []
    for groups in groups_by_path.values():
        for bundles in groups.values():
            for bundle in bundles:
                bundle_sizes.append(len(bundle))
    bundle_count = len(bundle_sizes)
    mean_bundle_size = (sum(bundle_sizes) / bundle_count) if bundle_count else 0.0
    max_bundle_size = max(bundle_sizes) if bundle_sizes else 0
    size_histogram: dict[int, int] = defaultdict(int)
    for size in bundle_sizes:
        size_histogram[size] += 1
    return {
        "files": file_count,
        "functions": function_count,
        "bundles": bundle_count,
        "mean_bundle_size": mean_bundle_size,
        "max_bundle_size": max_bundle_size,
        # JSON object keys are strings; use explicit conversion for stability.
        "bundle_size_histogram": {
            str(size): count for size, count in sorted(size_histogram.items())
        },
    }


def render_structure_snapshot(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    project_root: Path | None = None,
    invariant_propositions: list[InvariantProposition] | None = None,
) -> JSONObject:
    root = project_root or _infer_root(groups_by_path)
    invariant_map: dict[tuple[str, str], list[InvariantProposition]] = {}
    if invariant_propositions:
        for prop in invariant_propositions:
            if not prop.scope or ":" not in prop.scope:
                continue
            scope_path, fn_name = prop.scope.rsplit(":", 1)
            invariant_map.setdefault((scope_path, fn_name), []).append(prop)
    files: list[JSONObject] = []
    for path in sorted(
        groups_by_path, key=lambda p: _normalize_snapshot_path(p, root)
    ):
        groups = groups_by_path[path]
        functions: list[JSONObject] = []
        path_key = _normalize_snapshot_path(path, root)
        for fn_name in sorted(groups):
            bundles = groups[fn_name]
            normalized = [sorted(bundle) for bundle in bundles]
            normalized.sort(key=lambda bundle: (len(bundle), bundle))
            entry: JSONObject = {"name": fn_name, "bundles": normalized}
            invariants = invariant_map.get((path_key, fn_name))
            if invariants:
                entry["invariants"] = [
                    prop.as_dict()
                    for prop in sorted(
                        invariants,
                        key=lambda prop: (
                            prop.form,
                            prop.terms,
                            prop.source or "",
                            prop.scope or "",
                        ),
                    )
                ]
            functions.append(entry)
        files.append({"path": _normalize_snapshot_path(path, root), "functions": functions})
    return {
        "format_version": 1,
        "root": str(root) if root is not None else None,
        "files": files,
    }


# dataflow-bundle: decision_surfaces, value_decision_surfaces
def render_decision_snapshot(
    *,
    decision_surfaces: list[str],
    value_decision_surfaces: list[str],
    project_root: Path | None = None,
) -> JSONObject:
    return {
        "format_version": 1,
        "root": str(project_root) if project_root is not None else None,
        "decision_surfaces": sorted(decision_surfaces),
        "value_decision_surfaces": sorted(value_decision_surfaces),
        "summary": {
            "decision_surfaces": len(decision_surfaces),
            "value_decision_surfaces": len(value_decision_surfaces),
        },
    }


def load_decision_snapshot(path: Path) -> JSONObject:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid decision snapshot JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Decision snapshot must be a JSON object: {path}")
    return data


def diff_decision_snapshots(
    baseline_snapshot: JSONObject,
    current_snapshot: JSONObject,
) -> JSONObject:
    base_decisions = set(baseline_snapshot.get("decision_surfaces") or [])
    curr_decisions = set(current_snapshot.get("decision_surfaces") or [])
    base_value = set(baseline_snapshot.get("value_decision_surfaces") or [])
    curr_value = set(current_snapshot.get("value_decision_surfaces") or [])
    return {
        "format_version": 1,
        "baseline_root": baseline_snapshot.get("root"),
        "current_root": current_snapshot.get("root"),
        "decision_surfaces": {
            "added": sorted(curr_decisions - base_decisions),
            "removed": sorted(base_decisions - curr_decisions),
        },
        "value_decision_surfaces": {
            "added": sorted(curr_value - base_value),
            "removed": sorted(base_value - curr_value),
        },
    }


def _bundle_counts_from_snapshot(snapshot: JSONObject) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    files = snapshot.get("files") or []
    for file_entry in files:
        if not isinstance(file_entry, dict):
            continue
        functions = file_entry.get("functions") or []
        for fn_entry in functions:
            if not isinstance(fn_entry, dict):
                continue
            bundles = fn_entry.get("bundles") or []
            for bundle in bundles:
                if not isinstance(bundle, list):
                    continue
                counts[tuple(bundle)] += 1
    return counts


# dataflow-bundle: baseline_snapshot, current_snapshot
def diff_structure_snapshots(
    baseline_snapshot: JSONObject,
    current_snapshot: JSONObject,
) -> JSONObject:
    baseline_counts = _bundle_counts_from_snapshot(baseline_snapshot)
    current_counts = _bundle_counts_from_snapshot(current_snapshot)
    all_bundles = sorted(
        set(baseline_counts) | set(current_counts),
        key=lambda bundle: (len(bundle), list(bundle)),
    )
    added: list[JSONObject] = []
    removed: list[JSONObject] = []
    changed: list[JSONObject] = []
    for bundle in all_bundles:
        before = baseline_counts.get(bundle, 0)
        after = current_counts.get(bundle, 0)
        entry = {
            "bundle": list(bundle),
            "before": before,
            "after": after,
            "delta": after - before,
        }
        if before == 0:
            added.append(entry)
        elif after == 0:
            removed.append(entry)
        elif before != after:
            changed.append(entry)
    return {
        "format_version": 1,
        "baseline_root": baseline.get("root"),
        "current_root": current.get("root"),
        "added": added,
        "removed": removed,
        "changed": changed,
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "baseline_total": sum(baseline_counts.values()),
            "current_total": sum(current_counts.values()),
        },
    }


def diff_structure_snapshot_files(
    baseline_path: Path,
    current_path: Path,
) -> JSONObject:
    # dataflow-bundle: baseline_path, current_path
    baseline = load_structure_snapshot(baseline_path)
    current = load_structure_snapshot(current_path)
    return diff_structure_snapshots(baseline, current)


def compute_structure_reuse(
    snapshot: JSONObject,
    *,
    min_count: int = 2,
    hash_fn: Callable[[str, object | None, list[str]], str] | None = None,
) -> JSONObject:
    if min_count < 2:
        min_count = 2
    files = snapshot.get("files") or []
    root_value = snapshot.get("root")
    root_path = Path(root_value) if isinstance(root_value, str) else None
    bundle_name_map: dict[tuple[str, ...], set[str]] = {}
    if root_path is not None and root_path.exists():
        bundle_name_map = _bundle_name_registry(root_path)
    reuse_map: dict[str, JSONObject] = {}
    warnings: list[str] = []

    def _hash_node(kind: str, value: object | None, child_hashes: list[str]) -> str:
        payload = {
            "kind": kind,
            "value": value,
            "children": sorted(child_hashes),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest

    hasher = hash_fn or _hash_node

    def _record(
        *,
        node_hash: str,
        kind: str,
        location: str,
        value: object | None = None,
        child_count: int | None = None,
    ) -> None:
        entry = reuse_map.get(node_hash)
        if entry is None:
            entry = {
                "hash": node_hash,
                "kind": kind,
                "count": 0,
                "locations": [],
            }
            if value is not None:
                entry["value"] = value
            if child_count is not None:
                entry["child_count"] = child_count
            reuse_map[node_hash] = entry
        entry["count"] += 1
        entry["locations"].append(location)

    file_hashes: list[str] = []
    for file_entry in files:
        if not isinstance(file_entry, dict):
            continue
        file_path = file_entry.get("path")
        if not isinstance(file_path, str):
            continue
        function_hashes: list[str] = []
        functions = file_entry.get("functions") or []
        for fn_entry in functions:
            if not isinstance(fn_entry, dict):
                continue
            fn_name = fn_entry.get("name")
            if not isinstance(fn_name, str):
                continue
            bundle_hashes: list[str] = []
            bundles = fn_entry.get("bundles") or []
            for bundle in bundles:
                if not isinstance(bundle, list):
                    continue
                normalized = tuple(sorted(str(item) for item in bundle))
                bundle_hash = hasher("bundle", normalized, [])
                bundle_hashes.append(bundle_hash)
                _record(
                    node_hash=bundle_hash,
                    kind="bundle",
                    location=f"{file_path}::{fn_name}::bundle:{','.join(normalized)}",
                    value=list(normalized),
                )
            fn_hash = hasher("function", None, bundle_hashes)
            function_hashes.append(fn_hash)
            _record(
                node_hash=fn_hash,
                kind="function",
                location=f"{file_path}::{fn_name}",
                child_count=len(bundle_hashes),
            )
        file_hash = hasher("file", None, function_hashes)
        file_hashes.append(file_hash)
        _record(node_hash=file_hash, kind="file", location=f"{file_path}")

    root_hash = hasher("root", None, file_hashes)
    _record(
        node_hash=root_hash,
        kind="root",
        location="root",
        child_count=len(file_hashes),
    )

    reused = [
        entry
        for entry in reuse_map.values()
        if isinstance(entry.get("count"), int) and entry["count"] >= min_count
    ]
    reused.sort(
        key=lambda entry: (
            entry.get("kind", ""),
            -int(entry.get("count", 0)),
            entry.get("hash", ""),
        )
    )
    suggested: list[JSONObject] = []
    replacement_map: dict[str, list[JSONObject]] = {}
    for entry in reused:
        kind = entry.get("kind")
        if kind not in {"bundle", "function"}:
            continue
        count = int(entry.get("count", 0))
        hash_value = entry.get("hash")
        if not isinstance(hash_value, str) or not hash_value:
            continue
        suggestion = {
            "hash": hash_value,
            "kind": kind,
            "count": count,
            "suggested_name": f"_gabion_{kind}_lemma_{hash_value[:8]}",
            "locations": entry.get("locations", []),
        }
        if "value" in entry:
            suggestion["value"] = entry.get("value")
        if "child_count" in entry:
            suggestion["child_count"] = entry.get("child_count")
        if kind == "bundle" and "value" in entry:
            value = entry.get("value")
            if isinstance(value, list):
                key = tuple(sorted(str(item) for item in value))
                name_candidates = bundle_name_map.get(key)
                if name_candidates:
                    sorted_names = sorted(name_candidates)
                    if len(sorted_names) == 1:
                        suggestion["suggested_name"] = sorted_names[0]
                        suggestion["name_source"] = "declared_bundle"
                    else:
                        suggestion["name_candidates"] = sorted_names
                else:
                    warnings.append(
                        f"Missing declared bundle name for {list(key)}"
                    )
        suggested.append(suggestion)
    replacement_map = _build_reuse_replacement_map(suggested)
    return {
        "format_version": 1,
        "min_count": min_count,
        "reused": reused,
        "suggested_lemmas": suggested,
        "replacement_map": replacement_map,
        "warnings": warnings,
    }


def _build_reuse_replacement_map(
    suggested: list[JSONObject],
) -> dict[str, list[JSONObject]]:
    replacement_map: dict[str, list[JSONObject]] = {}
    for suggestion in suggested:
        locations = suggestion.get("locations") or []
        if not isinstance(locations, list):
            continue
        for location in locations:
            if not isinstance(location, str):
                continue
            replacement_map.setdefault(location, []).append(
                {
                    "kind": suggestion.get("kind"),
                    "hash": suggestion.get("hash"),
                    "suggested_name": suggestion.get("suggested_name"),
                }
            )
    return replacement_map


def render_reuse_lemma_stubs(reuse: JSONObject) -> str:
    suggested = reuse.get("suggested_lemmas") or []
    lines = [
        "# Generated by gabion structure-reuse",
        "# TODO: replace stubs with actual lemma definitions.",
        "",
    ]
    if not suggested:
        lines.append("# No lemma suggestions available.")
        lines.append("")
        return "\n".join(lines)
    for entry in sorted(
        (e for e in suggested if isinstance(e, dict)),
        key=lambda e: (str(e.get("kind", "")), str(e.get("suggested_name", ""))),
    ):
        name = entry.get("suggested_name")
        if not isinstance(name, str) or not name:
            continue
        kind = entry.get("kind", "lemma")
        count = entry.get("count", 0)
        value = entry.get("value")
        child_count = entry.get("child_count")
        lines.append(f"def {name}() -> None:")
        lines.append('    """Auto-generated lemma stub."""')
        lines.append(f"    # kind: {kind}")
        lines.append(f"    # count: {count}")
        if value is not None:
            lines.append(f"    # value: {value}")
        if child_count is not None:
            lines.append(f"    # child_count: {child_count}")
        lines.append("    ...")
        lines.append("")
    return "\n".join(lines)


def _bundle_counts(
    groups_by_path: dict[Path, dict[str, list[set[str]]]]
) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for groups in groups_by_path.values():
        for bundles in groups.values():
            for bundle in bundles:
                counts[tuple(sorted(bundle))] += 1
    return counts


def _merge_counts_by_knobs(
    counts: dict[tuple[str, ...], int],
    knob_names: set[str],
) -> dict[tuple[str, ...], int]:
    if not knob_names:
        return counts
    bundles = [set(bundle) for bundle in counts]
    merged: dict[tuple[str, ...], int] = defaultdict(int)
    for bundle_key, count in counts.items():
        bundle = set(bundle_key)
        target = bundle
        for other in bundles:
            if bundle and bundle.issubset(other):
                extra = set(other) - bundle
                if extra and extra.issubset(knob_names):
                    if len(other) < len(target) or target == bundle:
                        target = set(other)
        merged[tuple(sorted(target))] += count
    return merged


def _collect_declared_bundles(root: Path) -> set[tuple[str, ...]]:
    declared: set[tuple[str, ...]] = set()
    file_paths = sorted(root.rglob("*.py"))
    bundles_by_path = _collect_config_bundles(file_paths)
    for bundles in bundles_by_path.values():
        for fields in bundles.values():
            declared.add(tuple(sorted(fields)))
    return declared


def build_synthesis_plan(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    project_root: Path | None = None,
    max_tier: int = 2,
    min_bundle_size: int = 2,
    allow_singletons: bool = False,
    merge_overlap_threshold: float | None = None,
    config: AuditConfig | None = None,
) -> JSONObject:
    audit_config = config or AuditConfig(
        project_root=project_root or _infer_root(groups_by_path)
    )
    root = project_root or audit_config.project_root or _infer_root(groups_by_path)
    path_list = list(groups_by_path.keys())
    by_name, by_qual = _build_function_index(
        path_list,
        root,
        audit_config.ignore_params,
        audit_config.strictness,
        audit_config.transparent_decorators,
    )
    symbol_table = _build_symbol_table(
        path_list,
        root,
        external_filter=audit_config.external_filter,
    )
    class_index = _collect_class_index(path_list, root)
    knob_names = _compute_knob_param_names(
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=root,
        class_index=class_index,
        strictness=audit_config.strictness,
    )
    counts = _bundle_counts(groups_by_path)
    counts = _merge_counts_by_knobs(counts, knob_names)
    if not counts:
        response = SynthesisResponse(
            protocols=[],
            warnings=["No bundles observed for synthesis."],
            errors=[],
        )
        return response.model_dump()

    declared = _collect_declared_bundles(root)
    bundle_tiers: dict[frozenset[str], int] = {}
    frequency: dict[str, int] = defaultdict(int)
    bundle_fields: set[str] = set()
    for bundle, count in counts.items():
        tier = 1 if bundle in declared else (2 if count > 1 else 3)
        bundle_tiers[frozenset(bundle)] = tier
        for field in bundle:
            frequency[field] += count
            bundle_fields.add(field)

    merged_bundle_tiers: dict[frozenset[str], int] = {}
    original_bundles = [set(bundle) for bundle in counts]
    synth_config = SynthesisConfig(
        max_tier=max_tier,
        min_bundle_size=min_bundle_size,
        allow_singletons=allow_singletons,
        merge_overlap_threshold=(
            merge_overlap_threshold
            if merge_overlap_threshold is not None
            else SynthesisConfig().merge_overlap_threshold
        ),
    )
    merged_bundles = merge_bundles(
        original_bundles, min_overlap=synth_config.merge_overlap_threshold
    )
    if merged_bundles:
        for merged in merged_bundles:
            members = [
                bundle
                for bundle in original_bundles
                if bundle and bundle.issubset(merged)
            ]
            if not members:
                continue
            tier = min(
                bundle_tiers[frozenset(member)] for member in members
            )
            merged_bundle_tiers[frozenset(merged)] = tier
        if merged_bundle_tiers:
            bundle_tiers = merged_bundle_tiers

    naming_context = NamingContext(frequency=dict(frequency))
    field_types: dict[str, str] = {}
    type_warnings: list[str] = []
    if bundle_fields:
        inferred, _, _ = analyze_type_flow_repo_with_map(
            path_list,
            project_root=root,
            ignore_params=audit_config.ignore_params,
            strictness=audit_config.strictness,
            external_filter=audit_config.external_filter,
            transparent_decorators=audit_config.transparent_decorators,
        )
        type_sets: dict[str, set[str]] = defaultdict(set)
        for annots in inferred.values():
            for name, annot in annots.items():
                if name not in bundle_fields or not annot:
                    continue
                type_sets[name].add(annot)
        for infos in by_name.values():
            for info in infos:
                for call in info.calls:
                    if call.is_test:
                        continue
                    callee = _resolve_callee(
                        call.callee,
                        info,
                        by_name,
                        by_qual,
                        symbol_table,
                        root,
                        class_index,
                    )
                    if callee is None or not callee.transparent:
                        continue
                    callee_params = callee.params
                    for idx_str, value in call.const_pos.items():
                        idx = int(idx_str)
                        if idx >= len(callee_params):
                            continue
                        param = callee_params[idx]
                        if param not in bundle_fields:
                            continue
                        hint = _type_from_const_repr(value)
                        if hint:
                            type_sets[param].add(hint)
                    for kw, value in call.const_kw.items():
                        if kw not in callee_params or kw not in bundle_fields:
                            continue
                        hint = _type_from_const_repr(value)
                        if hint:
                            type_sets[kw].add(hint)
        for name, types in type_sets.items():
            combined, conflicted = _combine_type_hints(types)
            field_types[name] = combined
            if conflicted and len(types) > 1:
                type_warnings.append(
                    f"Conflicting type hints for '{name}': {sorted(types)} -> {combined}"
                )
    plan = Synthesizer(config=synth_config).plan(
        bundle_tiers=bundle_tiers,
        field_types=field_types,
        naming_context=naming_context,
    )
    response = SynthesisResponse(
        protocols=[
            {
                "name": spec.name,
                "fields": [
                    {
                        "name": field.name,
                        "type_hint": field.type_hint,
                        "source_params": sorted(field.source_params),
                    }
                    for field in spec.fields
                ],
                "bundle": sorted(spec.bundle),
                "tier": spec.tier,
                "rationale": spec.rationale,
            }
            for spec in plan.protocols
        ],
        warnings=plan.warnings + type_warnings,
        errors=plan.errors,
    )
    return response.model_dump()


def render_synthesis_section(plan: JSONObject) -> str:
    protocols = plan.get("protocols", [])
    warnings = plan.get("warnings", [])
    errors = plan.get("errors", [])
    lines = ["", "## Synthesis plan (prototype)", ""]
    if not protocols:
        lines.append("No protocol candidates.")
    else:
        for spec in protocols:
            name = spec.get("name", "Bundle")
            tier = spec.get("tier", "?")
            fields = spec.get("fields", [])
            parts = []
            for field in fields:
                fname = field.get("name", "")
                type_hint = field.get("type_hint") or "Any"
                if fname:
                    parts.append(f"{fname}: {type_hint}")
            field_list = ", ".join(parts) if parts else "(no fields)"
            lines.append(f"- {name} (tier {tier}): {field_list}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.append("```")
        lines.extend(str(w) for w in warnings)
        lines.append("```")
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.append("```")
        lines.extend(str(e) for e in errors)
        lines.append("```")
    return "\n".join(lines)


def render_protocol_stubs(plan: JSONObject, kind: str = "dataclass") -> str:
    protocols = plan.get("protocols", [])
    if kind not in {"dataclass", "protocol"}:
        kind = "dataclass"
    typing_names = {"Any"}
    if kind == "protocol":
        typing_names.add("Protocol")
    for spec in protocols:
        for field in spec.get("fields", []) or []:
            hint = field.get("type_hint") or "Any"
            if "Optional[" in hint:
                typing_names.add("Optional")
            if "Union[" in hint:
                typing_names.add("Union")
    typing_import = ", ".join(sorted(typing_names))
    lines = [
        "# Auto-generated by gabion dataflow audit.",
        "from __future__ import annotations",
        "",
        f"from typing import {typing_import}",
        "",
    ]
    if kind == "dataclass":
        lines.insert(3, "from dataclasses import dataclass")
    if not protocols:
        lines.append("# No protocol candidates.")
        return "\n".join(lines)
    placeholder_base = "TODO_Name_Me"
    for idx, spec in enumerate(protocols, start=1):
        name = placeholder_base if idx == 1 else f"{placeholder_base}{idx}"
        suggested = spec.get("name", "Bundle")
        tier = spec.get("tier", "?")
        bundle = spec.get("bundle", [])
        rationale = spec.get("rationale", "")
        if kind == "dataclass":
            lines.append("@dataclass")
            lines.append(f"class {name}:")
        else:
            lines.append(f"class {name}(Protocol):")
        doc_lines = [
            "TODO: Rename this Protocol.",
            f"Suggested name: {suggested}",
            f"Tier: {tier}",
        ]
        if bundle:
            doc_lines.append(f"Bundle: {', '.join(bundle)}")
        if rationale:
            doc_lines.append(f"Rationale: {rationale}")
        fields = spec.get("fields", [])
        if fields:
            field_summary = []
            for field in fields:
                fname = field.get("name") or "field"
                type_hint = field.get("type_hint") or "Any"
                field_summary.append(f"{fname}: {type_hint}")
            doc_lines.append("Fields: " + ", ".join(field_summary))
        lines.append('    """')
        for line in doc_lines:
            lines.append(f"    {line}")
        lines.append('    """')
        if not fields:
            lines.append("    pass")
        else:
            for field in fields:
                fname = field.get("name") or "field"
                type_hint = field.get("type_hint") or "Any"
                lines.append(f"    {fname}: {type_hint}")
        lines.append("")
    return "\n".join(lines)


def build_refactor_plan(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    paths: list[Path],
    *,
    config: AuditConfig,
) -> JSONObject:
    file_paths = _iter_paths([str(p) for p in paths], config)
    if not file_paths:
        return {"bundles": [], "warnings": ["No files available for refactor plan."]}

    by_name, by_qual = _build_function_index(
        file_paths,
        config.project_root,
        config.ignore_params,
        config.strictness,
        config.transparent_decorators,
    )
    symbol_table = _build_symbol_table(
        file_paths, config.project_root, external_filter=config.external_filter
    )
    class_index = _collect_class_index(file_paths, config.project_root)
    info_by_path_name: dict[tuple[Path, str], FunctionInfo] = {}
    for infos in by_name.values():
        for info in infos:
            key = _function_key(info.scope, info.name)
            info_by_path_name[(info.path, key)] = info

    bundle_map: dict[tuple[str, ...], dict[str, FunctionInfo]] = defaultdict(dict)
    for path, groups in groups_by_path.items():
        for fn, bundles in groups.items():
            for bundle in bundles:
                key = tuple(sorted(bundle))
                info = info_by_path_name.get((path, fn))
                if info is not None:
                    bundle_map[key][info.qual] = info

    plans: list[JSONObject] = []
    for bundle, infos in sorted(bundle_map.items(), key=lambda item: (len(item[0]), item[0])):
        comp = dict(infos)
        deps: dict[str, set[str]] = {qual: set() for qual in comp}
        for info in infos.values():
            for call in info.calls:
                callee = _resolve_callee(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table,
                    config.project_root,
                    class_index,
                )
                if callee is None:
                    continue
                if not callee.transparent:
                    continue
                if callee.qual in comp:
                    deps[info.qual].add(callee.qual)
        schedule = topological_schedule(deps)
        plans.append(
            {
                "bundle": list(bundle),
                "functions": sorted(comp.keys()),
                "order": schedule.order,
                "cycles": [sorted(list(cycle)) for cycle in schedule.cycles],
            }
        )

    warnings: list[str] = []
    if not plans:
        warnings.append("No bundle components available for refactor plan.")
    return {"bundles": plans, "warnings": warnings}


def render_refactor_plan(plan: JSONObject) -> str:
    bundles = plan.get("bundles", [])
    warnings = plan.get("warnings", [])
    lines = ["", "## Refactoring plan (prototype)", ""]
    if not bundles:
        lines.append("No refactoring plan available.")
    else:
        for entry in bundles:
            bundle = entry.get("bundle", [])
            title = ", ".join(bundle) if bundle else "(unknown bundle)"
            lines.append(f"### Bundle: {title}")
            order = entry.get("order", [])
            if order:
                lines.append("Order (callee-first):")
                lines.append("```")
                for item in order:
                    lines.append(f"- {item}")
                lines.append("```")
            cycles = entry.get("cycles", [])
            if cycles:
                lines.append("Cycles:")
                lines.append("```")
                for cycle in cycles:
                    lines.append(", ".join(cycle))
                lines.append("```")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.append("```")
        lines.extend(str(w) for w in warnings)
        lines.append("```")
    return "\n".join(lines)


def _render_type_mermaid(
    suggestions: list[str],
    ambiguities: list[str],
) -> str:
    lines = ["```mermaid", "flowchart LR"]
    node_id = 0
    def _node(label: str) -> str:
        nonlocal node_id
        node_id += 1
        node = f"type_{node_id}"
        safe = label.replace('"', "'")
        lines.append(f'  {node}["{safe}"]')
        return node

    for entry in suggestions:
        # Format: file:func.param can tighten to Type
        if " can tighten to " not in entry:
            continue
        lhs, rhs = entry.split(" can tighten to ", 1)
        src = _node(lhs)
        dst = _node(rhs)
        lines.append(f"  {src} --> {dst}")
    for entry in ambiguities:
        if " downstream types conflict: " not in entry:
            continue
        lhs, rhs = entry.split(" downstream types conflict: ", 1)
        src = _node(lhs)
        # rhs is a repr of list; keep as string nodes per type
        rhs = rhs.strip()
        if rhs.startswith("[") and rhs.endswith("]"):
            rhs = rhs[1:-1]
        type_names = []
        for item in rhs.split(","):
            item = item.strip()
            if not item:
                continue
            item = item.strip("'\"")
            type_names.append(item)
        for type_name in type_names:
            dst = _node(type_name)
            lines.append(f"  {src} -.-> {dst}")
    lines.append("```")
    return "\n".join(lines)


def _compute_violations(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    type_suggestions: list[str] | None = None,
    type_ambiguities: list[str] | None = None,
    decision_warnings: list[str] | None = None,
    fingerprint_warnings: list[str] | None = None,
) -> list[str]:
    _, violations = _emit_report(
        groups_by_path,
        max_components,
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
        constant_smells=[],
        unused_arg_smells=[],
        decision_surfaces=[],
        value_decision_surfaces=[],
        decision_warnings=decision_warnings,
        fingerprint_warnings=fingerprint_warnings,
        context_suggestions=[],
    )
    return violations


def _resolve_baseline_path(path: str | None, root: Path) -> Path | None:
    if not path:
        return None
    baseline = Path(path)
    if not baseline.is_absolute():
        baseline = root / baseline
    return baseline


def _resolve_synth_registry_path(path: str | None, root: Path) -> Path | None:
    if not path:
        return None
    value = str(path).strip()
    if not value:
        return None
    if value.endswith("/LATEST/fingerprint_synth.json"):
        marker = Path(root) / value.replace(
            "/LATEST/fingerprint_synth.json", "/LATEST.txt"
        )
        try:
            stamp = marker.read_text().strip()
        except Exception:
            return None
        return (marker.parent / stamp / "fingerprint_synth.json").resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        raw = path.read_text()
    except OSError:
        return set()
    entries: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _write_baseline(path: Path, violations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(violations))
    header = [
        "# gabion baseline (ratchet)",
        "# Lines list known violations to allow; new ones should fail.",
        "",
    ]
    path.write_text("\n".join(header + unique) + "\n")


def _apply_baseline(
    violations: list[str], baseline_allowlist: set[str]
) -> tuple[list[str], list[str]]:
    if not baseline_allowlist:
        return violations, []
    new = [line for line in violations if line not in baseline_allowlist]
    suppressed = [line for line in violations if line in baseline_allowlist]
    return new, suppressed


def resolve_baseline_path(path: str | None, root: Path) -> Path | None:
    return _resolve_baseline_path(path, root)


def load_baseline(path: Path) -> set[str]:
    return _load_baseline(path)


def write_baseline(path: Path, violations: list[str]) -> None:
    _write_baseline(path, violations)


def apply_baseline(
    violations: list[str], baseline_allowlist: set[str]
) -> tuple[list[str], list[str]]:
    return _apply_baseline(violations, baseline_allowlist)


def render_dot(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> str:
    return _emit_dot(groups_by_path)


def render_report(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]] | None = None,
    type_suggestions: list[str] | None = None,
    type_ambiguities: list[str] | None = None,
    type_callsite_evidence: list[str] | None = None,
    constant_smells: list[str] | None = None,
    unused_arg_smells: list[str] | None = None,
    deadness_witnesses: list[JSONObject] | None = None,
    coherence_witnesses: list[JSONObject] | None = None,
    rewrite_plans: list[JSONObject] | None = None,
    exception_obligations: list[JSONObject] | None = None,
    handledness_witnesses: list[JSONObject] | None = None,
    decision_surfaces: list[str] | None = None,
    value_decision_surfaces: list[str] | None = None,
    decision_warnings: list[str] | None = None,
    fingerprint_warnings: list[str] | None = None,
    fingerprint_matches: list[str] | None = None,
    fingerprint_synth: list[str] | None = None,
    fingerprint_provenance: list[JSONObject] | None = None,
    context_suggestions: list[str] | None = None,
    invariant_propositions: list[InvariantProposition] | None = None,
    value_decision_rewrites: list[str] | None = None,
) -> tuple[str, list[str]]:
    return _emit_report(
        groups_by_path,
        max_components,
        bundle_sites_by_path=bundle_sites_by_path,
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
        type_callsite_evidence=type_callsite_evidence,
        constant_smells=constant_smells,
        unused_arg_smells=unused_arg_smells,
        deadness_witnesses=deadness_witnesses,
        coherence_witnesses=coherence_witnesses,
        rewrite_plans=rewrite_plans,
        exception_obligations=exception_obligations,
        handledness_witnesses=handledness_witnesses,
        decision_surfaces=decision_surfaces,
        value_decision_surfaces=value_decision_surfaces,
        decision_warnings=decision_warnings,
        fingerprint_warnings=fingerprint_warnings,
        fingerprint_matches=fingerprint_matches,
        fingerprint_synth=fingerprint_synth,
        fingerprint_provenance=fingerprint_provenance,
        context_suggestions=context_suggestions,
        invariant_propositions=invariant_propositions,
        value_decision_rewrites=value_decision_rewrites,
    )


def compute_violations(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    type_suggestions: list[str] | None = None,
    type_ambiguities: list[str] | None = None,
    decision_warnings: list[str] | None = None,
    fingerprint_warnings: list[str] | None = None,
) -> list[str]:
    return _compute_violations(
        groups_by_path,
        max_components,
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
        decision_warnings=decision_warnings,
        fingerprint_warnings=fingerprint_warnings,
    )


def analyze_paths(
    paths: list[Path],
    *,
    recursive: bool,
    type_audit: bool,
    type_audit_report: bool,
    type_audit_max: int,
    include_constant_smells: bool,
    include_unused_arg_smells: bool,
    include_deadness_witnesses: bool = False,
    include_coherence_witnesses: bool = False,
    include_rewrite_plans: bool = False,
    include_exception_obligations: bool = False,
    include_handledness_witnesses: bool = False,
    include_decision_surfaces: bool = False,
    include_value_decision_surfaces: bool = False,
    include_invariant_propositions: bool = False,
    include_lint_lines: bool = False,
    config: AuditConfig | None = None,
) -> AnalysisResult:
    if config is None:
        config = AuditConfig()
    file_paths = _iter_paths([str(p) for p in paths], config)
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {}
    param_spans_by_path: dict[Path, dict[str, dict[str, tuple[int, int, int, int]]]] = {}
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]] = {}
    invariant_propositions: list[InvariantProposition] = []
    for path in file_paths:
        groups, spans, sites = _analyze_file_internal(
            path, recursive=recursive, config=config
        )
        groups_by_path[path] = groups
        param_spans_by_path[path] = spans
        bundle_sites_by_path[path] = sites
        if include_invariant_propositions:
            invariant_propositions.extend(
                _collect_invariant_propositions(
                    path,
                    ignore_params=config.ignore_params,
                    project_root=config.project_root,
                    emitters=config.invariant_emitters,
                )
            )

    type_suggestions: list[str] = []
    type_ambiguities: list[str] = []
    type_callsite_evidence: list[str] = []
    if type_audit or type_audit_report:
        type_suggestions, type_ambiguities, type_callsite_evidence = analyze_type_flow_repo_with_evidence(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
        )
        if type_audit_report:
            type_suggestions = type_suggestions[:type_audit_max]
            type_ambiguities = type_ambiguities[:type_audit_max]
            # Trim evidence opportunistically so reports remain reviewable.
            type_callsite_evidence = type_callsite_evidence[:type_audit_max]

    constant_smells: list[str] = []
    if include_constant_smells:
        constant_smells = analyze_constant_flow_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
        )

    unused_arg_smells: list[str] = []
    if include_unused_arg_smells:
        unused_arg_smells = analyze_unused_arg_flow_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
        )
    deadness_witnesses: list[JSONObject] = []
    if include_deadness_witnesses:
        deadness_witnesses = analyze_deadness_flow_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
        )

    decision_surfaces: list[str] = []
    decision_warnings: list[str] = []
    decision_lint_lines: list[str] = []
    if include_decision_surfaces:
        decision_surfaces, decision_warnings, decision_lint_lines = (
            analyze_decision_surfaces_repo(
                file_paths,
                project_root=config.project_root,
                ignore_params=config.ignore_params,
                strictness=config.strictness,
                external_filter=config.external_filter,
                transparent_decorators=config.transparent_decorators,
                decision_tiers=config.decision_tiers,
            )
        )
    value_decision_surfaces: list[str] = []
    value_decision_rewrites: list[str] = []
    if include_value_decision_surfaces:
        (
            value_decision_surfaces,
            value_warnings,
            value_decision_rewrites,
            value_lint_lines,
        ) = analyze_value_encoded_decisions_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
            decision_tiers=config.decision_tiers,
        )
        decision_warnings.extend(value_warnings)
        decision_lint_lines.extend(value_lint_lines)
    fingerprint_warnings: list[str] = []
    fingerprint_matches: list[str] = []
    fingerprint_synth: list[str] = []
    fingerprint_synth_registry: JSONObject | None = None
    fingerprint_provenance: list[JSONObject] = []
    coherence_witnesses: list[JSONObject] = []
    rewrite_plans: list[JSONObject] = []
    exception_obligations: list[JSONObject] = []
    handledness_witnesses: list[JSONObject] = []
    need_exception_obligations = include_exception_obligations or (
        include_lint_lines and bool(config.never_exceptions)
    )
    if need_exception_obligations or include_handledness_witnesses:
        handledness_witnesses = _collect_handledness_witnesses(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
        )
    if need_exception_obligations:
        exception_obligations = _collect_exception_obligations(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            handledness_witnesses=handledness_witnesses,
            deadness_witnesses=deadness_witnesses,
            never_exceptions=config.never_exceptions,
        )
    if config.fingerprint_registry is not None and config.fingerprint_index:
        annotations_by_path = _param_annotations_by_path(
            file_paths,
            ignore_params=config.ignore_params,
        )
        fingerprint_warnings = _compute_fingerprint_warnings(
            groups_by_path,
            annotations_by_path,
            registry=config.fingerprint_registry,
            index=config.fingerprint_index,
            ctor_registry=config.constructor_registry,
        )
        fingerprint_matches = _compute_fingerprint_matches(
            groups_by_path,
            annotations_by_path,
            registry=config.fingerprint_registry,
            index=config.fingerprint_index,
            ctor_registry=config.constructor_registry,
        )
        fingerprint_provenance = _compute_fingerprint_provenance(
            groups_by_path,
            annotations_by_path,
            registry=config.fingerprint_registry,
            project_root=config.project_root,
            index=config.fingerprint_index,
            ctor_registry=config.constructor_registry,
        )
        fingerprint_synth, fingerprint_synth_registry = _compute_fingerprint_synth(
            groups_by_path,
            annotations_by_path,
            registry=config.fingerprint_registry,
            ctor_registry=config.constructor_registry,
            min_occurrences=config.fingerprint_synth_min_occurrences,
            version=config.fingerprint_synth_version,
            existing=config.fingerprint_synth_registry,
        )
        if include_coherence_witnesses:
            coherence_witnesses = _compute_fingerprint_coherence(
                fingerprint_provenance,
                synth_version=config.fingerprint_synth_version,
            )
        if include_rewrite_plans:
            rewrite_plans = _compute_fingerprint_rewrite_plans(
                fingerprint_provenance,
                coherence_witnesses,
                synth_version=config.fingerprint_synth_version,
                exception_obligations=(
                    exception_obligations if include_exception_obligations else None
                ),
            )
    context_suggestions: list[str] = []
    if decision_surfaces:
        for entry in decision_surfaces:
            if "(internal callers:" in entry:
                context_suggestions.append(f"Consider contextvar for {entry}")
    lint_lines: list[str] = []
    if include_lint_lines:
        lint_lines = _compute_lint_lines(
            groups_by_path=groups_by_path,
            bundle_sites_by_path=bundle_sites_by_path,
            type_callsite_evidence=type_callsite_evidence,
            exception_obligations=exception_obligations,
            decision_lint_lines=decision_lint_lines,
            constant_smells=constant_smells,
            unused_arg_smells=unused_arg_smells,
        )

    return AnalysisResult(
        groups_by_path=groups_by_path,
        param_spans_by_path=param_spans_by_path,
        bundle_sites_by_path=bundle_sites_by_path,
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
        type_callsite_evidence=type_callsite_evidence,
        constant_smells=constant_smells,
        unused_arg_smells=unused_arg_smells,
        lint_lines=lint_lines,
        deadness_witnesses=deadness_witnesses,
        decision_surfaces=decision_surfaces,
        value_decision_surfaces=value_decision_surfaces,
        decision_warnings=sorted(set(decision_warnings)),
        fingerprint_warnings=fingerprint_warnings,
        fingerprint_matches=fingerprint_matches,
        fingerprint_synth=fingerprint_synth,
        fingerprint_synth_registry=fingerprint_synth_registry,
        fingerprint_provenance=fingerprint_provenance,
        coherence_witnesses=coherence_witnesses,
        rewrite_plans=rewrite_plans,
        exception_obligations=exception_obligations,
        handledness_witnesses=handledness_witnesses,
        context_suggestions=context_suggestions,
        invariant_propositions=invariant_propositions,
        value_decision_rewrites=value_decision_rewrites,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--root", default=".", help="Project root for module resolution.")
    parser.add_argument("--config", default=None, help="Path to gabion.toml.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Comma-separated directory names to exclude (repeatable).",
    )
    parser.add_argument(
        "--ignore-params",
        default=None,
        help="Comma-separated parameter names to ignore.",
    )
    parser.add_argument(
        "--transparent-decorators",
        default=None,
        help="Comma-separated decorator names treated as transparent.",
    )
    parser.add_argument(
        "--allow-external",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow resolving calls into external libraries.",
    )
    parser.add_argument(
        "--strictness",
        choices=["high", "low"],
        default=None,
        help="Wildcard forwarding strictness (default: high).",
    )
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dot", default=None, help="Write DOT graph to file or '-' for stdout.")
    parser.add_argument(
        "--emit-structure-tree",
        default=None,
        help="Write canonical structure snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-structure-metrics",
        default=None,
        help="Write structure metrics JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-synth-json",
        default=None,
        help="Write fingerprint synth registry JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-provenance-json",
        default=None,
        help="Write fingerprint provenance JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-deadness-json",
        default=None,
        help="Write fingerprint deadness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-coherence-json",
        default=None,
        help="Write fingerprint coherence JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-rewrite-plans-json",
        default=None,
        help="Write fingerprint rewrite plans JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-exception-obligations-json",
        default=None,
        help="Write fingerprint exception obligations JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-handledness-json",
        default=None,
        help="Write fingerprint handledness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-decision-snapshot",
        default=None,
        help="Write decision surface snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument("--report", default=None, help="Write Markdown report (mermaid) to file.")
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Emit lint-style lines (path:line:col: CODE message).",
    )
    parser.add_argument("--max-components", type=int, default=10, help="Max components in report.")
    parser.add_argument(
        "--type-audit",
        action="store_true",
        help="Emit type-tightening suggestions based on downstream annotations.",
    )
    parser.add_argument(
        "--type-audit-max",
        type=int,
        default=50,
        help="Max type-tightening entries to print.",
    )
    parser.add_argument(
        "--type-audit-report",
        action="store_true",
        help="Include type-flow audit summary in the markdown report.",
    )
    parser.add_argument(
        "--fail-on-type-ambiguities",
        action="store_true",
        help="Exit non-zero if type ambiguities are detected.",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if undocumented/undeclared bundle violations are detected.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline file of violations to allow (ratchet mode).",
    )
    parser.add_argument(
        "--baseline-write",
        action="store_true",
        help="Write the current violations to the baseline file and exit zero.",
    )
    parser.add_argument(
        "--synthesis-plan",
        default=None,
        help="Write synthesis plan JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-report",
        action="store_true",
        help="Include synthesis plan summary in the markdown report.",
    )
    parser.add_argument(
        "--synthesis-protocols",
        default=None,
        help="Write protocol/dataclass stubs to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-protocols-kind",
        choices=["dataclass", "protocol"],
        default="dataclass",
        help="Emit dataclass or typing.Protocol stubs (default: dataclass).",
    )
    parser.add_argument(
        "--refactor-plan",
        action="store_true",
        help="Include refactoring plan summary in the markdown report.",
    )
    parser.add_argument(
        "--refactor-plan-json",
        default=None,
        help="Write refactoring plan JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-max-tier",
        type=int,
        default=2,
        help="Max tier to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-min-bundle-size",
        type=int,
        default=2,
        help="Min bundle size to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-allow-singletons",
        action="store_true",
        help="Allow single-field bundles in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-merge-overlap",
        type=float,
        default=None,
        help="Jaccard overlap threshold for merging bundles (0.0-1.0).",
    )
    return parser


def _normalize_transparent_decorators(
    value: object,
) -> set[str] | None:
    if value is None:
        return None
    items: list[str] = []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
    if not items:
        return None
    return set(items)


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.fail_on_type_ambiguities:
        args.type_audit = True
    fingerprint_deadness_json = args.fingerprint_deadness_json
    fingerprint_coherence_json = args.fingerprint_coherence_json
    fingerprint_rewrite_plans_json = args.fingerprint_rewrite_plans_json
    fingerprint_exception_obligations_json = args.fingerprint_exception_obligations_json
    fingerprint_handledness_json = args.fingerprint_handledness_json
    exclude_dirs: list[str] | None = None
    if args.exclude is not None:
        exclude_dirs = []
        for entry in args.exclude:
            for part in entry.split(","):
                part = part.strip()
                if part:
                    exclude_dirs.append(part)
    ignore_params: list[str] | None = None
    if args.ignore_params is not None:
        ignore_params = [p.strip() for p in args.ignore_params.split(",") if p.strip()]
    transparent_decorators: list[str] | None = None
    if args.transparent_decorators is not None:
        transparent_decorators = [
            p.strip() for p in args.transparent_decorators.split(",") if p.strip()
        ]
    config_path = Path(args.config) if args.config else None
    defaults = dataflow_defaults(Path(args.root), config_path)
    synth_defaults = synthesis_defaults(Path(args.root), config_path)
    decision_section = decision_defaults(Path(args.root), config_path)
    decision_tiers = decision_tier_map(decision_section)
    exception_section = exception_defaults(Path(args.root), config_path)
    never_exceptions = set(exception_never_list(exception_section))
    fingerprint_section = fingerprint_defaults(Path(args.root), config_path)
    synth_min_occurrences = 0
    synth_version = "synth@1"
    synth_registry_path: str | None = None
    if isinstance(fingerprint_section, dict):
        try:
            synth_min_occurrences = int(
                fingerprint_section.get("synth_min_occurrences", 0) or 0
            )
        except (TypeError, ValueError):
            synth_min_occurrences = 0
        synth_version = str(
            fingerprint_section.get("synth_version", synth_version) or synth_version
        )
        synth_registry_path = fingerprint_section.get("synth_registry_path")
    fingerprint_registry: PrimeRegistry | None = None
    fingerprint_index: dict[Fingerprint, set[str]] = {}
    constructor_registry: TypeConstructorRegistry | None = None
    synth_registry: SynthRegistry | None = None
    fingerprint_spec: dict[str, JSONValue] = {}
    if isinstance(fingerprint_section, dict):
        # The [fingerprints] section mixes bundle specs with synth settings.
        # Filter out the settings so they do not pollute the registry/index.
        fingerprint_spec = {
            key: value
            for key, value in fingerprint_section.items()
            if not str(key).startswith("synth_")
        }
    if fingerprint_spec:
        registry, index = build_fingerprint_registry(fingerprint_spec)
        if index:
            fingerprint_registry = registry
            fingerprint_index = index
            constructor_registry = TypeConstructorRegistry(registry)
            if synth_registry_path:
                resolved = _resolve_synth_registry_path(
                    str(synth_registry_path), Path(args.root)
                )
                if resolved is not None:
                    try:
                        payload = json.loads(resolved.read_text())
                    except Exception:
                        payload = None
                else:
                    payload = None
                if isinstance(payload, dict):
                    synth_registry = build_synth_registry_from_payload(
                        payload, registry
                    )
    merged = merge_payload(
        {
            "exclude": exclude_dirs,
            "ignore_params": ignore_params,
            "allow_external": args.allow_external,
            "strictness": args.strictness,
            "baseline": args.baseline,
            "transparent_decorators": transparent_decorators,
        },
        defaults,
    )
    exclude_dirs = set(merged.get("exclude", []) or [])
    ignore_params_set = set(merged.get("ignore_params", []) or [])
    allow_external = bool(merged.get("allow_external", False))
    strictness = merged.get("strictness") or "high"
    if strictness not in {"high", "low"}:
        strictness = "high"
    transparent_decorators = _normalize_transparent_decorators(
        merged.get("transparent_decorators")
    )
    config = AuditConfig(
        project_root=Path(args.root),
        exclude_dirs=exclude_dirs,
        ignore_params=ignore_params_set,
        external_filter=not allow_external,
        strictness=strictness,
        transparent_decorators=transparent_decorators,
        decision_tiers=decision_tiers,
        never_exceptions=never_exceptions,
        fingerprint_registry=fingerprint_registry,
        fingerprint_index=fingerprint_index,
        constructor_registry=constructor_registry,
        fingerprint_synth_min_occurrences=synth_min_occurrences,
        fingerprint_synth_version=synth_version,
        fingerprint_synth_registry=synth_registry,
    )
    baseline_path = _resolve_baseline_path(merged.get("baseline"), Path(args.root))
    baseline_write = args.baseline_write
    if baseline_write and baseline_path is None:
        print("Baseline path required for --baseline-write.", file=sys.stderr)
        return 2
    paths = _iter_paths(args.paths, config)
    decision_snapshot_path = args.emit_decision_snapshot
    include_decisions = bool(args.report) or bool(decision_snapshot_path) or bool(
        args.fail_on_violations
    )
    if decision_tiers:
        include_decisions = True
    include_rewrite_plans = bool(args.report) or bool(fingerprint_rewrite_plans_json)
    include_exception_obligations = bool(args.report) or bool(
        fingerprint_exception_obligations_json
    )
    include_handledness_witnesses = bool(args.report) or bool(
        fingerprint_handledness_json
    )
    include_coherence = (
        bool(args.report)
        or bool(fingerprint_coherence_json)
        or include_rewrite_plans
    )
    analysis = analyze_paths(
        paths,
        recursive=not args.no_recursive,
        type_audit=args.type_audit or args.type_audit_report,
        type_audit_report=args.type_audit_report,
        type_audit_max=args.type_audit_max,
        include_constant_smells=bool(args.report),
        include_unused_arg_smells=bool(args.report),
        include_deadness_witnesses=bool(args.report) or bool(fingerprint_deadness_json),
        include_coherence_witnesses=include_coherence,
        include_rewrite_plans=include_rewrite_plans,
        include_exception_obligations=include_exception_obligations,
        include_handledness_witnesses=include_handledness_witnesses,
        include_decision_surfaces=include_decisions,
        include_value_decision_surfaces=include_decisions,
        include_invariant_propositions=bool(args.report),
        include_lint_lines=bool(args.lint),
        config=config,
    )

    if args.fingerprint_synth_json and analysis.fingerprint_synth_registry:
        payload_json = json.dumps(
            analysis.fingerprint_synth_registry, indent=2, sort_keys=True
        )
        if args.fingerprint_synth_json.strip() == "-":
            print(payload_json)
        else:
            Path(args.fingerprint_synth_json).write_text(payload_json)

    if args.fingerprint_provenance_json and analysis.fingerprint_provenance:
        payload_json = json.dumps(
            analysis.fingerprint_provenance, indent=2, sort_keys=True
        )
        if args.fingerprint_provenance_json.strip() == "-":
            print(payload_json)
        else:
            Path(args.fingerprint_provenance_json).write_text(payload_json)
    if fingerprint_deadness_json:
        payload_json = json.dumps(
            analysis.deadness_witnesses, indent=2, sort_keys=True
        )
        if fingerprint_deadness_json.strip() == "-":
            print(payload_json)
        else:
            Path(fingerprint_deadness_json).write_text(payload_json)
    if fingerprint_coherence_json:
        payload_json = json.dumps(
            analysis.coherence_witnesses, indent=2, sort_keys=True
        )
        if fingerprint_coherence_json.strip() == "-":
            print(payload_json)
        else:
            Path(fingerprint_coherence_json).write_text(payload_json)
    if fingerprint_rewrite_plans_json:
        payload_json = json.dumps(
            analysis.rewrite_plans, indent=2, sort_keys=True
        )
        if fingerprint_rewrite_plans_json.strip() == "-":
            print(payload_json)
        else:
            Path(fingerprint_rewrite_plans_json).write_text(payload_json)
    if fingerprint_exception_obligations_json:
        payload_json = json.dumps(
            analysis.exception_obligations, indent=2, sort_keys=True
        )
        if fingerprint_exception_obligations_json.strip() == "-":
            print(payload_json)
        else:
            Path(fingerprint_exception_obligations_json).write_text(payload_json)
    if fingerprint_handledness_json:
        payload_json = json.dumps(
            analysis.handledness_witnesses, indent=2, sort_keys=True
        )
        if fingerprint_handledness_json.strip() == "-":
            print(payload_json)
        else:
            Path(fingerprint_handledness_json).write_text(payload_json)
    if args.lint:
        for line in analysis.lint_lines:
            print(line)
    structure_tree_path = args.emit_structure_tree
    structure_metrics_path = args.emit_structure_metrics
    if structure_tree_path:
        snapshot = render_structure_snapshot(
            analysis.groups_by_path,
            project_root=config.project_root,
            invariant_propositions=analysis.invariant_propositions,
        )
        payload_json = json.dumps(snapshot, indent=2, sort_keys=True)
        if structure_tree_path.strip() == "-":
            print(payload_json)
        else:
            Path(structure_tree_path).write_text(payload_json)
        if (
            args.report is None
            and args.dot is None
            and structure_metrics_path is None
            and not (
                args.type_audit
                or args.synthesis_plan
                or args.synthesis_report
                or args.synthesis_protocols
                or args.refactor_plan
                or args.refactor_plan_json
            )
        ):
            return 0
    if structure_metrics_path:
        metrics = compute_structure_metrics(analysis.groups_by_path)
        payload_json = json.dumps(metrics, indent=2, sort_keys=True)
        if structure_metrics_path.strip() == "-":
            print(payload_json)
        else:
            Path(structure_metrics_path).write_text(payload_json)
        if args.report is None and args.dot is None and not (
            args.type_audit
            or args.synthesis_plan
            or args.synthesis_report
            or args.synthesis_protocols
            or args.refactor_plan
            or args.refactor_plan_json
            or structure_tree_path
        ):
            return 0
    if decision_snapshot_path:
        snapshot = render_decision_snapshot(
            decision_surfaces=analysis.decision_surfaces,
            value_decision_surfaces=analysis.value_decision_surfaces,
            project_root=config.project_root,
        )
        payload_json = json.dumps(snapshot, indent=2, sort_keys=True)
        if decision_snapshot_path.strip() == "-":
            print(payload_json)
        else:
            Path(decision_snapshot_path).write_text(payload_json)
        if args.report is None and args.dot is None and not (
            args.type_audit
            or args.synthesis_plan
            or args.synthesis_report
            or args.synthesis_protocols
            or args.refactor_plan
            or args.refactor_plan_json
            or structure_tree_path
            or structure_metrics_path
        ):
            return 0
    synthesis_plan: JSONObject | None = None
    merge_overlap_threshold = None
    if args.synthesis_merge_overlap is not None:
        merge_overlap_threshold = args.synthesis_merge_overlap
    else:
        value = synth_defaults.get("merge_overlap_threshold")
        if isinstance(value, (int, float)):
            merge_overlap_threshold = float(value)
    if merge_overlap_threshold is not None:
        merge_overlap_threshold = max(0.0, min(1.0, merge_overlap_threshold))
    if args.synthesis_plan or args.synthesis_report or args.synthesis_protocols:
        synthesis_plan = build_synthesis_plan(
            analysis.groups_by_path,
            project_root=config.project_root,
            max_tier=args.synthesis_max_tier,
            min_bundle_size=args.synthesis_min_bundle_size,
            allow_singletons=args.synthesis_allow_singletons,
            merge_overlap_threshold=merge_overlap_threshold,
            config=config,
        )
        if args.synthesis_plan:
            payload = json.dumps(synthesis_plan, indent=2, sort_keys=True)
            if args.synthesis_plan.strip() == "-":
                print(payload)
            else:
                Path(args.synthesis_plan).write_text(payload)
        if args.synthesis_protocols:
            stubs = render_protocol_stubs(
                synthesis_plan, kind=args.synthesis_protocols_kind
            )
            if args.synthesis_protocols.strip() == "-":
                print(stubs)
            else:
                Path(args.synthesis_protocols).write_text(stubs)
    refactor_plan: JSONObject | None = None
    if args.refactor_plan or args.refactor_plan_json:
        refactor_plan = build_refactor_plan(
            analysis.groups_by_path,
            paths,
            config=config,
        )
        if args.refactor_plan_json:
            payload = json.dumps(refactor_plan, indent=2, sort_keys=True)
            if args.refactor_plan_json.strip() == "-":
                print(payload)
            else:
                Path(args.refactor_plan_json).write_text(payload)
    if args.dot is not None:
        dot = _emit_dot(analysis.groups_by_path)
        if args.dot.strip() == "-":
            print(dot)
        else:
            Path(args.dot).write_text(dot)
        if args.report is None and not (
            args.type_audit
            or args.synthesis_plan
            or args.synthesis_report
            or args.synthesis_protocols
            or args.refactor_plan
            or args.refactor_plan_json
            or structure_tree_path
        ):
            return 0
    if args.type_audit:
        if analysis.type_suggestions:
            print("Type tightening candidates:")
            for line in analysis.type_suggestions[: args.type_audit_max]:
                print(f"- {line}")
        if analysis.type_ambiguities:
            print("Type ambiguities (conflicting downstream expectations):")
            for line in analysis.type_ambiguities[: args.type_audit_max]:
                print(f"- {line}")
        if args.report is None and not (
            args.synthesis_plan
            or args.synthesis_report
            or args.synthesis_protocols
            or args.refactor_plan
            or args.refactor_plan_json
        ):
            return 0
    if args.report is not None:
        report, violations = _emit_report(
            analysis.groups_by_path,
            args.max_components,
            type_suggestions=analysis.type_suggestions if args.type_audit_report else None,
            type_ambiguities=analysis.type_ambiguities if args.type_audit_report else None,
            constant_smells=analysis.constant_smells,
            unused_arg_smells=analysis.unused_arg_smells,
            deadness_witnesses=analysis.deadness_witnesses,
            coherence_witnesses=analysis.coherence_witnesses,
            rewrite_plans=analysis.rewrite_plans,
            exception_obligations=analysis.exception_obligations,
            handledness_witnesses=analysis.handledness_witnesses,
            decision_surfaces=analysis.decision_surfaces,
            value_decision_surfaces=analysis.value_decision_surfaces,
            value_decision_rewrites=analysis.value_decision_rewrites,
            decision_warnings=analysis.decision_warnings,
            fingerprint_warnings=analysis.fingerprint_warnings,
            fingerprint_matches=analysis.fingerprint_matches,
            context_suggestions=analysis.context_suggestions,
            invariant_propositions=analysis.invariant_propositions,
        )
        suppressed: list[str] = []
        new_violations = violations
        if baseline_path is not None:
            baseline_entries = _load_baseline(baseline_path)
            if baseline_write:
                _write_baseline(baseline_path, violations)
                baseline_entries = set(violations)
                new_violations = []
            else:
                new_violations, suppressed = _apply_baseline(
                    violations, baseline_entries
                )
            report = (
                report
                + "\n\nBaseline/Ratchet:\n```\n"
                + f"Baseline: {baseline_path}\n"
                + f"Baseline entries: {len(baseline_entries)}\n"
                + f"Suppressed: {len(suppressed)}\n"
                + f"New violations: {len(new_violations)}\n"
                + "```\n"
            )
        if synthesis_plan and (
            args.synthesis_report or args.synthesis_plan or args.synthesis_protocols
        ):
            report = report + render_synthesis_section(synthesis_plan)
        if refactor_plan and (args.refactor_plan or args.refactor_plan_json):
            report = report + render_refactor_plan(refactor_plan)
        Path(args.report).write_text(report)
        if args.fail_on_violations and violations:
            if baseline_write:
                return 0
            if new_violations:
                return 1
        return 0
    for path, groups in analysis.groups_by_path.items():
        print(f"# {path}")
        for fn, bundles in groups.items():
            if not bundles:
                continue
            print(f"{fn}:")
            for bundle in bundles:
                print(f"  bundle: {sorted(bundle)}")
        print()
    if args.fail_on_type_ambiguities and analysis.type_ambiguities:
        return 1
    if args.fail_on_violations:
        violations = _compute_violations(
            analysis.groups_by_path,
            args.max_components,
            type_suggestions=analysis.type_suggestions if args.type_audit_report else None,
            type_ambiguities=analysis.type_ambiguities if args.type_audit_report else None,
            decision_warnings=analysis.decision_warnings,
            fingerprint_warnings=analysis.fingerprint_warnings,
        )
        if baseline_path is not None:
            baseline_entries = _load_baseline(baseline_path)
            if baseline_write:
                _write_baseline(baseline_path, violations)
                return 0
            new_violations, _ = _apply_baseline(violations, baseline_entries)
            if new_violations:
                return 1
        elif violations:
            return 1
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
