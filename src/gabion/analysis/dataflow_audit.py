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
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
import re

from gabion.config import dataflow_defaults, merge_payload

@dataclass
class ParamUse:
    direct_forward: set[tuple[str, str]]
    non_forward: bool
    current_aliases: set[str]


@dataclass(frozen=True)
class CallArgs:
    callee: str
    pos_map: dict[str, str]
    kw_map: dict[str, str]
    const_pos: dict[str, str]
    const_kw: dict[str, str]
    non_const_pos: set[str]
    non_const_kw: set[str]
    is_test: bool


@dataclass
class SymbolTable:
    imports: dict[tuple[str, str], str] = field(default_factory=dict)
    internal_roots: set[str] = field(default_factory=set)
    external_filter: bool = True

    def resolve(self, current_module: str, name: str) -> str | None:
        if (current_module, name) in self.imports:
            fqn = self.imports[(current_module, name)]
            if self.external_filter:
                root = fqn.split(".")[0]
                if root not in self.internal_roots:
                    return None
            return fqn
        return f"{current_module}.{name}"


@dataclass
class AuditConfig:
    project_root: Path | None = None
    exclude_dirs: set[str] = field(default_factory=set)
    ignore_params: set[str] = field(default_factory=set)
    external_filter: bool = True
    strictness: str = "high"

    def is_ignored_path(self, path: Path) -> bool:
        parts = set(path.parts)
        return bool(self.exclude_dirs & parts)


class ParentAnnotator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self.visit(child)


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


class ImportVisitor(ast.NodeVisitor):
    def __init__(self, module_name: str, table: SymbolTable) -> None:
        self.module = module_name
        self.table = table

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name
            self.table.imports[(self.module, local)] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module and node.level == 0:
            return
        if node.level > 0:
            parts = self.module.split(".")
            if node.level > len(parts):
                return
            base = parts[:-node.level]
            if node.module:
                base.append(node.module)
            source = ".".join(base)
        else:
            source = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            fqn = f"{source}.{alias.name}" if source else alias.name
            self.table.imports[(self.module, local)] = fqn


@dataclass
class AnalysisResult:
    groups_by_path: dict[Path, dict[str, list[set[str]]]]
    type_suggestions: list[str]
    type_ambiguities: list[str]
    constant_smells: list[str]
    unused_arg_smells: list[str]


def _callee_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except Exception:
        return "<call>"


def _iter_paths(paths: Iterable[str], config: AuditConfig) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for candidate in sorted(path.rglob("*.py")):
                if config.is_ignored_path(candidate):
                    continue
                out.append(candidate)
        else:
            if config.is_ignored_path(path):
                continue
            out.append(path)
    return out


def _collect_functions(tree: ast.AST):
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return funcs


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
        annots[fn.args.vararg.arg] = None
    if fn.args.kwarg:
        annots[fn.args.kwarg.arg] = None
    if names and names[0] in {"self", "cls"}:
        annots.pop(names[0], None)
    if ignore_params:
        for name in list(annots.keys()):
            if name in ignore_params:
                annots.pop(name, None)
    return annots


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
) -> tuple[dict[str, ParamUse], list[CallArgs]]:
    params = _param_names(fn, ignore_params)
    use_map = {p: ParamUse(set(), False, {p}) for p in params}
    alias_to_param: dict[str, str] = {p: p for p in params}
    call_args: list[CallArgs] = []

    class UseVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            callee = _callee_name(node)
            pos_map = {}
            kw_map = {}
            const_pos: dict[str, str] = {}
            const_kw: dict[str, str] = {}
            non_const_pos: set[str] = set()
            non_const_kw: set[str] = set()
            for idx, arg in enumerate(node.args):
                const = _const_repr(arg)
                if const is not None:
                    const_pos[str(idx)] = const
                    continue
                if isinstance(arg, ast.Name) and arg.id in alias_to_param:
                    pos_map[str(idx)] = alias_to_param[arg.id]
                else:
                    non_const_pos.add(str(idx))
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                const = _const_repr(kw.value)
                if const is not None:
                    const_kw[kw.arg] = const
                    continue
                if isinstance(kw.value, ast.Name) and kw.value.id in alias_to_param:
                    kw_map[kw.arg] = alias_to_param[kw.value.id]
                else:
                    non_const_kw.add(kw.arg)
            call_args.append(
                CallArgs(
                    callee=callee,
                    pos_map=pos_map,
                    kw_map=kw_map,
                    const_pos=const_pos,
                    const_kw=const_kw,
                    non_const_pos=non_const_pos,
                    non_const_kw=non_const_kw,
                    is_test=is_test,
                )
            )
            self.generic_visit(node)

        def _check_write(self, target: ast.AST) -> None:
            for node in ast.walk(target):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    name = node.id
                    if name in alias_to_param:
                        param = alias_to_param.pop(name)
                        if param in use_map:
                            use_map[param].current_aliases.discard(name)
                            use_map[param].non_forward = True

        def visit_Assign(self, node: ast.Assign) -> None:
            rhs_param = None
            if isinstance(node.value, ast.Name) and node.value.id in alias_to_param:
                rhs_param = alias_to_param[node.value.id]

            for target in node.targets:
                if rhs_param and isinstance(target, ast.Name):
                    alias_to_param[target.id] = rhs_param
                    use_map[rhs_param].current_aliases.add(target.id)
                else:
                    self._check_write(target)

            self.visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is None:
                return
            rhs_param = None
            if isinstance(node.value, ast.Name) and node.value.id in alias_to_param:
                rhs_param = alias_to_param[node.value.id]
            if isinstance(node.target, ast.Name) and rhs_param:
                alias_to_param[node.target.id] = rhs_param
                use_map[rhs_param].current_aliases.add(node.target.id)
            else:
                self._check_write(node.target)
            self.visit(node.value)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self._check_write(node.target)
            self.visit(node.value)

        def visit_Name(self, node: ast.Name) -> None:
            if not isinstance(node.ctx, ast.Load):
                return
            if node.id not in alias_to_param:
                return
            parent = parents.get(node)
            if isinstance(parent, ast.Starred):
                param_name = alias_to_param[node.id]
                if strictness == "high":
                    use_map[param_name].non_forward = True
                    return
                use_map[param_name].direct_forward.add(("args[*]", "arg[*]"))
                return
            if isinstance(parent, ast.keyword) and parent.arg is None:
                param_name = alias_to_param[node.id]
                if strictness == "high":
                    use_map[param_name].non_forward = True
                    return
                use_map[param_name].direct_forward.add(("kwargs[*]", "kw[*]"))
                return
            param_name = alias_to_param[node.id]
            call, direct = _call_context(node, parents)
            if call is None or not direct:
                use_map[param_name].non_forward = True
                return
            callee = _callee_name(call)
            # Determine arg slot.
            slot = None
            for idx, arg in enumerate(call.args):
                if arg is node:
                    slot = f"arg[{idx}]"
                    break
            if slot is None:
                for kw in call.keywords:
                    if kw.value is node and kw.arg is not None:
                        slot = f"kw[{kw.arg}]"
                        break
            if slot is None:
                slot = "arg[?]"
            use_map[param_name].direct_forward.add((callee, slot))

    UseVisitor().visit(fn)
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
) -> list[set[str]]:
    groups: list[set[str]] = []
    for call in call_args:
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
        for group in callee_groups[call.callee]:
            mapped = {callee_to_caller.get(p) for p in group}
            mapped.discard(None)
            if len(mapped) > 1:
                groups.append(set(mapped))
    return groups


def analyze_file(
    path: Path,
    recursive: bool = True,
    *,
    config: AuditConfig | None = None,
) -> dict[str, list[set[str]]]:
    if config is None:
        config = AuditConfig()
    tree = ast.parse(path.read_text())
    parent = ParentAnnotator()
    parent.visit(tree)
    parents = parent.parents
    is_test = _is_test_path(path)

    funcs = _collect_functions(tree)
    fn_param_orders = {f.name: _param_names(f) for f in funcs}
    fn_use = {}
    fn_calls = {}
    for f in funcs:
        use_map, call_args = _analyze_function(
            f,
            parents,
            is_test=is_test,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
        )
        fn_use[f.name] = use_map
        fn_calls[f.name] = call_args

    groups_by_fn = {fn: _group_by_signature(use_map) for fn, use_map in fn_use.items()}

    if not recursive:
        return groups_by_fn

    changed = True
    while changed:
        changed = False
        for fn in fn_use:
            propagated = _propagate_groups(
                fn_calls[fn],
                groups_by_fn,
                fn_param_orders,
            )
            if not propagated:
                continue
            combined = _union_groups(groups_by_fn.get(fn, []) + propagated)
            if combined != groups_by_fn.get(fn, []):
                groups_by_fn[fn] = combined
                changed = True
    return groups_by_fn


def _callee_key(name: str) -> str:
    if not name:
        return name
    return name.split(".")[-1]


def _is_broad_type(annot: str | None) -> bool:
    if annot is None:
        return True
    base = annot.replace("typing.", "")
    return base in {"Any", "object"}


@dataclass
class FunctionInfo:
    name: str
    qual: str
    path: Path
    params: list[str]
    annots: dict[str, str | None]
    calls: list[CallArgs]
    unused_params: set[str]


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
    return table


def _build_function_index(
    paths: list[Path],
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
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
        for fn in funcs:
            use_map, call_args = _analyze_function(
                fn,
                parent_map,
                is_test=_is_test_path(path),
                ignore_params=ignore_params,
                strictness=strictness,
            )
            unused_params = _unused_params(use_map)
            info = FunctionInfo(
                name=fn.name,
                qual=f"{module}.{fn.name}",
                path=path,
                params=_param_names(fn, ignore_params),
                annots=_param_annotations(fn, ignore_params),
                calls=call_args,
                unused_params=unused_params,
            )
            by_name[fn.name].append(info)
            by_qual[info.qual] = info
    return by_name, by_qual


def _resolve_callee(
    callee_name: str,
    caller: FunctionInfo,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable | None = None,
    project_root: Path | None = None,
) -> FunctionInfo | None:
    if not callee_name:
        return None
    if symbol_table is not None:
        caller_module = _module_name(caller.path, project_root=project_root)
        if "." not in callee_name:
            if (caller_module, callee_name) in symbol_table.imports:
                fqn = symbol_table.resolve(caller_module, callee_name)
                if fqn is None:
                    return None
                if fqn in by_qual:
                    return by_qual[fqn]
        else:
            parts = callee_name.split(".")
            base = parts[0]
            if base in ("self", "cls"):
                method = parts[-1]
                candidate = f"{caller_module}.{method}"
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
    if callee_name in by_qual:
        return by_qual[callee_name]
    # If call uses module.func, try match by module suffix.
    if "." in callee_name:
        parts = callee_name.split(".")
        func = parts[-1]
        module = ".".join(parts[:-1])
        candidates = [
            info
            for info in by_name.get(func, [])
            if info.qual.endswith(f"{module}.{func}")
        ]
        if len(candidates) == 1:
            return candidates[0]
        # If caller's module matches a candidate, prefer it.
        same_module = [
            info
            for info in candidates
            if info.path == caller.path
        ]
        if len(same_module) == 1:
            return same_module[0]
        return None
    # Fallback: unique function name across repo.
    candidates = by_name.get(callee_name, [])
    if len(candidates) == 1:
        return candidates[0]
    # Prefer same-module definition when ambiguous.
    same_module = [info for info in candidates if info.path == caller.path]
    if len(same_module) == 1:
        return same_module[0]
    # If callee is self.foo/cls.foo, prefer same-module foo.
    if callee_name.startswith(("self.", "cls.")):
        func = callee_name.split(".")[-1]
        same_module = [
            info for info in by_name.get(func, []) if info.path == caller.path
        ]
        if len(same_module) == 1:
            return same_module[0]
    return None


def analyze_type_flow_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
) -> tuple[list[str], list[str]]:
    """Repo-wide fixed-point pass for downstream type tightening."""
    by_name, by_qual = _build_function_index(
        paths, project_root, ignore_params, strictness
    )
    symbol_table = _build_symbol_table(
        paths, project_root, external_filter=external_filter
    )
    inferred: dict[str, dict[str, str | None]] = {}
    for infos in by_name.values():
        for info in infos:
            inferred[info.qual] = dict(info.annots)

    def _get_annot(info: FunctionInfo, param: str) -> str | None:
        return inferred.get(info.qual, {}).get(param)

    suggestions: set[str] = set()
    ambiguities: set[str] = set()
    changed = True
    while changed:
        changed = False
        for infos in by_name.values():
            for info in infos:
                if _is_test_path(info.path):
                    continue
                downstream: dict[str, set[str]] = defaultdict(set)
                for call in info.calls:
                    callee = _resolve_callee(
                        call.callee,
                        info,
                        by_name,
                        by_qual,
                        symbol_table,
                        project_root,
                    )
                    if callee is None:
                        continue
                    callee_params = callee.params
                    for pos_idx, param in call.pos_map.items():
                        try:
                            idx = int(pos_idx)
                        except ValueError:
                            continue
                        if idx >= len(callee_params):
                            continue
                        callee_param = callee_params[idx]
                        annot = _get_annot(callee, callee_param)
                        if annot:
                            downstream[param].add(annot)
                    for kw_name, param in call.kw_map.items():
                        annot = _get_annot(callee, kw_name)
                        if annot:
                            downstream[param].add(annot)
                for param, annots in downstream.items():
                    if not annots:
                        continue
                    if len(annots) > 1:
                        ambiguities.add(
                            f"{info.path.name}:{info.name}.{param} downstream types conflict: {sorted(annots)}"
                        )
                        continue
                    downstream_annot = next(iter(annots))
                    current = _get_annot(info, param)
                    if _is_broad_type(current) and downstream_annot:
                        if inferred[info.qual].get(param) != downstream_annot:
                            inferred[info.qual][param] = downstream_annot
                            changed = True
                        suggestions.add(
                            f"{info.path.name}:{info.name}.{param} can tighten to {downstream_annot}"
                        )
    return sorted(suggestions), sorted(ambiguities)


def analyze_constant_flow_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
) -> list[str]:
    """Detect parameters that only receive a single constant value (non-test)."""
    by_name, by_qual = _build_function_index(
        paths, project_root, ignore_params, strictness
    )
    symbol_table = _build_symbol_table(
        paths, project_root, external_filter=external_filter
    )
    const_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    non_const: dict[tuple[str, str], bool] = defaultdict(bool)
    call_counts: dict[tuple[str, str], int] = defaultdict(int)

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
                )
                if callee is None:
                    continue
                callee_params = callee.params

                for idx_str, value in call.const_pos.items():
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    if idx >= len(callee_params):
                        continue
                    key = (callee.qual, callee_params[idx])
                    const_values[key].add(value)
                    call_counts[key] += 1
                for idx_str in call.pos_map:
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    if idx >= len(callee_params):
                        continue
                    key = (callee.qual, callee_params[idx])
                    non_const[key] = True
                    call_counts[key] += 1
                for idx_str in call.non_const_pos:
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    if idx >= len(callee_params):
                        continue
                    key = (callee.qual, callee_params[idx])
                    non_const[key] = True
                    call_counts[key] += 1

                for kw, value in call.const_kw.items():
                    if kw not in callee_params:
                        continue
                    key = (callee.qual, kw)
                    const_values[key].add(value)
                    call_counts[key] += 1
                for kw in call.kw_map:
                    if kw not in callee_params:
                        continue
                    key = (callee.qual, kw)
                    non_const[key] = True
                    call_counts[key] += 1
                for kw in call.non_const_kw:
                    if kw not in callee_params:
                        continue
                    key = (callee.qual, kw)
                    non_const[key] = True
                    call_counts[key] += 1

    smells: list[str] = []
    for key, values in const_values.items():
        if non_const.get(key):
            continue
        if not values:
            continue
        if len(values) == 1:
            qual, param = key
            info = by_qual.get(qual)
            path_name = info.path.name if info is not None else qual
            count = call_counts.get(key, 0)
            smells.append(
                f"{path_name}:{qual.split('.')[-1]}.{param} only observed constant {next(iter(values))} across {count} non-test call(s)"
            )
    return sorted(smells)


def analyze_unused_arg_flow_repo(
    paths: list[Path],
    *,
    project_root: Path | None,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
) -> list[str]:
    """Detect non-constant arguments passed into unused callee parameters."""
    by_name, by_qual = _build_function_index(
        paths, project_root, ignore_params, strictness
    )
    symbol_table = _build_symbol_table(
        paths, project_root, external_filter=external_filter
    )
    smells: set[str] = set()

    def _format(
        caller: FunctionInfo,
        callee: FunctionInfo,
        callee_param: str,
        arg_desc: str,
    ) -> str:
        return (
            f"{caller.path.name}:{caller.name} passes {arg_desc} "
            f"to unused {callee.path.name}:{callee.name}.{callee_param}"
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
                )
                if callee is None:
                    continue
                if not callee.unused_params:
                    continue
                callee_params = callee.params

                for idx_str, caller_param in call.pos_map.items():
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
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
                            )
                        )
                for idx_str in call.non_const_pos:
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
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
                            )
                        )
    return sorted(smells)


def _iter_config_fields(path: Path) -> dict[str, set[str]]:
    """Best-effort extraction of @dataclass config bundles (fields ending with _fn)."""
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return {}
    bundles: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {getattr(d, "id", None) for d in node.decorator_list}
        if "dataclass" not in decorators and not node.name.endswith("Config"):
            continue
        fields: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                if name.endswith("_fn"):
                    fields.add(name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_fn"):
                        fields.add(target.id)
        if fields:
            bundles[node.name] = fields
    return bundles


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


def _iter_dataclass_call_bundles(path: Path) -> set[tuple[str, ...]]:
    """Return bundles promoted via local @dataclass constructor calls."""
    bundles: set[tuple[str, ...]] = set()
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return bundles
    dataclass_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = {
            ast.unparse(dec) if hasattr(ast, "unparse") else ""
            for dec in node.decorator_list
        }
        if any("dataclass" in dec for dec in decorators):
            dataclass_names.add(node.name)
    if not dataclass_names:
        return bundles

    def _callee_name(call: ast.Call) -> str | None:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_name(node)
        if callee not in dataclass_names:
            continue
        names: list[str] = []
        ok = True
        for arg in node.args:
            if isinstance(arg, ast.Name):
                names.append(arg.id)
            else:
                ok = False
                break
        if not ok:
            continue
        for kw in node.keywords:
            if kw.arg is None:
                ok = False
                break
            if isinstance(kw.value, ast.Name):
                names.append(kw.value.id)
            else:
                ok = False
                break
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
    declared = set()
    documented = set()
    for path in component_paths:
        config_path = path.parent / "config.py"
        bundles = config_bundles_by_path.get(config_path)
        if bundles:
            for fields in bundles.values():
                declared.add(tuple(sorted(fields)))
        documented |= documented_bundles_by_path.get(path, set())
    observed_norm = {tuple(sorted(b)) for b in observed}
    observed_only = sorted(observed_norm - declared) if declared else sorted(observed_norm)
    declared_only = sorted(declared - observed_norm)
    documented_only = sorted(observed_norm & documented)
    def _tier(bundle: tuple[str, ...]) -> str:
        count = bundle_counts.get(bundle, 1)
        if declared:
            return "tier-1"
        if count > 1:
            return "tier-2"
        return "tier-3"
    summary_lines = [
        f"Functions: {len(fn_nodes)}",
        f"Observed bundles: {len(observed_norm)}",
    ]
    if not declared:
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


def _emit_report(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    type_suggestions: list[str] | None = None,
    type_ambiguities: list[str] | None = None,
    constant_smells: list[str] | None = None,
    unused_arg_smells: list[str] | None = None,
) -> tuple[str, list[str]]:
    nodes, adj, bundle_map = _component_graph(groups_by_path)
    components = _connected_components(nodes, adj)
    if groups_by_path:
        common = os.path.commonpath([str(p) for p in groups_by_path])
        root = Path(common)
    else:
        root = Path(".")
    config_bundles_by_path = {}
    documented_bundles_by_path = {}
    for path in sorted(root.rglob("config.py")):
        config_bundles_by_path[path] = _iter_config_fields(path)
    protocols = root / "prism_vm_core" / "protocols.py"
    if protocols.exists():
        config_bundles_by_path[protocols] = _iter_config_fields(protocols)
    for path in sorted(root.rglob("*.py")):
        documented = _iter_documented_bundles(path)
        promoted = _iter_dataclass_call_bundles(path)
        documented_bundles_by_path[path] = documented | promoted
    lines = [
        "<!-- dataflow-grammar -->",
        "Dataflow grammar audit (observed forwarding bundles).",
        "",
    ]
    if not components:
        return "\n".join(lines + ["No bundle components detected."]), []
    if len(components) > max_components:
        lines.append(
            f"Showing top {max_components} components of {len(components)}."
        )
    violations: list[str] = []
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
        for line in summary.splitlines():
            if "(tier-3, undocumented)" in line:
                violations.append(line.strip())
            if "(tier-1," in line or "(tier-2," in line:
                if "undocumented" in line:
                    violations.append(line.strip())
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
    return "\n".join(lines), violations


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
) -> list[str]:
    _, violations = _emit_report(
        groups_by_path,
        max_components,
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
        constant_smells=[],
        unused_arg_smells=[],
    )
    return violations


def render_dot(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> str:
    return _emit_dot(groups_by_path)


def render_report(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    type_suggestions: list[str] | None = None,
    type_ambiguities: list[str] | None = None,
    constant_smells: list[str] | None = None,
    unused_arg_smells: list[str] | None = None,
) -> tuple[str, list[str]]:
    return _emit_report(
        groups_by_path,
        max_components,
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
        constant_smells=constant_smells,
        unused_arg_smells=unused_arg_smells,
    )


def compute_violations(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    type_suggestions: list[str] | None = None,
    type_ambiguities: list[str] | None = None,
) -> list[str]:
    return _compute_violations(
        groups_by_path,
        max_components,
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
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
    config: AuditConfig | None = None,
) -> AnalysisResult:
    if config is None:
        config = AuditConfig()
    file_paths = _iter_paths([str(p) for p in paths], config)
    groups_by_path: dict[Path, dict[str, list[set[str]]]] = {}
    for path in file_paths:
        groups_by_path[path] = analyze_file(path, recursive=recursive, config=config)

    type_suggestions: list[str] = []
    type_ambiguities: list[str] = []
    if type_audit or type_audit_report:
        type_suggestions, type_ambiguities = analyze_type_flow_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
        )
        if type_audit_report:
            type_suggestions = type_suggestions[:type_audit_max]
            type_ambiguities = type_ambiguities[:type_audit_max]

    constant_smells: list[str] = []
    if include_constant_smells:
        constant_smells = analyze_constant_flow_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
        )

    unused_arg_smells: list[str] = []
    if include_unused_arg_smells:
        unused_arg_smells = analyze_unused_arg_flow_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
        )

    return AnalysisResult(
        groups_by_path=groups_by_path,
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
        constant_smells=constant_smells,
        unused_arg_smells=unused_arg_smells,
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
    parser.add_argument("--report", default=None, help="Write Markdown report (mermaid) to file.")
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
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if undocumented/undeclared bundle violations are detected.",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
    config_path = Path(args.config) if args.config else None
    defaults = dataflow_defaults(Path(args.root), config_path)
    merged = merge_payload(
        {
            "exclude": exclude_dirs,
            "ignore_params": ignore_params,
            "allow_external": args.allow_external,
            "strictness": args.strictness,
        },
        defaults,
    )
    exclude_dirs = set(merged.get("exclude", []) or [])
    ignore_params_set = set(merged.get("ignore_params", []) or [])
    allow_external = bool(merged.get("allow_external", False))
    strictness = merged.get("strictness") or "high"
    if strictness not in {"high", "low"}:
        strictness = "high"
    config = AuditConfig(
        project_root=Path(args.root),
        exclude_dirs=exclude_dirs,
        ignore_params=ignore_params_set,
        external_filter=not allow_external,
        strictness=strictness,
    )
    paths = _iter_paths(args.paths, config)
    analysis = analyze_paths(
        paths,
        recursive=not args.no_recursive,
        type_audit=args.type_audit or args.type_audit_report,
        type_audit_report=args.type_audit_report,
        type_audit_max=args.type_audit_max,
        include_constant_smells=bool(args.report),
        include_unused_arg_smells=bool(args.report),
        config=config,
    )
    if args.dot is not None:
        dot = _emit_dot(analysis.groups_by_path)
        if args.dot.strip() == "-":
            print(dot)
        else:
            Path(args.dot).write_text(dot)
        if args.report is None:
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
        return 0
    if args.report is not None:
        report, violations = _emit_report(
            analysis.groups_by_path,
            args.max_components,
            type_suggestions=analysis.type_suggestions if args.type_audit_report else None,
            type_ambiguities=analysis.type_ambiguities if args.type_audit_report else None,
            constant_smells=analysis.constant_smells,
            unused_arg_smells=analysis.unused_arg_smells,
        )
        Path(args.report).write_text(report)
        if args.fail_on_violations and violations:
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
    if args.fail_on_violations:
        violations = _compute_violations(
            analysis.groups_by_path,
            args.max_components,
            type_suggestions=analysis.type_suggestions if args.type_audit_report else None,
            type_ambiguities=analysis.type_ambiguities if args.type_audit_report else None,
        )
        if violations:
            return 1
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
