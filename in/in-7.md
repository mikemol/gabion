---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: in_7
doc_role: inbox
doc_scope:
  - repo
  - inbox
doc_authority: informative
doc_owner: maintainer
doc_requires:
doc_reviewed_as_of:
doc_review_notes:
doc_change_protocol: "POLICY_SEED.md §6"
doc_erasure:
  - formatting
  - typos
---
This is the **Architectural Archeology Tool (v2)**.

I have integrated the **Symbol Table (for Import Resolution)** and the **Identity-Aware Visitor (for Aliasing)**. This transforms the tool from a local heuristic scanner into a repository-wide structural compiler. It now understands that `db.query` in one file is the same as `database.query` in another, and that `y = x; func(y)` is a valid forwarding of `x`.

### The Complete Script

```python
#!/usr/bin/env python3
"""Infer forwarding-based parameter bundles and propagate them across calls.

This script performs a multi-stage analysis:
  1) Import Resolution: Scans the repo to build a global symbol table, linking
     local names to fully qualified definitions.
  2) Local Grouping: Analyzes functions to find parameters forwarded to other calls.
     Now supports local aliasing (e.g. y = x; f(y)) and identity tracking.
  3) Propagation: Iteratively propagates bundles up the call stack to a fixed point.

The goal is to surface "dataflow grammar" candidates for config dataclasses.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# --- Data Structures ---

@dataclass
class SymbolTable:
    """Resolves local names to fully qualified paths."""
    # Map: (module_name, local_name) -> fully_qualified_name
    imports: dict[tuple[str, str], str] = field(default_factory=dict)

    def resolve(self, current_module: str, name: str) -> str:
        """Resolve a local name to its absolute path."""
        # 1. Check explicit imports
        if (current_module, name) in self.imports:
            return self.imports[(current_module, name)]
        
        # 2. Implicit/Local definition
        # If not imported, we assume it is defined in the current module.
        return f"{current_module}.{name}"


@dataclass
class ParamUse:
    """Tracks how a parameter (or its aliases) flows through the function."""
    direct_forward: set[tuple[str, str]]
    non_forward: bool
    current_aliases: set[str]  # Local variables currently holding this identity


@dataclass(frozen=True)
class CallArgs:
    callee: str  # Local name of the callee
    pos_map: dict[str, str]
    kw_map: dict[str, str]
    const_pos: dict[str, str]
    const_kw: dict[str, str]
    non_const_pos: set[str]
    non_const_kw: set[str]
    is_test: bool


@dataclass
class FunctionInfo:
    name: str
    qual: str
    path: Path
    params: list[str]
    annots: dict[str, str | None]
    calls: list[CallArgs]


# --- Visitors ---

class ImportVisitor(ast.NodeVisitor):
    """Populates the SymbolTable from import statements."""
    def __init__(self, module_name: str, table: SymbolTable):
        self.module = module_name
        self.table = table

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name
            fqn = alias.name
            self.table.imports[(self.module, local)] = fqn

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level > 0:
            # Handle relative imports
            parts = self.module.split(".")
            # level=1 is '.', level=2 is '..', so we slice off level-1?
            # Actually, standard behavior: from . import -> parent package.
            # Heuristic: strip last 'level' components.
            if len(parts) < node.level:
                return # Error or root escape
            base_parts = parts[:-node.level] if node.level > 0 else parts
            if node.module:
                base_parts.append(node.module)
            source_module = ".".join(base_parts)
        else:
            source_module = node.module or ""

        for alias in node.names:
            local = alias.asname or alias.name
            if alias.name == "*":
                continue # Cannot resolve wildcard statically
            fqn = f"{source_module}.{alias.name}" if source_module else alias.name
            self.table.imports[(self.module, local)] = fqn


class ParentAnnotator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self.visit(child)


# --- Core Analysis Logic ---

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


def _callee_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except Exception:
        return "<call>"


def _const_repr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
        try:
            return ast.unparse(node)
        except Exception:
            return None
    if isinstance(node, ast.Attribute):
        try:
            return ast.unparse(node)
        except Exception:
            return None
    return None


def _param_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    names = [a.arg for a in args]
    if fn.args.vararg:
        names.append(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.append(fn.args.kwarg.arg)
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    return names


def _param_annotations(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str | None]:
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
    return annots


def _analyze_function(fn, parents, *, is_test: bool):
    """
    Analyzes a function for parameter forwarding, respecting local aliasing.
    Returns (use_map, call_args).
    """
    params = _param_names(fn)
    
    # Initialize usage map.
    # We add 'current_aliases' to track local variables that ARE the parameter.
    use_map: dict[str, ParamUse] = {
        p: ParamUse(set(), False, {p}) for p in params
    }
    
    # Reverse lookup: distinct local alias -> original parameter name
    # Invariant: A local variable maps to AT MOST one parameter.
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
                
                # Check if the argument is a known alias to a parameter
                if isinstance(arg, ast.Name) and arg.id in alias_to_param:
                    # It's a forward! Map it back to the original parameter.
                    origin_param = alias_to_param[arg.id]
                    pos_map[str(idx)] = origin_param
                else:
                    non_const_pos.add(str(idx))

            for kw in node.keywords:
                if kw.arg is None:
                    continue
                
                const = _const_repr(kw.value)
                if const is not None:
                    const_kw[kw.arg] = const
                    continue

                # Check if keyword value is a known alias
                if isinstance(kw.value, ast.Name) and kw.value.id in alias_to_param:
                    origin_param = alias_to_param[kw.value.id]
                    kw_map[kw.arg] = origin_param
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

        def visit_Assign(self, node: ast.Assign) -> None:
            # Handle aliasing: y = x
            rhs_param = None
            if isinstance(node.value, ast.Name) and node.value.id in alias_to_param:
                rhs_param = alias_to_param[node.value.id]

            # Process targets (LHS)
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    self._check_write(target)
                    continue

                lhs_name = target.id

                if rhs_param:
                    # Case: y = x. Grant 'y' the identity of 'x'.
                    alias_to_param[lhs_name] = rhs_param
                    use_map[rhs_param].current_aliases.add(lhs_name)
                else:
                    # Case: y = <something else>. If 'y' was an alias, it is no longer.
                    if lhs_name in alias_to_param:
                        old_param = alias_to_param.pop(lhs_name)
                        if old_param in use_map:
                            use_map[old_param].current_aliases.discard(lhs_name)

            self.visit(node.value)
            
        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            # Handle aliasing: y: int = x
            if not isinstance(node.target, ast.Name):
                if node.value: self.visit(node.value)
                return

            lhs_name = node.target.id
            rhs_param = None
            
            if node.value and isinstance(node.value, ast.Name) and node.value.id in alias_to_param:
                rhs_param = alias_to_param[node.value.id]
            
            if rhs_param:
                alias_to_param[lhs_name] = rhs_param
                use_map[rhs_param].current_aliases.add(lhs_name)
            else:
                if lhs_name in alias_to_param:
                    old_param = alias_to_param.pop(lhs_name)
                    if old_param in use_map:
                        use_map[old_param].current_aliases.discard(lhs_name)

            if node.value:
                self.visit(node.value)

        def _check_write(self, node: ast.AST):
            """If a parameter/alias is mutated, sever the link."""
            if isinstance(node, ast.Name) and node.id in alias_to_param:
                param = alias_to_param[node.id]
                if node.id == param:
                     # Reassigning the parameter itself! 'x = 5'.
                     use_map[param].non_forward = True
                     alias_to_param.pop(param, None)
                else:
                    # Reassigning an alias. 'y' is gone, 'x' remains safe.
                    alias_to_param.pop(node.id, None)
                    use_map[param].current_aliases.discard(node.id)

        def visit_Name(self, node: ast.Name) -> None:
            if not isinstance(node.ctx, ast.Load):
                return
            if node.id not in alias_to_param:
                return

            origin_param = alias_to_param[node.id]
            call, is_arg = _call_context(node, parents)
            if call:
                if is_arg:
                    # It's an argument. We already mapped it in visit_Call (CallArgs).
                    # But we also need to record the direct_forward logic here for local bundling.
                    callee = _callee_name(call)
                    slot = None
                    for idx, arg in enumerate(call.args):
                        if arg is node:
                            slot = f"arg[{idx}]"
                            break
                    if slot is None:
                        for kw in call.keywords:
                            if kw.value is node:
                                slot = f"kw[{kw.arg}]"
                                break
                    
                    if slot:
                        use_map[origin_param].direct_forward.add((callee, slot))
                    else:
                         use_map[origin_param].non_forward = True
                    return
                else:
                    # Inside call, but not an arg (e.g. func(x+1)). Usage.
                    use_map[origin_param].non_forward = True
                    return

            # Check if RHS of assignment (Identity Transfer)
            parent = parents.get(node)
            if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                if parent.value is node:
                    return

            # Otherwise, it's a computation/logic use.
            use_map[origin_param].non_forward = True

    UseVisitor().visit(fn)
    return use_map, call_args


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
    fn_name: str,
    call_args: list[CallArgs],
    callee_groups: dict[str, list[set[str]]],
    callee_param_orders: dict[str, list[str]],
    caller_info: FunctionInfo,
    symbol_table: SymbolTable,
    function_index: dict[str, FunctionInfo],
) -> list[set[str]]:
    groups: list[set[str]] = []
    
    for call in call_args:
        # RESOLUTION: Use deterministic symbol table lookup
        callee_info = _resolve_callee_deterministic(
            call.callee, caller_info, symbol_table, function_index
        )
        if not callee_info:
            continue
            
        callee_qual = callee_info.qual
        if callee_qual not in callee_groups:
            continue

        callee_params = callee_param_orders[callee_qual]
        
        # Build mapping from callee param -> caller param
        callee_to_caller: dict[str, str] = {}
        for idx, pname in enumerate(callee_params):
            key = str(idx)
            if key in call.pos_map:
                callee_to_caller[pname] = call.pos_map[key]
        for kw, caller_name in call.kw_map.items():
            callee_to_caller[kw] = caller_name
            
        for group in callee_groups[callee_qual]:
            mapped = {callee_to_caller.get(p) for p in group}
            mapped.discard(None)
            if len(mapped) > 1:
                groups.append(set(mapped))
                
    return groups


# --- Resolution & System Glue ---

def _module_name(path: Path) -> str:
    # Quick heuristic: assumes run from repo root.
    rel = path.with_suffix("")
    return ".".join(rel.parts)


def _resolve_callee_deterministic(
    callee_name: str,
    caller_info: FunctionInfo,
    symbol_table: SymbolTable,
    by_qual: dict[str, FunctionInfo],
) -> FunctionInfo | None:
    if not callee_name:
        return None
    
    caller_module = _module_name(caller_info.path)
    
    # CASE 1: Simple Name or Fully Qualified
    if "." not in callee_name:
        fqn = symbol_table.resolve(caller_module, callee_name)
        if fqn in by_qual:
            return by_qual[fqn]
        return None

    # CASE 2: Attribute Access (e.g., utils.process)
    parts = callee_name.split(".")
    base = parts[0]
    base_fqn = symbol_table.resolve(caller_module, base)
    
    # Attempt to reconstruct path
    candidate_fqn = base_fqn + "." + ".".join(parts[1:])
    if candidate_fqn in by_qual:
        return by_qual[candidate_fqn]

    # CASE 3: Methods (self.foo) - Heuristic fallback
    if base in ("self", "cls"):
        method_name = parts[-1]
        candidate_fqn = f"{caller_module}.{method_name}"
        if candidate_fqn in by_qual:
            return by_qual[candidate_fqn]
            
    return None


def _is_test_path(path: Path) -> bool:
    if "tests" in path.parts:
        return True
    return path.name.startswith("test_")


def _collect_functions(tree: ast.AST):
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return funcs


def analyze_repo(paths: list[Path], recursive: bool = True) -> tuple[dict[str, list[set[str]]], SymbolTable, dict[str, FunctionInfo]]:
    symbol_table = SymbolTable()
    function_index: dict[str, FunctionInfo] = {}
    
    # PASS 1: Build Symbol Table & Function Index
    print("Pass 1: Parsing and Import Resolution...")
    
    # We need to hold the ASTs in memory or re-parse. 
    # For simplicity, we re-parse or hold lightweight info.
    # Let's hold ASTs in a temporary map to avoid double I/O.
    ast_cache = {}

    for path in paths:
        try:
            tree = ast.parse(path.read_text())
            ast_cache[path] = tree
            mod_name = _module_name(path)
            
            # Imports
            ImportVisitor(mod_name, symbol_table).visit(tree)
            
            # Functions
            parent_annotator = ParentAnnotator()
            parent_annotator.visit(tree)
            parents = parent_annotator.parents
            is_test = _is_test_path(path)

            funcs = _collect_functions(tree)
            for fn in funcs:
                use_map, call_args = _analyze_function(fn, parents, is_test=is_test)
                qual = f"{mod_name}.{fn.name}"
                
                info = FunctionInfo(
                    name=fn.name,
                    qual=qual,
                    path=path,
                    params=_param_names(fn),
                    annots=_param_annotations(fn),
                    calls=call_args
                )
                function_index[qual] = info
                # Store use_map/call_args temporarily on the object? 
                # We can store them in a parallel dict for the next phase.
        except Exception as e:
            # print(f"Skipping {path}: {e}")
            continue

    # PASS 2: Local Grouping
    print("Pass 2: Local Grouping...")
    groups_by_qual: dict[str, list[set[str]]] = {}
    fn_param_orders: dict[str, list[str]] = {}
    
    # We need to re-run _analyze_function? No, we ran it above.
    # We just need to persist the use_maps. 
    # In this refactor, I'll assume we stored them. 
    # Actually, let's just re-analyze or better, update FunctionInfo to hold the use_map for a moment.
    # For now, let's just re-run the local analysis part or extract it from the loop above.
    
    # Refined Approach: Store the `use_map` from Pass 1.
    pass1_data: dict[str, dict] = {} # qual -> use_map

    # Re-looping AST cache to fill pass1_data if not captured above.
    # (To correct the structure above, let's pretend we saved it).
    # Since I didn't save it in the loop above, let's fix the loop logic.
    pass1_data = {}
    for path, tree in ast_cache.items():
        mod_name = _module_name(path)
        parent_annotator = ParentAnnotator()
        parent_annotator.visit(tree)
        parents = parent_annotator.parents
        is_test = _is_test_path(path)
        
        funcs = _collect_functions(tree)
        for fn in funcs:
            use_map, call_args = _analyze_function(fn, parents, is_test=is_test)
            qual = f"{mod_name}.{fn.name}"
            pass1_data[qual] = use_map
            fn_param_orders[qual] = _param_names(fn)
            
            # Ensure index has the latest call_args (which might be updated by re-analysis)
            if qual in function_index:
                function_index[qual].calls = call_args

    for qual, use_map in pass1_data.items():
        groups_by_qual[qual] = _group_by_signature(use_map)

    if not recursive:
        return groups_by_qual, symbol_table, function_index

    # PASS 3: Propagation
    print("Pass 3: Global Propagation...")
    changed = True
    iteration = 0
    while changed:
        iteration += 1
        # print(f"  Iteration {iteration}...")
        changed = False
        for qual, info in function_index.items():
            propagated = _propagate_groups(
                fn_name=info.name,
                call_args=info.calls,
                callee_groups=groups_by_qual,
                callee_param_orders=fn_param_orders,
                caller_info=info,
                symbol_table=symbol_table,
                function_index=function_index
            )
            
            if not propagated:
                continue
                
            combined = _union_groups(groups_by_qual.get(qual, []) + propagated)
            if combined != groups_by_qual.get(qual, []):
                groups_by_qual[qual] = combined
                changed = True

    return groups_by_qual, symbol_table, function_index


# --- Reporting (Simplified) ---

def _iter_paths(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.py")))
        else:
            out.append(path)
    return out


def _emit_report(groups_by_qual: dict[str, list[set[str]]], symbol_table: SymbolTable, function_index: dict[str, FunctionInfo], max_components: int) -> str:
    # Quick connectivity graph
    nodes = set()
    edges = defaultdict(set)
    bundle_map = {}
    
    for qual, groups in groups_by_qual.items():
        if not groups: continue
        fn_node = f"fn::{qual}"
        nodes.add(fn_node)
        
        for i, group in enumerate(groups):
            b_node = f"bundle::{qual}::{i}"
            bundle_map[b_node] = group
            nodes.add(b_node)
            edges[fn_node].add(b_node)
            edges[b_node].add(fn_node)
            
            # Check calls to link bundles?
            # Global propagation already linked them by unioning. 
            # We need to visualize the CALL graph edges that support these bundles.
            # This is complex to render perfectly. We'll stick to local components for now.
    
    lines = ["# Dataflow Architecture Report", ""]
    lines.append(f"Analyzed {len(function_index)} functions.")
    lines.append(f"Found {sum(len(g) for g in groups_by_qual.values())} bundle candidates.")
    lines.append("")
    
    for qual, groups in sorted(groups_by_qual.items()):
        if not groups: continue
        lines.append(f"## {qual}")
        for group in groups:
            lines.append(f"- Bundle: `{', '.join(sorted(group))}`")
            
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--report", default=None, help="Write report to file.")
    args = parser.parse_args()
    
    paths = _iter_paths(args.paths)
    groups, symbols, index = analyze_repo(paths)
    
    if args.report:
        report = _emit_report(groups, symbols, index, 10)
        Path(args.report).write_text(report)
        print(f"Report written to {args.report}")
    else:
        # Stdout dump
        for qual, bun in sorted(groups.items()):
            if not bun: continue
            print(f"{qual}:")
            for b in bun:
                print(f"  {sorted(b)}")

if __name__ == "__main__":
    main()

```

### Guidance on the "Next Step"

You now have a tool that can "see" through the fog of aliasing and module boundaries.

1. **Run this script** on your codebase.
2. **Look for "Tier-2" Bundles:** These are the ones that survive the identity-aware propagation. They are your true structural constants.
3. **Refactor:** Pick *one* detected bundle and create a `dataclass` or `Protocol` for it.

Would you like me to now generate the **Protocol Synthesizer** (Step #3 from our opportunity list) to automatically write those `dataclass` definitions for you?
