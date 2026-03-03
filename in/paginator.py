#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Literal, Optional, Set, Tuple

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider


_COMMENT_ONLY_RE = re.compile(r"^\s*#")
BudgetMetric = Literal["physical", "code"]


def iter_py_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if path.is_file():
            yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def code_line_mask(lines: List[str]) -> Set[int]:
    """Return 1-based line numbers counted as code LOC."""
    keep: Set[int] = set()
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if _COMMENT_ONLY_RE.match(stripped):
            continue
        keep.add(idx)
    return keep


def span_lines(start: int, end: int, mask: Optional[Set[int]] = None) -> Set[int]:
    lines = set(range(start, end + 1))
    return lines if mask is None else (lines & mask)


def select_budget_value(*, loc_phys: int, loc_code: int, budget_metric: BudgetMetric) -> int:
    if budget_metric == "physical":
        return loc_phys
    return loc_code


@dataclass(frozen=True)
class NodeId:
    kind: str
    file: str
    qualname: str

    def __str__(self) -> str:
        return f"{self.kind}:{self.file}:{self.qualname}"


@dataclass
class NodeInfo:
    node_id: NodeId
    start_line: int
    end_line: int
    loc_introduced: int = 0
    loc_span_code: int = 0
    loc_span_phys: int = 0
    sym_name: Optional[str] = None
    sym_type: Optional[str] = None


@dataclass
class Edge:
    src: str
    dst: str
    weight: int
    kind: str


@dataclass
class FileReport:
    file: str
    module: str
    loc_phys: int
    loc_code: int
    budget_metric: str
    budget_value: int
    budget_over_by: int
    over_budget: bool
    symbols: int
    scc_atoms: int
    largest_atom_loc: int


def tarjan_scc(nodes: List[str], succ: Dict[str, Set[str]]) -> List[List[str]]:
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    out: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in succ.get(v, set()):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in nodes:
        if v not in idx:
            strongconnect(v)

    return out


class SymbolCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, file_id: NodeId) -> None:
        self.file_id = file_id
        self.stack: List[str] = []
        self.symbols: Dict[str, NodeInfo] = {}
        self.children: DefaultDict[str, List[str]] = defaultdict(list)

    def _push(self, name: str) -> None:
        self.stack.append(name)

    def _pop(self) -> None:
        self.stack.pop()

    def _qual(self, leaf: str) -> str:
        return ".".join([*self.stack, leaf]) if self.stack else leaf

    def _parent_sym_str(self) -> str:
        if not self.stack:
            return str(self.file_id)
        parent = NodeId(kind="sym", file=self.file_id.file, qualname=".".join(self.stack))
        return str(parent)

    def _record(self, *, name: str, node: cst.CSTNode, sym_type: str) -> str:
        pos = self.get_metadata(PositionProvider, node)
        nid = NodeId(kind="sym", file=self.file_id.file, qualname=self._qual(name))
        key = str(nid)
        self.symbols[key] = NodeInfo(
            node_id=nid,
            start_line=pos.start.line,
            end_line=pos.end.line,
            sym_name=name,
            sym_type=sym_type,
        )
        self.children[self._parent_sym_str()].append(key)
        return key

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._record(name=node.name.value, node=node, sym_type="class")
        self._push(node.name.value)
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self._pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._record(name=node.name.value, node=node, sym_type="def")
        self._push(node.name.value)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._pop()

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> bool:
        self._record(name=node.name.value, node=node, sym_type="async_def")
        self._push(node.name.value)
        return True

    def leave_AsyncFunctionDef(self, original_node: cst.AsyncFunctionDef) -> None:
        self._pop()


class ImportCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.imports: Set[str] = set()

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            module_name = module_expr_to_str(alias.name)
            if module_name:
                self.imports.add(module_name)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return
        module_name = module_expr_to_str(node.module)
        if module_name:
            self.imports.add(module_name)


def module_expr_to_str(expr: cst.BaseExpression) -> Optional[str]:
    if isinstance(expr, cst.Name):
        return expr.value
    if isinstance(expr, cst.Attribute):
        parts: List[str] = []
        cur: cst.BaseExpression = expr
        while isinstance(cur, cst.Attribute):
            parts.append(cur.attr.value)
            cur = cur.value
        if isinstance(cur, cst.Name):
            parts.append(cur.value)
        return ".".join(reversed(parts))
    return None


class NameFinder(cst.CSTVisitor):
    def __init__(self) -> None:
        self.names: List[str] = []

    def visit_Name(self, node: cst.Name) -> None:
        self.names.append(node.value)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)


class IntraFileRefCollector(cst.CSTVisitor):
    """Collect weighted symbol reference edges inside one file."""

    def __init__(self, *, file_id: NodeId, known_top_level: Set[str]) -> None:
        self.file_id = file_id
        self.known = known_top_level
        self.stack: List[str] = []
        self.edges: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)

    def _cur_sym(self) -> str:
        if not self.stack:
            return str(self.file_id)
        nid = NodeId(kind="sym", file=self.file_id.file, qualname=".".join(self.stack))
        return str(nid)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self.stack.append(node.name.value)
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self.stack.append(node.name.value)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> bool:
        self.stack.append(node.name.value)
        return True

    def leave_AsyncFunctionDef(self, original_node: cst.AsyncFunctionDef) -> None:
        self.stack.pop()

    def visit_Call(self, node: cst.Call) -> None:
        src = self._cur_sym()
        if isinstance(node.func, cst.Name):
            name = node.func.value
            if name in self.known:
                dst = str(NodeId(kind="sym", file=self.file_id.file, qualname=name))
                self.edges[(src, dst, "call")] += 5
        elif isinstance(node.func, cst.Attribute):
            attr = node.func.attr.value
            if attr in self.known:
                dst = str(NodeId(kind="sym", file=self.file_id.file, qualname=attr))
                self.edges[(src, dst, "call")] += 4

    def visit_Name(self, node: cst.Name) -> None:
        src = self._cur_sym()
        name = node.value
        if name in self.known:
            dst = str(NodeId(kind="sym", file=self.file_id.file, qualname=name))
            self.edges[(src, dst, "name")] += 2

    def visit_Annotation(self, node: cst.Annotation) -> None:
        src = self._cur_sym()
        finder = NameFinder()
        node.annotation.visit(finder)
        for name in finder:
            if name in self.known:
                dst = str(NodeId(kind="sym", file=self.file_id.file, qualname=name))
                self.edges[(src, dst, "type")] += 1


def relpath(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def module_name_from_path(src_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(src_root).with_suffix("")
    return ".".join(rel.parts)


def resolve_import_to_file(
    *,
    import_mod: str,
    module_to_file: Dict[str, str],
) -> Optional[str]:
    if import_mod in module_to_file:
        return module_to_file[import_mod]
    pkg_init = f"{import_mod}.__init__"
    if pkg_init in module_to_file:
        return module_to_file[pkg_init]
    return None


def compute_introduced_loc(
    node: NodeInfo,
    child_nodes: List[NodeInfo],
    *,
    mask_code_lines: Set[int],
) -> None:
    node_lines_code = span_lines(node.start_line, node.end_line, mask_code_lines)
    node.loc_span_code = len(node_lines_code)
    node.loc_span_phys = len(span_lines(node.start_line, node.end_line, None))

    child_union: Set[int] = set()
    for child_info in child_nodes:
        child_union |= span_lines(child_info.start_line, child_info.end_line, mask_code_lines)

    node.loc_introduced = len(node_lines_code - child_union)


def scout(
    *,
    repo_root: Path,
    tree_root: Path,
    budget: int,
    budget_metric: BudgetMetric,
) -> Tuple[Dict[str, NodeInfo], List[Edge], List[FileReport]]:
    py_files = sorted(iter_py_files(tree_root))

    module_to_file: Dict[str, str] = {}
    for path in py_files:
        module = module_name_from_path(tree_root, path)
        module_to_file[module] = relpath(repo_root, path)

    nodes: Dict[str, NodeInfo] = {}
    edges: List[Edge] = []
    reports: List[FileReport] = []

    for path in py_files:
        path_rel = relpath(repo_root, path)
        module = module_name_from_path(tree_root, path)
        source = read_text(path)
        lines = source.splitlines()
        loc_phys = len(lines)
        mask = code_line_mask(lines)
        loc_code = len(mask)

        file_id = NodeId(kind="file", file=path_rel, qualname=module)
        file_key = str(file_id)
        file_info = NodeInfo(node_id=file_id, start_line=1, end_line=max(1, loc_phys))
        nodes[file_key] = file_info

        wrapper = MetadataWrapper(cst.parse_module(source))

        import_collector = ImportCollector()
        wrapper.module.visit(import_collector)
        for import_mod in sorted(import_collector.imports):
            target = resolve_import_to_file(import_mod=import_mod, module_to_file=module_to_file)
            if not target:
                continue
            dst_id = NodeId(kind="file", file=target, qualname="(module?)")
            edges.append(Edge(src=file_key, dst=str(dst_id), weight=1, kind="import"))

        symbol_collector = SymbolCollector(file_id)
        wrapper.visit(symbol_collector)

        for sym_key, sym_info in symbol_collector.symbols.items():
            nodes[sym_key] = sym_info

        for parent_key, child_keys in symbol_collector.children.items():
            for child_key in child_keys:
                edges.append(Edge(src=parent_key, dst=child_key, weight=0, kind="containment"))

        child_infos: DefaultDict[str, List[NodeInfo]] = defaultdict(list)
        for parent_key, child_keys in symbol_collector.children.items():
            for child_key in child_keys:
                child_infos[parent_key].append(nodes[child_key])

        syms_sorted = sorted(
            symbol_collector.symbols.values(),
            key=lambda node_info: (node_info.end_line - node_info.start_line, node_info.start_line),
            reverse=True,
        )
        for sym_info in syms_sorted:
            compute_introduced_loc(
                sym_info,
                child_infos.get(str(sym_info.node_id), []),
                mask_code_lines=mask,
            )

        compute_introduced_loc(
            file_info,
            child_infos.get(file_key, []),
            mask_code_lines=mask,
        )

        top_level_names: Set[str] = set()
        for sym_info in symbol_collector.symbols.values():
            if "." not in sym_info.node_id.qualname:
                top_level_names.add(sym_info.node_id.qualname)

        ref_collector = IntraFileRefCollector(file_id=file_id, known_top_level=top_level_names)
        wrapper.module.visit(ref_collector)
        for (src_key, dst_key, kind), weight in ref_collector.edges.items():
            edges.append(Edge(src=src_key, dst=dst_key, weight=weight, kind=kind))

        top_nodes = [
            str(NodeId(kind="sym", file=path_rel, qualname=name))
            for name in sorted(top_level_names)
        ]
        succ: Dict[str, Set[str]] = defaultdict(set)
        for edge in edges:
            if edge.kind == "containment":
                continue
            if edge.src in top_nodes and edge.dst in top_nodes and edge.weight > 0:
                succ[edge.src].add(edge.dst)

        sccs = tarjan_scc(top_nodes, succ) if top_nodes else []
        atom_locs = [sum(nodes[n].loc_introduced for n in comp if n in nodes) for comp in sccs]
        largest_atom = max(atom_locs) if atom_locs else 0

        budget_value = select_budget_value(
            loc_phys=loc_phys,
            loc_code=loc_code,
            budget_metric=budget_metric,
        )
        budget_over_by = max(0, budget_value - budget)
        reports.append(
            FileReport(
                file=path_rel,
                module=module,
                loc_phys=loc_phys,
                loc_code=loc_code,
                budget_metric=budget_metric,
                budget_value=budget_value,
                budget_over_by=budget_over_by,
                over_budget=budget_over_by > 0,
                symbols=len(symbol_collector.symbols),
                scc_atoms=len(sccs),
                largest_atom_loc=largest_atom,
            )
        )

    return nodes, edges, reports


def write_outputs(
    *,
    out_dir: Path,
    budget: int,
    budget_metric: BudgetMetric,
    nodes: Dict[str, NodeInfo],
    edges: List[Edge],
    reports: List[FileReport],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(reports, key=lambda report: (-report.budget_value, report.file))
    payload = {
        "budget": budget,
        "budget_metric": budget_metric,
        "files": [asdict(report) for report in ordered],
        "nodes": {key: asdict(value) for key, value in nodes.items()},
        "edges": [asdict(edge) for edge in edges],
    }
    (out_dir / "scout.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    offenders = [report for report in ordered if report.over_budget]
    lines: List[str] = []
    lines.append(f"Budget ({budget_metric} LOC): {budget}")
    lines.append(f"Python files scanned: {len(reports)}")
    lines.append(f"Offenders (> budget): {len(offenders)}")
    lines.append("")

    lines.append(f"Top files by {budget_metric} LOC:")
    for report in ordered[:30]:
        flag = " !!" if report.over_budget else ""
        lines.append(
            f"  {report.budget_value:6d} budgetLOC  {report.loc_code:6d} codeLOC  "
            f"{report.loc_phys:6d} physLOC  atoms={report.scc_atoms:3d}  "
            f"largest_atom={report.largest_atom_loc:6d}  {report.file}{flag}"
        )

    if offenders:
        lines.append("")
        lines.append("Offenders (over budget):")
        for report in offenders:
            lines.append(
                f"  {report.budget_value:6d} budgetLOC  over_by={report.budget_over_by:6d}  "
                f"{report.loc_code:6d} codeLOC  {report.loc_phys:6d} physLOC  "
                f"atoms={report.scc_atoms:3d}  largest_atom={report.largest_atom_loc:6d}  "
                f"{report.file}"
            )

    (out_dir / "scout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scout a Python tree: LOC, LibCST spans, intra-file coupling graph, SCC atoms."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(".").resolve())
    parser.add_argument(
        "--tree-root",
        type=Path,
        default=None,
        help="Root of python tree to scan (default: <repo-root>/src).",
    )
    parser.add_argument("--budget", type=int, default=3000, help="Max allowed LOC per file.")
    parser.add_argument(
        "--budget-metric",
        choices=("physical", "code"),
        default="physical",
        help="Metric used to evaluate the line budget (default: physical).",
    )
    parser.add_argument("--out", type=Path, default=Path("out/scout"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root: Path = args.repo_root.resolve()
    tree_root: Path = args.tree_root.resolve() if args.tree_root else (repo_root / "src")
    budget_metric: BudgetMetric = args.budget_metric

    nodes, edges, reports = scout(
        repo_root=repo_root,
        tree_root=tree_root,
        budget=args.budget,
        budget_metric=budget_metric,
    )
    write_outputs(
        out_dir=args.out,
        budget=args.budget,
        budget_metric=budget_metric,
        nodes=nodes,
        edges=edges,
        reports=reports,
    )

    print(f"Wrote: {args.out / 'scout.json'}")
    print(f"Wrote: {args.out / 'scout.txt'}")


if __name__ == "__main__":
    main()
