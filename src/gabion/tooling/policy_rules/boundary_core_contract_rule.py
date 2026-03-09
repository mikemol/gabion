#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from gabion.analysis.foundation.event_algebra import CanonicalRunContext
from gabion.tooling.policy_rules.fiber_diagnostics import (
    FiberApplicabilityBounds,
    FiberCounterfactualBoundary,
    FiberTraceEvent,
)
from gabion.tooling.policy_substrate import (
    build_aspf_union_view,
    cst_failure_seeds,
    decorate_site,
    new_run_context,
)
from gabion.tooling.runtime.policy_scan_batch import (
    PolicyScanBatch,
    iter_failure_seeds,
)

RULE_NAME = "boundary_core_contract"
TARGET_GLOB = "src/gabion/**/*.py"
BOUNDARY_MARKER = "gabion:boundary_normalization_module"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    input_slot: str
    flow_identity: str
    fiber_trace: tuple[FiberTraceEvent, ...]
    applicability_bounds: FiberApplicabilityBounds
    counterfactual_boundary: FiberCounterfactualBoundary
    fiber_id: str
    taint_interval_id: str
    condition_overlap_id: str
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
class _CoreImport:
    module_dotted: str
    alias_names: tuple[str, ...]


def collect_violations(
    *,
    batch: PolicyScanBatch,
    run_context: CanonicalRunContext | None = None,
) -> list[Violation]:
    context = run_context if run_context is not None else new_run_context(rule_name=RULE_NAME)
    union_view = build_aspf_union_view(batch=batch)
    violations = list(
        _iter_failure_violations(
            batch=batch,
            union_view=union_view,
            run_context=context,
        )
    )
    for module in filter(_is_boundary_module, union_view.modules):
        violations.extend(
            _iter_boundary_module_contract_violations(
                root=batch.root,
                module=module,
                run_context=context,
            )
        )
    return _dedupe_exact_violations(violations)


def _is_boundary_module(module: object) -> bool:
    return _module_has_boundary_marker(module.source.splitlines())


def _iter_boundary_module_contract_violations(
    *,
    root: Path,
    module,
    run_context: CanonicalRunContext,
) -> Iterable[Violation]:
    core_imports = _collect_core_imports(tree=module.pyast_tree, rel_path=module.rel_path)
    if not core_imports:
        yield _violation(
            rel_path=module.rel_path,
            line=1,
            column=1,
            qualname="<module>",
            kind="missing_paired_core_module",
            message="boundary normalization module must import at least one paired *_core module",
            run_context=run_context,
        )
        return
    if not _has_explicit_single_hop_core_call(tree=module.pyast_tree, core_imports=core_imports):
        yield _violation(
            rel_path=module.rel_path,
            line=1,
            column=1,
            qualname="<module>",
            kind="missing_single_hop_core_call",
            message=(
                "boundary module must call paired core via "
                "explicit single-hop boundary->core call"
            ),
            run_context=run_context,
        )
    resolved_paths = tuple(_iter_resolved_core_paths(root=root, core_imports=core_imports))
    yield from _iter_missing_core_module_file_violations(
        rel_path=module.rel_path,
        resolved_paths=resolved_paths,
        run_context=run_context,
    )
    yield from _iter_existing_core_module_violations(
        root=root,
        boundary_rel_path=module.rel_path,
        resolved_paths=resolved_paths,
        run_context=run_context,
    )


def _iter_missing_core_module_file_violations(
    *,
    rel_path: str,
    resolved_paths: Sequence[tuple[str, Path | None]],
    run_context: CanonicalRunContext,
) -> Iterable[Violation]:
    for module_dotted in map(itemgetter(0), filter(_is_missing_core_resolution, resolved_paths)):
        yield _violation(
            rel_path=rel_path,
            line=1,
            column=1,
            qualname="<module>",
            kind="missing_core_module_file",
            message=(
                "paired core module "
                f"'{module_dotted}' must resolve "
                "to an on-disk module"
            ),
            run_context=run_context,
        )


def _is_missing_core_resolution(resolution: tuple[str, Path | None]) -> bool:
    return resolution[1] is None


def _iter_existing_core_module_violations(
    *,
    root: Path,
    boundary_rel_path: str,
    resolved_paths: Sequence[tuple[str, Path | None]],
    run_context: CanonicalRunContext,
) -> Iterable[Violation]:
    for core_path in map(itemgetter(1), filter(_is_existing_core_resolution, resolved_paths)):
        yield from _iter_core_module_contract_violations(
            root=root,
            boundary_rel_path=boundary_rel_path,
            core_path=core_path,
            run_context=run_context,
        )


def _is_existing_core_resolution(resolution: tuple[str, Path | None]) -> bool:
    return resolution[1] is not None


def _iter_failure_violations(
    *,
    batch: PolicyScanBatch,
    union_view,
    run_context: CanonicalRunContext,
) -> Iterable[Violation]:
    for seed in (*iter_failure_seeds(batch=batch), *cst_failure_seeds(union_view=union_view)):
        yield _violation(
            rel_path=seed.path,
            line=seed.line,
            column=seed.column,
            qualname="<module>",
            kind=seed.kind,
            message=_failure_message(seed.kind),
            run_context=run_context,
        )


def _failure_message(kind: str) -> str:
    return {
        "read_error": "unable to read boundary module for boundary/core contract checks",
        "syntax_error": "unable to parse boundary module for boundary/core contract checks",
        "cst_parse_failure": "unable to parse boundary module for boundary/core contract checks",
    }.get(kind, "unable to parse boundary module for boundary/core contract checks")


def _iter_resolved_core_paths(
    *,
    root: Path,
    core_imports: Sequence[_CoreImport],
) -> Iterable[tuple[str, Path | None]]:
    for core_import in core_imports:
        yield (
            core_import.module_dotted,
            next(
                _iter_module_path_from_dotted(root=root, dotted=core_import.module_dotted),
                None,
            ),
        )


def _iter_core_module_contract_violations(
    *,
    root: Path,
    boundary_rel_path: str,
    core_path: Path,
    run_context: CanonicalRunContext,
) -> Iterable[Violation]:
    core_source = next(_iter_read_source(core_path), None)
    if core_source is None:
        yield _violation(
            rel_path=boundary_rel_path,
            line=1,
            column=1,
            qualname="<module>",
            kind="core_read_error",
            message="unable to read paired core module for contract checks",
            run_context=run_context,
        )
        return

    core_rel = core_path.relative_to(root).as_posix()
    core_tree = next(_iter_parsed_tree(core_source), None)
    if core_tree is None:
        yield _violation(
            rel_path=core_rel,
            line=1,
            column=1,
            qualname="<module>",
            kind="core_syntax_error",
            message="paired core module must parse cleanly for contract checks",
            run_context=run_context,
        )
        return

    if _module_has_boundary_marker(core_source.splitlines()):
        yield _violation(
            rel_path=core_rel,
            line=1,
            column=1,
            qualname="<module>",
            kind="core_marked_as_boundary",
            message="paired core module must not be marked as boundary_normalization_module",
            run_context=run_context,
        )
    yield from _core_annotation_violations(rel_path=core_rel, tree=core_tree, run_context=run_context)
    yield from _core_narrowing_violations(rel_path=core_rel, tree=core_tree, run_context=run_context)
    yield from _core_branch_violations(rel_path=core_rel, tree=core_tree, run_context=run_context)


def _iter_read_source(path: Path) -> Iterable[str]:
    try:
        yield path.read_text(encoding="utf-8")
    except OSError:
        return


def _iter_parsed_tree(source: str) -> Iterable[ast.AST]:
    try:
        yield ast.parse(source)
    except SyntaxError:
        return


def _module_has_boundary_marker(source_lines: Sequence[str]) -> bool:
    return any(
        _is_boundary_marker_comment(stripped_line)
        for stripped_line in filter(None, map(str.strip, source_lines[:100]))
    )


def _is_boundary_marker_comment(stripped_line: str) -> bool:
    return stripped_line.startswith("#") and BOUNDARY_MARKER in stripped_line


def _collect_core_imports(*, tree: ast.AST, rel_path: str) -> tuple[_CoreImport, ...]:
    deduped: dict[str, set[str]] = {}
    for item in _iter_core_imports(tree=tree, package_parts=_package_parts_from_rel_path(rel_path)):
        deduped.setdefault(item.module_dotted, set()).update(item.alias_names)
    return tuple(_iter_deduped_core_imports(deduped))


def _iter_core_imports(
    *,
    tree: ast.AST,
    package_parts: tuple[str, ...],
) -> Iterable[_CoreImport]:
    for node in ast.walk(tree):
        yield from _iter_core_imports_from_node(node, package_parts)


def _iter_core_imports_from_node(
    node: ast.AST,
    package_parts: tuple[str, ...],
) -> Iterable[_CoreImport]:
    match node:
        case ast.Import():
            yield from _iter_core_imports_from_import(node, package_parts)
        case ast.ImportFrom():
            yield from _iter_core_imports_from_import_from(node, package_parts)
        case _:
            return ()


def _iter_core_imports_from_import(
    node: ast.Import,
    package_parts: tuple[str, ...],
) -> Iterable[_CoreImport]:
    del package_parts
    for alias in node.names:
        dotted = alias.name.strip()
        if dotted.endswith("_core"):
            alias_name = (alias.asname or dotted.rsplit(".", 1)[-1]).strip()
            if alias_name:
                yield _CoreImport(module_dotted=dotted, alias_names=(alias_name,))


def _iter_core_imports_from_import_from(
    node: ast.ImportFrom,
    package_parts: tuple[str, ...],
) -> Iterable[_CoreImport]:
    module_name = (node.module or "").strip()
    for resolved_module in _iter_resolved_import_from_module(
        module=module_name,
        level=int(node.level or 0),
        package_parts=package_parts,
    ):
        if resolved_module.endswith("_core"):
            alias_names = tuple(
                (alias.asname or alias.name).strip()
                for alias in node.names
                if alias.name != "*"
            )
            if alias_names:
                yield _CoreImport(module_dotted=resolved_module, alias_names=alias_names)
            continue
        yield from _iter_nested_core_imports_from_aliases(
            resolved_module=resolved_module,
            import_names=node.names,
        )
    return ()


def _iter_nested_core_imports_from_aliases(
    *,
    resolved_module: str,
    import_names: Sequence[ast.alias],
) -> Iterable[_CoreImport]:
    for alias in import_names:
        imported_name = alias.name.strip()
        if imported_name != "*" and imported_name.endswith("_core"):
            yield _CoreImport(
                module_dotted=f"{resolved_module}.{imported_name}",
                alias_names=((alias.asname or imported_name).strip(),),
            )


def _iter_deduped_core_imports(deduped: dict[str, set[str]]) -> Iterable[_CoreImport]:
    for module, names in sorted(deduped.items()):
        alias_names = tuple(sorted(name for name in names if name))
        if alias_names:
            yield _CoreImport(module_dotted=module, alias_names=alias_names)


def _package_parts_from_rel_path(rel_path: str) -> tuple[str, ...]:
    path = Path(rel_path)
    parts = list(path.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        return tuple(parts)
    return tuple(parts[:-1])


def _iter_resolved_import_from_module(
    *,
    module: str,
    level: int,
    package_parts: tuple[str, ...],
) -> Iterable[str]:
    if level <= 0:
        if module:
            yield module
        return
    if not package_parts:
        return
    if level == 1:
        base = package_parts
    else:
        trim = level - 1
        if trim > len(package_parts):
            return
        base = package_parts[: len(package_parts) - trim]
    base_text = ".".join(base)
    if module:
        if base_text:
            yield f"{base_text}.{module}"
            return
        yield module
        return
    if base_text:
        yield base_text


def _iter_module_path_from_dotted(*, root: Path, dotted: str) -> Iterable[Path]:
    if not dotted.startswith("gabion."):
        return
    rel = Path("src") / Path(*dotted.split("."))
    module_path = root / rel.with_suffix(".py")
    if module_path.exists():
        yield module_path
        return
    package_path = root / rel
    if package_path.is_dir():
        yield package_path


def _has_explicit_single_hop_core_call(*, tree: ast.AST, core_imports: Sequence[_CoreImport]) -> bool:
    module_aliases = {name for item in core_imports for name in item.alias_names}
    return any(_call_targets_core_alias(call, module_aliases) for call in _iter_call_nodes(tree))


def _iter_call_nodes(tree: ast.AST) -> Iterable[ast.Call]:
    for node in ast.walk(tree):
        match node:
            case ast.Call() as call:
                yield call
            case _:
                pass


def _call_targets_core_alias(call: ast.Call, module_aliases: set[str]) -> bool:
    match call.func:
        case ast.Attribute(value=ast.Name(id=module_alias_id)):
            return module_alias_id in module_aliases
        case ast.Name(id=callable_name):
            return callable_name in module_aliases
        case _:
            return False


def _core_annotation_violations(
    *,
    rel_path: str,
    tree: ast.AST,
    run_context: CanonicalRunContext,
) -> list[Violation]:
    violations: list[Violation] = []
    for node in _iter_function_nodes(tree):
        for line, col, text in filter(
            lambda site: _is_raw_ingress_annotation(site[2]),
            _iter_function_annotation_sites(node),
        ):
            violations.append(
                _violation(
                    rel_path=rel_path,
                    line=line,
                    column=col,
                    qualname=node.name,
                    kind="raw_ingress_type_in_core",
                    message=(
                        "core signature must not expose raw ingress types "
                        "(Any/object/dict[str, object])"
                    ),
                    run_context=run_context,
                )
            )
    return violations


def _iter_function_nodes(tree: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef() as function_node:
                yield function_node
            case _:
                pass


def _iter_function_annotation_sites(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterable[tuple[int, int, str]]:
    args = tuple(node.args.args) + tuple(node.args.kwonlyargs)
    for arg in args:
        yield from _iter_annotation_site_from_arg(arg=arg, fallback=node)
    if node.args.vararg is not None:
        yield from _iter_annotation_site_from_arg(arg=node.args.vararg, fallback=node)
    if node.args.kwarg is not None:
        yield from _iter_annotation_site_from_arg(arg=node.args.kwarg, fallback=node)
    if node.returns is not None:
        yield (
            int(node.returns.lineno or node.lineno),
            int(node.returns.col_offset or node.col_offset) + 1,
            ast.unparse(node.returns),
        )


def _iter_annotation_site_from_arg(
    *,
    arg: ast.arg,
    fallback: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterable[tuple[int, int, str]]:
    if arg.annotation is not None:
        yield (
            int(arg.lineno or fallback.lineno),
            int(arg.col_offset or fallback.col_offset) + 1,
            ast.unparse(arg.annotation),
        )


def _is_raw_ingress_annotation(annotation_text: str) -> bool:
    compact = annotation_text.replace(" ", "")
    return annotation_text in {"Any", "object"} or compact == "dict[str,object]"


def _core_narrowing_violations(
    *,
    rel_path: str,
    tree: ast.AST,
    run_context: CanonicalRunContext,
) -> list[Violation]:
    return [
        _violation(
            rel_path=rel_path,
            line=int(getattr(node, "lineno", 1) or 1),
            column=int(getattr(node, "col_offset", 0) or 0) + 1,
            qualname="<module>",
            kind="ingress_narrowing_in_core",
            message="core module must not perform runtime ingress narrowing",
            run_context=run_context,
        )
        for node in filter(_is_runtime_narrowing_node, ast.walk(tree))
    ]


def _is_runtime_narrowing_node(node: ast.AST) -> bool:
    match node:
        case ast.Call(func=ast.Name(id="isinstance")) | ast.Call(
            func=ast.Name(id="cast")
        ):
            return True
        case _:
            return False


def _core_branch_violations(
    *,
    rel_path: str,
    tree: ast.AST,
    run_context: CanonicalRunContext,
) -> list[Violation]:
    return [
        _violation(
            rel_path=rel_path,
            line=int(getattr(node, "lineno", 1) or 1),
            column=int(getattr(node, "col_offset", 0) or 0) + 1,
            qualname="<module>",
            kind="branch_in_core_module",
            message="paired core module must remain branchless",
            run_context=run_context,
        )
        for node in filter(_is_core_branch_node, ast.walk(tree))
    ]


def _is_core_branch_node(node: ast.AST) -> bool:
    match node:
        case (
            ast.If()
            | ast.IfExp()
            | ast.Match()
            | ast.For()
            | ast.AsyncFor()
            | ast.While()
            | ast.Try()
            | ast.TryStar()
        ):
            return True
        case _:
            return False


def _violation(
    *,
    rel_path: str,
    line: int,
    column: int,
    qualname: str,
    kind: str,
    message: str,
    run_context: CanonicalRunContext,
) -> Violation:
    input_slot = f"boundary_core:{kind}"
    decoration = decorate_site(
        run_context=run_context,
        rule_name=RULE_NAME,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=f"boundary_core:{kind}",
        input_slot=input_slot,
        taint_class="boundary_core_contract",
        intro_kind=f"syntax:boundary_core_taint:{kind}",
        condition_kind=f"syntax:boundary_core_condition:{kind}",
        erase_kind=f"syntax:boundary_core_erase:{kind}",
        rationale=(
            "Move boundary/core contract checks to explicit substrate boundaries so "
            "core modules stay normalized and branch-minimal."
        ),
    )
    structured_hash = _structured_hash(rel_path, qualname, kind, str(column), message)
    return Violation(
        path=rel_path,
        line=line,
        column=column,
        qualname=qualname,
        kind=kind,
        message=message,
        input_slot=input_slot,
        flow_identity=decoration.flow_identity,
        fiber_trace=decoration.fiber_trace,
        applicability_bounds=decoration.applicability_bounds,
        counterfactual_boundary=decoration.counterfactual_boundary,
        fiber_id=decoration.fiber_id,
        taint_interval_id=decoration.taint_interval_id,
        condition_overlap_id=decoration.condition_overlap_id,
        structured_hash=structured_hash,
    )


def _dedupe_exact_violations(violations: Sequence[Violation]) -> list[Violation]:
    return list(_iter_deduped_violations(violations))


def _iter_deduped_violations(violations: Sequence[Violation]) -> Iterable[Violation]:
    deduped_by_signature: dict[tuple[str, int, int, str, str, str], Violation] = {}
    for violation in violations:
        deduped_by_signature.setdefault(
            (
                violation.path,
                violation.line,
                violation.column,
                violation.qualname,
                violation.kind,
                violation.message,
            ),
            violation,
        )
    for violation in deduped_by_signature.values():
        yield violation


def _structured_hash(*parts: str) -> str:
    payload = "\x00".join(parts) + "\x00"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "Violation",
    "collect_violations",
]
