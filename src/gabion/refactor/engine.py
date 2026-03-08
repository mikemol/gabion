from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from pathlib import Path
from typing import cast

import libcst as cst
from libcst import matchers as cst_matchers

from gabion.refactor.model import (
    CompatibilityShimConfig, FieldSpec, LoopGeneratorRequest, RefactorPlan, RefactorPlanOutcome, RefactorRequest, RewritePlanEntry, TextEdit, normalize_compatibility_shim)
from gabion.refactor.loop_generator import plan_loop_generator_rewrite as _plan_loop_generator_rewrite
import gabion.refactor.cst_shared as cst_shared
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.order_contract import sort_once


class RefactorEngine:
    def __init__(self, project_root = None) -> None:
        self.project_root = project_root

    def plan_loop_generator_rewrite(self, request: LoopGeneratorRequest) -> RefactorPlan:
        return _plan_loop_generator_rewrite(
            request=request,
            project_root=self.project_root,
        )

    def plan_protocol_extraction(self, request: RefactorRequest) -> RefactorPlan:
        check_deadline()
        path = Path(request.target_path)
        if self.project_root and not path.is_absolute():
            path = self.project_root / path
        try:
            source = path.read_text()
        except Exception as exc:
            return RefactorPlan(outcome=RefactorPlanOutcome.ERROR, errors=[f"Failed to read {path}: {exc}"])
        try:
            module = cst.parse_module(source)
        except Exception as exc:
            return RefactorPlan(outcome=RefactorPlanOutcome.ERROR, errors=[f"LibCST parse failed for {path}: {exc}"])
        protocol = (request.protocol_name or "").strip()
        if not protocol:
            return RefactorPlan(outcome=RefactorPlanOutcome.ERROR, errors=["Protocol name is required for extraction."])
        bundle = [name.strip() for name in request.bundle or [] if name.strip()]
        field_specs: list[FieldSpec] = []
        seen_fields: set[str] = set()
        for spec in request.fields or []:
            check_deadline()
            name = (spec.name or "").strip()
            if not name or name in seen_fields:
                continue
            seen_fields.add(name)
            field_specs.append(spec)
        if bundle:
            for name in bundle:
                check_deadline()
                if name in seen_fields:
                    continue
                seen_fields.add(name)
                field_specs.append(FieldSpec(name=name))
        elif field_specs:
            bundle = [spec.name for spec in field_specs]
        if not bundle:
            return RefactorPlan(outcome=RefactorPlanOutcome.ERROR, errors=["Bundle fields are required for extraction."])

        body = list(module.body)

        insert_idx = _find_import_insert_index(body)

        protocol_base: cst.CSTNode
        import_stmt = None
        if _has_typing_import(body):
            protocol_base = cst.Attribute(cst.Name("typing"), cst.Name("Protocol"))
        elif _has_typing_protocol_import(body):
            protocol_base = cst.Name("Protocol")
        else:
            protocol_base = cst.Name("Protocol")
            import_stmt = cst.SimpleStatementLine(
                [cst.ImportFrom(module=cst.Name("typing"), names=[cst.ImportAlias(name=cst.Name("Protocol"))])]
            )

        doc_lines = []
        if request.rationale:
            doc_lines.append(f"Rationale: {request.rationale}")
        doc_lines.append(f"Bundle: {', '.join(bundle)}")
        docstring = cst.SimpleStatementLine(
            [cst.Expr(cst.SimpleString('"""' + "\\n".join(doc_lines) + '"""'))]
        )
        warnings: list[str] = []
        rewrite_plans: list[RewritePlanEntry] = []

        def _annotation_for(hint) -> cst.BaseExpression:
            if not hint:
                return cst.Name("object")
            try:
                return cst.parse_expression(hint)
            except Exception as exc:
                warnings.append(f"Failed to parse type hint '{hint}': {exc}")
                return cst.Name("object")

        field_lines = []
        for spec in field_specs:
            check_deadline()
            field_lines.append(
                cst.SimpleStatementLine(
                    [
                        cst.AnnAssign(
                            target=cst.Name(spec.name),
                            annotation=cst.Annotation(_annotation_for(spec.type_hint)),
                            value=None,
                        )
                    ]
                )
            )
        class_body = [docstring] + field_lines
        class_def = cst.ClassDef(
            name=cst.Name(protocol),
            bases=[cst.Arg(protocol_base)],
            body=cst.IndentedBlock(body=class_body),
        )

        new_body = list(body)
        if import_stmt is not None:
            new_body.insert(insert_idx, import_stmt)
            insert_idx += 1
        new_body.insert(insert_idx, cst.EmptyLine())
        insert_idx += 1
        new_body.insert(insert_idx, class_def)

        new_module = module.with_changes(body=new_body)

        targets = {name.strip() for name in request.target_functions or [] if name.strip()}
        bundle_fields = [spec.name for spec in field_specs]
        protocol_hint = protocol
        if targets:
            compat_shim = normalize_compatibility_shim(request.compatibility_shim)
            if compat_shim.enabled:
                new_module = _ensure_compat_imports(new_module, compat_shim)
            if request.ambient_rewrite:
                ambient_transformer = _AmbientRewriteTransformer(
                    targets=targets,
                    bundle_fields=bundle_fields,
                    protocol_hint=protocol_hint,
                )
                rewritten_module = new_module.visit(ambient_transformer)
                rewrite_plans.extend(ambient_transformer.plan_entries)
                warnings.extend(ambient_transformer.warnings)
                if ambient_transformer.changed:
                    new_module = _ensure_ambient_scaffolding(
                        rewritten_module,
                        context_names=ambient_transformer.rewritten_context_names,
                        protocol_hint=protocol_hint,
                    )
                else:
                    new_module = rewritten_module
            else:
                transformer = _RefactorTransformer(
                    targets=targets,
                    bundle_fields=bundle_fields,
                    protocol_hint=protocol_hint,
                    compat_shim=compat_shim,
                )
                new_module = new_module.visit(transformer)
                warnings.extend(transformer.warnings)

        project_callsite_edits: list[TextEdit] = []
        project_callsite_warnings: list[str] = []
        if targets and not request.ambient_rewrite:
            target_module = _module_name(path, self.project_root)
            module_validation = _validated_module_identifier(target_module)
            if module_validation.status is _ModuleIdentifierValidationStatus.INVALID:
                return RefactorPlan(
                    outcome=RefactorPlanOutcome.ERROR,
                    errors=[f"Invalid target module identifier: {target_module}"],
                )
            validated_target_module_identifier = module_validation.identifier
            call_warnings, call_edits = _rewrite_call_sites(
                new_module,
                file_path=path,
                target_path=path,
                target_module=validated_target_module_identifier,
                protocol_name=protocol,
                bundle_fields=bundle_fields,
                targets=targets,
            )
            warnings.extend(call_warnings)
            if call_edits is None:
                new_source = new_module.code
            else:
                new_module = call_edits
                new_source = new_module.code
            if self.project_root:
                project_callsite_edits, project_callsite_warnings = _rewrite_call_sites_in_project(
                    project_root=self.project_root,
                    target_path=path,
                    target_module=validated_target_module_identifier,
                    protocol_name=protocol,
                    bundle_fields=bundle_fields,
                    targets=targets,
                )
        else:
            new_source = new_module.code
        end_line = len(source.splitlines())
        edits = [
            TextEdit(
                path=str(path),
                start=(0, 0),
                end=(end_line, 0),
                replacement=new_source,
            )
        ]
        edits.extend(project_callsite_edits)
        warnings.extend(project_callsite_warnings)
        return RefactorPlan(edits=edits, rewrite_plans=rewrite_plans, warnings=warnings)


@dataclass(frozen=True)
class _ValidatedModuleIdentifier:
    value: str
    expression: cst.BaseExpression


class _ModuleIdentifierValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class _ModuleIdentifierValidation:
    status: _ModuleIdentifierValidationStatus
    identifier: _ValidatedModuleIdentifier


def _validated_module_identifier(module_name: str) -> _ModuleIdentifierValidation:
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        return _ModuleIdentifierValidation(
            status=_ModuleIdentifierValidationStatus.INVALID,
            identifier=_ValidatedModuleIdentifier(
                value=module_name,
                expression=cst.Name("_invalid_module_identifier"),
            ),
        )
    if any(not part.isidentifier() for part in parts):
        return _ModuleIdentifierValidation(
            status=_ModuleIdentifierValidationStatus.INVALID,
            identifier=_ValidatedModuleIdentifier(
                value=module_name,
                expression=cst.Name("_invalid_module_identifier"),
            ),
        )
    expression: cst.BaseExpression = cst.Name(parts[0])
    for part in parts[1:]:
        expression = cst.Attribute(value=expression, attr=cst.Name(part))
    return _ModuleIdentifierValidation(
        status=_ModuleIdentifierValidationStatus.VALID,
        identifier=_ValidatedModuleIdentifier(value=".".join(parts), expression=expression),
    )


def _module_name(path: Path, project_root) -> str:
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


def _is_docstring(stmt: cst.CSTNode) -> bool:
    return cst_shared.is_docstring_statement(stmt)


@singledispatch
def _simple_statement_line_or_none(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_simple_statement_line_or_none.register(cst.SimpleStatementLine)
def _(stmt: cst.SimpleStatementLine):
    return stmt


def _simple_statement_line_none(value: object):
    _ = value
    return None


for _stmt_type in (
    cst.For,
    cst.While,
    cst.If,
    cst.With,
    cst.Try,
    cst.TryStar,
    cst.Match,
    cst.FunctionDef,
    cst.ClassDef,
    cst.EmptyLine,
):
    _simple_statement_line_or_none.register(_stmt_type)(_simple_statement_line_none)


@singledispatch
def _import_or_none(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_import_or_none.register(cst.Import)
def _(stmt: cst.Import):
    return stmt


@singledispatch
def _import_from_or_none(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_import_from_or_none.register(cst.ImportFrom)
def _(stmt: cst.ImportFrom):
    return stmt


def _small_statement_none(value: object):
    _ = value
    return None


for _small_stmt_type in (
    cst.AnnAssign,
    cst.Assert,
    cst.Assign,
    cst.AugAssign,
    cst.Break,
    cst.Continue,
    cst.Del,
    cst.Expr,
    cst.Global,
    cst.ImportFrom,
    cst.Nonlocal,
    cst.Pass,
    cst.Raise,
    cst.Return,
    cst.TypeAlias,
):
    _import_or_none.register(_small_stmt_type)(_small_statement_none)


for _import_from_nonmatch_type in (
    cst.AnnAssign,
    cst.Assert,
    cst.Assign,
    cst.AugAssign,
    cst.Break,
    cst.Continue,
    cst.Del,
    cst.Expr,
    cst.Global,
    cst.Import,
    cst.Nonlocal,
    cst.Pass,
    cst.Raise,
    cst.Return,
    cst.TypeAlias,
):
    _import_from_or_none.register(_import_from_nonmatch_type)(_small_statement_none)


@singledispatch
def _import_alias_or_none(alias: object):
    never("unregistered runtime type", value_type=type(alias).__name__)


@_import_alias_or_none.register(cst.ImportAlias)
def _(alias: cst.ImportAlias):
    return alias


@_import_alias_or_none.register(cst.ImportStar)
def _(alias: cst.ImportStar):
    _ = alias
    return None


@singledispatch
def _import_alias_sequence_or_none(names: object):
    never("unregistered runtime type", value_type=type(names).__name__)


@_import_alias_sequence_or_none.register(list)
def _(names: list[object]):
    return names


@_import_alias_sequence_or_none.register(tuple)
def _(names: tuple[object, ...]):
    return list(names)


@_import_alias_sequence_or_none.register(cst.ImportStar)
def _(names: cst.ImportStar):
    _ = names
    return None


@singledispatch
def _indented_block_or_none(body: object):
    never("unregistered runtime type", value_type=type(body).__name__)


@_indented_block_or_none.register(cst.IndentedBlock)
def _(body: cst.IndentedBlock):
    return body


@_indented_block_or_none.register(cst.SimpleStatementSuite)
def _(body: cst.SimpleStatementSuite):
    _ = body
    return None


@_indented_block_or_none.register(cst.SimpleStatementLine)
def _(body: cst.SimpleStatementLine):
    _ = body
    return None


@singledispatch
def _top_level_function_name_or_none(node: object) -> str:
    never("unregistered runtime type", value_type=type(node).__name__)


@_top_level_function_name_or_none.register(cst.FunctionDef)
def _(node: cst.FunctionDef) -> str:
    return node.name.value


def _empty_function_name(value: object) -> str:
    _ = value
    return ""


for _node_type in (
    cst.SimpleStatementLine,
    cst.For,
    cst.While,
    cst.If,
    cst.With,
    cst.Try,
    cst.TryStar,
    cst.Match,
    cst.ClassDef,
    cst.EmptyLine,
):
    _top_level_function_name_or_none.register(_node_type)(_empty_function_name)


def _is_import(stmt: cst.CSTNode) -> bool:
    line = _simple_statement_line_or_none(stmt)
    if line is None:
        return False
    for item in line.body:
        check_deadline()
        if _import_or_none(item) is not None or _import_from_or_none(item) is not None:
            return True
    return False


def _find_import_insert_index(body: list[cst.CSTNode]) -> int:
    return cst_shared.find_import_insert_index(
        body,
        check_deadline_fn=check_deadline,
    )


def _module_expr_to_str(expr):
    return cst_shared.module_expr_to_str(expr, check_deadline_fn=check_deadline)


def _has_typing_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        check_deadline()
        line = _simple_statement_line_or_none(stmt)
        if line is not None:
            for item in line.body:
                check_deadline()
                import_item = _import_or_none(item)
                if import_item is not None:
                    for alias in import_item.names:
                        check_deadline()
                        import_alias = _import_alias_or_none(alias)
                        if (
                            import_alias is not None
                            and _module_expr_to_str(import_alias.name) == "typing"
                        ):
                            return True
    return False


def _has_typing_protocol_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        check_deadline()
        line = _simple_statement_line_or_none(stmt)
        if line is not None:
            for item in line.body:
                check_deadline()
                import_from = _import_from_or_none(item)
                if import_from is not None:
                    module = _module_expr_to_str(import_from.module)
                    aliases = _import_alias_sequence_or_none(import_from.names)
                    if module == "typing" and aliases is not None:
                        for alias in aliases:
                            check_deadline()
                            import_alias = _import_alias_or_none(alias)
                            if (
                                import_alias is not None
                                and _module_expr_to_str(import_alias.name) == "Protocol"
                            ):
                                return True
    return False


def _has_typing_overload_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        check_deadline()
        line = _simple_statement_line_or_none(stmt)
        if line is not None:
            for item in line.body:
                check_deadline()
                import_from = _import_from_or_none(item)
                if import_from is not None:
                    module = _module_expr_to_str(import_from.module)
                    aliases = _import_alias_sequence_or_none(import_from.names)
                    if module == "typing" and aliases is not None:
                        for alias in aliases:
                            check_deadline()
                            import_alias = _import_alias_or_none(alias)
                            if (
                                import_alias is not None
                                and _module_expr_to_str(import_alias.name) == "overload"
                            ):
                                return True
    return False


def _has_warnings_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        check_deadline()
        line = _simple_statement_line_or_none(stmt)
        if line is not None:
            for item in line.body:
                check_deadline()
                import_item = _import_or_none(item)
                if import_item is not None:
                    for alias in import_item.names:
                        check_deadline()
                        import_alias = _import_alias_or_none(alias)
                        if (
                            import_alias is not None
                            and _module_expr_to_str(import_alias.name) == "warnings"
                        ):
                            return True
    return False


def _ensure_compat_imports(
    module: cst.Module, shim: CompatibilityShimConfig
) -> cst.Module:
    body = list(module.body)
    insert_idx = _find_import_insert_index(body)
    if shim.emit_deprecation_warning and not _has_warnings_import(body):
        body.insert(
            insert_idx,
            cst.SimpleStatementLine(
                [cst.Import(names=[cst.ImportAlias(name=cst.Name("warnings"))])]
            ),
        )
        insert_idx += 1
    if shim.emit_overload_stubs and not _has_typing_overload_import(body):
        body.insert(
            insert_idx,
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name("typing"),
                        names=[cst.ImportAlias(name=cst.Name("overload"))],
                    )
                ]
            ),
        )
        insert_idx += 1
    return module.with_changes(body=body)


def _collect_import_context(
    module: cst.Module,
    *,
    target_module: _ValidatedModuleIdentifier,
    protocol_name: str,
):
    check_deadline()
    module_aliases: dict[str, str] = {}
    imported_targets: dict[str, str] = {}
    protocol_alias = None
    for stmt in module.body:
        check_deadline()
        line = _simple_statement_line_or_none(stmt)
        if line is not None:
            for item in line.body:
                check_deadline()
                import_item = _import_or_none(item)
                if import_item is not None:
                    for alias in import_item.names:
                        check_deadline()
                        import_alias = _import_alias_or_none(alias)
                        if import_alias is not None:
                            module_name = _module_expr_to_str(import_alias.name)
                            if module_name and module_name == target_module.value:
                                local = (
                                    import_alias.asname.name.value
                                    if import_alias.asname
                                    else module_name
                                )
                                module_aliases[local] = module_name
                else:
                    import_from = _import_from_or_none(item)
                    if import_from is not None:
                        module_name = _module_expr_to_str(import_from.module)
                        aliases = _import_alias_sequence_or_none(import_from.names)
                        if module_name == target_module.value and aliases is not None:
                            for alias in aliases:
                                check_deadline()
                                import_alias = _import_alias_or_none(alias)
                                if import_alias is not None and cst_matchers.matches(
                                    import_alias.name, cst_matchers.Name()
                                ):
                                    alias_name = cast(cst.Name, import_alias.name)
                                    local = (
                                        import_alias.asname.name.value
                                        if import_alias.asname
                                        else alias_name.value
                                    )
                                    imported_targets[local] = alias_name.value
                                    if alias_name.value == protocol_name:
                                        protocol_alias = local
    return module_aliases, imported_targets, protocol_alias


def _rewrite_call_sites(
    module: cst.Module,
    *,
    file_path: Path,
    target_path: Path,
    target_module: _ValidatedModuleIdentifier,
    protocol_name: str,
    bundle_fields: list[str],
    targets: set[str],
):
    check_deadline()
    warnings: list[str] = []
    file_is_target = file_path == target_path
    if not targets:
        return warnings, None
    target_simple = {name for name in targets if "." not in name}
    target_methods: dict[str, set[str]] = {}
    for name in targets:
        check_deadline()
        if "." not in name:
            continue
        parts = name.split(".")
        class_name = ".".join(parts[:-1])
        method = parts[-1]
        target_methods.setdefault(class_name, set()).add(method)
    module_aliases: dict[str, str] = {}
    imported_targets: dict[str, str] = {}
    protocol_alias = None
    if not file_is_target:
        module_aliases, imported_targets, protocol_alias = _collect_import_context(
            module, target_module=target_module, protocol_name=protocol_name
        )

    constructor_expr: cst.BaseExpression
    needs_import = False
    if file_is_target:
        constructor_expr = cst.Name(protocol_name)
    else:
        if protocol_alias:
            constructor_expr = cst.Name(protocol_alias)
        elif module_aliases:
            alias = sort_once(
                module_aliases.keys(),
                source="_rewrite_call_sites.module_aliases",
            )[0]
            constructor_expr = cst.Attribute(cst.Name(alias), cst.Name(protocol_name))
        else:
            constructor_expr = cst.Name(protocol_name)
            needs_import = True

    transformer = _CallSiteTransformer(
        file_is_target=file_is_target,
        target_simple=target_simple,
        target_methods=target_methods,
        module_aliases=set(module_aliases.keys()),
        imported_targets=set(
            name for name, original in imported_targets.items() if original in target_simple
        ),
        bundle_fields=bundle_fields,
        constructor_expr=constructor_expr,
    )
    new_module = module.visit(transformer)
    warnings.extend(transformer.warnings)
    if not transformer.changed:
        return warnings, None

    if not file_is_target and needs_import:
        body = list(new_module.body)
        insert_idx = _find_import_insert_index(body)
        import_stmt = cst.SimpleStatementLine(
            [
                cst.ImportFrom(
                    module=target_module.expression,
                    names=[cst.ImportAlias(name=cst.Name(protocol_name))],
                )
            ]
        )
        body.insert(insert_idx, import_stmt)
        new_module = new_module.with_changes(body=body)
    return warnings, new_module


def _rewrite_call_sites_in_project(
    *,
    project_root: Path,
    target_path: Path,
    target_module: _ValidatedModuleIdentifier,
    protocol_name: str,
    bundle_fields: list[str],
    targets: set[str],
) -> tuple[list[TextEdit], list[str]]:
    check_deadline()
    edits: list[TextEdit] = []
    warnings: list[str] = []
    scan_root = project_root / "src"
    if not scan_root.exists():
        scan_root = project_root
    for path in sort_once(
        scan_root.rglob("*.py"),
        source="_rewrite_call_sites_in_project.scan_root",
        key=lambda item: str(item),
    ):
        check_deadline()
        if path == target_path:
            continue
        try:
            source = path.read_text()
        except Exception as exc:
            warnings.append(f"Failed to read {path}: {exc}")
            continue
        try:
            module = cst.parse_module(source)
        except Exception as exc:
            warnings.append(f"LibCST parse failed for {path}: {exc}")
            continue
        call_warnings, updated_module = _rewrite_call_sites(
            module,
            file_path=path,
            target_path=target_path,
            target_module=target_module,
            protocol_name=protocol_name,
            bundle_fields=bundle_fields,
            targets=targets,
        )
        warnings.extend(call_warnings)
        if updated_module is not None:
            new_source = updated_module.code
            end_line = len(source.splitlines())
            edits.append(
                TextEdit(
                    path=str(path),
                    start=(0, 0),
                    end=(end_line, 0),
                    replacement=new_source,
                )
            )
    return edits, warnings




def _ensure_ambient_scaffolding(
    module: cst.Module,
    *,
    context_names: set[str],
    protocol_hint: str,
) -> cst.Module:
    if not context_names:
        return module
    body = list(module.body)
    insert_idx = _find_import_insert_index(body)
    if not _has_contextvars_import(body):
        body.insert(
            insert_idx,
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name("contextvars"),
                        names=[cst.ImportAlias(name=cst.Name("ContextVar"))],
                    )
                ]
            ),
        )
        insert_idx += 1

    existing_top_level: set[str] = set()
    for node in body:
        check_deadline()
        function_name = _top_level_function_name_or_none(node)
        if function_name:
            existing_top_level.add(function_name)
    for context_name in sort_once(
        context_names,
        source="_ensure_ambient_scaffolding.context_names",
    ):
        check_deadline()
        ambient_name = _ambient_var_name(context_name)
        getter = _ambient_getter_name(context_name)
        setter = _ambient_setter_name(context_name)
        if getter in existing_top_level and setter in existing_top_level:
            continue
        annotation = protocol_hint or "object"
        statements = cst.parse_module(
            f"""
{ambient_name}: ContextVar[{annotation} | None] = ContextVar("{ambient_name}", default=None)

def {getter}() -> {annotation}:
    value = {ambient_name}.get()
    if value is None:
        raise RuntimeError("Ambient context '{context_name}' is not set.")
    return value

def {setter}(value: {annotation}) -> None:
    {ambient_name}.set(value)
"""
        ).body
        body[insert_idx:insert_idx] = statements
        insert_idx += len(statements)
    return module.with_changes(body=body)


def _has_contextvars_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    for stmt in body:
        check_deadline()
        line = _simple_statement_line_or_none(stmt)
        if line is not None:
            for item in line.body:
                check_deadline()
                import_from = _import_from_or_none(item)
                if import_from is not None:
                    aliases = _import_alias_sequence_or_none(import_from.names)
                    if _module_expr_to_str(import_from.module) == "contextvars" and aliases is not None:
                        for alias in aliases:
                            check_deadline()
                            import_alias = _import_alias_or_none(alias)
                            if (
                                import_alias is not None
                                and _module_expr_to_str(import_alias.name) == "ContextVar"
                            ):
                                return True
    return False


def _ambient_var_name(name: str) -> str:
    return f"_AMBIENT_{name.upper()}"


def _ambient_getter_name(name: str) -> str:
    return f"_ambient_get_{name}"


def _ambient_setter_name(name: str) -> str:
    return f"_ambient_set_{name}"


class _AmbientArgThreadingRewriter(cst.CSTTransformer):
    def __init__(self, *, targets: set[str], context_name: str, current: str) -> None:
        self.targets = {name.split(".")[-1] for name in targets}
        self.context_name = context_name
        self.current = current
        self.changed = False
        self.skipped_reasons: list[str] = []

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        func = updated_node.func
        target_name = None
        if cst_matchers.matches(func, cst_matchers.Name()):
            target_name = cast(cst.Name, func).value
        elif cst_matchers.matches(func, cst_matchers.Attribute()):
            target_name = cast(cst.Attribute, func).attr.value
        if target_name not in self.targets:
            return updated_node
        if target_name == self.current:
            return updated_node
        if any(arg.star in {"*", "**"} for arg in updated_node.args):
            self.skipped_reasons.append(
                f"{self.current}: skipped call rewrite with dynamic star args for ambient parameter '{self.context_name}'."
            )
            return updated_node
        new_args: list[cst.Arg] = []
        removed = False
        for arg in updated_node.args:
            check_deadline()
            arg_name = (
                cast(cst.Name, arg.value).value
                if cst_matchers.matches(arg.value, cst_matchers.Name())
                else None
            )
            is_name = arg_name == self.context_name
            if arg.keyword is not None and arg.keyword.value == self.context_name and is_name:
                removed = True
                continue
            if arg.keyword is None and is_name and len(updated_node.args) == 1:
                removed = True
                continue
            if arg.keyword is None and is_name and len(updated_node.args) > 1:
                self.skipped_reasons.append(
                    f"{self.current}: skipped positional ambient argument rewrite for '{self.context_name}' due to ambiguous arity."
                )
            new_args.append(arg)
        if removed:
            self.changed = True
            return updated_node.with_changes(args=new_args)
        return updated_node


class _AmbientSafetyVisitor(cst.CSTVisitor):
    def __init__(self, context_name: str) -> None:
        self.context_name = context_name
        self.reasons: list[str] = []

    def visit_AssignTarget(self, node: cst.AssignTarget) -> None:
        if not cst_matchers.matches(node.target, cst_matchers.Name()):
            return
        target_name = cast(cst.Name, node.target).value
        if target_name == self.context_name:
            self.reasons.append(
                f"writes to parameter '{self.context_name}' prevent a safe ambient rewrite"
            )


class _AmbientRewriteTransformer(cst.CSTTransformer):
    def __init__(self, *, targets: set[str], bundle_fields: list[str], protocol_hint: str) -> None:
        self.targets = targets
        self.bundle_fields = bundle_fields
        self.protocol_hint = protocol_hint
        self.changed = False
        self.warnings: list[str] = []
        self.plan_entries: list[RewritePlanEntry] = []
        self.rewritten_context_names: set[str] = set()

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        return self._rewrite_function(updated_node)

    def leave_AsyncFunctionDef(
        self, original_node: cst.AsyncFunctionDef, updated_node: cst.AsyncFunctionDef
    ) -> cst.CSTNode:
        return self._rewrite_function(updated_node)

    def _rewrite_function(self, node) -> cst.CSTNode:
        check_deadline()
        name = node.name.value
        if name not in self.targets:
            return node
        if not self.bundle_fields:
            self.plan_entries.append(RewritePlanEntry(
                kind="AMBIENT_REWRITE",
                status="skipped",
                target=name,
                summary="No bundle fields were provided.",
                non_rewrite_reasons=["missing bundle field context parameter"],
            ))
            return node
        context_name = self.bundle_fields[0]
        params = list(node.params.params)
        match_param = next((param for param in params if param.name.value == context_name), None)
        if match_param is None:
            self.plan_entries.append(RewritePlanEntry(
                kind="AMBIENT_REWRITE",
                status="noop",
                target=name,
                summary=f"No threaded context parameter '{context_name}' found.",
            ))
            return node
        if node.params.star_arg is not cst.MaybeSentinel.DEFAULT or node.params.star_kwarg is not None:
            self.plan_entries.append(RewritePlanEntry(
                kind="AMBIENT_REWRITE",
                status="skipped",
                target=name,
                summary="Dynamic argument capture prevents safe ambient rewrite.",
                non_rewrite_reasons=["function uses *args or **kwargs"],
            ))
            return node
        safety = _AmbientSafetyVisitor(context_name)
        node.body.visit(safety)
        if safety.reasons:
            self.plan_entries.append(RewritePlanEntry(
                kind="AMBIENT_REWRITE",
                status="skipped",
                target=name,
                summary="Aliasing/dynamic writes prevent safe ambient rewrite.",
                non_rewrite_reasons=safety.reasons,
            ))
            return node

        rewriter = _AmbientArgThreadingRewriter(targets=self.targets, context_name=context_name, current=name)
        updated_body = node.body.visit(rewriter)
        preamble = list(
            cst.parse_module(
                f"""if {context_name} is None:
    {context_name} = {_ambient_getter_name(context_name)}()
else:
    {_ambient_setter_name(context_name)}({context_name})
"""
            ).body
        )
        updated_block = _indented_block_or_none(updated_body)
        if updated_block is not None:
            existing = list(updated_block.body)
            insert_at = 1 if existing and _is_docstring(existing[0]) else 0
            updated_body = updated_block.with_changes(body=existing[:insert_at] + preamble + existing[insert_at:])

        updated_params: list[cst.Param] = []
        for param in node.params.params:
            if param.name.value != context_name:
                updated_params.append(param)
                continue
            annotation = param.annotation
            if annotation is None:
                try:
                    annotation = cst.Annotation(cst.parse_expression(f"{self.protocol_hint} | None"))
                except Exception:
                    annotation = cst.Annotation(cst.parse_expression("object | None"))
            updated_params.append(param.with_changes(default=cst.Name("None"), annotation=annotation))

        updated_node = node.with_changes(
            params=node.params.with_changes(params=updated_params),
            body=updated_body,
        )
        self.changed = True
        self.rewritten_context_names.add(context_name)
        reasons = rewriter.skipped_reasons
        summary = f"Migrated threaded parameter '{context_name}' to ambient accessor scaffold."
        self.plan_entries.append(RewritePlanEntry(
            kind="AMBIENT_REWRITE",
            status="applied",
            target=name,
            summary=summary,
            non_rewrite_reasons=reasons,
        ))
        if reasons:
            self.warnings.extend(reasons)
        return updated_node

class _RefactorTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        targets: set[str],
        bundle_fields: list[str],
        protocol_hint: str,
        compat_shim: CompatibilityShimConfig = CompatibilityShimConfig(enabled=False),
    ) -> None:
        # dataflow-bundle: bundle_fields, compat_shim, protocol_hint, targets
        self.targets = targets
        self.bundle_fields = bundle_fields
        self.protocol_hint = protocol_hint
        self.compat_shim = compat_shim
        self.warnings: list[str] = []
        self._stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.CSTNode:
        if self._stack:
            self._stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        updated = self._maybe_rewrite_function(original_node, updated_node)
        if self._stack:
            self._stack.pop()
        return updated

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_AsyncFunctionDef(
        self, original_node: cst.AsyncFunctionDef, updated_node: cst.AsyncFunctionDef
    ) -> cst.CSTNode:
        updated = self._maybe_rewrite_function(original_node, updated_node)
        if self._stack:
            self._stack.pop()
        return updated

    def _maybe_rewrite_function(
        self,
        original_node: cst.FunctionDef,
        updated_node,
    ) -> cst.CSTNode:
        qualname = ".".join(self._stack)
        name = original_node.name.value
        if name not in self.targets and qualname not in self.targets:
            return updated_node

        original_params = self._ordered_param_names(original_node.params)
        if not original_params:
            return updated_node

        keep_self = original_params[0] in {"self", "cls"}
        self_param = None
        if keep_self:
            self_param = self._find_self_param(original_node.params, original_params[0])

        bundle_param_name = self._choose_bundle_name(original_params)
        new_params = self._build_parameters(self_param, bundle_param_name)

        bundle_set = set(self.bundle_fields)
        target_fields = [
            name
            for name in original_params
            if name in bundle_set and name not in {"self", "cls"}
        ]
        new_body = self._inject_preamble(
            updated_node.body, bundle_param_name, target_fields
        )
        updated_impl = updated_node.with_changes(params=new_params, body=new_body)
        if not self.compat_shim.enabled:
            return updated_impl
        impl_name = f"_{name}_bundle"
        impl_node = updated_impl.with_changes(
            name=cst.Name(impl_name),
            decorators=[],
        )
        shim_nodes = self._build_compat_shim(
            original_node=original_node,
            impl_name=impl_name,
            bundle_param=bundle_param_name,
            self_param=self_param,
        )
        return cst.FlattenSentinel([*shim_nodes, impl_node])

    def _build_compat_shim(
        self,
        *,
        original_node,
        impl_name: str,
        bundle_param: str,
        self_param,
    ) -> list[cst.CSTNode]:
        decorators = list(original_node.decorators)
        overload_decorators = [cst.Decorator(cst.Name("overload")), *decorators]
        bundle_params = self._build_parameters(self_param, bundle_param)
        legacy_params = original_node.params
        wrapper_params = self._build_shim_parameters(self_param)
        wrapper_body = self._build_shim_body(
            impl_name=impl_name,
            bundle_type=self.protocol_hint,
            self_param=self_param,
            is_async=bool(original_node.asynchronous),
            public_name=original_node.name.value,
        )
        wrapper = self._build_wrapper(
            original_node=original_node,
            params=wrapper_params,
            body=wrapper_body,
            decorators=decorators,
        )
        shim_nodes: list[cst.CSTNode] = []
        if self.compat_shim.emit_overload_stubs:
            bundle_stub = self._build_overload_stub(
                original_node=original_node,
                params=bundle_params,
                decorators=overload_decorators,
            )
            legacy_stub = self._build_overload_stub(
                original_node=original_node,
                params=legacy_params,
                decorators=overload_decorators,
            )
            shim_nodes.extend([bundle_stub, legacy_stub])
        shim_nodes.append(wrapper)
        return shim_nodes

    def _build_overload_stub(
        self,
        *,
        original_node: cst.FunctionDef,
        params: cst.Parameters,
        decorators: list[cst.Decorator],
    ) -> cst.CSTNode:
        body = cst.IndentedBlock(
            body=[cst.SimpleStatementLine([cst.Expr(cst.Ellipsis())])]
        )
        return original_node.with_changes(
            params=params,
            body=body,
            decorators=decorators,
        )

    def _build_wrapper(
        self,
        *,
        original_node: cst.FunctionDef,
        params: cst.Parameters,
        body: cst.BaseSuite,
        decorators: list[cst.Decorator],
    ) -> cst.CSTNode:
        return original_node.with_changes(
            params=params,
            body=body,
            decorators=decorators,
        )

    def _build_shim_parameters(self, self_param) -> cst.Parameters:
        params: list[cst.Param] = []
        if self_param is not None:
            params.append(self_param)
        return cst.Parameters(
            params=params,
            star_arg=cst.Param(name=cst.Name("args")),
            kwonly_params=[],
            star_kwarg=cst.Param(name=cst.Name("kwargs")),
            posonly_params=[],
            posonly_ind=cst.MaybeSentinel.DEFAULT,
        )

    def _build_shim_body(
        self,
        *,
        impl_name: str,
        bundle_type: str,
        self_param,
        is_async: bool,
        public_name: str,
    ) -> cst.BaseSuite:
        receiver = self_param.name.value if self_param is not None else None
        impl_call = f"{receiver}.{impl_name}" if receiver else impl_name
        return_prefix = "return await" if is_async else "return"
        guard = (
            "if args:\n"
            "    match args[0]:\n"
            f"        case {bundle_type}() as bundle:\n"
            f"            if len(args) != 1 or kwargs:\n"
            f"                raise TypeError(\"{public_name}() bundle call expects a single {bundle_type} argument\")\n"
            f"            {return_prefix} {impl_call}(bundle)\n"
        )
        build = f"bundle = {bundle_type}(*args, **kwargs)"
        tail = f"{return_prefix} {impl_call}(bundle)"
        body: list[cst.BaseStatement] = [cst.parse_statement(guard)]
        if self.compat_shim.emit_deprecation_warning:
            warn = (
                f"warnings.warn(\"{public_name}() is deprecated; use {public_name}({bundle_type}(...))\", "
                "DeprecationWarning, stacklevel=2)"
            )
            body.append(cst.parse_statement(warn))
        body.extend([cst.parse_statement(build), cst.parse_statement(tail)])
        return cst.IndentedBlock(body=body)

    def _ordered_param_names(self, params: cst.Parameters) -> list[str]:
        check_deadline()
        names: list[str] = []
        for param in params.posonly_params:
            check_deadline()
            names.append(param.name.value)
        for param in params.params:
            check_deadline()
            names.append(param.name.value)
        for param in params.kwonly_params:
            check_deadline()
            names.append(param.name.value)
        return names

    def _find_self_param(
        self, params: cst.Parameters, name: str
    ):
        check_deadline()
        for param in params.posonly_params:
            check_deadline()
            if param.name.value == name:
                return param
        for param in params.params:
            check_deadline()
            if param.name.value == name:
                return param
        return None

    def _choose_bundle_name(self, existing: list[str]) -> str:
        check_deadline()
        candidate = "bundle"
        if candidate not in existing:
            return candidate
        idx = 1
        while f"bundle_{idx}" in existing:
            check_deadline()
            idx += 1
        return f"bundle_{idx}"

    def _build_parameters(
        self, self_param, bundle_name: str
    ) -> cst.Parameters:
        params: list[cst.Param] = []
        if self_param is not None:
            params.append(self_param)
        annotation = None
        if self.protocol_hint:
            try:
                annotation = cst.Annotation(cst.parse_expression(self.protocol_hint))
            except Exception as exc:
                self.warnings.append(
                    f"Failed to parse protocol type hint '{self.protocol_hint}': {exc}"
                )
        params.append(
            cst.Param(
                name=cst.Name(bundle_name),
                annotation=annotation,
            )
        )
        return cst.Parameters(
            params=params,
            star_arg=cst.MaybeSentinel.DEFAULT,
            kwonly_params=[],
            star_kwarg=None,
            posonly_params=[],
            posonly_ind=cst.MaybeSentinel.DEFAULT,
        )

    def _inject_preamble(
        self, body: cst.BaseSuite, bundle_name: str, fields: list[str]
    ) -> cst.BaseSuite:
        if not fields:
            return body
        body_block = _indented_block_or_none(body)
        if body_block is None:
            return body
        assign_lines = [
            cst.SimpleStatementLine(
                [
                    cst.Assign(
                        targets=[cst.AssignTarget(cst.Name(name))],
                        value=cst.Attribute(
                            value=cst.Name(bundle_name),
                            attr=cst.Name(name),
                        ),
                    )
                ]
            )
            for name in fields
        ]
        existing = list(body_block.body)
        insert_at = 1 if existing and _is_docstring(existing[0]) else 0
        new_body = existing[:insert_at] + assign_lines + existing[insert_at:]
        return body_block.with_changes(body=new_body)


class _CallSiteTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        file_is_target: bool,
        target_simple: set[str],
        target_methods: dict[str, set[str]],
        module_aliases: set[str],
        imported_targets: set[str],
        bundle_fields: list[str],
        constructor_expr: cst.BaseExpression,
    ) -> None:
        # dataflow-bundle: bundle_fields, constructor_expr, file_is_target, imported_targets, module_aliases, target_methods, target_simple
        self.file_is_target = file_is_target
        self.target_simple = target_simple
        self.target_methods = target_methods
        self.module_aliases = module_aliases
        self.imported_targets = imported_targets
        self.bundle_fields = bundle_fields
        self.constructor_expr = constructor_expr
        self.changed = False
        self.warnings: list[str] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._class_stack.append(node.name.value)
        return True

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.CSTNode:
        if self._class_stack:
            self._class_stack.pop()
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        if not self._is_target_call(updated_node.func):
            return updated_node
        if self._already_wrapped(updated_node):
            return updated_node
        bundle_args = self._build_bundle_args(updated_node)
        if bundle_args is None:
            return updated_node
        bundle_call = cst.Call(func=self.constructor_expr, args=bundle_args)
        self.changed = True
        return updated_node.with_changes(args=[cst.Arg(value=bundle_call)])

    def _is_target_call(self, func: cst.BaseExpression) -> bool:
        if cst_matchers.matches(func, cst_matchers.Name()):
            func_name = cast(cst.Name, func)
            if self.file_is_target and func_name.value in self.target_simple:
                return True
            if not self.file_is_target and func_name.value in self.imported_targets:
                return True
            return False
        if cst_matchers.matches(func, cst_matchers.Attribute()):
            func_attr = cast(cst.Attribute, func)
            attr = func_attr.attr.value
            if self.file_is_target and self._class_stack:
                class_name = ".".join(self._class_stack)
                methods = self.target_methods.get(class_name, set())
                if attr in methods and cst_matchers.matches(func_attr.value, cst_matchers.Name()):
                    value_name = cast(cst.Name, func_attr.value).value
                    if value_name in {"self", "cls", self._class_stack[-1]}:
                        return True
            if not self.file_is_target and cst_matchers.matches(func_attr.value, cst_matchers.Name()):
                value_name = cast(cst.Name, func_attr.value).value
                if value_name in self.module_aliases and attr in self.target_simple:
                    return True
        return False

    def _already_wrapped(self, call: cst.Call) -> bool:
        if len(call.args) != 1:
            return False
        arg = call.args[0]
        if arg.star:
            return False
        value = arg.value
        if not cst_matchers.matches(value, cst_matchers.Call()):
            return False
        value_call = cast(cst.Call, value)
        if (
            cst_matchers.matches(value_call.func, cst_matchers.Name())
            and cst_matchers.matches(self.constructor_expr, cst_matchers.Name())
        ):
            return cast(cst.Name, value_call.func).value == cast(cst.Name, self.constructor_expr).value
        if (
            cst_matchers.matches(value_call.func, cst_matchers.Attribute())
            and cst_matchers.matches(self.constructor_expr, cst_matchers.Attribute())
        ):
            return cast(cst.Attribute, value_call.func).attr.value == cast(
                cst.Attribute, self.constructor_expr
            ).attr.value
        return False

    def _build_bundle_args(self, call: cst.Call):
        check_deadline()
        if any(arg.star in {"*", "**"} for arg in call.args):
            self.warnings.append("Skipped call with star args/kwargs during refactor.")
            return None
        positional = [arg for arg in call.args if arg.keyword is None]
        keyword_args: dict[str, cst.BaseExpression] = {}
        for arg in call.args:
            check_deadline()
            if arg.keyword is not None:
                keyword_args[arg.keyword.value] = arg.value
        for key in keyword_args:
            check_deadline()
            if key not in self.bundle_fields:
                self.warnings.append(
                    f"Skipped call with unknown keyword '{key}' during refactor."
                )
                return None
        mapping: dict[str, cst.BaseExpression] = {}
        mapping.update(keyword_args)
        remaining = [field for field in self.bundle_fields if field not in mapping]
        if len(positional) > len(remaining):
            self.warnings.append("Skipped call with extra positional args during refactor.")
            return None
        for field, arg in zip(remaining, positional):
            check_deadline()
            mapping[field] = arg.value
        if len(mapping) != len(self.bundle_fields):
            self.warnings.append("Skipped call with missing bundle fields during refactor.")
            return None
        return [
            cst.Arg(keyword=cst.Name(field), value=mapping[field])
            for field in self.bundle_fields
        ]
