from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import reduce, singledispatch
import itertools
from pathlib import Path
from typing import Iterable, cast

import libcst as cst
from libcst import matchers as cst_matchers

from gabion.refactor.model import (
    CompatibilityShimConfig, FieldSpec, LoopGeneratorRequest, RefactorPlan, RefactorPlanOutcome, RefactorRequest, RewritePlanEntry, TextEdit, normalize_compatibility_shim)
from gabion.refactor.loop_generator import plan_loop_generator_rewrite as _plan_loop_generator_rewrite
import gabion.refactor.cst_shared as cst_shared
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import decision_protocol, grade_boundary, never
from gabion.order_contract import sort_once


class RefactorEngine:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def plan_loop_generator_rewrite(self, request: LoopGeneratorRequest) -> RefactorPlan:
        return _plan_loop_generator_rewrite(
            request=request,
        )

    def plan_protocol_extraction(self, request: RefactorRequest) -> RefactorPlan:
        check_deadline()
        path = request.target_path
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
        bundle = _normalized_stripped_names(request.bundle or ())
        field_spec_collection = _field_spec_collection_from_specs(request.fields or ())
        field_spec_collection = _field_spec_collection_with_bundle_names(
            collection=field_spec_collection,
            bundle_names=bundle,
        )
        field_specs = list(field_spec_collection.field_specs)
        if not bundle and field_specs:
            bundle = _field_spec_names(field_specs)
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

        targets = set(_normalized_stripped_names(request.target_functions or ()))
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
    parts = tuple(filter(_nonempty_text, module_name.split(".")))
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


def _nonempty_text(value: str) -> bool:
    return bool(value)


def _stripped_text(value: str) -> str:
    return value.strip()


def _iter_normalized_stripped_names(values: Iterable[str]) -> Iterable[str]:
    stripped_values = map(_stripped_text, values)
    return filter(_nonempty_text, stripped_values)


def _normalized_stripped_names(values: Iterable[str]) -> list[str]:
    return list(_iter_normalized_stripped_names(values))


@dataclass(frozen=True)
class _FieldSpecCollection:
    field_specs: tuple[FieldSpec, ...]
    seen_fields: frozenset[str]


def _field_spec_collection_from_specs(specs: Iterable[FieldSpec]) -> _FieldSpecCollection:
    return reduce(
        lambda collection, spec: _field_spec_collection_after_spec(
            collection=collection,
            spec=spec,
        ),
        specs,
        _FieldSpecCollection(field_specs=(), seen_fields=frozenset()),
    )


def _field_spec_collection_after_spec(
    *,
    collection: _FieldSpecCollection,
    spec: FieldSpec,
) -> _FieldSpecCollection:
    check_deadline()
    normalized_name = _stripped_text(spec.name or "")
    if (not normalized_name) or (normalized_name in collection.seen_fields):
        return collection
    return _field_spec_collection_with_new_field(
        collection=collection,
        name=normalized_name,
        spec=spec,
    )


def _field_spec_collection_with_bundle_names(
    *,
    collection: _FieldSpecCollection,
    bundle_names: Iterable[str],
) -> _FieldSpecCollection:
    return reduce(
        lambda current, name: _field_spec_collection_after_bundle_name(
            collection=current,
            name=name,
        ),
        bundle_names,
        collection,
    )


def _field_spec_collection_after_bundle_name(
    *,
    collection: _FieldSpecCollection,
    name: str,
) -> _FieldSpecCollection:
    check_deadline()
    if name in collection.seen_fields:
        return collection
    return _field_spec_collection_with_new_field(
        collection=collection,
        name=name,
        spec=FieldSpec(name=name),
    )


def _field_spec_collection_with_new_field(
    *,
    collection: _FieldSpecCollection,
    name: str,
    spec: FieldSpec,
) -> _FieldSpecCollection:
    return _FieldSpecCollection(
        field_specs=((*collection.field_specs, spec)),
        seen_fields=(collection.seen_fields | frozenset({name})),
    )


def _field_spec_names(field_specs: list[FieldSpec]) -> list[str]:
    return [spec.name for spec in field_specs]


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="refactor.module_name",
)
@decision_protocol
def _module_name(path: Path, project_root: Path) -> str:
    rel = path.with_suffix("")
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
def _simple_statement_line_candidate(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_simple_statement_line_candidate.register(cst.SimpleStatementLine)
def _sd_reg_1(stmt: cst.SimpleStatementLine):
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
    _simple_statement_line_candidate.register(_stmt_type)(_simple_statement_line_none)


@singledispatch
def _import_candidate(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_import_candidate.register(cst.Import)
def _sd_reg_2(stmt: cst.Import):
    return stmt


@singledispatch
def _import_from_candidate(stmt: object):
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_import_from_candidate.register(cst.ImportFrom)
def _sd_reg_3(stmt: cst.ImportFrom):
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
    _import_candidate.register(_small_stmt_type)(_small_statement_none)


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
    _import_from_candidate.register(_import_from_nonmatch_type)(_small_statement_none)


@singledispatch
def _import_alias_candidate(alias: object):
    never("unregistered runtime type", value_type=type(alias).__name__)


@_import_alias_candidate.register(cst.ImportAlias)
def _sd_reg_4(alias: cst.ImportAlias):
    return alias


@_import_alias_candidate.register(cst.ImportStar)
def _sd_reg_5(alias: cst.ImportStar):
    _ = alias
    return None


@singledispatch
def _import_alias_sequence(names: object):
    never("unregistered runtime type", value_type=type(names).__name__)


@_import_alias_sequence.register(list)
def _sd_reg_6(names: list[object]):
    return names


@_import_alias_sequence.register(tuple)
def _sd_reg_7(names: tuple[object, ...]):
    return list(names)


@_import_alias_sequence.register(cst.ImportStar)
def _sd_reg_8(names: cst.ImportStar):
    _ = names
    return None


@singledispatch
def _indented_block_candidate(body: object):
    never("unregistered runtime type", value_type=type(body).__name__)


@_indented_block_candidate.register(cst.IndentedBlock)
def _sd_reg_9(body: cst.IndentedBlock):
    return body


@_indented_block_candidate.register(cst.SimpleStatementSuite)
def _sd_reg_10(body: cst.SimpleStatementSuite):
    _ = body
    return None


@_indented_block_candidate.register(cst.SimpleStatementLine)
def _sd_reg_11(body: cst.SimpleStatementLine):
    _ = body
    return None


@singledispatch
def _top_level_function_name_candidate(node: object) -> str:
    never("unregistered runtime type", value_type=type(node).__name__)


@_top_level_function_name_candidate.register(cst.FunctionDef)
def _sd_reg_12(node: cst.FunctionDef) -> str:
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
    _top_level_function_name_candidate.register(_node_type)(_empty_function_name)


def _is_import(stmt: cst.CSTNode) -> bool:
    return any(map(_statement_item_is_import, _iter_statement_items(stmt)))


def _find_import_insert_index(body: list[cst.CSTNode]) -> int:
    return cst_shared.find_import_insert_index(
        body,
        check_deadline_fn=check_deadline,
    )


def _module_expr_to_str(expr):
    return cst_shared.module_expr_to_str(expr, check_deadline_fn=check_deadline)


def _name_value_candidate(expr: object):
    if cst_matchers.matches(expr, cst_matchers.Name()):
        return _module_expr_to_str(expr)
    return None


def _has_typing_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    return any(map(_is_typing_module_name, _iter_import_module_names(body)))


def _has_typing_protocol_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    return any(
        map(
            _is_typing_protocol_alias,
            _iter_import_from_alias_pairs(body),
        )
    )


def _has_typing_overload_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    return any(
        map(
            _is_typing_overload_alias,
            _iter_import_from_alias_pairs(body),
        )
    )


def _has_warnings_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    return any(map(_is_warnings_module_name, _iter_import_module_names(body)))


def _statement_item_is_import(item: cst.BaseSmallStatement) -> bool:
    return _import_candidate(item) is not None or _import_from_candidate(item) is not None


def _iter_statement_items(stmt: cst.CSTNode) -> Iterable[cst.BaseSmallStatement]:
    line = _simple_statement_line_candidate(stmt)
    if line is None:
        return ()
    return _iter_checked_statement_items(line.body)


def _iter_statement_items_from_body(
    body: list[cst.CSTNode],
) -> Iterable[cst.BaseSmallStatement]:
    for stmt in body:
        check_deadline()
        yield from _iter_statement_items(stmt)


def _iter_checked_statement_items(
    body: list[cst.BaseSmallStatement],
) -> Iterable[cst.BaseSmallStatement]:
    for item in body:
        check_deadline()
        yield item


def _iter_import_module_names(body: list[cst.CSTNode]) -> Iterable[str]:
    for item in _iter_statement_items_from_body(body):
        yield from _import_module_names_for_item(item)


def _import_module_names_for_item(
    item: cst.BaseSmallStatement,
) -> Iterable[str]:
    import_item = _import_candidate(item)
    if import_item is None:
        return ()
    return _iter_import_module_names_from_aliases(import_item.names)


def _iter_import_module_names_from_aliases(
    aliases: list[cst.ImportAlias],
) -> Iterable[str]:
    for alias in aliases:
        yield from _import_module_name_from_alias(alias)


def _import_module_name_from_alias(alias: cst.ImportAlias) -> tuple[str, ...]:
    check_deadline()
    import_alias = _import_alias_candidate(alias)
    if import_alias is None:
        return ()
    return (_module_expr_to_str(import_alias.name),)


def _iter_import_from_alias_pairs(
    body: list[cst.CSTNode],
) -> Iterable[tuple[str, str]]:
    for item in _iter_statement_items_from_body(body):
        yield from _import_from_alias_pairs_for_item(item)


def _import_from_alias_pairs_for_item(
    item: cst.BaseSmallStatement,
) -> Iterable[tuple[str, str]]:
    import_from = _import_from_candidate(item)
    if import_from is None:
        return ()
    aliases = _import_alias_sequence(import_from.names)
    if aliases is None:
        return ()
    module_name = _module_expr_to_str(import_from.module)
    return _iter_module_alias_pairs(module_name, aliases)


def _iter_module_alias_pairs(
    module_name: str,
    aliases: list[cst.ImportAlias],
) -> Iterable[tuple[str, str]]:
    for alias in aliases:
        yield from _module_alias_pair(module_name, alias)


def _module_alias_pair(
    module_name: str,
    alias: cst.ImportAlias,
) -> tuple[tuple[str, str], ...]:
    check_deadline()
    import_alias = _import_alias_candidate(alias)
    if import_alias is None:
        return ()
    return ((module_name, _module_expr_to_str(import_alias.name)),)


def _is_typing_module_name(module_name: str) -> bool:
    return module_name == "typing"


def _is_warnings_module_name(module_name: str) -> bool:
    return module_name == "warnings"


def _is_typing_protocol_alias(module_alias: tuple[str, str]) -> bool:
    module_name, alias_name = module_alias
    return module_name == "typing" and alias_name == "Protocol"


def _is_typing_overload_alias(module_alias: tuple[str, str]) -> bool:
    module_name, alias_name = module_alias
    return module_name == "typing" and alias_name == "overload"


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
    initial = _ImportContextState(
        module_aliases={},
        imported_targets={},
        protocol_aliases=(),
    )
    final_state = _reduce_import_context(
        initial=initial,
        statement_items=_iter_statement_items_from_body(module.body),
        target_module_value=target_module.value,
        protocol_name=protocol_name,
    )
    return (
        final_state.module_aliases,
        final_state.imported_targets,
        _first_alias_text(final_state.protocol_aliases),
    )


@dataclass(frozen=True)
class _ImportContextState:
    module_aliases: dict[str, str]
    imported_targets: dict[str, str]
    protocol_aliases: tuple[str, ...]


@dataclass(frozen=True)
class _ImportContextContribution:
    module_alias_pairs: tuple[tuple[str, str], ...]
    imported_target_pairs: tuple[tuple[str, str], ...]
    protocol_aliases: tuple[str, ...]


def _reduce_import_context(
    *,
    initial: _ImportContextState,
    statement_items: Iterable[cst.BaseSmallStatement],
    target_module_value: str,
    protocol_name: str,
) -> _ImportContextState:
    return reduce(
        lambda state, item: _import_context_state_after(
            state=state,
            contribution=_import_context_contribution_for_item(
                item=item,
                target_module_value=target_module_value,
                protocol_name=protocol_name,
            ),
        ),
        statement_items,
        initial,
    )


def _import_context_state_after(
    *,
    state: _ImportContextState,
    contribution: _ImportContextContribution,
) -> _ImportContextState:
    module_aliases = dict(state.module_aliases)
    module_aliases.update(dict(contribution.module_alias_pairs))
    imported_targets = dict(state.imported_targets)
    imported_targets.update(dict(contribution.imported_target_pairs))
    return _ImportContextState(
        module_aliases=module_aliases,
        imported_targets=imported_targets,
        protocol_aliases=(state.protocol_aliases or contribution.protocol_aliases),
    )


def _import_context_contribution_for_item(
    *,
    item: cst.BaseSmallStatement,
    target_module_value: str,
    protocol_name: str,
) -> _ImportContextContribution:
    module_alias_pairs = _module_alias_pairs_for_import_item(
        item=item,
        target_module_value=target_module_value,
    )
    imported_target_pairs = _imported_target_pairs_for_item(
        item=item,
        target_module_value=target_module_value,
    )
    return _ImportContextContribution(
        module_alias_pairs=module_alias_pairs,
        imported_target_pairs=imported_target_pairs,
        protocol_aliases=_protocol_aliases_for_imported_targets(
            imported_target_pairs=imported_target_pairs,
            protocol_name=protocol_name,
        ),
    )


def _module_alias_pairs_for_import_item(
    *,
    item: cst.BaseSmallStatement,
    target_module_value: str,
) -> tuple[tuple[str, str], ...]:
    import_item = _import_candidate(item)
    if import_item is None:
        return ()
    pair_groups = map(
        lambda alias: _module_alias_pair_for_import_alias(
            alias=alias,
            target_module_value=target_module_value,
        ),
        import_item.names,
    )
    return tuple(itertools.chain.from_iterable(pair_groups))


def _module_alias_pair_for_import_alias(
    *,
    alias: cst.ImportAlias,
    target_module_value: str,
) -> tuple[tuple[str, str], ...]:
    check_deadline()
    import_alias = _import_alias_candidate(alias)
    if import_alias is None:
        return ()
    module_name = _module_expr_to_str(import_alias.name)
    if not module_name or module_name != target_module_value:
        return ()
    local_name = _import_alias_local_name(import_alias=import_alias, default_name=module_name)
    return ((local_name, module_name),)


def _imported_target_pairs_for_item(
    *,
    item: cst.BaseSmallStatement,
    target_module_value: str,
) -> tuple[tuple[str, str], ...]:
    import_from = _import_from_candidate(item)
    if import_from is None:
        return ()
    module_name = _module_expr_to_str(import_from.module)
    aliases = _import_alias_sequence(import_from.names)
    if module_name != target_module_value or aliases is None:
        return ()
    pair_groups = map(_imported_target_pair_for_alias, aliases)
    return tuple(itertools.chain.from_iterable(pair_groups))


def _imported_target_pair_for_alias(
    alias: cst.ImportAlias,
) -> tuple[tuple[str, str], ...]:
    check_deadline()
    import_alias = _import_alias_candidate(alias)
    if import_alias is None or not cst_matchers.matches(import_alias.name, cst_matchers.Name()):
        return ()
    alias_name = _module_expr_to_str(import_alias.name)
    local_name = _import_alias_local_name(import_alias=import_alias, default_name=alias_name)
    return ((local_name, alias_name),)


def _import_alias_local_name(*, import_alias: cst.ImportAlias, default_name: str) -> str:
    return (
        import_alias.asname.name.value
        if import_alias.asname
        else default_name
    )


def _protocol_aliases_for_imported_targets(
    *,
    imported_target_pairs: tuple[tuple[str, str], ...],
    protocol_name: str,
) -> tuple[str, ...]:
    matches = filter(lambda item: item[1] == protocol_name, imported_target_pairs)
    aliases = tuple(map(lambda item: item[0], matches))
    return _first_alias_tuple(aliases)


def _first_alias_tuple(aliases: tuple[str, ...]) -> tuple[str, ...]:
    if not aliases:
        return ()
    return (aliases[0],)


def _first_alias_text(aliases: tuple[str, ...]) -> str:
    if not aliases:
        return ""
    return aliases[0]


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
    target_simple = set(filter(_target_name_is_simple, targets))
    target_methods = _target_methods_by_class(targets)
    module_aliases: dict[str, str] = {}
    imported_targets: dict[str, str] = {}
    protocol_alias = ""
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
        imported_targets=_imported_target_names_in_scope(
            imported_targets=imported_targets,
            target_simple=target_simple,
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


def _target_name_is_simple(name: str) -> bool:
    return "." not in name


def _target_terminal_name(name: str) -> str:
    parts = name.split(".")
    return parts[-1]


def _target_method_entries(name: str) -> tuple[tuple[str, str], ...]:
    check_deadline()
    parts = name.split(".")
    if len(parts) < 2:
        return ()
    class_name = ".".join(parts[:-1])
    method_name = parts[-1]
    return ((class_name, method_name),)


def _target_methods_by_class(targets: set[str]) -> dict[str, set[str]]:
    entries = itertools.chain.from_iterable(map(_target_method_entries, targets))
    return reduce(_target_methods_accumulate, entries, {})


def _target_methods_accumulate(
    grouped: dict[str, set[str]],
    entry: tuple[str, str],
) -> dict[str, set[str]]:
    class_name, method_name = entry
    updated = dict(grouped)
    methods = set(updated.get(class_name, set()))
    methods.add(method_name)
    updated[class_name] = methods
    return updated


def _imported_target_names_in_scope(
    *,
    imported_targets: dict[str, str],
    target_simple: set[str],
) -> set[str]:
    in_scope = filter(
        lambda item: item[1] in target_simple,
        imported_targets.items(),
    )
    return set(map(lambda item: item[0], in_scope))


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
    outcomes = tuple(
        _iter_project_callsite_rewrite_outcomes(
            scan_root=_project_scan_root(project_root),
            target_path=target_path,
            target_module=target_module,
            protocol_name=protocol_name,
            bundle_fields=bundle_fields,
            targets=targets,
        )
    )
    warnings = list(itertools.chain.from_iterable(map(_rewrite_outcome_warnings, outcomes)))
    edits = list(itertools.chain.from_iterable(map(_rewrite_outcome_edits, outcomes)))
    return edits, warnings


@dataclass(frozen=True)
class _ProjectCallsiteRewriteOutcome:
    warnings: tuple[str, ...]
    edits: tuple[TextEdit, ...]


def _project_scan_root(project_root: Path) -> Path:
    scan_root = project_root / "src"
    if scan_root.exists():
        return scan_root
    return project_root


def _iter_project_callsite_rewrite_outcomes(
    *,
    scan_root: Path,
    target_path: Path,
    target_module: _ValidatedModuleIdentifier,
    protocol_name: str,
    bundle_fields: list[str],
    targets: set[str],
) -> Iterable[_ProjectCallsiteRewriteOutcome]:
    for path in sort_once(
        scan_root.rglob("*.py"),
        source="_rewrite_call_sites_in_project.scan_root",
        key=lambda item: str(item),
    ):
        check_deadline()
        yield from _project_callsite_rewrite_outcome_for_path(
            path=path,
            target_path=target_path,
            target_module=target_module,
            protocol_name=protocol_name,
            bundle_fields=bundle_fields,
            targets=targets,
        )


def _project_callsite_rewrite_outcome_for_path(
    *,
    path: Path,
    target_path: Path,
    target_module: _ValidatedModuleIdentifier,
    protocol_name: str,
    bundle_fields: list[str],
    targets: set[str],
) -> tuple[_ProjectCallsiteRewriteOutcome, ...]:
    if path == target_path:
        return ()
    source_outcome = _source_read_outcome(path)
    if source_outcome.status is _SourceReadStatus.ERROR:
        return (
            _ProjectCallsiteRewriteOutcome(
                warnings=(source_outcome.error_message,),
                edits=(),
            ),
        )
    parse_outcome = _module_parse_outcome(path=path, source=source_outcome.source)
    if parse_outcome.status is _ModuleParseStatus.ERROR:
        return (
            _ProjectCallsiteRewriteOutcome(
                warnings=(parse_outcome.error_message,),
                edits=(),
            ),
        )
    call_warnings, updated_module = _rewrite_call_sites(
        parse_outcome.module,
        file_path=path,
        target_path=target_path,
        target_module=target_module,
        protocol_name=protocol_name,
        bundle_fields=bundle_fields,
        targets=targets,
    )
    if updated_module is None:
        return (
            _ProjectCallsiteRewriteOutcome(
                warnings=tuple(call_warnings),
                edits=(),
            ),
        )
    edit = _full_module_replacement_edit(path=path, old_source=source_outcome.source, updated_module=updated_module)
    return (
        _ProjectCallsiteRewriteOutcome(
            warnings=tuple(call_warnings),
            edits=(edit,),
        ),
    )


class _SourceReadStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class _SourceReadOutcome:
    status: _SourceReadStatus
    source: str
    error_message: str


def _source_read_outcome(path: Path) -> _SourceReadOutcome:
    try:
        source = path.read_text()
    except Exception as exc:
        return _SourceReadOutcome(
            status=_SourceReadStatus.ERROR,
            source="",
            error_message=f"Failed to read {path}: {exc}",
        )
    return _SourceReadOutcome(
        status=_SourceReadStatus.OK,
        source=source,
        error_message="",
    )


class _ModuleParseStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class _ModuleParseOutcome:
    status: _ModuleParseStatus
    module: cst.Module
    error_message: str


def _module_parse_outcome(*, path: Path, source: str) -> _ModuleParseOutcome:
    try:
        module = cst.parse_module(source)
    except Exception as exc:
        return _ModuleParseOutcome(
            status=_ModuleParseStatus.ERROR,
            module=cst.parse_module(""),
            error_message=f"LibCST parse failed for {path}: {exc}",
        )
    return _ModuleParseOutcome(
        status=_ModuleParseStatus.OK,
        module=module,
        error_message="",
    )


def _full_module_replacement_edit(
    *,
    path: Path,
    old_source: str,
    updated_module: cst.Module,
) -> TextEdit:
    new_source = updated_module.code
    end_line = len(old_source.splitlines())
    return TextEdit(
        path=str(path),
        start=(0, 0),
        end=(end_line, 0),
        replacement=new_source,
    )


def _rewrite_outcome_warnings(
    outcome: _ProjectCallsiteRewriteOutcome,
) -> tuple[str, ...]:
    return outcome.warnings


def _rewrite_outcome_edits(
    outcome: _ProjectCallsiteRewriteOutcome,
) -> tuple[TextEdit, ...]:
    return outcome.edits




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

    existing_top_level = set(
        filter(
            _nonempty_text,
            map(_top_level_function_name_candidate, _iter_checked_nodes(body)),
        )
    )
    missing_context_names = tuple(
        filter(
            lambda context_name: not _ambient_scaffold_exists(
                context_name=context_name,
                existing_top_level=existing_top_level,
            ),
            sort_once(
                context_names,
                source="_ensure_ambient_scaffolding.context_names",
            ),
        )
    )
    state = reduce(
        lambda current, context_name: _ambient_scaffolding_state_after(
            state=current,
            context_name=context_name,
            protocol_hint=protocol_hint,
        ),
        missing_context_names,
        _AmbientScaffoldingState(body=body, insert_idx=insert_idx),
    )
    return module.with_changes(body=state.body)


@dataclass(frozen=True)
class _AmbientScaffoldingState:
    body: list[cst.CSTNode]
    insert_idx: int


def _iter_checked_nodes(body: list[cst.CSTNode]) -> Iterable[cst.CSTNode]:
    for node in body:
        check_deadline()
        yield node


def _ambient_scaffold_exists(
    *,
    context_name: str,
    existing_top_level: set[str],
) -> bool:
    getter = _ambient_getter_name(context_name)
    setter = _ambient_setter_name(context_name)
    return getter in existing_top_level and setter in existing_top_level


def _ambient_scaffolding_state_after(
    *,
    state: _AmbientScaffoldingState,
    context_name: str,
    protocol_hint: str,
) -> _AmbientScaffoldingState:
    check_deadline()
    statements = _ambient_scaffolding_statements(
        context_name=context_name,
        protocol_hint=protocol_hint,
    )
    updated_body = list(state.body)
    updated_body[state.insert_idx:state.insert_idx] = statements
    return _AmbientScaffoldingState(
        body=updated_body,
        insert_idx=(state.insert_idx + len(statements)),
    )


def _ambient_scaffolding_statements(
    *,
    context_name: str,
    protocol_hint: str,
) -> list[cst.CSTNode]:
    ambient_name = _ambient_var_name(context_name)
    getter = _ambient_getter_name(context_name)
    setter = _ambient_setter_name(context_name)
    annotation = protocol_hint or "object"
    return list(
        cst.parse_module(
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
    )


def _has_contextvars_import(body: list[cst.CSTNode]) -> bool:
    check_deadline()
    return any(
        map(
            _is_contextvars_contextvar_alias,
            _iter_import_from_alias_pairs(body),
        )
    )


def _is_contextvars_contextvar_alias(module_alias: tuple[str, str]) -> bool:
    module_name, alias_name = module_alias
    return module_name == "contextvars" and alias_name == "ContextVar"


def _ambient_var_name(name: str) -> str:
    return f"_AMBIENT_{name.upper()}"


def _ambient_getter_name(name: str) -> str:
    return f"_ambient_get_{name}"


def _ambient_setter_name(name: str) -> str:
    return f"_ambient_set_{name}"


class _AmbientArgThreadingRewriter(cst.CSTTransformer):
    def __init__(self, *, targets: set[str], context_name: str, current: str) -> None:
        self.targets = set(map(_target_terminal_name, targets))
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
        if _call_has_dynamic_star_args(updated_node):
            self.skipped_reasons.append(
                f"{self.current}: skipped call rewrite with dynamic star args for ambient parameter '{self.context_name}'."
            )
            return updated_node
        rewrite_state = reduce(
            lambda state, arg: _ambient_arg_rewrite_state_after(
                state=state,
                arg=arg,
                context_name=self.context_name,
                args_len=len(updated_node.args),
                current_name=self.current,
            ),
            updated_node.args,
            _AmbientArgRewriteState(new_args=(), removed=False, skipped_reasons=()),
        )
        self.skipped_reasons.extend(rewrite_state.skipped_reasons)
        if rewrite_state.removed:
            self.changed = True
            return updated_node.with_changes(args=list(rewrite_state.new_args))
        return updated_node


class _AmbientSafetyVisitor(cst.CSTVisitor):
    def __init__(self, context_name: str) -> None:
        self.context_name = context_name
        self.reasons: list[str] = []

    def visit_AssignTarget(self, node: cst.AssignTarget) -> None:
        if not cst_matchers.matches(node.target, cst_matchers.Name()):
            return
        target_name = _module_expr_to_str(node.target)
        if target_name == self.context_name:
            self.reasons.append(
                f"writes to parameter '{self.context_name}' prevent a safe ambient rewrite"
            )


@dataclass(frozen=True)
class _AmbientArgRewriteState:
    new_args: tuple[cst.Arg, ...]
    removed: bool
    skipped_reasons: tuple[str, ...]


def _ambient_arg_rewrite_state_after(
    *,
    state: _AmbientArgRewriteState,
    arg: cst.Arg,
    context_name: str,
    args_len: int,
    current_name: str,
) -> _AmbientArgRewriteState:
    check_deadline()
    action = _ambient_arg_action(
        arg=arg,
        context_name=context_name,
        args_len=args_len,
        current_name=current_name,
    )
    return _AmbientArgRewriteState(
        new_args=_ambient_new_args(state.new_args, arg=arg, keep=action.keep),
        removed=(state.removed or action.removed),
        skipped_reasons=(*state.skipped_reasons, *action.skipped_reasons),
    )


@dataclass(frozen=True)
class _AmbientArgAction:
    keep: bool
    removed: bool
    skipped_reasons: tuple[str, ...]


def _ambient_arg_action(
    *,
    arg: cst.Arg,
    context_name: str,
    args_len: int,
    current_name: str,
) -> _AmbientArgAction:
    arg_name = _name_value_candidate(arg.value)
    is_name = arg_name == context_name
    if arg.keyword is not None and arg.keyword.value == context_name and is_name:
        return _AmbientArgAction(keep=False, removed=True, skipped_reasons=())
    if arg.keyword is None and is_name and args_len == 1:
        return _AmbientArgAction(keep=False, removed=True, skipped_reasons=())
    if arg.keyword is None and is_name and args_len > 1:
        return _AmbientArgAction(
            keep=True,
            removed=False,
            skipped_reasons=(
                f"{current_name}: skipped positional ambient argument rewrite for '{context_name}' due to ambiguous arity.",
            ),
        )
    return _AmbientArgAction(keep=True, removed=False, skipped_reasons=())


def _ambient_new_args(
    existing: tuple[cst.Arg, ...],
    *,
    arg: cst.Arg,
    keep: bool,
) -> tuple[cst.Arg, ...]:
    if keep:
        return (*existing, arg)
    return existing


@dataclass(frozen=True)
class _ParamLookupOutcome:
    found: bool
    param: cst.Param


_MISSING_PARAM = cst.Param(name=cst.Name("_missing"))


def _param_lookup(*, params: list[cst.Param], name: str) -> _ParamLookupOutcome:
    candidate = next(filter(lambda param: param.name.value == name, params), _MISSING_PARAM)
    return _param_lookup_outcome(candidate)


def _param_lookup_outcome(candidate: cst.Param) -> _ParamLookupOutcome:
    if candidate is _MISSING_PARAM:
        return _ParamLookupOutcome(found=False, param=_MISSING_PARAM)
    return _ParamLookupOutcome(found=True, param=candidate)


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="refactor.ambient_rewrite_param_ingress",
)
@decision_protocol
def _ambient_rewrite_param_ingress(
    *,
    param: cst.Param,
    context_name: str,
    protocol_hint: str,
) -> cst.Param:
    if param.name.value != context_name:
        return param
    if param.annotation is not None:
        return param
    return param.with_changes(annotation=_ambient_default_annotation(protocol_hint))


def _ambient_rewrite_param(
    *,
    param: cst.Param,
    context_name: str,
    protocol_hint: str,
) -> cst.Param:
    if param.name.value != context_name:
        return param
    return param.with_changes(
        default=cst.Name("None"),
        annotation=_ambient_param_annotation(param.annotation, protocol_hint),
    )


def _ambient_param_annotation(annotation, protocol_hint: str) -> cst.Annotation:
    match annotation:
        case cst.Annotation() as existing:
            return existing
        case _:
            pass
            never("unreachable wildcard match fall-through")
    return _ambient_default_annotation(protocol_hint)


def _ambient_default_annotation(protocol_hint: str) -> cst.Annotation:
    try:
        return cst.Annotation(cst.parse_expression(f"{protocol_hint} | None"))
    except Exception:
        return cst.Annotation(cst.parse_expression("object | None"))


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
        match_param_lookup = _param_lookup(params=params, name=context_name)
        if not match_param_lookup.found:
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
        updated_block = _indented_block_candidate(updated_body)
        if updated_block is not None:
            existing = list(updated_block.body)
            insert_at = 1 if existing and _is_docstring(existing[0]) else 0
            updated_body = updated_block.with_changes(body=existing[:insert_at] + preamble + existing[insert_at:])

        updated_params = list(
            map(
                lambda param: _ambient_rewrite_param(
                    param=_ambient_rewrite_param_ingress(
                        param=param,
                        context_name=context_name,
                        protocol_hint=self.protocol_hint,
                    ),
                    context_name=context_name,
                    protocol_hint=self.protocol_hint,
                ),
                node.params.params,
            )
        )

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
        target_fields = list(
            filter(
                lambda name: _is_bundle_field_target(name=name, bundle_set=bundle_set),
                original_params,
            )
        )
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
        return next(
            filter(
                lambda param: param.name.value == name,
                _iter_named_params(params),
            ),
            None,
        )

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
        body_block = _indented_block_candidate(body)
        if body_block is None:
            return body
        assign_lines = list(
            map(
                lambda name: _bundle_field_assignment_line(
                    bundle_name=bundle_name,
                    field_name=name,
                ),
                fields,
            )
        )
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
        if _call_has_dynamic_star_args(call):
            self.warnings.append("Skipped call with star args/kwargs during refactor.")
            return None
        positional = tuple(_iter_positional_args(call.args))
        keyword_args = _keyword_arg_mapping(call.args)
        unknown_keywords = tuple(
            _unknown_keyword_names(
                keyword_args=keyword_args,
                bundle_fields=self.bundle_fields,
            )
        )
        if unknown_keywords:
            self.warnings.append(
                f"Skipped call with unknown keyword '{unknown_keywords[0]}' during refactor."
            )
            return None
        mapping: dict[str, cst.BaseExpression] = dict(keyword_args)
        remaining = tuple(filter(lambda field: field not in mapping, self.bundle_fields))
        if len(positional) > len(remaining):
            self.warnings.append("Skipped call with extra positional args during refactor.")
            return None
        mapping.update(dict(zip(remaining, map(_arg_value, positional))))
        if len(mapping) != len(self.bundle_fields):
            self.warnings.append("Skipped call with missing bundle fields during refactor.")
            return None
        return list(
            map(
                lambda field: _bundle_arg_for_field(field=field, mapping=mapping),
                self.bundle_fields,
            )
        )


def _call_has_dynamic_star_args(call: cst.Call) -> bool:
    return any(map(_arg_is_dynamic_star, call.args))


def _arg_is_dynamic_star(arg: cst.Arg) -> bool:
    return arg.star in {"*", "**"}


def _arg_is_positional(arg: cst.Arg) -> bool:
    return arg.keyword is None


def _iter_positional_args(args: tuple[cst.Arg, ...]) -> Iterable[cst.Arg]:
    return filter(_arg_is_positional, args)


def _arg_has_keyword(arg: cst.Arg) -> bool:
    return arg.keyword is not None


def _keyword_arg_pair(arg: cst.Arg) -> tuple[str, cst.BaseExpression]:
    check_deadline()
    return (arg.keyword.value, arg.value)  # type: ignore[union-attr]


def _keyword_arg_mapping(args: tuple[cst.Arg, ...]) -> dict[str, cst.BaseExpression]:
    keyword_args = filter(_arg_has_keyword, args)
    return dict(map(_keyword_arg_pair, keyword_args))


def _unknown_keyword_names(
    *,
    keyword_args: dict[str, cst.BaseExpression],
    bundle_fields: list[str],
) -> Iterable[str]:
    return filter(lambda key: key not in bundle_fields, keyword_args.keys())


def _arg_value(arg: cst.Arg) -> cst.BaseExpression:
    return arg.value


def _bundle_arg_for_field(
    *,
    field: str,
    mapping: dict[str, cst.BaseExpression],
) -> cst.Arg:
    return cst.Arg(keyword=cst.Name(field), value=mapping[field])


def _is_bundle_field_target(*, name: str, bundle_set: set[str]) -> bool:
    return name in bundle_set and name not in {"self", "cls"}


def _iter_named_params(params: cst.Parameters) -> Iterable[cst.Param]:
    yield from _iter_checked_params(params.posonly_params)
    yield from _iter_checked_params(params.params)


def _iter_checked_params(params: list[cst.Param]) -> Iterable[cst.Param]:
    for param in params:
        check_deadline()
        yield param


def _bundle_field_assignment_line(
    *,
    bundle_name: str,
    field_name: str,
) -> cst.SimpleStatementLine:
    return cst.SimpleStatementLine(
        [
            cst.Assign(
                targets=[cst.AssignTarget(cst.Name(field_name))],
                value=cst.Attribute(
                    value=cst.Name(bundle_name),
                    attr=cst.Name(field_name),
                ),
            )
        ]
    )
