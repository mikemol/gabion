# gabion:ambiguity_boundary_module
from __future__ import annotations

import ast
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from functools import singledispatch
from pathlib import Path
from typing import Iterator, Mapping, Sequence, cast

from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.projection.pattern_schema import (
    PatternAxis, PatternInstance, PatternResidue, PatternSchema, execution_signature, mismatch_residue_payload)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.order_contract import sort_once

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_PARSE_MODULE_ERROR_TYPES = (
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)


def _ast_leaf_node_types() -> tuple[type[ast.AST], ...]:
    stack: list[type[ast.AST]] = [ast.AST]
    leaves: set[type[ast.AST]] = set()
    while stack:
        node_type = stack.pop()
        subclasses = tuple(
            candidate
            for candidate in node_type.__subclasses__()
            if issubclass(candidate, ast.AST)
        )
        if subclasses:
            stack.extend(subclasses)
            continue
        if node_type is not ast.AST:
            leaves.add(node_type)
    return tuple(
        sort_once(
            leaves,
            source="pattern_schema_projection._ast_leaf_node_types",
            key=lambda node_type: node_type.__name__,
        )
    )


_AST_LEAF_NODE_TYPES = _ast_leaf_node_types()


@singledispatch
def _is_name_node(node: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_name_node.register(ast.Name)
def _is_name_node_name(node: ast.Name) -> bool:
    return True


def _is_not_name_node(node: ast.AST) -> bool:
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type is ast.Name:
        continue
    _is_name_node.register(_runtime_type)(_is_not_name_node)


@singledispatch
def _name_node_id(node: ast.AST) -> str:
    never("unregistered runtime type", value_type=type(node).__name__)


@_name_node_id.register(ast.Name)
def _name_node_id_name(node: ast.Name) -> str:
    return node.id


@singledispatch
def _is_attribute_node(node: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_attribute_node.register(ast.Attribute)
def _is_attribute_node_attribute(node: ast.Attribute) -> bool:
    return True


def _is_not_attribute_node(node: ast.AST) -> bool:
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type is ast.Attribute:
        continue
    _is_attribute_node.register(_runtime_type)(_is_not_attribute_node)


@singledispatch
def _attribute_node(node: ast.AST) -> ast.Attribute:
    never("unregistered runtime type", value_type=type(node).__name__)


@_attribute_node.register(ast.Attribute)
def _attribute_node_attribute(node: ast.Attribute) -> ast.Attribute:
    return node


@singledispatch
def _is_function_node(node: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_function_node.register(ast.FunctionDef)
def _is_function_node_function(node: ast.FunctionDef) -> bool:
    return True


@_is_function_node.register(ast.AsyncFunctionDef)
def _is_function_node_async_function(node: ast.AsyncFunctionDef) -> bool:
    return True


def _is_not_function_node(node: ast.AST) -> bool:
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type in {ast.FunctionDef, ast.AsyncFunctionDef}:
        continue
    _is_function_node.register(_runtime_type)(_is_not_function_node)


@singledispatch
def _function_node(node: ast.AST) -> FunctionNode:
    never("unregistered runtime type", value_type=type(node).__name__)


@_function_node.register(ast.FunctionDef)
def _function_node_function(node: ast.FunctionDef) -> ast.FunctionDef:
    return node


@_function_node.register(ast.AsyncFunctionDef)
def _function_node_async_function(node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
    return node


@singledispatch
def _is_assign_node(node: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_assign_node.register(ast.Assign)
def _is_assign_node_assign(node: ast.Assign) -> bool:
    return True


def _is_not_assign_node(node: ast.AST) -> bool:
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type is ast.Assign:
        continue
    _is_assign_node.register(_runtime_type)(_is_not_assign_node)


@singledispatch
def _assign_node(node: ast.AST) -> ast.Assign:
    never("unregistered runtime type", value_type=type(node).__name__)


@_assign_node.register(ast.Assign)
def _assign_node_assign(node: ast.Assign) -> ast.Assign:
    return node


@singledispatch
def _is_call_node(node: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_call_node.register(ast.Call)
def _is_call_node_call(node: ast.Call) -> bool:
    return True


def _is_not_call_node(node: ast.AST) -> bool:
    return False


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type is ast.Call:
        continue
    _is_call_node.register(_runtime_type)(_is_not_call_node)


@singledispatch
def _call_node(node: ast.AST) -> ast.Call:
    never("unregistered runtime type", value_type=type(node).__name__)


@_call_node.register(ast.Call)
def _call_node_call(node: ast.Call) -> ast.Call:
    return node


@singledispatch
def _source_text_optional(source: str | None) -> str | None:
    never("unregistered runtime type", value_type=type(source).__name__)


@_source_text_optional.register(str)
def _source_text_or_none_str(source: str) -> str:
    return source


@_source_text_optional.register(type(None))
def _source_text_or_none_none(source: None) -> None:
    return None


@dataclass(frozen=True)
class _ExecutionPatternMatch:
    pattern_id: str
    kind: str
    schema_family: str
    members: tuple[str, ...]
    suggestion: str


@dataclass(frozen=True)
class _ExecutionPatternRule:
    pattern_id: str
    kind: str
    schema_family: str
    predicates: tuple["_ExecutionPatternPredicate", ...]
    min_members: int
    candidate: str
    description: str


@dataclass(frozen=True)
class _ExecutionCallShape:
    positional_args: int
    keyword_names: frozenset[str]


@dataclass(frozen=True)
class _ExecutionFunctionFact:
    function_name: str
    param_names: frozenset[str]
    called_names: frozenset[str]
    call_shapes: Mapping[str, tuple[_ExecutionCallShape, ...]]


class _ExecutionPatternPredicateKind(StrEnum):
    PARAMS = "params"
    CALLEE = "callee"
    CALL_SHAPE = "call_shape"


@dataclass(frozen=True)
class _ExecutionPatternPredicate:
    kind: _ExecutionPatternPredicateKind
    required_params: frozenset[str] = frozenset()
    callee_names: frozenset[str] = frozenset()
    required_keywords: frozenset[str] = frozenset()
    min_positional_args: int = 0

    def matches(self, *, fact: _ExecutionFunctionFact) -> bool:
        check_deadline()
        if self.kind is _ExecutionPatternPredicateKind.PARAMS:
            return self.required_params.issubset(fact.param_names)
        if self.kind is _ExecutionPatternPredicateKind.CALLEE:
            return bool(fact.called_names.intersection(self.callee_names))
        if self.kind is _ExecutionPatternPredicateKind.CALL_SHAPE:
            for callee_name in self.callee_names:
                check_deadline()
                for shape in fact.call_shapes.get(callee_name, ()):  # boundary alias normalization
                    check_deadline()
                    if shape.positional_args < self.min_positional_args:
                        continue
                    if not self.required_keywords.issubset(shape.keyword_names):
                        continue
                    return True
            return False
        return never(self.kind)  # pragma: no cover - invariant sink

    def payload(self) -> JSONObject:
        check_deadline()
        payload: JSONObject = {"kind": self.kind.value}
        if self.required_params:
            payload["required_params"] = list(
                sort_once(
                    self.required_params,
                    source="pattern_schema_projection._ExecutionPatternPredicate.payload.required_params",
                )
            )
        if self.callee_names:
            payload["callee_names"] = list(
                sort_once(
                    self.callee_names,
                    source="pattern_schema_projection._ExecutionPatternPredicate.payload.callee_names",
                )
            )
        if self.required_keywords:
            payload["required_keywords"] = list(
                sort_once(
                    self.required_keywords,
                    source="pattern_schema_projection._ExecutionPatternPredicate.payload.required_keywords",
                )
            )
        if self.min_positional_args:
            payload["min_positional_args"] = self.min_positional_args
        return payload


_INDEXED_PASS_INGRESS_CORE_PARAMS = frozenset(
    {
        "paths",
        "project_root",
        "ignore_params",
        "strictness",
        "external_filter",
        "transparent_decorators",
        "parse_failure_witnesses",
        "analysis_index",
    }
)
_INDEXED_PASS_CALL_SHAPE_KEYWORDS = frozenset(
    {
        "project_root",
        "ignore_params",
        "strictness",
        "external_filter",
        "transparent_decorators",
        "parse_failure_witnesses",
    }
)
_INDEXED_PASS_INGRESS_RULE = _ExecutionPatternRule(
    pattern_id="indexed_pass_ingress",
    kind="execution_pattern",
    schema_family="indexed_pass_cluster",
    predicates=(
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.PARAMS,
            required_params=_INDEXED_PASS_INGRESS_CORE_PARAMS,
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALLEE,
            callee_names=frozenset({"_build_analysis_index", "_build_call_graph"}),
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALL_SHAPE,
            callee_names=frozenset({"_build_analysis_index", "_build_call_graph"}),
            required_keywords=_INDEXED_PASS_CALL_SHAPE_KEYWORDS,
            min_positional_args=1,
        ),
    ),
    min_members=3,
    candidate="IndexedPassSpec[T] metafactory",
    description=(
        "Functions sharing the indexed-pass ingress contract should be reified "
        "behind a typed pass metafactory."
    ),
)
_INDEXED_PASS_RUNNER_RULE = _ExecutionPatternRule(
    pattern_id="indexed_pass_runner",
    kind="execution_pattern",
    schema_family="indexed_pass_cluster",
    predicates=(
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.PARAMS,
            required_params=_INDEXED_PASS_INGRESS_CORE_PARAMS,
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALLEE,
            callee_names=frozenset({"_run_indexed_pass"}),
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALL_SHAPE,
            callee_names=frozenset({"_run_indexed_pass"}),
            required_keywords=frozenset({"analysis_index"}),
            min_positional_args=1,
        ),
    ),
    min_members=2,
    candidate="IndexedPassSpec[T] runner Protocol",
    description=(
        "Functions repeatedly invoking _run_indexed_pass should consolidate "
        "the execution carrier into a shared runner Protocol."
    ),
)
_INDEXED_PASS_GRAPH_RULE = _ExecutionPatternRule(
    pattern_id="indexed_pass_graph_builder",
    kind="execution_pattern",
    schema_family="indexed_pass_cluster",
    predicates=(
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.PARAMS,
            required_params=_INDEXED_PASS_INGRESS_CORE_PARAMS,
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALLEE,
            callee_names=frozenset({"_build_call_graph"}),
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALL_SHAPE,
            callee_names=frozenset({"_build_call_graph"}),
            required_keywords=frozenset({"analysis_index"}),
            min_positional_args=1,
        ),
    ),
    min_members=2,
    candidate="IndexedPassSpec[T] graph-builder Protocol",
    description=(
        "Functions repeatedly rebuilding call graph projections should "
        "share one indexed-pass graph-builder execution contract."
    ),
)
_PARSE_FAILURE_SINK_RULE = _ExecutionPatternRule(
    pattern_id="parse_failure_sink_plumbing",
    kind="execution_pattern",
    schema_family="parse_failure_cluster",
    predicates=(
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.PARAMS,
            required_params=frozenset({"parse_failure_witnesses"}),
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALLEE,
            callee_names=frozenset({"_parse_failure_sink"}),
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALL_SHAPE,
            callee_names=frozenset({"_parse_failure_sink"}),
            min_positional_args=1,
        ),
    ),
    min_members=2,
    candidate="ParseFailureSinkCarrier Protocol",
    description=(
        "Functions normalizing parse-failure witnesses should share one "
        "typed sink-carrier boundary contract."
    ),
)
_PARSE_MODULE_TREE_RULE = _ExecutionPatternRule(
    pattern_id="parse_module_tree_stage",
    kind="execution_pattern",
    schema_family="parse_module_tree_cluster",
    predicates=(
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.PARAMS,
            required_params=frozenset({"parse_failure_witnesses"}),
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALLEE,
            callee_names=frozenset({"_parse_module_tree"}),
        ),
        _ExecutionPatternPredicate(
            kind=_ExecutionPatternPredicateKind.CALL_SHAPE,
            callee_names=frozenset({"_parse_module_tree"}),
            required_keywords=frozenset({"parse_failure_witnesses", "stage"}),
            min_positional_args=1,
        ),
    ),
    min_members=2,
    candidate="ParseModuleStageSpec Protocol",
    description=(
        "Functions reusing parse-module stage dispatch should centralize "
        "stage + witness carriage into a shared protocol."
    ),
)

EXECUTION_PATTERN_RULES: tuple[_ExecutionPatternRule, ...] = (
    _INDEXED_PASS_INGRESS_RULE,
    _INDEXED_PASS_RUNNER_RULE,
    _INDEXED_PASS_GRAPH_RULE,
    _PARSE_FAILURE_SINK_RULE,
    _PARSE_MODULE_TREE_RULE,
)


_DEFAULT_EXECUTION_SOURCE_BASENAMES: tuple[str, ...] = (
    "dataflow_pipeline.py",
    "dataflow_reporting.py",
    "dataflow_obligations.py",
    "dataflow_bundle_iteration.py",
    "dataflow_callee_resolution.py",
    "dataflow_run_outputs.py",
    "dataflow_raw_runtime.py",
)


def _default_execution_source_candidates() -> tuple[Path, ...]:
    base_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    for basename in _DEFAULT_EXECUTION_SOURCE_BASENAMES:
        check_deadline()
        path = base_dir / basename
        if path.exists():
            candidates.append(path)
    return tuple(candidates)


def _default_execution_source_path() -> Path:
    candidates = _default_execution_source_candidates()
    if candidates:
        return candidates[0]
    # Fallback for partially-checked-out states.
    return Path(__file__).with_name("dataflow_pipeline.py")


def _default_execution_source_blob() -> tuple[str, Path]:
    candidates = _default_execution_source_candidates()
    if not candidates:
        return "", _default_execution_source_path()
    chunks: list[str] = []
    for path in candidates:
        check_deadline()
        try:
            source_text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        chunks.append(f"# <gabion-execution-source:{path.name}>\n{source_text}\n")
    if not chunks:
        return "", candidates[0]
    return "\n".join(chunks), candidates[0]


def _function_param_names(node: FunctionNode) -> tuple[str, ...]:
    params: list[str] = []
    params.extend(arg.arg for arg in node.args.posonlyargs)
    params.extend(arg.arg for arg in node.args.args)
    params.extend(arg.arg for arg in node.args.kwonlyargs)
    return tuple(params)


def _callable_name_variants(node: ast.AST) -> tuple[str, ...]:
    if _is_name_node(node):
        return (_name_node_id(node),)
    if not _is_attribute_node(node):
        return ()
    attribute = _attribute_node(node)
    parts: list[str] = [attribute.attr]
    cursor: ast.AST = attribute.value
    while _is_attribute_node(cursor):
        nested = _attribute_node(cursor)
        parts.append(nested.attr)
        cursor = nested.value
    if _is_name_node(cursor):
        parts.append(_name_node_id(cursor))
    dotted = ".".join(reversed(parts))
    return (attribute.attr, dotted)


def _iter_execution_function_facts(tree: ast.Module) -> Iterator[_ExecutionFunctionFact]:
    for node_index, node in enumerate(tree.body, start=1):
        if node_index % 32 == 0:
            check_deadline()
        if not _is_function_node(node):
            continue
        function_node = _function_node(node)
        param_names = frozenset(_function_param_names(function_node))
        alias_map: dict[str, tuple[str, ...]] = {}
        for statement_index, statement in enumerate(function_node.body, start=1):
            if statement_index % 32 == 0:
                check_deadline()
            if not _is_assign_node(statement):
                continue
            assign_node = _assign_node(statement)
            if len(assign_node.targets) != 1:
                continue
            target = assign_node.targets[0]
            if not _is_name_node(target):
                continue
            variants = _callable_name_variants(assign_node.value)
            if not variants:
                continue
            alias_map[_name_node_id(target)] = variants
        called_names: set[str] = set()
        call_shapes: dict[str, set[_ExecutionCallShape]] = defaultdict(set)
        for index, child in enumerate(ast.walk(function_node), start=1):
            if index % 64 == 0:
                check_deadline()
            if not _is_call_node(child):
                continue
            call_node = _call_node(child)
            variants = _callable_name_variants(call_node.func)
            if not variants:
                continue
            shape = _ExecutionCallShape(
                positional_args=len(call_node.args),
                keyword_names=frozenset(
                    keyword.arg
                    for keyword in call_node.keywords
                    if keyword.arg is not None
                ),
            )
            for variant in variants:
                called_names.add(variant)
                call_shapes[variant].add(shape)
                for alias_variant in alias_map.get(variant, ()):  # boundary alias normalization
                    called_names.add(alias_variant)
                    call_shapes[alias_variant].add(shape)
        yield _ExecutionFunctionFact(
            function_name=function_node.name,
            param_names=param_names,
            called_names=frozenset(called_names),
            call_shapes={
                name: tuple(
                    sort_once(
                        shapes,
                        source="pattern_schema_projection._iter_execution_function_facts.call_shapes",
                        key=lambda entry: (
                            entry.positional_args,
                            tuple(
                                sort_once(
                                    entry.keyword_names,
                                    source="pattern_schema_projection._iter_execution_function_facts.call_shape_keywords",
                                )
                            ),
                        ),
                    )
                )
                for name, shapes in call_shapes.items()
            },
        )


def _execution_pattern_members_from_facts(
    *,
    facts: Sequence[_ExecutionFunctionFact],
    rule: _ExecutionPatternRule,
) -> tuple[str, ...]:
    members: list[str] = []
    for fact_index, fact in enumerate(facts, start=1):
        if fact_index % 64 == 0:
            check_deadline()
        if any(not predicate.matches(fact=fact) for predicate in rule.predicates):
            continue
        members.append(fact.function_name)
    return tuple(sort_once(members, source=f"pattern_schema_projection._execution_pattern_members.{rule.pattern_id}"))


def _execution_pattern_members_by_rule(
    *,
    tree: ast.Module,
) -> dict[str, tuple[str, ...]]:
    facts = tuple(_iter_execution_function_facts(tree))
    members_by_rule: dict[str, tuple[str, ...]] = {}
    for rule in EXECUTION_PATTERN_RULES:
        check_deadline()
        members_by_rule[rule.pattern_id] = _execution_pattern_members_from_facts(
            facts=facts,
            rule=rule,
        )
    return members_by_rule


def _execution_pattern_match_suggestion(
    *,
    rule: _ExecutionPatternRule,
    members: Sequence[str],
) -> str:
    return (
        f"{rule.pattern_id} members={len(members)} "
        + ", ".join(members[:8])
        + (" ..." if len(members) > 8 else "")
        + f"; candidate={rule.candidate}"
    )


def _execution_rule_residue_payload(
    *,
    rule: _ExecutionPatternRule,
    members: Sequence[str],
    observed_member_count: int,
) -> JSONObject:
    check_deadline()
    return {
        "pattern_id": rule.pattern_id,
        "schema_family": rule.schema_family,
        "predicate_contract": [predicate.payload() for predicate in rule.predicates],
        "members": list(
            sort_once(
                {str(member) for member in members},
                source="pattern_schema_projection._execution_rule_residue_payload.members",
            )
        ),
        "member_count": observed_member_count,
    }


def detect_execution_pattern_matches(
    *,
    source: object = None,
    source_path: Path | None = None,
) -> list[_ExecutionPatternMatch]:
    source_text = _source_text_optional(source)
    module_path = source_path or _default_execution_source_path()
    if source_text is None and source_path is None:
        source_text, module_path = _default_execution_source_blob()
    elif source_text is None:
        try:
            source_text = module_path.read_text(encoding="utf-8")
        except OSError:
            return []
    if source_text is None:
        never("unregistered runtime type", value_type=type(source).__name__)
    if not source_text.strip():
        return []
    try:
        tree = ast.parse(source_text)
    except _PARSE_MODULE_ERROR_TYPES:
        return []
    members_by_rule = _execution_pattern_members_by_rule(tree=tree)
    matches: list[_ExecutionPatternMatch] = []
    for rule in EXECUTION_PATTERN_RULES:
        check_deadline()
        members = members_by_rule.get(rule.pattern_id, ())
        if len(members) < rule.min_members:
            continue
        matches.append(
            _ExecutionPatternMatch(
                pattern_id=rule.pattern_id,
                kind=rule.kind,
                schema_family=rule.schema_family,
                members=members,
                suggestion=_execution_pattern_match_suggestion(
                    rule=rule,
                    members=members,
                ),
            )
        )
    return matches


def execution_pattern_instances(
    *,
    source: object = None,
    source_path: Path | None = None,
) -> list[PatternInstance]:
    instances: list[PatternInstance] = []
    source_text = _source_text_optional(source)
    module_path = source_path or _default_execution_source_path()
    if source_text is None and source_path is None:
        source_text, module_path = _default_execution_source_blob()
    elif source_text is None:
        try:
            source_text = module_path.read_text(encoding="utf-8")
        except OSError:
            return instances
    if source_text is None:
        never("unregistered runtime type", value_type=type(source).__name__)
    if not source_text.strip():
        return instances
    try:
        tree = ast.parse(source_text)
    except _PARSE_MODULE_ERROR_TYPES:
        return instances
    members_by_rule = _execution_pattern_members_by_rule(tree=tree)
    for rule in EXECUTION_PATTERN_RULES:
        check_deadline()
        members = members_by_rule.get(rule.pattern_id, ())
        if len(members) < rule.min_members:
            continue
        signature = execution_signature(
            family=rule.schema_family,
            members=members,
        )
        schema = PatternSchema.build(
            axis=PatternAxis.EXECUTION,
            kind=rule.kind,
            signature=signature,
            normalization={
                "members": list(members),
                "pattern_id": rule.pattern_id,
            },
        )
        instances.append(
            PatternInstance.build(
                schema=schema,
                members=members,
                suggestion=(
                    "execution_pattern "
                    + _execution_pattern_match_suggestion(
                        rule=rule,
                        members=members,
                    )
                ),
                residue=(
                    PatternResidue(
                        schema_id=schema.schema_id,
                        reason="unreified_metafactory",
                        payload=mismatch_residue_payload(
                            axis=PatternAxis.EXECUTION,
                            kind=rule.pattern_id,
                            expected={
                                "min_members": rule.min_members,
                                "candidate": rule.candidate,
                                "rule": _execution_rule_residue_payload(
                                    rule=rule,
                                    members=members,
                                    observed_member_count=len(members),
                                ),
                            },
                            observed={
                                "members": list(members),
                                "member_count": len(members),
                                "rule": _execution_rule_residue_payload(
                                    rule=rule,
                                    members=members,
                                    observed_member_count=len(members),
                                ),
                            },
                        ),
                    ),
                ),
            )
        )
    for rule in EXECUTION_PATTERN_RULES:
        check_deadline()
        near_miss_members = members_by_rule.get(rule.pattern_id, ())
        if len(near_miss_members) != max(rule.min_members - 1, 1):
            continue
        signature = execution_signature(
            family=rule.schema_family,
            members=near_miss_members,
        )
        schema = PatternSchema.build(
            axis=PatternAxis.EXECUTION,
            kind=rule.kind,
            signature=signature,
            normalization={
                "members": list(near_miss_members),
                "pattern_id": rule.pattern_id,
            },
        )
        instances.append(
            PatternInstance.build(
                schema=schema,
                members=near_miss_members,
                suggestion=(
                    "execution_pattern near_miss "
                    + f"{rule.pattern_id} members={len(near_miss_members)}"
                ),
                residue=(
                    PatternResidue(
                        schema_id=schema.schema_id,
                        reason="schema_contract_mismatch",
                        payload=mismatch_residue_payload(
                            axis=PatternAxis.EXECUTION,
                            kind=rule.pattern_id,
                            expected={
                                "min_members": rule.min_members,
                                "rule": _execution_rule_residue_payload(
                                    rule=rule,
                                    members=near_miss_members,
                                    observed_member_count=len(near_miss_members),
                                ),
                            },
                            observed={
                                "members": list(near_miss_members),
                                "member_count": len(near_miss_members),
                                "rule": _execution_rule_residue_payload(
                                    rule=rule,
                                    members=near_miss_members,
                                    observed_member_count=len(near_miss_members),
                                ),
                            },
                        ),
                    ),
                ),
            )
        )
    return instances


def bundle_pattern_instances(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
) -> list[PatternInstance]:
    if not groups_by_path:
        return []
    occurrences: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for path, by_fn in groups_by_path.items():
        check_deadline()
        for fn_name, bundles in by_fn.items():
            check_deadline()
            for bundle in bundles:
                check_deadline()
                key = tuple(
                    sort_once(
                        bundle,
                        source="pattern_schema_projection.bundle_pattern_instances.bundle",
                    )
                )
                if len(key) < 2:
                    continue
                occurrences[key].append(f"{path.name}:{fn_name}")
    instances: list[PatternInstance] = []
    for bundle_key in sort_once(
        occurrences,
        source="pattern_schema_projection.bundle_pattern_instances.occurrences",
    ):
        check_deadline()
        members = tuple(
            sort_once(
                set(occurrences[bundle_key]),
                source="pattern_schema_projection.bundle_pattern_instances.members",
            )
        )
        count = len(members)
        if count <= 1:
            schema = PatternSchema.build(
                axis=PatternAxis.DATAFLOW,
                kind="bundle_signature",
                signature={
                    "bundle": list(bundle_key),
                    "tier": 3,
                    "site_count": count,
                },
                normalization={"bundle": list(bundle_key)},
            )
            instances.append(
                PatternInstance.build(
                    schema=schema,
                    members=members,
                    suggestion=(
                        "dataflow_pattern near_miss "
                        + f"bundle={','.join(bundle_key)} sites={count}"
                    ),
                    residue=(
                        PatternResidue(
                            schema_id=schema.schema_id,
                            reason="schema_contract_mismatch",
                            payload=mismatch_residue_payload(
                                axis=PatternAxis.DATAFLOW,
                                kind="bundle_signature",
                                expected={"min_sites": 2, "tier": 2},
                                observed={
                                    "site_count": count,
                                    "tier": 3,
                                    "bundle": list(bundle_key),
                                },
                            ),
                        ),
                    ),
                )
            )
            continue
        signature: JSONObject = {
            "bundle": list(bundle_key),
            "tier": 2 if count > 1 else 3,
            "site_count": count,
        }
        schema = PatternSchema.build(
            axis=PatternAxis.DATAFLOW,
            kind="bundle_signature",
            signature=signature,
            normalization={"bundle": list(bundle_key)},
        )
        instances.append(
            PatternInstance.build(
                schema=schema,
                members=members,
                suggestion=(
                    "dataflow_pattern "
                    + f"bundle={','.join(bundle_key)} sites={count}; "
                    + "candidate=Protocol/dataclass reification"
                ),
                residue=(
                    PatternResidue(
                        schema_id=schema.schema_id,
                        reason="unreified_protocol",
                        payload=mismatch_residue_payload(
                            axis=PatternAxis.DATAFLOW,
                            kind="bundle_signature",
                            expected={
                                "candidate": "Protocol/dataclass reification",
                                "tier": 2,
                            },
                            observed={
                                "bundle": list(bundle_key),
                                "site_count": count,
                                "tier": 2 if count > 1 else 3,
                            },
                        ),
                    ),
                ),
            )
        )
    return instances


def pattern_schema_matches(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    include_execution: bool = True,
    source: object = None,
    source_path: Path | None = None,
) -> list[PatternInstance]:
    instances: list[PatternInstance] = []
    if include_execution:
        instances.extend(
            execution_pattern_instances(
                source=source,
                source_path=source_path,
            )
        )
    instances.extend(
        bundle_pattern_instances(
            groups_by_path=groups_by_path,
        )
    )
    return sort_once(
        instances,
        source="pattern_schema_projection.pattern_schema_matches.instances",
        key=lambda entry: (
            entry.schema.axis.value,
            entry.schema.kind,
            entry.schema.schema_id,
            entry.suggestion,
        ),
    )


def pattern_schema_suggestions(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    include_execution: bool = True,
    source: object = None,
    source_path: Path | None = None,
) -> list[str]:
    instances = pattern_schema_matches(
        groups_by_path=groups_by_path,
        include_execution=include_execution,
        source=source,
        source_path=source_path,
    )
    return pattern_schema_suggestions_from_instances(instances)


def pattern_schema_suggestions_from_instances(
    instances: Sequence[PatternInstance],
) -> list[str]:
    suggestions: list[str] = []
    for instance in instances:
        check_deadline()
        suggestions.append(
            f"pattern_schema axis={instance.schema.axis.value} {instance.suggestion}"
        )
    return sort_once(
        set(suggestions),
        source="pattern_schema_projection.pattern_schema_suggestions_from_instances.suggestions",
    )


def pattern_schema_residue_entries(
    instances: Sequence[PatternInstance],
) -> list[PatternResidue]:
    entries: list[PatternResidue] = []
    for instance in instances:
        check_deadline()
        entries.extend(instance.residue)
    return sort_once(
        entries,
        source="pattern_schema_projection.pattern_schema_residue_entries.entries",
        key=lambda entry: (
            entry.schema_id,
            entry.reason,
            json.dumps(entry.payload, sort_keys=False, separators=(",", ":")),
        ),
    )


def tier2_unreified_residue_entries(
    entries: Sequence[PatternResidue],
) -> list[PatternResidue]:
    candidates: list[PatternResidue] = []
    for entry in entries:
        check_deadline()
        expected_raw = entry.payload.get("expected")
        match expected_raw:
            case dict() as expected_mapping:
                expected = expected_mapping
            case _:
                expected = {}
        tier_value = expected.get("tier")
        min_members_raw = str(expected.get("min_members", "")).strip()
        min_members = (
            int(min_members_raw)
            if min_members_raw.lstrip("-").isdigit()
            else -1
        )
        tier2_reason = entry.reason in {"unreified_protocol", "unreified_metafactory"}
        tier2_expected = tier_value == 2 or min_members >= 2
        if tier2_reason and tier2_expected:
            candidates.append(entry)
    return sort_once(
        candidates,
        source="pattern_schema_projection.tier2_unreified_residue_entries.candidates",
        key=lambda entry: (
            entry.schema_id,
            entry.reason,
            json.dumps(entry.payload, sort_keys=False, separators=(",", ":")),
        ),
    )


def pattern_schema_residue_lines(entries: Sequence[PatternResidue]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        payload = json.dumps(entry.payload, sort_keys=False)
        lines.append(
            f"schema_id={entry.schema_id} reason={entry.reason} payload={payload}"
        )
    return lines


def pattern_schema_snapshot_entries(
    instances: Sequence[PatternInstance],
) -> tuple[list[JSONObject], list[JSONObject]]:
    serialized_instances: list[JSONObject] = []
    for instance in instances:
        check_deadline()
        serialized_instances.append(
            {
                "schema": {
                    "schema_id": instance.schema.schema_id,
                    "legacy_schema_id": instance.schema.legacy_schema_id,
                    "schema_contract": instance.schema.schema_contract,
                    "axis": instance.schema.axis.value,
                    "kind": instance.schema.kind,
                    "signature": instance.schema.normalized_signature,
                    "normalization": instance.schema.normalization,
                },
                "members": list(instance.members),
                "suggestion": instance.suggestion,
                "residue": [
                    {
                        "schema_id": residue.schema_id,
                        "reason": residue.reason,
                        "payload": residue.payload,
                    }
                    for residue in sort_once(
                        instance.residue,
                        source="pattern_schema_projection.pattern_schema_snapshot_entries.instance_residue",
                        key=lambda entry: (
                            entry.schema_id,
                            entry.reason,
                            json.dumps(entry.payload, sort_keys=False, separators=(",", ":")),
                        ),
                    )
                ],
            }
        )
    residues = pattern_schema_residue_entries(instances)
    serialized_residue: list[JSONObject] = []
    for entry in residues:
        check_deadline()
        serialized_residue.append(
            {
                "schema_id": entry.schema_id,
                "reason": entry.reason,
                "payload": entry.payload,
            }
        )
    return serialized_instances, serialized_residue


def execution_pattern_suggestions(
    *,
    source: object = None,
    source_path: Path | None = None,
) -> list[str]:
    suggestions: list[str] = []
    for instance in execution_pattern_instances(
        source=source,
        source_path=source_path,
    ):
        check_deadline()
        suggestions.append(instance.suggestion)
    return sort_once(
        set(suggestions),
        source="pattern_schema_projection.execution_pattern_suggestions.suggestions",
    )


def pattern_schema_surface_payloads(
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    include_execution: bool = True,
    source: object = None,
    source_path: Path | None = None,
) -> tuple[list[PatternInstance], list[JSONObject], list[JSONObject]]:
    instances = pattern_schema_matches(
        groups_by_path=groups_by_path,
        include_execution=include_execution,
        source=source,
        source_path=source_path,
    )
    snapshot_instances, snapshot_residue = pattern_schema_snapshot_entries(instances)
    return instances, snapshot_instances, snapshot_residue
