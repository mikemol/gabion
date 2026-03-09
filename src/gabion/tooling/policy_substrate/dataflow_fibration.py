# gabion:decision_protocol_module
from __future__ import annotations

import ast
from functools import reduce
import hashlib
from itertools import chain, groupby
from dataclasses import dataclass
from typing import Iterable, Iterator

from gabion.tooling.policy_substrate.site_identity import canonical_site_identity


@dataclass(frozen=True)
class DataflowEvent:
    site_id: str
    site_identity: str
    rel_path: str
    qualname: str
    line: int
    column: int
    ordinal: int
    symbol: str
    event_kind: str
    node_kind: str


@dataclass(frozen=True)
class DataflowEdge:
    edge_id: str
    symbol: str
    source_site_id: str
    target_site_id: str
    source_site_identity: str
    target_site_identity: str
    source_ordinal: int
    target_ordinal: int


@dataclass(frozen=True)
class ExecutionEvent:
    site_id: str
    site_identity: str
    rel_path: str
    qualname: str
    line: int
    column: int
    ordinal: int
    node_kind: str
    event_kind: str


@dataclass(frozen=True)
class ExecutionEdge:
    edge_id: str
    source_site_id: str
    target_site_id: str
    source_site_identity: str
    target_site_identity: str
    source_ordinal: int
    target_ordinal: int


@dataclass(frozen=True)
class DataflowFiberBundle:
    rel_path: str
    qualname: str
    entry_site_id: str
    entry_site_identity: str
    events: tuple[DataflowEvent, ...]
    edges: tuple[DataflowEdge, ...]
    execution_events: tuple[ExecutionEvent, ...]
    execution_edges: tuple[ExecutionEdge, ...]


@dataclass(frozen=True)
class RecombinationFrontier:
    branch_site_id: str
    branch_site_identity: str
    branch_line: int
    branch_column: int
    branch_node_kind: str
    required_symbols: tuple[str, ...]
    unresolved_symbols: tuple[str, ...]
    anchor_site_id: str
    anchor_site_identity: str
    anchor_line: int
    anchor_column: int
    anchor_ordinal: int
    upstream_site_ids: tuple[str, ...]
    upstream_site_identities: tuple[str, ...]
    upstream_edge_ids: tuple[str, ...]
    execution_frontier_site_id: str
    execution_frontier_site_identity: str
    execution_frontier_line: int
    execution_frontier_column: int
    execution_frontier_ordinal: int
    execution_upstream_site_ids: tuple[str, ...]
    execution_upstream_site_identities: tuple[str, ...]
    execution_upstream_edge_ids: tuple[str, ...]
    bundle_event_count: int
    bundle_edge_count: int
    execution_event_count: int
    execution_edge_count: int


def build_dataflow_fiber_bundle_for_qualname(
    *,
    rel_path: str,
    module_tree: ast.AST,
    qualname: str,
) -> DataflowFiberBundle:
    scope = _resolve_scope(module_tree=module_tree, qualname=qualname)
    collector = _DataflowCollector(
        rel_path=rel_path,
        qualname=scope.qualname,
        entry_line=scope.start_line,
        entry_column=scope.start_column,
    )
    collector.record_argument_definitions(scope.arguments)
    tuple(map(collector.visit, scope.body))
    execution_collector = _ExecutionCollector(
        rel_path=rel_path,
        qualname=scope.qualname,
        entry_line=scope.start_line,
        entry_column=scope.start_column,
    )
    tuple(execution_collector.record_scope_statements(scope.body))
    execution_events = tuple(
        sorted(
            execution_collector.events,
            key=lambda event: (event.ordinal, event.line, event.column, event.node_kind),
        )
    )
    execution_edges = tuple(
        sorted(
            execution_collector.edges,
            key=lambda edge: (edge.source_ordinal, edge.target_ordinal),
        )
    )
    return DataflowFiberBundle(
        rel_path=rel_path,
        qualname=scope.qualname,
        entry_site_id=collector.entry_site_id,
        entry_site_identity=collector.entry_site_identity,
        events=tuple(collector.events),
        edges=tuple(
            sorted(
                collector.edges,
                key=lambda edge: (edge.source_ordinal, edge.target_ordinal, edge.symbol),
            )
        ),
        execution_events=execution_events,
        execution_edges=execution_edges,
    )


def branch_required_symbols(node: ast.AST) -> Iterator[str]:
    expression_nodes = tuple(_branch_condition_nodes(node))
    all_walked_nodes = chain.from_iterable(map(ast.walk, expression_nodes))
    name_ids = map(
        lambda item: str(item.id),
        filter(_is_loaded_name_node, all_walked_nodes),
    )
    for name in sorted(dict.fromkeys(name_ids)):
        yield name


def compute_recombination_frontier(
    *,
    rel_path: str,
    qualname: str,
    bundle: DataflowFiberBundle,
    branch_line: int,
    branch_column: int,
    branch_node_kind: str,
    required_symbols: Iterable[str],
) -> RecombinationFrontier:
    normalized_required = tuple(sorted(dict.fromkeys(required_symbols)))
    defs_by_symbol = dict(
        _iter_latest_definitions_before_branch(
            events=bundle.events,
            branch_line=branch_line,
            branch_column=branch_column,
            symbols=normalized_required,
        )
    )
    unresolved = tuple(
        filter(
            _symbol_missing_in(defs_by_symbol),
            normalized_required,
        )
    )
    upstream_events = tuple(
        sorted(defs_by_symbol.values(), key=lambda event: event.ordinal)
    )
    anchor_event = _anchor_event(
        upstream_events=upstream_events,
        events=bundle.events,
    )
    upstream_site_ids = tuple(map(_site_id_from_event, upstream_events))
    upstream_site_identities = tuple(map(_site_identity_from_event, upstream_events))
    upstream_edge_ids = tuple(
        map(
            _edge_id_from_edge,
            filter(
                _edge_targets_in(upstream_site_ids),
                bundle.edges,
            ),
        )
    )
    branch_site_identity = _branch_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=branch_line,
        column=branch_column,
        node_kind=branch_node_kind,
    )
    execution_frontier = _execution_recombination_frontier(
        bundle=bundle,
        branch_site_identity=branch_site_identity,
        required_dataflow_events=upstream_events,
    )
    return RecombinationFrontier(
        branch_site_id=_site_id(
            rel_path=rel_path,
            qualname=qualname,
            line=branch_line,
            column=branch_column,
            event_kind="branch",
            symbol="",
            node_kind=branch_node_kind,
        ),
        branch_site_identity=branch_site_identity,
        branch_line=branch_line,
        branch_column=branch_column,
        branch_node_kind=branch_node_kind,
        required_symbols=normalized_required,
        unresolved_symbols=unresolved,
        anchor_site_id=anchor_event.site_id,
        anchor_site_identity=anchor_event.site_identity,
        anchor_line=anchor_event.line,
        anchor_column=anchor_event.column,
        anchor_ordinal=anchor_event.ordinal,
        upstream_site_ids=upstream_site_ids,
        upstream_site_identities=upstream_site_identities,
        upstream_edge_ids=tuple(sorted(upstream_edge_ids)),
        execution_frontier_site_id=execution_frontier.site_id,
        execution_frontier_site_identity=execution_frontier.site_identity,
        execution_frontier_line=execution_frontier.line,
        execution_frontier_column=execution_frontier.column,
        execution_frontier_ordinal=execution_frontier.ordinal,
        execution_upstream_site_ids=execution_frontier.upstream_site_ids,
        execution_upstream_site_identities=execution_frontier.upstream_site_identities,
        execution_upstream_edge_ids=execution_frontier.upstream_edge_ids,
        bundle_event_count=len(bundle.events),
        bundle_edge_count=len(bundle.edges),
        execution_event_count=len(bundle.execution_events),
        execution_edge_count=len(bundle.execution_edges),
    )


def empty_recombination_frontier(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
) -> RecombinationFrontier:
    site_identity = _branch_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=node_kind,
    )
    site_id = _site_id(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        event_kind="branch",
        symbol="",
        node_kind=node_kind,
    )
    return RecombinationFrontier(
        branch_site_id=site_id,
        branch_site_identity=site_identity,
        branch_line=line,
        branch_column=column,
        branch_node_kind=node_kind,
        required_symbols=(),
        unresolved_symbols=(),
        anchor_site_id=site_id,
        anchor_site_identity=site_identity,
        anchor_line=line,
        anchor_column=column,
        anchor_ordinal=0,
        upstream_site_ids=(),
        upstream_site_identities=(),
        upstream_edge_ids=(),
        execution_frontier_site_id=site_id,
        execution_frontier_site_identity=site_identity,
        execution_frontier_line=line,
        execution_frontier_column=column,
        execution_frontier_ordinal=0,
        execution_upstream_site_ids=(),
        execution_upstream_site_identities=(),
        execution_upstream_edge_ids=(),
        bundle_event_count=0,
        bundle_edge_count=0,
        execution_event_count=0,
        execution_edge_count=0,
    )


@dataclass(frozen=True)
class _ResolvedScope:
    qualname: str
    start_line: int
    start_column: int
    arguments: tuple[ast.arg, ...]
    body: tuple[ast.stmt, ...]


def _resolve_scope(*, module_tree: ast.AST, qualname: str) -> _ResolvedScope:
    root_statements = tuple(_module_statements(module_tree))
    if qualname == "<module>":
        return _ResolvedScope(
            qualname="<module>",
            start_line=1,
            start_column=1,
            arguments=(),
            body=root_statements,
        )

    segments = tuple(filter(bool, qualname.split(".")))
    resolved_scope = _resolve_function_scope(segments=segments, root_statements=root_statements)
    if not resolved_scope.resolved:
        return _ResolvedScope(
            qualname=qualname,
            start_line=1,
            start_column=1,
            arguments=(),
            body=(),
        )
    return _ResolvedScope(
        qualname=qualname,
        start_line=resolved_scope.start_line,
        start_column=resolved_scope.start_column,
        arguments=resolved_scope.arguments,
        body=resolved_scope.body,
    )


@dataclass(frozen=True)
class _ResolvedFunctionScope:
    resolved: bool
    start_line: int
    start_column: int
    arguments: tuple[ast.arg, ...]
    body: tuple[ast.stmt, ...]


def _module_statements(module_tree: ast.AST) -> Iterator[ast.stmt]:
    match module_tree:
        case ast.Module(body=body):
            for statement in body:
                yield statement
        case _:
            return


def _resolve_function_scope(
    *,
    segments: tuple[str, ...],
    root_statements: tuple[ast.stmt, ...],
) -> _ResolvedFunctionScope:
    if not segments:
        return _empty_function_scope()
    return _resolve_function_scope_recursive(
        segments=segments,
        statements=root_statements,
    )


def _resolve_function_scope_recursive(
    *,
    segments: tuple[str, ...],
    statements: tuple[ast.stmt, ...],
) -> _ResolvedFunctionScope:
    current = segments[0]
    rest = segments[1:]
    first_match = next(filter(_statement_name_equals(current), statements), None)
    if first_match is None:
        return _empty_function_scope()
    if not rest:
        return _ResolvedFunctionScope(
            resolved=True,
            start_line=int(getattr(first_match, "lineno", 1)),
            start_column=int(getattr(first_match, "col_offset", 0)) + 1,
            arguments=tuple(_iter_function_arguments(first_match)),
            body=tuple(first_match.body),
        )
    return _resolve_function_scope_recursive(
        segments=rest,
        statements=tuple(first_match.body),
    )


def _empty_function_scope() -> _ResolvedFunctionScope:
    return _ResolvedFunctionScope(
        resolved=False,
        start_line=1,
        start_column=1,
        arguments=(),
        body=(),
    )


def _statement_name_matches(*, statement: ast.stmt, name: str) -> bool:
    match statement:
        case ast.FunctionDef(name=fn_name):
            return fn_name == name
        case ast.AsyncFunctionDef(name=fn_name):
            return fn_name == name
        case _:
            return False


def _statement_name_equals(name: str):
    return lambda statement: _statement_name_matches(statement=statement, name=name)


def _iter_function_arguments(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.arg]:
    arguments = function_node.args
    for argument in chain(arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
        yield argument
    for optional in filter(_is_ast_arg, (arguments.vararg, arguments.kwarg)):
        yield optional


class _DataflowCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        qualname: str,
        entry_line: int,
        entry_column: int,
    ) -> None:
        self.rel_path = rel_path
        self.qualname = qualname
        self._next_ordinal = 1
        self.events: list[DataflowEvent] = []
        self.edges: list[DataflowEdge] = []
        self._last_def_by_symbol: dict[str, DataflowEvent] = {}
        self.entry_site_identity = _site_identity(
            rel_path=rel_path,
            qualname=qualname,
            line=entry_line,
            column=entry_column,
            node_kind="entry",
        )
        self.entry_site_id = _site_id(
            rel_path=rel_path,
            qualname=qualname,
            line=entry_line,
            column=entry_column,
            event_kind="entry",
            symbol="",
            node_kind="entry",
        )
        self.events.append(
            DataflowEvent(
                site_id=self.entry_site_id,
                site_identity=self.entry_site_identity,
                rel_path=rel_path,
                qualname=qualname,
                line=entry_line,
                column=entry_column,
                ordinal=0,
                symbol="",
                event_kind="entry",
                node_kind="entry",
            )
        )

    def record_argument_definitions(self, arguments: tuple[ast.arg, ...]) -> None:
        tuple(map(self._record_argument_definition, arguments))

    def _record_argument_definition(self, arg_node: ast.arg) -> None:
        self._record_definition(
            symbol=str(arg_node.arg),
            line=int(getattr(arg_node, "lineno", 1)),
            column=int(getattr(arg_node, "col_offset", 0)) + 1,
            node_kind="arg",
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        _ = node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        _ = node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        _ = node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        _ = node

    def visit_Name(self, node: ast.Name) -> None:
        symbol = str(node.id)
        line = int(getattr(node, "lineno", 1))
        column = int(getattr(node, "col_offset", 0)) + 1
        match node.ctx:
            case ast.Store() | ast.Del():
                self._record_definition(
                    symbol=symbol,
                    line=line,
                    column=column,
                    node_kind="name_store",
                )
            case ast.Load():
                self._record_use(
                    symbol=symbol,
                    line=line,
                    column=column,
                    node_kind="name_load",
                )
            case _:
                return

    def _record_definition(
        self,
        *,
        symbol: str,
        line: int,
        column: int,
        node_kind: str,
    ) -> None:
        site_identity = _site_identity(
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            node_kind=node_kind,
        )
        event = DataflowEvent(
            site_id=_site_id(
                rel_path=self.rel_path,
                qualname=self.qualname,
                line=line,
                column=column,
                event_kind="def",
                symbol=symbol,
                node_kind=node_kind,
            ),
            site_identity=site_identity,
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            ordinal=self._next_ordinal,
            symbol=symbol,
            event_kind="def",
            node_kind=node_kind,
        )
        self.events.append(event)
        self._last_def_by_symbol[symbol] = event
        self._next_ordinal += 1

    def _record_use(
        self,
        *,
        symbol: str,
        line: int,
        column: int,
        node_kind: str,
    ) -> None:
        site_identity = _site_identity(
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            node_kind=node_kind,
        )
        event = DataflowEvent(
            site_id=_site_id(
                rel_path=self.rel_path,
                qualname=self.qualname,
                line=line,
                column=column,
                event_kind="use",
                symbol=symbol,
                node_kind=node_kind,
            ),
            site_identity=site_identity,
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            ordinal=self._next_ordinal,
            symbol=symbol,
            event_kind="use",
            node_kind=node_kind,
        )
        self.events.append(event)
        self._next_ordinal += 1
        if symbol not in self._last_def_by_symbol:
            return
        source_event = self._last_def_by_symbol[symbol]
        self.edges.append(
            DataflowEdge(
                edge_id=_edge_id(
                    symbol=symbol,
                    source_site_id=source_event.site_id,
                    target_site_id=event.site_id,
                ),
                symbol=symbol,
                source_site_id=source_event.site_id,
                target_site_id=event.site_id,
                source_site_identity=source_event.site_identity,
                target_site_identity=event.site_identity,
                source_ordinal=source_event.ordinal,
                target_ordinal=event.ordinal,
            )
        )


class _ExecutionCollector:
    def __init__(
        self,
        *,
        rel_path: str,
        qualname: str,
        entry_line: int,
        entry_column: int,
    ) -> None:
        self.rel_path = rel_path
        self.qualname = qualname
        self.events: list[ExecutionEvent] = []
        self.edges: list[ExecutionEdge] = []
        self._next_ordinal = 1
        self.entry_site_identity = _site_identity(
            rel_path=rel_path,
            qualname=qualname,
            line=entry_line,
            column=entry_column,
            node_kind="entry",
        )
        self.entry_site_id = _execution_site_id(
            rel_path=rel_path,
            qualname=qualname,
            line=entry_line,
            column=entry_column,
            node_kind="entry",
            event_kind="entry",
        )
        self.events.append(
            ExecutionEvent(
                site_id=self.entry_site_id,
                site_identity=self.entry_site_identity,
                rel_path=rel_path,
                qualname=qualname,
                line=entry_line,
                column=entry_column,
                ordinal=0,
                node_kind="entry",
                event_kind="entry",
            )
        )

    def record_scope_statements(
        self,
        statements: tuple[ast.stmt, ...],
    ) -> Iterator[ExecutionEvent]:
        predecessor = self.events[0]
        for statement in statements:
            current = self._record_statement_event(statement=statement)
            self._record_execution_edge(source=predecessor, target=current)
            self._record_nested_statements(statement=statement, branch_event=current)
            predecessor = current
            yield current

    def _record_statement_event(self, *, statement: ast.stmt) -> ExecutionEvent:
        line = int(getattr(statement, "lineno", 1))
        column = int(getattr(statement, "col_offset", 0)) + 1
        node_kind = _execution_node_kind(statement)
        site_identity = _site_identity(
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            node_kind=node_kind,
        )
        event = ExecutionEvent(
            site_id=_execution_site_id(
                rel_path=self.rel_path,
                qualname=self.qualname,
                line=line,
                column=column,
                node_kind=node_kind,
                event_kind="exec",
            ),
            site_identity=site_identity,
            rel_path=self.rel_path,
            qualname=self.qualname,
            line=line,
            column=column,
            ordinal=self._next_ordinal,
            node_kind=node_kind,
            event_kind="exec",
        )
        self.events.append(event)
        self._next_ordinal += 1
        return event

    def _record_nested_statements(
        self,
        *,
        statement: ast.stmt,
        branch_event: ExecutionEvent,
    ) -> None:
        tuple(
            chain.from_iterable(
                map(
                    lambda group: self._iter_nested_group_events(
                        group=group,
                        branch_event=branch_event,
                    ),
                    _nested_statement_groups(statement),
                )
            )
        )

    def _iter_nested_group_events(
        self,
        *,
        group: tuple[ast.stmt, ...],
        branch_event: ExecutionEvent,
    ) -> Iterator[ExecutionEvent]:
        predecessor = branch_event
        for nested_statement in group:
            nested_event = self._record_statement_event(statement=nested_statement)
            self._record_execution_edge(source=predecessor, target=nested_event)
            self._record_nested_statements(
                statement=nested_statement,
                branch_event=nested_event,
            )
            predecessor = nested_event
            yield nested_event

    def _record_execution_edge(
        self,
        *,
        source: ExecutionEvent,
        target: ExecutionEvent,
    ) -> None:
        if source.site_id == target.site_id:
            return
        self.edges.append(
            ExecutionEdge(
                edge_id=_execution_edge_id(
                    source_site_id=source.site_id,
                    target_site_id=target.site_id,
                ),
                source_site_id=source.site_id,
                target_site_id=target.site_id,
                source_site_identity=source.site_identity,
                target_site_identity=target.site_identity,
                source_ordinal=source.ordinal,
                target_ordinal=target.ordinal,
            )
        )


@dataclass(frozen=True)
class _ExecutionFrontierAnchor:
    site_id: str
    site_identity: str
    line: int
    column: int
    ordinal: int
    upstream_site_ids: tuple[str, ...]
    upstream_site_identities: tuple[str, ...]
    upstream_edge_ids: tuple[str, ...]


def _iter_latest_definitions_before_branch(
    *,
    events: tuple[DataflowEvent, ...],
    branch_line: int,
    branch_column: int,
    symbols: tuple[str, ...],
) -> Iterator[tuple[str, DataflowEvent]]:
    symbol_set = set(symbols)
    sorted_candidates = tuple(
        sorted(
            filter(
                _definition_candidate_filter(
                    symbol_set=symbol_set,
                    branch_line=branch_line,
                    branch_column=branch_column,
                ),
                events,
            ),
            key=lambda event: (event.symbol, event.ordinal),
        )
    )
    for symbol, grouped_events in groupby(
        sorted_candidates,
        key=lambda event: event.symbol,
    ):
        yield symbol, tuple(grouped_events)[-1]


def _event_before_or_at_branch(
    *,
    event: DataflowEvent,
    branch_line: int,
    branch_column: int,
) -> bool:
    if event.line < branch_line:
        return True
    if event.line > branch_line:
        return False
    return event.column <= branch_column


def _anchor_event(
    *,
    upstream_events: tuple[DataflowEvent, ...],
    events: tuple[DataflowEvent, ...],
) -> DataflowEvent:
    if upstream_events:
        return max(upstream_events, key=lambda event: event.ordinal)
    entry_candidates = tuple(filter(_is_entry_event, events))
    if entry_candidates:
        return entry_candidates[0]
    return DataflowEvent(
        site_id="",
        site_identity="",
        rel_path="",
        qualname="",
        line=1,
        column=1,
        ordinal=0,
        symbol="",
        event_kind="entry",
        node_kind="entry",
    )


def _execution_recombination_frontier(
    *,
    bundle: DataflowFiberBundle,
    branch_site_identity: str,
    required_dataflow_events: tuple[DataflowEvent, ...],
) -> _ExecutionFrontierAnchor:
    if not bundle.execution_events:
        return _empty_execution_frontier_anchor(branch_site_identity=branch_site_identity)
    execution_by_identity = {
        event.site_identity: event for event in bundle.execution_events
    }
    execution_by_site_id = {event.site_id: event for event in bundle.execution_events}
    predecessor_map: dict[str, list[str]] = {}
    for edge in bundle.execution_edges:
        predecessor_map.setdefault(edge.target_site_id, []).append(edge.source_site_id)
    predecessors = {
        site_id: tuple(dict.fromkeys(predecessor_map[site_id]))
        for site_id in predecessor_map
    }
    branch_event = _match_execution_event_for_branch(
        execution_events=bundle.execution_events,
        branch_site_identity=branch_site_identity,
    )
    required_execution_events = tuple(
        map(
            lambda event: _match_execution_event_for_dataflow(
                execution_events=bundle.execution_events,
                dataflow_event=event,
            ),
            required_dataflow_events,
        )
    )
    origin_events = tuple(
        dict.fromkeys((branch_event, *required_execution_events))
    )
    if not origin_events:
        origin_events = (bundle.execution_events[0],)
    ancestor_sets = tuple(
        _execution_ancestors(site_id=origin.site_id, predecessors=predecessors)
        for origin in origin_events
    )
    common_site_ids = set.intersection(*ancestor_sets) if ancestor_sets else set()
    frontier_event = _latest_execution_event(
        event_ids=common_site_ids,
        execution_by_site_id=execution_by_site_id,
        fallback=branch_event,
    )
    upstream_events = tuple(
        sorted(
            filter(
                lambda event: event.ordinal <= frontier_event.ordinal,
                tuple(execution_by_identity.values()),
            ),
            key=lambda event: event.ordinal,
        )
    )
    upstream_site_ids = tuple(map(_execution_site_id_from_event, upstream_events))
    upstream_site_identities = tuple(
        map(_execution_site_identity_from_event, upstream_events)
    )
    upstream_edge_ids = tuple(
        sorted(
            map(
                _execution_edge_id_from_edge,
                filter(
                    _execution_edge_targets_in(upstream_site_ids),
                    bundle.execution_edges,
                ),
            )
        )
    )
    return _ExecutionFrontierAnchor(
        site_id=frontier_event.site_id,
        site_identity=frontier_event.site_identity,
        line=frontier_event.line,
        column=frontier_event.column,
        ordinal=frontier_event.ordinal,
        upstream_site_ids=upstream_site_ids,
        upstream_site_identities=upstream_site_identities,
        upstream_edge_ids=upstream_edge_ids,
    )


def _empty_execution_frontier_anchor(*, branch_site_identity: str) -> _ExecutionFrontierAnchor:
    return _ExecutionFrontierAnchor(
        site_id="",
        site_identity=branch_site_identity,
        line=1,
        column=1,
        ordinal=0,
        upstream_site_ids=(),
        upstream_site_identities=(),
        upstream_edge_ids=(),
    )


def _execution_ancestors(
    *,
    site_id: str,
    predecessors: dict[str, tuple[str, ...]],
) -> set[str]:
    visited: set[str] = set()
    stack = [site_id]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(predecessors.get(current, ()))
    return visited


def _latest_execution_event(
    *,
    event_ids: set[str],
    execution_by_site_id: dict[str, ExecutionEvent],
    fallback: ExecutionEvent,
) -> ExecutionEvent:
    resolved_ids = event_ids.intersection(execution_by_site_id)
    candidates = tuple(map(execution_by_site_id.__getitem__, resolved_ids))
    if not candidates:
        return fallback
    return max(candidates, key=lambda event: event.ordinal)


def _match_execution_event_for_branch(
    *,
    execution_events: tuple[ExecutionEvent, ...],
    branch_site_identity: str,
) -> ExecutionEvent:
    direct = tuple(
        filter(
            lambda event: event.site_identity == branch_site_identity,
            execution_events,
        )
    )
    if direct:
        return max(direct, key=lambda event: event.ordinal)
    return execution_events[0]


def _match_execution_event_for_dataflow(
    *,
    execution_events: tuple[ExecutionEvent, ...],
    dataflow_event: DataflowEvent,
) -> ExecutionEvent:
    direct = tuple(
        filter(
            lambda event: event.site_identity == dataflow_event.site_identity,
            execution_events,
        )
    )
    if direct:
        return direct[0]
    same_line = tuple(
        filter(
            lambda event: event.line == dataflow_event.line
            and event.column <= dataflow_event.column,
            execution_events,
        )
    )
    if same_line:
        return same_line[-1]
    return execution_events[0]


def _branch_condition_nodes(node: ast.AST) -> Iterator[ast.AST]:
    match node:
        case ast.If(test=test):
            yield test
        case ast.While(test=test):
            yield test
        case ast.IfExp(test=test):
            yield test
        case ast.For(iter=iter_expr):
            yield iter_expr
        case ast.AsyncFor(iter=iter_expr):
            yield iter_expr
        case ast.Match(subject=subject):
            yield subject
        case _:
            return


def _is_loaded_name_node(node: ast.AST) -> bool:
    match node:
        case ast.Name(ctx=ast.Load()):
            return True
        case _:
            return False


def _is_ast_arg(node: object) -> bool:
    match node:
        case ast.arg():
            return True
        case _:
            return False


def _symbol_missing_in(lookup: dict[str, DataflowEvent]):
    return lambda symbol: symbol not in lookup


def _edge_targets_in(targets: tuple[str, ...]):
    target_set = set(targets)
    return lambda edge: edge.target_site_id in target_set


def _edge_id_from_edge(edge: DataflowEdge) -> str:
    return edge.edge_id


def _site_id_from_event(event: DataflowEvent) -> str:
    return event.site_id


def _site_identity_from_event(event: DataflowEvent) -> str:
    return event.site_identity


def _execution_site_id_from_event(event: ExecutionEvent) -> str:
    return event.site_id


def _execution_site_identity_from_event(event: ExecutionEvent) -> str:
    return event.site_identity


def _execution_edge_id_from_edge(edge: ExecutionEdge) -> str:
    return edge.edge_id


def _execution_edge_targets_in(targets: tuple[str, ...]):
    target_set = set(targets)
    return lambda edge: edge.target_site_id in target_set


def _definition_candidate_filter(
    *,
    symbol_set: set[str],
    branch_line: int,
    branch_column: int,
):
    return lambda event: (
        event.event_kind == "def"
        and event.symbol in symbol_set
        and _event_before_or_at_branch(
            event=event,
            branch_line=branch_line,
            branch_column=branch_column,
        )
    )


def _is_entry_event(event: DataflowEvent) -> bool:
    return event.event_kind == "entry"


def _site_identity(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
) -> str:
    return canonical_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=node_kind,
        surface="pyast",
    )


def _branch_site_identity(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
) -> str:
    return _site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=_execution_node_kind_from_branch(node_kind),
    )


def _execution_node_kind_from_branch(node_kind: str) -> str:
    if node_kind.startswith("branch:"):
        return f"stmt:{node_kind.removeprefix('branch:')}"
    return node_kind


def _site_id(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    event_kind: str,
    symbol: str,
    node_kind: str,
) -> str:
    return _stable_hash(
        rel_path,
        qualname,
        str(line),
        str(column),
        event_kind,
        symbol,
        node_kind,
    )


def _execution_site_id(
    *,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
    event_kind: str,
) -> str:
    return _stable_hash(
        "execution",
        rel_path,
        qualname,
        str(line),
        str(column),
        node_kind,
        event_kind,
    )


def _edge_id(*, symbol: str, source_site_id: str, target_site_id: str) -> str:
    return _stable_hash("edge", symbol, source_site_id, target_site_id)


def _execution_edge_id(*, source_site_id: str, target_site_id: str) -> str:
    return _stable_hash("execution_edge", source_site_id, target_site_id)


def _nested_statement_groups(statement: ast.stmt) -> Iterator[tuple[ast.stmt, ...]]:
    match statement:
        case ast.If(body=body, orelse=orelse):
            yield from _non_empty_statement_groups((tuple(body), tuple(orelse)))
        case ast.For(body=body, orelse=orelse):
            yield from _non_empty_statement_groups((tuple(body), tuple(orelse)))
        case ast.AsyncFor(body=body, orelse=orelse):
            yield from _non_empty_statement_groups((tuple(body), tuple(orelse)))
        case ast.While(body=body, orelse=orelse):
            yield from _non_empty_statement_groups((tuple(body), tuple(orelse)))
        case ast.With(body=body):
            yield from _non_empty_statement_groups((tuple(body),))
        case ast.AsyncWith(body=body):
            yield from _non_empty_statement_groups((tuple(body),))
        case ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody):
            handler_groups = tuple(tuple(handler.body) for handler in handlers)
            yield from _non_empty_statement_groups(
                (tuple(body), *handler_groups, tuple(orelse), tuple(finalbody))
            )
        case ast.TryStar(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody):
            handler_groups = tuple(tuple(handler.body) for handler in handlers)
            yield from _non_empty_statement_groups(
                (tuple(body), *handler_groups, tuple(orelse), tuple(finalbody))
            )
        case ast.Match(cases=cases):
            yield from _non_empty_statement_groups(
                tuple(tuple(case.body) for case in cases)
            )
        case _:
            return


def _non_empty_statement_groups(
    groups: tuple[tuple[ast.stmt, ...], ...],
) -> Iterator[tuple[ast.stmt, ...]]:
    for group in filter(bool, groups):
        yield group


def _execution_node_kind(node: ast.stmt) -> str:
    return f"stmt:{node.__class__.__name__.lower()}"


def _stable_hash(*parts: str) -> str:
    digest = reduce(_digest_with_part, parts, hashlib.sha256())
    return digest.hexdigest()


def _digest_with_part(digest: object, part: str):
    digest.update(part.encode("utf-8"))
    digest.update(b"\x00")
    return digest


__all__ = [
    "DataflowEdge",
    "DataflowEvent",
    "DataflowFiberBundle",
    "ExecutionEdge",
    "ExecutionEvent",
    "RecombinationFrontier",
    "branch_required_symbols",
    "build_dataflow_fiber_bundle_for_qualname",
    "compute_recombination_frontier",
    "empty_recombination_frontier",
]
