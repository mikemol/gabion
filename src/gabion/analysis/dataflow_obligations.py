# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field

_BOUND = False


def _bind_audit_symbols() -> None:
    global _BOUND
    if _BOUND:
        return
    from gabion.analysis import dataflow_audit as _audit

    module_globals = globals()
    for name, value in _audit.__dict__.items():
        module_globals.setdefault(name, value)
    _BOUND = True


@dataclass(frozen=True)
class _DeadlineObligationContext:
    by_name: Mapping[str, list[FunctionInfo]]
    by_qual: Mapping[str, FunctionInfo]
    facts_by_qual: Mapping[str, _DeadlineFunctionFacts]
    deadline_params: Mapping[str, set[str]]
    call_infos: Mapping[
        str,
        list[tuple[CallArgs, FunctionInfo, dict[str, _DeadlineArgInfo]]],
    ]
    trusted_params: Mapping[str, set[str]]
    forwarded_params: Mapping[str, set[str]]
    roots: set[str]
    forest: Forest
    collect_call_edges_from_forest_fn: Callable[..., dict[NodeId, set[NodeId]]]
    collect_call_resolution_obligations_from_forest_fn: Callable[
        ...,
        list[tuple[NodeId, NodeId, tuple[int, int, int, int] | None, str]],
    ]
    reachable_from_roots_fn: Callable[
        [Mapping[NodeId, set[NodeId]], set[NodeId]],
        set[NodeId],
    ]
    collect_recursive_nodes_fn: Callable[
        [Mapping[NodeId, set[NodeId]]],
        set[NodeId],
    ]


@dataclass
class _DeadlineObligationBuilder:
    by_qual: Mapping[str, FunctionInfo]
    facts_by_qual: Mapping[str, _DeadlineFunctionFacts]
    forest: Forest
    project_root: Path | None
    obligations: list[JSONObject] = field(default_factory=list)
    normalized_snapshot_path_cache: dict[Path, str] = field(default_factory=dict)
    suite_path_name_cache: dict[str, str] = field(default_factory=dict)

    def normalized_snapshot_path(self, path: Path) -> str:
        cached = self.normalized_snapshot_path_cache.get(path)
        if cached is None:
            cached = _normalize_snapshot_path(path, self.project_root)
            self.normalized_snapshot_path_cache[path] = cached
        return cached

    def _suite_path_name(self, path: str) -> str:
        cached = self.suite_path_name_cache.get(path)
        if cached is None:
            cached = Path(path).name
            self.suite_path_name_cache[path] = cached
        return cached

    def _fallback_span(
        self,
        function: str,
        param: str | None,
        span: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if span is not None:
            return span
        info = self.by_qual.get(function)
        if param and info is not None:
            candidate = info.param_spans.get(param)
            if candidate is not None:
                return candidate
        facts = self.facts_by_qual.get(function)
        if facts is not None and facts.span is not None:
            return facts.span
        return span

    def add_obligation(
        self,
        *,
        path: str,
        function: str,
        param: str | None,
        status: str,
        kind: str,
        detail: str,
        span: tuple[int, int, int, int] | None = None,
        caller: str | None = None,
        callee: str | None = None,
        suite_kind: str = "function",
    ) -> None:
        function_name = str(function)
        span = self._fallback_span(function_name, param, span)
        span = cast(
            tuple[int, int, int, int],
            require_not_none(
                span,
                reason="deadline obligation missing span",
                strict=True,
                kind=kind,
            ),
        )
        bundle = [param] if param else []
        span_line, span_col, span_end_line, span_end_col = span
        if param:
            deadline_id = (
                f"deadline:{path}:{function_name}:{kind}:{param}:"
                f"{span_line}:{span_col}:{span_end_line}:{span_end_col}"
            )
        else:
            deadline_id = (
                f"deadline:{path}:{function_name}:{kind}:"
                f"{span_line}:{span_col}:{span_end_line}:{span_end_col}"
            )
        entry: JSONObject = {
            "deadline_id": deadline_id,
            "site": {
                "path": path,
                "function": function_name,
                "bundle": bundle,
            },
            "status": status,
            "kind": kind,
            "detail": detail,
            "span": list(span),
        }
        if caller:
            entry["caller"] = caller
        if callee:
            entry["callee"] = callee
        self.obligations.append(entry)
        suite_path = self._suite_path_name(path)
        suite_id = self.forest.add_suite_site(
            suite_path,
            function_name,
            suite_kind,
            span=span,
        )
        suite_node = self.forest.nodes.get(suite_id)
        suite_meta = suite_node.meta if suite_node is not None else {}
        site_payload = cast(dict[str, object], entry["site"])
        suite_identity = suite_meta.get("suite_id")
        if isinstance(suite_identity, str) and suite_identity:
            site_payload["suite_id"] = suite_identity
        site_payload["suite_kind"] = suite_kind
        paramset_id = self.forest.add_paramset(bundle)
        evidence: dict[str, object] = {
            "deadline_id": deadline_id,
            "status": status,
            "kind": kind,
            "detail": detail,
        }
        if caller:
            evidence["caller"] = caller
        if callee:
            evidence["callee"] = callee
        self.forest.add_alt("DeadlineObligation", (suite_id, paramset_id), evidence=evidence)


def _append_origin_obligations(
    *,
    context: _DeadlineObligationContext,
    builder: _DeadlineObligationBuilder,
    emit_progress_fn: Callable[..., None],
) -> None:
    for qual, facts in context.facts_by_qual.items():
        check_deadline()
        emit_progress_fn("origin_obligations")
        if facts is None:
            continue
        if qual not in context.by_qual:
            continue
        if _is_test_path(facts.path):
            continue
        if qual in context.roots:
            continue
        for name, span in facts.local_info.origin_spans.items():
            check_deadline()
            if name not in facts.local_info.origin_vars:
                continue
            builder.add_obligation(
                path=builder.normalized_snapshot_path(facts.path),
                function=qual,
                param=name,
                status="VIOLATION",
                kind="origin_not_allowlisted",
                detail=f"local Deadline origin '{name}' outside allowlist",
                span=span,
                suite_kind="function",
            )
    emit_progress_fn("origin_obligations_done", force=True)


def _append_default_param_obligations(
    *,
    context: _DeadlineObligationContext,
    builder: _DeadlineObligationBuilder,
    emit_progress_fn: Callable[..., None],
) -> None:
    for qual, params in context.deadline_params.items():
        check_deadline()
        emit_progress_fn("default_param_obligations")
        info = context.by_qual.get(qual)
        if info is None or _is_test_path(info.path):
            continue
        for param in sort_once(
            params,
            source="src/gabion/analysis/dataflow_obligations.py:default_param_obligations",
        ):
            check_deadline()
            if param in info.defaults:
                span = info.param_spans.get(param)
                builder.add_obligation(
                    path=builder.normalized_snapshot_path(info.path),
                    function=qual,
                    param=param,
                    status="VIOLATION",
                    kind="default_param",
                    detail=f"deadline param '{param}' has default",
                    span=span,
                    suite_kind="function",
                )
    emit_progress_fn("default_param_obligations_done", force=True)


def _append_resolution_obligations(
    *,
    context: _DeadlineObligationContext,
    builder: _DeadlineObligationBuilder,
    emit_progress_fn: Callable[..., None],
) -> tuple[set[str], set[NodeId]]:
    edges = context.collect_call_edges_from_forest_fn(context.forest, by_name=context.by_name)
    emit_progress_fn("call_edges_ready")
    raw_resolution_obligations = context.collect_call_resolution_obligations_from_forest_fn(
        context.forest
    )
    resolution_obligation_kind_by_site: dict[
        tuple[NodeId, NodeId, tuple[int, int, int, int] | None, str],
        str,
    ] = {}
    for (
        caller_id,
        suite_id,
        span,
        callee_key,
        obligation_kind,
    ) in _collect_call_resolution_obligation_details_from_forest(context.forest):
        check_deadline()
        emit_progress_fn("resolution_obligation_details")
        resolution_obligation_kind_by_site[(caller_id, suite_id, span, callee_key)] = (
            obligation_kind
        )
    resolution_obligations = [
        (
            caller_id,
            suite_id,
            span,
            callee_key,
            resolution_obligation_kind_by_site.get(
                (caller_id, suite_id, span, callee_key),
                "unresolved_internal_callee",
            ),
        )
        for caller_id, suite_id, span, callee_key in raw_resolution_obligations
    ]
    recursive_nodes = context.collect_recursive_nodes_fn(edges)

    def _deadline_exempt(qual: str) -> bool:
        return any(qual.startswith(prefix) for prefix in _DEADLINE_EXEMPT_PREFIXES)

    root_site_ids: set[NodeId] = set()
    for qual in context.roots:
        check_deadline()
        emit_progress_fn("root_site_resolution")
        info = context.by_qual.get(qual)
        if info is None:
            continue
        root_site_ids.add(_function_suite_id(_function_suite_key(info.path.name, qual)))

    reachable_from_roots = context.reachable_from_roots_fn(edges, root_site_ids)
    emit_progress_fn("reachability_ready", force=True)

    resolved_call_suites: set[NodeId] = set()
    for alt in context.forest.alts:
        check_deadline()
        emit_progress_fn("resolved_call_sites")
        if alt.kind != "CallCandidate" or len(alt.inputs) < 2:
            continue
        suite_id = alt.inputs[0]
        suite_node = context.forest.nodes.get(suite_id)
        if suite_node is None or suite_node.kind != "SuiteSite":
            continue
        if str(suite_node.meta.get("suite_kind", "") or "") != "call":
            continue
        resolved_call_suites.add(suite_id)
    emit_progress_fn("resolved_call_sites_done", force=True)

    resolution_obligation_kind_map = {
        "unresolved_dynamic_callee": "call_dynamic_resolution_required",
        "unresolved_internal_callee": "call_resolution_required",
    }

    for caller_id, suite_id, span, callee_key, obligation_kind in sort_once(
        resolution_obligations,
        key=lambda entry: (
            entry[0].sort_key(),
            entry[1].sort_key(),
            entry[2] or (-1, -1, -1, -1),
            entry[3],
            entry[4],
        ),
        source="src/gabion/analysis/dataflow_obligations.py:resolution_obligations",
    ):
        check_deadline()
        emit_progress_fn("resolution_obligations")
        if suite_id in resolved_call_suites:
            continue
        if caller_id not in reachable_from_roots:
            continue
        if caller_id.kind != "SuiteSite" or len(caller_id.key) < 2:
            continue
        caller_qual = str(caller_id.key[1] or "")
        if not caller_qual:
            continue
        if _deadline_exempt(caller_qual):
            continue
        caller_info = context.by_qual.get(caller_qual)
        if caller_info is None or _is_test_path(caller_info.path):
            continue
        output_kind = resolution_obligation_kind_map.get(
            obligation_kind,
            "call_resolution_required",
        )
        if obligation_kind == "unresolved_dynamic_callee":
            detail = (
                f"call '{callee_key}' appears to use dynamic dispatch; "
                "add explicit routing evidence"
            )
        else:
            detail = f"call '{callee_key}' requires resolution"
        builder.add_obligation(
            path=builder.normalized_snapshot_path(caller_info.path),
            function=caller_qual,
            param=None,
            status="OBLIGATION",
            kind=output_kind,
            detail=detail,
            span=span,
            caller=caller_qual,
            callee=callee_key,
            suite_kind="call",
        )
    emit_progress_fn("resolution_obligations_done", force=True)

    recursive_required: set[str] = set()
    for function_id in recursive_nodes:
        check_deadline()
        emit_progress_fn("recursive_required")
        if function_id.kind != "SuiteSite" or len(function_id.key) < 2:
            continue
        qual = str(function_id.key[1] or "")
        if not qual or _deadline_exempt(qual):
            continue
        recursive_required.add(qual)
    emit_progress_fn("recursive_required_done", force=True)
    return recursive_required, reachable_from_roots


def _append_recursive_obligations(
    *,
    context: _DeadlineObligationContext,
    builder: _DeadlineObligationBuilder,
    recursive_required: set[str],
    reachable_from_roots: set[NodeId],
    emit_progress_fn: Callable[..., None],
) -> None:
    def _deadline_carrier_status(
        *,
        qual: str,
        info: FunctionInfo,
        has_carrier_signal: bool,
    ) -> str:
        if not has_carrier_signal:
            return "OBLIGATION"
        function_id = _function_suite_id(_function_suite_key(info.path.name, qual))
        return "VIOLATION" if function_id in reachable_from_roots else "OBLIGATION"

    for qual in sort_once(
        recursive_required,
        source="src/gabion/analysis/dataflow_obligations.py:recursive_required",
    ):
        check_deadline()
        emit_progress_fn("recursive_obligations")
        facts = context.facts_by_qual.get(qual)
        info = context.by_qual.get(qual)
        if facts is None or info is None or _is_test_path(info.path):
            continue
        carriers = context.deadline_params.get(qual, set())
        carrier_status = _deadline_carrier_status(
            qual=qual,
            info=info,
            has_carrier_signal=bool(carriers),
        )
        if facts.loop_sites:
            for loop_fact in facts.loop_sites:
                check_deadline()
                if not carriers:
                    if loop_fact.ambient_check:
                        continue
                    builder.add_obligation(
                        path=builder.normalized_snapshot_path(info.path),
                        function=qual,
                        param=None,
                        status=carrier_status,
                        kind="missing_carrier",
                        detail="recursion loop requires Deadline carrier",
                        span=loop_fact.span,
                        suite_kind="loop",
                    )
                    continue
                checked = loop_fact.check_params & carriers
                if loop_fact.ambient_check:
                    checked = set(carriers)
                forwarded = _deadline_loop_forwarded_params(
                    qual=qual,
                    loop_fact=loop_fact,
                    deadline_params=context.deadline_params,
                    call_infos=context.call_infos,
                ) & carriers
                if checked or forwarded:
                    continue
                builder.add_obligation(
                    path=builder.normalized_snapshot_path(info.path),
                    function=qual,
                    param=None,
                    status=carrier_status,
                    kind="unchecked_deadline",
                    detail=(
                        "deadline carrier not checked or forwarded "
                        f"in recursion loop depth {loop_fact.depth}"
                    ),
                    span=loop_fact.span,
                    suite_kind="loop",
                )
            continue
        if not carriers:
            if facts.ambient_check:
                continue
            builder.add_obligation(
                path=builder.normalized_snapshot_path(info.path),
                function=qual,
                param=None,
                status=carrier_status,
                kind="missing_carrier",
                detail="recursion requires Deadline carrier",
                span=facts.span,
                suite_kind="function",
            )
            continue
        checked = facts.check_params & carriers
        if facts.ambient_check:
            checked = set(carriers)
        forwarded = context.forwarded_params.get(qual, set()) & carriers
        if checked or forwarded:
            continue
        builder.add_obligation(
            path=builder.normalized_snapshot_path(info.path),
            function=qual,
            param=None,
            status=carrier_status,
            kind="unchecked_deadline",
            detail="deadline carrier not checked or forwarded (recursion)",
            span=facts.span,
            suite_kind="function",
        )
    emit_progress_fn("recursive_obligations_done", force=True)


def _append_loop_obligations(
    *,
    context: _DeadlineObligationContext,
    builder: _DeadlineObligationBuilder,
    recursive_required: set[str],
    reachable_from_roots: set[NodeId],
    emit_progress_fn: Callable[..., None],
) -> None:
    def _deadline_exempt(qual: str) -> bool:
        return any(qual.startswith(prefix) for prefix in _DEADLINE_EXEMPT_PREFIXES)

    def _deadline_carrier_status(
        *,
        qual: str,
        info: FunctionInfo,
        has_carrier_signal: bool,
    ) -> str:
        if not has_carrier_signal:
            return "OBLIGATION"
        function_id = _function_suite_id(_function_suite_key(info.path.name, qual))
        return "VIOLATION" if function_id in reachable_from_roots else "OBLIGATION"

    for qual, facts in context.facts_by_qual.items():
        check_deadline()
        emit_progress_fn("loop_obligations")
        if _deadline_exempt(qual):
            continue
        if qual in recursive_required:
            continue
        if facts is None:
            continue
        info = context.by_qual.get(qual)
        if info is None or _is_test_path(info.path):
            continue
        if not facts.loop_sites:
            continue
        carriers = context.deadline_params.get(qual, set())
        carrier_status = _deadline_carrier_status(
            qual=qual,
            info=info,
            has_carrier_signal=bool(carriers),
        )
        for loop_fact in facts.loop_sites:
            check_deadline()
            if not carriers:
                if loop_fact.ambient_check:
                    continue
                builder.add_obligation(
                    path=builder.normalized_snapshot_path(info.path),
                    function=qual,
                    param=None,
                    status=carrier_status,
                    kind="missing_carrier",
                    detail="loop requires Deadline carrier",
                    span=loop_fact.span,
                    suite_kind="loop",
                )
                continue
            checked = loop_fact.check_params & carriers
            if loop_fact.ambient_check:
                checked = set(carriers)
            forwarded = _deadline_loop_forwarded_params(
                qual=qual,
                loop_fact=loop_fact,
                deadline_params=context.deadline_params,
                call_infos=context.call_infos,
            ) & carriers
            if not checked and not forwarded:
                builder.add_obligation(
                    path=builder.normalized_snapshot_path(info.path),
                    function=qual,
                    param=None,
                    status=carrier_status,
                    kind="unchecked_deadline",
                    detail="deadline carrier not checked or forwarded in loop",
                    span=loop_fact.span,
                    suite_kind="loop",
                )
    emit_progress_fn("loop_obligations_done", force=True)


def _append_call_arg_obligations(
    *,
    context: _DeadlineObligationContext,
    builder: _DeadlineObligationBuilder,
    emit_progress_fn: Callable[..., None],
) -> None:
    for caller_qual, entries in context.call_infos.items():
        check_deadline()
        emit_progress_fn("call_arg_obligations")
        caller_info = context.by_qual.get(caller_qual)
        if caller_info is None or _is_test_path(caller_info.path):
            continue
        for call, callee, arg_info in entries:
            check_deadline()
            callee_deadlines = context.deadline_params.get(callee.qual, set())
            if not callee_deadlines:
                continue
            span = call.span
            for callee_param in sort_once(
                callee_deadlines,
                source="src/gabion/analysis/dataflow_obligations.py:call_arg_obligations",
            ):
                check_deadline()
                info = arg_info.get(callee_param)
                if info is None:
                    missing_unknown = bool(
                        call.star_pos
                        or call.star_kw
                        or call.non_const_pos
                        or call.non_const_kw
                    )
                    status = "OBLIGATION" if missing_unknown else "VIOLATION"
                    kind = "missing_arg_unknown" if missing_unknown else "missing_arg"
                    builder.add_obligation(
                        path=builder.normalized_snapshot_path(caller_info.path),
                        function=caller_qual,
                        param=callee_param,
                        status=status,
                        kind=kind,
                        detail=f"missing deadline arg for {callee.qual}.{callee_param}",
                        span=span,
                        caller=caller_qual,
                        callee=callee.qual,
                        suite_kind="call",
                    )
                    continue
                if info.kind == "none":
                    builder.add_obligation(
                        path=builder.normalized_snapshot_path(caller_info.path),
                        function=caller_qual,
                        param=callee_param,
                        status="VIOLATION",
                        kind="none_arg",
                        detail=f"None passed to {callee.qual}.{callee_param}",
                        span=span,
                        caller=caller_qual,
                        callee=callee.qual,
                        suite_kind="call",
                    )
                    continue
                if info.kind == "const":
                    builder.add_obligation(
                        path=builder.normalized_snapshot_path(caller_info.path),
                        function=caller_qual,
                        param=callee_param,
                        status="VIOLATION",
                        kind="const_arg",
                        detail=f"constant {info.const} passed to {callee.qual}.{callee_param}",
                        span=span,
                        caller=caller_qual,
                        callee=callee.qual,
                        suite_kind="call",
                    )
                    continue
                if info.kind == "origin":
                    if caller_qual not in context.roots:
                        builder.add_obligation(
                            path=builder.normalized_snapshot_path(caller_info.path),
                            function=caller_qual,
                            param=callee_param,
                            status="VIOLATION",
                            kind="origin_not_allowlisted",
                            detail=f"origin deadline passed outside allowlist to {callee.qual}.{callee_param}",
                            span=span,
                            caller=caller_qual,
                            callee=callee.qual,
                            suite_kind="call",
                        )
                    continue
                if info.kind == "param":
                    if info.param in context.trusted_params.get(caller_qual, set()):
                        continue
                    builder.add_obligation(
                        path=builder.normalized_snapshot_path(caller_info.path),
                        function=caller_qual,
                        param=callee_param,
                        status="OBLIGATION",
                        kind="untrusted_param",
                        detail=f"deadline param '{info.param}' not proven from allowlist",
                        span=span,
                        caller=caller_qual,
                        callee=callee.qual,
                        suite_kind="call",
                    )
                    continue
                if info.kind == "unknown":
                    builder.add_obligation(
                        path=builder.normalized_snapshot_path(caller_info.path),
                        function=caller_qual,
                        param=callee_param,
                        status="OBLIGATION",
                        kind="unknown_arg",
                        detail=f"deadline arg not proven for {callee.qual}.{callee_param}",
                        span=span,
                        caller=caller_qual,
                        callee=callee.qual,
                        suite_kind="call",
                    )
    emit_progress_fn("call_arg_obligations_done", force=True)


@dataclass(frozen=True)
class _DeadlineCollectionFns:
    materialize_call_candidates_fn: Callable[..., None]
    collect_call_nodes_by_path_fn: Callable[
        ...,
        dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]],
    ]
    collect_deadline_function_facts_fn: Callable[
        ...,
        dict[str, "_DeadlineFunctionFacts"],
    ]
    collect_call_edges_from_forest_fn: Callable[..., dict[NodeId, set[NodeId]]]
    collect_call_resolution_obligations_from_forest_fn: Callable[
        ...,
        list[tuple[NodeId, NodeId, tuple[int, int, int, int] | None, str]],
    ]
    reachable_from_roots_fn: Callable[
        [Mapping[NodeId, set[NodeId]], set[NodeId]],
        set[NodeId],
    ]
    collect_recursive_nodes_fn: Callable[
        [Mapping[NodeId, set[NodeId]]],
        set[NodeId],
    ]
    resolve_callee_outcome_fn: Callable[..., _CalleeResolutionOutcome]


def _resolve_deadline_collection_fns(
    *,
    materialize_call_candidates_fn: Callable[..., None] | None,
    collect_call_nodes_by_path_fn: Callable[
        ...,
        dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]],
    ]
    | None,
    collect_deadline_function_facts_fn: Callable[
        ...,
        dict[str, "_DeadlineFunctionFacts"],
    ]
    | None,
    collect_call_edges_from_forest_fn: Callable[..., dict[NodeId, set[NodeId]]] | None,
    collect_call_resolution_obligations_from_forest_fn: Callable[
        ...,
        list[tuple[NodeId, NodeId, tuple[int, int, int, int] | None, str]],
    ]
    | None,
    reachable_from_roots_fn: Callable[
        [Mapping[NodeId, set[NodeId]], set[NodeId]],
        set[NodeId],
    ]
    | None,
    collect_recursive_nodes_fn: Callable[
        [Mapping[NodeId, set[NodeId]]],
        set[NodeId],
    ]
    | None,
    resolve_callee_outcome_fn: Callable[..., _CalleeResolutionOutcome] | None,
) -> _DeadlineCollectionFns:
    if materialize_call_candidates_fn is None:
        materialize_call_candidates_fn = _materialize_call_candidates
    if collect_call_nodes_by_path_fn is None:
        collect_call_nodes_by_path_fn = _collect_call_nodes_by_path
    if collect_deadline_function_facts_fn is None:
        collect_deadline_function_facts_fn = _collect_deadline_function_facts
    if collect_call_edges_from_forest_fn is None:
        collect_call_edges_from_forest_fn = _collect_call_edges_from_forest
    if collect_call_resolution_obligations_from_forest_fn is None:
        collect_call_resolution_obligations_from_forest_fn = (
            _collect_call_resolution_obligations_from_forest
        )
    if reachable_from_roots_fn is None:
        reachable_from_roots_fn = _reachable_from_roots
    if collect_recursive_nodes_fn is None:
        collect_recursive_nodes_fn = _collect_recursive_nodes
    if resolve_callee_outcome_fn is None:
        resolve_callee_outcome_fn = _resolve_callee_outcome
    return _DeadlineCollectionFns(
        materialize_call_candidates_fn=materialize_call_candidates_fn,
        collect_call_nodes_by_path_fn=collect_call_nodes_by_path_fn,
        collect_deadline_function_facts_fn=collect_deadline_function_facts_fn,
        collect_call_edges_from_forest_fn=collect_call_edges_from_forest_fn,
        collect_call_resolution_obligations_from_forest_fn=collect_call_resolution_obligations_from_forest_fn,
        reachable_from_roots_fn=reachable_from_roots_fn,
        collect_recursive_nodes_fn=collect_recursive_nodes_fn,
        resolve_callee_outcome_fn=resolve_callee_outcome_fn,
    )


@dataclass(frozen=True)
class _DeadlineCollectionContext:
    by_name: Mapping[str, list[FunctionInfo]]
    by_qual: Mapping[str, FunctionInfo]
    symbol_table: object
    class_index: object
    project_root: Path | None
    config: AuditConfig
    resolve_callee_outcome_fn: Callable[..., _CalleeResolutionOutcome]


def _collect_deadline_params(
    *,
    context: _DeadlineCollectionContext,
    extra_deadline_params: dict[str, set[str]] | None,
    emit_progress_fn: Callable[..., None],
) -> defaultdict[str, set[str]]:
    deadline_params: defaultdict[str, set[str]] = defaultdict(set)
    for info in context.by_qual.values():
        check_deadline()
        emit_progress_fn("deadline_params")
        if _is_test_path(info.path):
            continue
        for name in info.params:
            check_deadline()
            if _is_deadline_param(name, info.annots.get(name)):
                deadline_params[info.qual].add(name)
    if extra_deadline_params:
        for qual, params in extra_deadline_params.items():
            check_deadline()
            if params:
                deadline_params[qual].update(params)
    for helper in _DEADLINE_HELPER_QUALS:
        check_deadline()
        deadline_params.pop(helper, None)
    emit_progress_fn("deadline_params_ready", force=True)
    return deadline_params


def _propagate_deadline_params(
    *,
    context: _DeadlineCollectionContext,
    deadline_params: defaultdict[str, set[str]],
    emit_progress_fn: Callable[..., None],
) -> None:
    changed = True
    while changed:
        check_deadline()
        changed = False
        for infos in context.by_name.values():
            check_deadline()
            emit_progress_fn("deadline_param_propagation")
            for info in infos:
                check_deadline()
                if _is_test_path(info.path):
                    continue
                for call in info.calls:
                    check_deadline()
                    resolution = context.resolve_callee_outcome_fn(
                        call.callee,
                        info,
                        context.by_name,
                        context.by_qual,
                        symbol_table=context.symbol_table,
                        project_root=context.project_root,
                        class_index=context.class_index,
                        call=call,
                    )
                    if not resolution.candidates:
                        continue
                    for callee in resolution.candidates:
                        check_deadline()
                        mapping = _caller_param_bindings_for_call(
                            call,
                            callee,
                            strictness=context.config.strictness,
                        )
                        for callee_param in deadline_params.get(callee.qual, set()):
                            check_deadline()
                            for caller_param in mapping.get(callee_param, set()):
                                check_deadline()
                                if caller_param not in deadline_params[info.qual]:
                                    deadline_params[info.qual].add(caller_param)
                                    changed = True
    emit_progress_fn("deadline_param_propagation_done", force=True)


def _collect_call_infos(
    *,
    context: _DeadlineCollectionContext,
    call_nodes_by_path: Mapping[Path, Mapping[tuple[int, int, int, int], list[ast.Call]]],
    facts_by_qual: Mapping[str, _DeadlineFunctionFacts],
    emit_progress_fn: Callable[..., None],
) -> defaultdict[str, list[tuple[CallArgs, FunctionInfo, dict[str, _DeadlineArgInfo]]]]:
    call_infos: defaultdict[str, list[tuple[CallArgs, FunctionInfo, dict[str, _DeadlineArgInfo]]]] = (
        defaultdict(list)
    )
    for infos in context.by_name.values():
        check_deadline()
        emit_progress_fn("call_info_collection")
        for info in infos:
            check_deadline()
            if _is_test_path(info.path):
                continue
            facts = facts_by_qual.get(info.qual)
            alias_to_param = facts.local_info.alias_to_param if facts else {p: p for p in info.params}
            origin_vars = facts.local_info.origin_vars if facts else set()
            span_index = call_nodes_by_path.get(info.path, {})
            for call in info.calls:
                check_deadline()
                resolution = context.resolve_callee_outcome_fn(
                    call.callee,
                    info,
                    context.by_name,
                    context.by_qual,
                    symbol_table=context.symbol_table,
                    project_root=context.project_root,
                    class_index=context.class_index,
                    call=call,
                )
                if not resolution.candidates:
                    continue
                call_node = None
                if call.span is not None:
                    nodes = span_index.get(call.span)
                    if nodes:
                        call_node = nodes[0]
                for callee in resolution.candidates:
                    check_deadline()
                    arg_info = _deadline_arg_info_map(
                        call,
                        callee,
                        call_node=call_node,
                        alias_to_param=alias_to_param,
                        origin_vars=origin_vars,
                        strictness=context.config.strictness,
                    )
                    call_infos[info.qual].append((call, callee, arg_info))
    emit_progress_fn("call_info_collection_done", force=True)
    return call_infos


def _collect_trusted_params(
    *,
    roots: set[str],
    deadline_params: Mapping[str, set[str]],
    call_infos: Mapping[str, list[tuple[CallArgs, FunctionInfo, dict[str, _DeadlineArgInfo]]]],
    emit_progress_fn: Callable[..., None],
) -> defaultdict[str, set[str]]:
    trusted_params: defaultdict[str, set[str]] = defaultdict(set)
    for qual, params in deadline_params.items():
        check_deadline()
        emit_progress_fn("trusted_seed")
        if qual in roots:
            trusted_params[qual].update(params)

    changed = True
    while changed:
        check_deadline()
        changed = False
        for caller_qual, entries in call_infos.items():
            check_deadline()
            emit_progress_fn("trusted_propagation")
            for _, callee, arg_info in entries:
                check_deadline()
                for callee_param in deadline_params.get(callee.qual, set()):
                    check_deadline()
                    info = arg_info.get(callee_param)
                    if info is None:
                        continue
                    if (
                        info.kind == "param"
                        and info.param in trusted_params.get(caller_qual, set())
                        and callee_param not in trusted_params[callee.qual]
                    ):
                        trusted_params[callee.qual].add(callee_param)
                        changed = True
                    if (
                        info.kind == "origin"
                        and caller_qual in roots
                        and callee_param not in trusted_params[callee.qual]
                    ):
                        trusted_params[callee.qual].add(callee_param)
                        changed = True
    emit_progress_fn("trusted_propagation_done", force=True)
    return trusted_params


def _collect_forwarded_params(
    *,
    deadline_params: Mapping[str, set[str]],
    call_infos: Mapping[str, list[tuple[CallArgs, FunctionInfo, dict[str, _DeadlineArgInfo]]]],
    emit_progress_fn: Callable[..., None],
) -> defaultdict[str, set[str]]:
    forwarded_params: defaultdict[str, set[str]] = defaultdict(set)
    for caller_qual, entries in call_infos.items():
        check_deadline()
        emit_progress_fn("forwarded_params")
        caller_params = deadline_params.get(caller_qual, set())
        if not caller_params:
            continue
        for _, callee, arg_info in entries:
            check_deadline()
            for callee_param in deadline_params.get(callee.qual, set()):
                check_deadline()
                info = arg_info.get(callee_param)
                if info is None:
                    continue
                if info.kind == "param" and info.param in caller_params:
                    forwarded_params[caller_qual].add(info.param)
    emit_progress_fn("forwarded_params_done", force=True)
    return forwarded_params


def collect_deadline_obligations(
    paths: list[Path],
    *,
    project_root: Path | None,
    config: AuditConfig,
    forest: Forest,
    extra_facts_by_qual: dict[str, "_DeadlineFunctionFacts"] | None = None,
    extra_call_infos: dict[str, list[tuple[CallArgs, FunctionInfo, dict[str, "_DeadlineArgInfo"]]]] | None = None,
    extra_deadline_params: dict[str, set[str]] | None = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index: AnalysisIndex | None = None,
    materialize_call_candidates_fn: Callable[..., None] | None = None,
    collect_call_nodes_by_path_fn: Callable[
        ..., dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]]
    ] | None = None,
    collect_deadline_function_facts_fn: Callable[
        ..., dict[str, "_DeadlineFunctionFacts"]
    ] | None = None,
    collect_call_edges_from_forest_fn: Callable[..., dict[NodeId, set[NodeId]]] | None = None,
    collect_call_resolution_obligations_from_forest_fn: Callable[
        ..., list[tuple[NodeId, NodeId, tuple[int, int, int, int] | None, str]]
    ] | None = None,
    reachable_from_roots_fn: Callable[
        [Mapping[NodeId, set[NodeId]], set[NodeId]], set[NodeId]
    ] | None = None,
    collect_recursive_nodes_fn: Callable[
        [Mapping[NodeId, set[NodeId]]], set[NodeId]
    ] | None = None,
    resolve_callee_outcome_fn: Callable[..., _CalleeResolutionOutcome] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[JSONObject]:
    _bind_audit_symbols()
    check_deadline()
    collection_fns = _resolve_deadline_collection_fns(
        materialize_call_candidates_fn=materialize_call_candidates_fn,
        collect_call_nodes_by_path_fn=collect_call_nodes_by_path_fn,
        collect_deadline_function_facts_fn=collect_deadline_function_facts_fn,
        collect_call_edges_from_forest_fn=collect_call_edges_from_forest_fn,
        collect_call_resolution_obligations_from_forest_fn=collect_call_resolution_obligations_from_forest_fn,
        reachable_from_roots_fn=reachable_from_roots_fn,
        collect_recursive_nodes_fn=collect_recursive_nodes_fn,
        resolve_callee_outcome_fn=resolve_callee_outcome_fn,
    )
    if not config.deadline_roots:
        return []
    roots = set(config.deadline_roots)
    last_progress_emit_monotonic: float | None = None
    progress_emit_counter = 0

    def _emit_progress(stage: str, *, force: bool = False) -> None:
        nonlocal progress_emit_counter
        nonlocal last_progress_emit_monotonic
        if on_progress is None:
            return
        now = time.monotonic()
        if (
            not force
            and last_progress_emit_monotonic is not None
            and now - last_progress_emit_monotonic < _PROGRESS_EMIT_MIN_INTERVAL_SECONDS
        ):
            return
        progress_emit_counter += 1
        last_progress_emit_monotonic = now
        on_progress(f"{stage}:{progress_emit_counter}")

    index = analysis_index
    if index is None:
        index = _build_analysis_index(
            paths,
            project_root=project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    _emit_progress("index_ready", force=True)
    by_name = index.by_name
    by_qual = index.by_qual
    symbol_table = index.symbol_table
    class_index = index.class_index
    collection_fns.materialize_call_candidates_fn(
        forest=forest,
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
    )
    _emit_progress("call_candidates_materialized")
    call_nodes_by_path = collection_fns.collect_call_nodes_by_path_fn(
        paths,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=index,
    )
    _emit_progress("call_nodes_collected")
    facts_by_qual = collection_fns.collect_deadline_function_facts_fn(
        paths,
        project_root=project_root,
        ignore_params=config.ignore_params,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=index,
    )
    _emit_progress("function_facts_collected")
    if extra_facts_by_qual:
        facts_by_qual = dict(facts_by_qual)
        facts_by_qual.update(extra_facts_by_qual)

    collection_context = _DeadlineCollectionContext(
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        class_index=class_index,
        project_root=project_root,
        config=config,
        resolve_callee_outcome_fn=collection_fns.resolve_callee_outcome_fn,
    )
    deadline_params = _collect_deadline_params(
        context=collection_context,
        extra_deadline_params=extra_deadline_params,
        emit_progress_fn=_emit_progress,
    )
    _propagate_deadline_params(
        context=collection_context,
        deadline_params=deadline_params,
        emit_progress_fn=_emit_progress,
    )
    call_infos = _collect_call_infos(
        context=collection_context,
        call_nodes_by_path=call_nodes_by_path,
        facts_by_qual=facts_by_qual,
        emit_progress_fn=_emit_progress,
    )
    if extra_call_infos:
        for qual, entries in extra_call_infos.items():
            check_deadline()
            call_infos[qual].extend(entries)
    _emit_progress("call_info_overrides_applied")

    trusted_params = _collect_trusted_params(
        roots=roots,
        deadline_params=deadline_params,
        call_infos=call_infos,
        emit_progress_fn=_emit_progress,
    )

    forwarded_params = _collect_forwarded_params(
        deadline_params=deadline_params,
        call_infos=call_infos,
        emit_progress_fn=_emit_progress,
    )

    context = _DeadlineObligationContext(
        by_name=by_name,
        by_qual=by_qual,
        facts_by_qual=facts_by_qual,
        deadline_params=deadline_params,
        call_infos=call_infos,
        trusted_params=trusted_params,
        forwarded_params=forwarded_params,
        roots=roots,
        forest=forest,
        collect_call_edges_from_forest_fn=collection_fns.collect_call_edges_from_forest_fn,
        collect_call_resolution_obligations_from_forest_fn=collection_fns.collect_call_resolution_obligations_from_forest_fn,
        reachable_from_roots_fn=collection_fns.reachable_from_roots_fn,
        collect_recursive_nodes_fn=collection_fns.collect_recursive_nodes_fn,
    )
    builder = _DeadlineObligationBuilder(
        by_qual=by_qual,
        facts_by_qual=facts_by_qual,
        forest=forest,
        project_root=project_root,
    )
    _append_origin_obligations(
        context=context,
        builder=builder,
        emit_progress_fn=_emit_progress,
    )
    _append_default_param_obligations(
        context=context,
        builder=builder,
        emit_progress_fn=_emit_progress,
    )
    recursive_required, reachable_from_roots = _append_resolution_obligations(
        context=context,
        builder=builder,
        emit_progress_fn=_emit_progress,
    )
    _append_recursive_obligations(
        context=context,
        builder=builder,
        recursive_required=recursive_required,
        reachable_from_roots=reachable_from_roots,
        emit_progress_fn=_emit_progress,
    )
    _append_loop_obligations(
        context=context,
        builder=builder,
        recursive_required=recursive_required,
        reachable_from_roots=reachable_from_roots,
        emit_progress_fn=_emit_progress,
    )
    _append_call_arg_obligations(
        context=context,
        builder=builder,
        emit_progress_fn=_emit_progress,
    )
    return sort_once(
        builder.obligations,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            str(entry.get("kind", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("deadline_id", "")),
        ),
        source="src/gabion/analysis/dataflow_obligations.py:collect_deadline_obligations.return",
    )
