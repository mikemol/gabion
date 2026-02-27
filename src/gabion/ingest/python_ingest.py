# gabion:decision_protocol_module
from __future__ import annotations

import ast
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


def _default_deadline() -> None:
    return None


@dataclass(frozen=True)
class ParseFailureWitness:
    path: Path
    stage: str
    error: str


@dataclass(frozen=True)
class PythonFunctionIngestCarrier:
    key: str
    name: str
    params: tuple[str, ...]
    annotations: Mapping[str, str | None]
    callsites: Sequence[object]
    param_spans: Mapping[str, tuple[int, int, int, int]]
    decision_evidence: Mapping[str, set[str]]


@dataclass(frozen=True)
class PythonFileIngestCarrier:
    path: Path
    functions: tuple[PythonFunctionIngestCarrier, ...]
    function_use: Mapping[str, Mapping[str, object]]
    function_calls: Mapping[str, list[object]]
    function_param_orders: Mapping[str, list[str]]
    function_param_spans: Mapping[str, Mapping[str, tuple[int, int, int, int]]]
    function_names: Mapping[str, str]
    function_lexical_scopes: Mapping[str, tuple[str, ...]]
    function_class_names: Mapping[str, str | None]
    opaque_callees: set[str]
    parse_failure_witnesses: tuple[ParseFailureWitness, ...] = ()


def iter_python_paths(
    paths: Iterable[str],
    *,
    config,
    check_deadline: Callable[[], None] = _default_deadline,
    sort_once: Callable[..., list[object]],
) -> list[Path]:
    """Expand input paths to python files, pruning ignored directories early."""
    check_deadline()
    out: list[Path] = []
    for p in paths:
        check_deadline()
        path = Path(p)
        if path.is_dir():
            for root, dirnames, filenames in os.walk(path, topdown=True):
                check_deadline()
                if config.exclude_dirs:
                    dirnames[:] = [d for d in dirnames if d not in config.exclude_dirs]
                dirnames[:] = sort_once(
                    dirnames,
                    source="iter_python_paths.dirnames",
                    key=lambda name: name,
                )
                for filename in sort_once(filenames, source="iter_python_paths.filenames"):
                    check_deadline()
                    if not str(filename).endswith(".py"):
                        continue
                    candidate = Path(root) / str(filename)
                    if config.is_ignored_path(candidate):
                        continue
                    out.append(candidate)
        else:
            if config.is_ignored_path(path):
                continue
            out.append(path)
    return sort_once(out, source="iter_python_paths.out")


def ingest_python_file(
    path: Path,
    *,
    config,
    recursive: bool,
    parse_module: Callable[[Path], ast.Module],
    collect_functions: Callable[[ast.AST], list[object]],
    collect_return_aliases: Callable[..., object],
    load_resume_state: Callable[..., tuple[dict[str, object], ...]],
    serialize_resume_state: Callable[..., dict[str, object]],
    profiling_payload: Callable[..., dict[str, object]],
    analyze_function: Callable[..., tuple[object, list[object]]],
    enclosing_class: Callable[..., str | None],
    enclosing_scopes: Callable[..., list[str]],
    enclosing_function_scopes: Callable[..., list[str]],
    function_key: Callable[[list[str], str], str],
    decorators_transparent: Callable[..., bool],
    param_names: Callable[..., list[str]],
    param_spans: Callable[..., dict[str, tuple[int, int, int, int]]],
    collect_local_class_bases: Callable[..., dict[str, set[str]]],
    resolve_local_method_in_hierarchy: Callable[..., str | None],
    is_test_path: Callable[[Path], bool],
    check_deadline: Callable[[], None],
    parent_annotator_factory: Callable[[], object],
    progress_emit_interval: int,
    progress_min_interval_seconds: float,
    on_progress: Callable[[dict[str, object]], None] | None = None,
    on_profile: Callable[[dict[str, object]], None] | None = None,
    resume_state: dict[str, object] | None = None,
) -> PythonFileIngestCarrier:
    check_deadline()
    profile_stage_ns: dict[str, int] = {
        "file_scan.read_parse": 0,
        "file_scan.parent_annotation": 0,
        "file_scan.collect_functions": 0,
        "file_scan.function_scan": 0,
        "file_scan.resolve_local_calls": 0,
        "file_scan.resolve_local_methods": 0,
    }
    profile_counters: Counter[str] = Counter()
    parse_started_ns = time.monotonic_ns()
    tree = parse_module(path)
    profile_stage_ns["file_scan.read_parse"] += time.monotonic_ns() - parse_started_ns

    parent_started_ns = time.monotonic_ns()
    parent = parent_annotator_factory()
    parent.visit(tree)
    profile_stage_ns["file_scan.parent_annotation"] += time.monotonic_ns() - parent_started_ns
    parents = parent.parents
    is_test = is_test_path(path)

    collect_started_ns = time.monotonic_ns()
    funcs = collect_functions(tree)
    profile_stage_ns["file_scan.collect_functions"] += time.monotonic_ns() - collect_started_ns
    profile_counters["file_scan.functions_total"] = len(funcs)
    fn_keys_in_file: set[str] = set()
    for function_node in funcs:
        check_deadline()
        scopes = enclosing_scopes(function_node, parents)
        fn_keys_in_file.add(function_key(scopes, function_node.name))
    return_aliases = collect_return_aliases(funcs, parents, ignore_params=config.ignore_params)

    (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        fn_names,
        fn_lexical_scopes,
        fn_class_names,
        opaque_callees,
    ) = load_resume_state(payload=resume_state, valid_fn_keys=fn_keys_in_file)

    scanned_since_emit = 0
    last_scan_progress_emit_monotonic: float | None = None

    # gabion:boundary_normalization
    def _emit_scan_progress(*, force: bool = False) -> bool:
        nonlocal last_scan_progress_emit_monotonic
        if on_progress is None:
            return False
        now = time.monotonic()
        if (
            not force
            and last_scan_progress_emit_monotonic is not None
            and now - last_scan_progress_emit_monotonic < progress_min_interval_seconds
        ):
            return False
        progress_payload = serialize_resume_state(
            fn_use=fn_use,
            fn_calls=fn_calls,
            fn_param_orders=fn_param_orders,
            fn_param_spans=fn_param_spans,
            fn_names=fn_names,
            fn_lexical_scopes=fn_lexical_scopes,
            fn_class_names=fn_class_names,
            opaque_callees=opaque_callees,
        )
        progress_payload["profiling_v1"] = profiling_payload(
            stage_ns=profile_stage_ns,
            counters=profile_counters,
        )
        on_progress(progress_payload)
        last_scan_progress_emit_monotonic = now
        return True

    scan_started_ns = time.monotonic_ns()
    try:
        for f in funcs:
            check_deadline()
            class_name = enclosing_class(f, parents)
            scopes = enclosing_scopes(f, parents)
            lexical_scopes = enclosing_function_scopes(f, parents)
            fn_key = function_key(scopes, f.name)
            if (
                fn_key in fn_use
                and fn_key in fn_calls
                and fn_key in fn_param_orders
                and fn_key in fn_param_spans
                and fn_key in fn_names
                and fn_key in fn_lexical_scopes
                and fn_key in fn_class_names
            ):
                continue
            if not decorators_transparent(f, config.transparent_decorators):
                opaque_callees.add(fn_key)
            use_map, call_args = analyze_function(
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
            fn_param_orders[fn_key] = param_names(f, config.ignore_params)
            fn_param_spans[fn_key] = param_spans(f, config.ignore_params)
            fn_names[fn_key] = f.name
            fn_lexical_scopes[fn_key] = tuple(lexical_scopes)
            fn_class_names[fn_key] = class_name
            scanned_since_emit += 1
            if scanned_since_emit >= progress_emit_interval and _emit_scan_progress():
                scanned_since_emit = 0
        profile_stage_ns["file_scan.function_scan"] += time.monotonic_ns() - scan_started_ns
    except Exception:
        _emit_scan_progress(force=True)
        if on_profile is not None:
            on_profile(profiling_payload(stage_ns=profile_stage_ns, counters=profile_counters))
        raise
    if scanned_since_emit > 0:
        _emit_scan_progress(force=True)

    local_by_name: dict[str, list[str]] = defaultdict(list)
    for key, name in fn_names.items():
        check_deadline()
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

    local_resolve_started_ns = time.monotonic_ns()
    for caller_key, calls in list(fn_calls.items()):
        check_deadline()
        resolved_calls = []
        for call in calls:
            check_deadline()
            resolved = _resolve_local_callee(call.callee, caller_key)
            resolved_calls.append(call if not resolved else replace(call, callee=resolved))
        fn_calls[caller_key] = resolved_calls
    profile_stage_ns["file_scan.resolve_local_calls"] += time.monotonic_ns() - local_resolve_started_ns

    class_bases = collect_local_class_bases(tree, parents)
    profile_counters["file_scan.class_bases_count"] = len(class_bases)
    if class_bases:
        method_resolve_started_ns = time.monotonic_ns()
        local_functions = set(fn_use.keys())

        def _resolve_local_method(callee: str) -> str | None:
            class_part, method = callee.rsplit(".", 1)
            return resolve_local_method_in_hierarchy(
                class_part,
                method,
                class_bases=class_bases,
                local_functions=local_functions,
                seen=set(),
            )

        for caller_key, calls in list(fn_calls.items()):
            resolved_calls = []
            for call in calls:
                check_deadline()
                if "." in call.callee:
                    resolved = _resolve_local_method(call.callee)
                    if resolved and resolved != call.callee:
                        resolved_calls.append(replace(call, callee=resolved))
                        continue
                resolved_calls.append(call)
            fn_calls[caller_key] = resolved_calls
        profile_stage_ns["file_scan.resolve_local_methods"] += time.monotonic_ns() - method_resolve_started_ns

    functions: list[PythonFunctionIngestCarrier] = []
    for key in fn_use:
        check_deadline()
        functions.append(
            PythonFunctionIngestCarrier(
                key=key,
                name=fn_names.get(key, ""),
                params=tuple(fn_param_orders.get(key, [])),
                annotations={},
                callsites=tuple(fn_calls.get(key, [])),
                param_spans=fn_param_spans.get(key, {}),
                decision_evidence={},
            )
        )
    if on_profile is not None:
        on_profile(profiling_payload(stage_ns=profile_stage_ns, counters=profile_counters))

    return PythonFileIngestCarrier(
        path=path,
        functions=tuple(functions),
        function_use=fn_use,
        function_calls=fn_calls,
        function_param_orders=fn_param_orders,
        function_param_spans=fn_param_spans,
        function_names=fn_names,
        function_lexical_scopes=fn_lexical_scopes,
        function_class_names=fn_class_names,
        opaque_callees=opaque_callees,
        parse_failure_witnesses=(),
    )
