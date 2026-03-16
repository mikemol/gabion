from __future__ import annotations

from gabion.invariants import todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    declared_touchsite_definition,
    registry_marker_metadata,
)


@todo_decorator(
    reason="RCI remains active until invariant-graph runtime context is constructed explicitly instead of being inferred from repo-global state and monkeypatched providers.",
    reasoning={
        "summary": "Phase5 touchsite scanning and invariant-graph runtime-policy tests still leak repo-root and imported-provider assumptions instead of using injected runtime context.",
        "control": "runtime_context_injection.root",
        "blocking_dependencies": (
            "RCI-SQ-001",
            "RCI-SQ-002",
            "RCI-SQ-003",
            "RCI-SQ-004",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="RCI closure",
    links=[{"kind": "object_id", "value": "RCI"}],
)
def _rci_root() -> None:
    return None


@todo_decorator(
    reason="RCI-SQ-001 remains active until phase5 touchsite scanning resolves source from the active invariant-graph build root.",
    reasoning={
        "summary": "Phase5 touchsite scanning still binds source reads to the repository root instead of the runtime build root used to construct the graph.",
        "control": "runtime_context_injection.phase5_root_scanning",
        "blocking_dependencies": ("RCI-TP-001",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-001"},
    ],
)
def _rci_sq_phase5_root_injection() -> None:
    return None


@todo_decorator(
    reason="RCI-SQ-002 remains active until invariant-graph library tests construct registry context explicitly instead of mutating imported providers.",
    reasoning={
        "summary": "Invariant-graph library tests still disable or replace declared registry providers by monkeypatch instead of passing explicit registry tuples to the builder seam.",
        "control": "runtime_context_injection.library_test_di",
        "blocking_dependencies": ("RCI-TP-002", "RCI-TP-003"),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-002"},
    ],
)
def _rci_sq_library_di_cleanup() -> None:
    return None


@todo_decorator(
    reason="RCI-SQ-003 remains active until command and runtime entrypoints accept injected invariant-graph runtime context without new CLI flags.",
    reasoning={
        "summary": "policy_check and tooling.runtime.invariant_graph still internalize repo-root and declared-registry selection, forcing tests to patch globals instead of passing context programmatically.",
        "control": "runtime_context_injection.command_runtime_di",
        "blocking_dependencies": ("RCI-TP-004", "RCI-TP-005"),
    },
    owner="gabion.tooling.runtime",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-003"},
    ],
)
def _rci_sq_command_di_cleanup() -> None:
    return None


@todo_decorator(
    reason="RCI-SQ-004 remains active until live-repo invariant-graph sentinel assertions are separated from synthetic-root code tests.",
    reasoning={
        "summary": "Runtime-policy invariant-graph tests still mix live repository state assertions with code-targeted synthetic-root tests, conflating repo-state signals and code signals.",
        "control": "runtime_context_injection.repo_state_sentinels",
        "blocking_dependencies": ("RCI-TP-006",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-004"},
    ],
)
def _rci_sq_repo_state_sentinels() -> None:
    return None


@todo_decorator(
    reason="RCI-TP-001 tracks active-root phase5 touchsite scanning.",
    reasoning={
        "summary": "Invariant-graph phase5 scanning should resolve touchsite source from the active build root and fail closed when the runtime root does not contain the declared path.",
        "control": "runtime_context_injection.phase5_touchsite_root",
        "blocking_dependencies": ("RCI-SQ-001",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-001"},
        {"kind": "object_id", "value": "RCI-TP-001"},
    ],
)
def _rci_tp_phase5_touchsite_root() -> None:
    return None


@todo_decorator(
    reason="RCI-TP-002 tracks explicit empty-registry injection for pure invariant-graph unit tests.",
    reasoning={
        "summary": "Pure synthetic-root invariant-graph tests should use declared_registries=() instead of mutating imported registry providers to simulate no declared workstreams.",
        "control": "runtime_context_injection.empty_registry_tests",
        "blocking_dependencies": ("RCI-SQ-002",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-002"},
        {"kind": "object_id", "value": "RCI-TP-002"},
    ],
)
def _rci_tp_empty_registry_tests() -> None:
    return None


@todo_decorator(
    reason="RCI-TP-003 tracks explicit synthetic/connectivity registry injection for invariant-graph library tests.",
    reasoning={
        "summary": "Invariant-graph library tests that depend on declared workstream context should pass synthetic or connectivity registry tuples directly instead of monkeypatching provider functions.",
        "control": "runtime_context_injection.synthetic_registry_tests",
        "blocking_dependencies": ("RCI-SQ-002",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-002"},
        {"kind": "object_id", "value": "RCI-TP-003"},
    ],
)
def _rci_tp_synthetic_registry_tests() -> None:
    return None


@todo_decorator(
    reason="RCI-TP-004 tracks repo-root and invariant-registry injection for policy_check programmatic callers.",
    reasoning={
        "summary": "policy_check.main should accept injected repo root and invariant declared registries for tests and synthetic runtime construction without changing CLI shape.",
        "control": "runtime_context_injection.policy_check_main",
        "blocking_dependencies": ("RCI-SQ-003",),
    },
    owner="scripts.policy",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-003"},
        {"kind": "object_id", "value": "RCI-TP-004"},
    ],
)
def _rci_tp_policy_check_main() -> None:
    return None


@todo_decorator(
    reason="RCI-TP-005 tracks declared-registry injection for tooling.runtime.invariant_graph programmatic callers.",
    reasoning={
        "summary": "tooling.runtime.invariant_graph.main should accept injected declared registries for tests and synthetic graph builds while preserving the CLI contract.",
        "control": "runtime_context_injection.runtime_invariant_graph_main",
        "blocking_dependencies": ("RCI-SQ-003",),
    },
    owner="gabion.tooling.runtime",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-003"},
        {"kind": "object_id", "value": "RCI-TP-005"},
    ],
)
def _rci_tp_runtime_invariant_graph_main() -> None:
    return None


@todo_decorator(
    reason="RCI-TP-006 tracks the split between live-repo invariant-graph sentinels and synthetic-root code tests.",
    reasoning={
        "summary": "Live repository state assertions should live in a dedicated sentinel module so repo-state signals and synthetic code-targeted tests do not share the same file.",
        "control": "runtime_context_injection.live_repo_test_split",
        "blocking_dependencies": ("RCI-SQ-004",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-004"},
        {"kind": "object_id", "value": "RCI-TP-006"},
    ],
)
def _rci_tp_live_repo_test_split() -> None:
    return None


def _root_definition(
    *,
    root_id: str,
    title: str,
    subqueue_ids: tuple[str, ...],
    symbol,
    status_hint: str = "",
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="runtime_context_injection_root",
        structural_path=f"runtime_context_injection.root::{root_id}",
    )
    return RegisteredRootDefinition(
        root_id=root_id,
        title=title,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        subqueue_ids=subqueue_ids,
        status_hint=status_hint,
    )


def _subqueue_definition(
    *,
    root_id: str,
    subqueue_id: str,
    title: str,
    touchpoint_ids: tuple[str, ...],
    symbol,
    status_hint: str = "",
) -> RegisteredSubqueueDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="runtime_context_injection_subqueue",
        structural_path=f"runtime_context_injection.subqueue::{subqueue_id}",
    )
    return RegisteredSubqueueDefinition(
        root_id=root_id,
        subqueue_id=subqueue_id,
        title=title,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        touchpoint_ids=touchpoint_ids,
        status_hint=status_hint,
    )


def _touchpoint_definition(
    *,
    root_id: str,
    subqueue_id: str,
    touchpoint_id: str,
    title: str,
    symbol,
    declared_touchsites: tuple = (),
    status_hint: str = "",
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="runtime_context_injection_touchpoint",
        structural_path=f"runtime_context_injection.touchpoint::{touchpoint_id}",
    )
    return RegisteredTouchpointDefinition(
        root_id=root_id,
        touchpoint_id=touchpoint_id,
        subqueue_id=subqueue_id,
        title=title,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        declared_touchsites=declared_touchsites,
        status_hint=status_hint,
    )


def _module_touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=1,
        node_kind="module",
        surface="runtime_context_injection_touchsite",
        structural_path=f"runtime_context_injection.touchsite::{touchsite_id}",
        seam_class="surviving_carrier_seam",
    )


def _script_touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=1,
        node_kind="script",
        surface="runtime_context_injection_touchsite",
        structural_path=f"runtime_context_injection.touchsite::{touchsite_id}",
        seam_class="surviving_carrier_seam",
    )


def runtime_context_injection_workstream_registry() -> WorkstreamRegistry:
    root_id = "RCI"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Runtime-context injection for invariant-graph surfaces",
            symbol=_rci_root,
            subqueue_ids=(
                "RCI-SQ-001",
                "RCI-SQ-002",
                "RCI-SQ-003",
                "RCI-SQ-004",
            ),
            status_hint="landed",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-001",
                title="Phase5 touchsite active-root injection",
                symbol=_rci_sq_phase5_root_injection,
                touchpoint_ids=("RCI-TP-001",),
                status_hint="landed",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-002",
                title="Invariant-graph library dependency injection cleanup",
                symbol=_rci_sq_library_di_cleanup,
                touchpoint_ids=("RCI-TP-002", "RCI-TP-003"),
                status_hint="landed",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-003",
                title="Command and runtime DI seams for invariant-graph callers",
                symbol=_rci_sq_command_di_cleanup,
                touchpoint_ids=("RCI-TP-004", "RCI-TP-005"),
                status_hint="landed",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-004",
                title="Live-repo sentinel separation for invariant-graph tests",
                symbol=_rci_sq_repo_state_sentinels,
                touchpoint_ids=("RCI-TP-006",),
                status_hint="landed",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-001",
                touchpoint_id="RCI-TP-001",
                title="Active-root phase5 touchsite scanning",
                symbol=_rci_tp_phase5_touchsite_root,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="RCI-TS-001-A",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="invariant_graph",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-002",
                touchpoint_id="RCI-TP-002",
                title="Explicit empty-registry injection for invariant-graph unit tests",
                symbol=_rci_tp_empty_registry_tests,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="RCI-TS-002-A",
                        rel_path="tests/gabion/tooling/runtime_policy/test_invariant_graph.py",
                        qualname="test_invariant_graph",
                    ),
                    _module_touchsite(
                        touchsite_id="RCI-TS-002-B",
                        rel_path="tests/gabion/tooling/runtime_policy/invariant_graph_test_support.py",
                        qualname="invariant_graph_test_support",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-002",
                touchpoint_id="RCI-TP-003",
                title="Explicit synthetic/connectivity registry injection for invariant-graph tests",
                symbol=_rci_tp_synthetic_registry_tests,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="RCI-TS-003-A",
                        rel_path="tests/gabion/tooling/runtime_policy/test_invariant_graph.py",
                        qualname="test_invariant_graph",
                    ),
                    _module_touchsite(
                        touchsite_id="RCI-TS-003-B",
                        rel_path="tests/gabion/tooling/runtime_policy/invariant_graph_test_support.py",
                        qualname="invariant_graph_test_support",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-003",
                touchpoint_id="RCI-TP-004",
                title="Injected repo-root and registry seam for policy_check.main",
                symbol=_rci_tp_policy_check_main,
                status_hint="landed",
                declared_touchsites=(
                    _script_touchsite(
                        touchsite_id="RCI-TS-004-A",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="policy_check",
                    ),
                    _module_touchsite(
                        touchsite_id="RCI-TS-004-B",
                        rel_path="tests/gabion/tooling/runtime_policy/test_policy_check_output.py",
                        qualname="test_policy_check_output",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-003",
                touchpoint_id="RCI-TP-005",
                title="Injected declared-registry seam for tooling.runtime.invariant_graph.main",
                symbol=_rci_tp_runtime_invariant_graph_main,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="RCI-TS-005-A",
                        rel_path="src/gabion/tooling/runtime/invariant_graph.py",
                        qualname="runtime_invariant_graph",
                    ),
                    _module_touchsite(
                        touchsite_id="RCI-TS-005-B",
                        rel_path="tests/gabion/tooling/runtime_policy/test_runtime_invariant_graph_perf.py",
                        qualname="test_runtime_invariant_graph_perf",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-004",
                touchpoint_id="RCI-TP-006",
                title="Dedicated live-repo sentinel module for invariant-graph tests",
                symbol=_rci_tp_live_repo_test_split,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="RCI-TS-006-A",
                        rel_path="tests/gabion/tooling/runtime_policy/test_invariant_graph.py",
                        qualname="test_invariant_graph",
                    ),
                    _module_touchsite(
                        touchsite_id="RCI-TS-006-B",
                        rel_path="tests/gabion/tooling/runtime_policy/test_invariant_graph_live_repo.py",
                        qualname="test_invariant_graph_live_repo",
                    ),
                ),
            ),
        ),
        tags=("runtime_context_injection",),
    )


__all__ = ["runtime_context_injection_workstream_registry"]
