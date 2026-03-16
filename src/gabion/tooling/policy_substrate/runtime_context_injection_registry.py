from __future__ import annotations

from gabion.invariants import landed_todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    declared_touchsite_definition,
    registry_marker_metadata,
)


@landed_todo_decorator(
    reason="RCI is recorded as landed metadata after invariant-graph runtime context construction converged on explicit injected context instead of repo-global state and monkeypatched providers.",
    reasoning={
        "summary": "Phase5 touchsite scanning and invariant-graph runtime-policy tests now use explicit injected runtime context rather than repo-root and imported-provider assumptions, and the completed convergence is recorded as closed queue state.",
        "control": "runtime_context_injection.root",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="RCI closure",
    links=[{"kind": "object_id", "value": "RCI"}],
)
def _rci_root() -> None:
    return None


@landed_todo_decorator(
    reason="RCI-SQ-001 is recorded as landed metadata after phase5 touchsite scanning converged on the active invariant-graph build root.",
    reasoning={
        "summary": "Phase5 touchsite scanning now resolves source from the runtime build root used to construct the graph, and the completed convergence is recorded as closed subqueue state.",
        "control": "runtime_context_injection.phase5_root_scanning",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-SQ-002 is recorded as landed metadata after invariant-graph library tests converged on explicit registry-context construction instead of mutating imported providers.",
    reasoning={
        "summary": "Invariant-graph library tests now pass explicit registry tuples to the builder seam instead of disabling or replacing declared registry providers by monkeypatch, and the completed cleanup is recorded as closed subqueue state.",
        "control": "runtime_context_injection.library_test_di",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-SQ-003 is recorded as landed metadata after command and runtime entrypoints accepted injected invariant-graph runtime context without new CLI flags.",
    reasoning={
        "summary": "policy_check and tooling.runtime.invariant_graph now accept programmatic runtime context injection instead of internalizing repo-root and declared-registry selection, and the completed cleanup is recorded as closed subqueue state.",
        "control": "runtime_context_injection.command_runtime_di",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-SQ-004 is recorded as landed metadata after live-repo invariant-graph sentinel assertions were separated from synthetic-root code tests.",
    reasoning={
        "summary": "Runtime-policy invariant-graph tests now isolate live repository state assertions from code-targeted synthetic-root tests, and the completed separation is recorded as closed subqueue state.",
        "control": "runtime_context_injection.repo_state_sentinels",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-TP-001 is recorded as landed metadata for active-root phase5 touchsite scanning.",
    reasoning={
        "summary": "Invariant-graph phase5 scanning now resolves touchsite source from the active build root and fails closed when the runtime root does not contain the declared path, and the completed change is recorded as closed touchpoint state.",
        "control": "runtime_context_injection.phase5_touchsite_root",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-TP-002 is recorded as landed metadata for explicit empty-registry injection in pure invariant-graph unit tests.",
    reasoning={
        "summary": "Pure synthetic-root invariant-graph tests now use declared_registries=() instead of mutating imported registry providers to simulate no declared workstreams, and the completed cleanup is recorded as closed touchpoint state.",
        "control": "runtime_context_injection.empty_registry_tests",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-TP-003 is recorded as landed metadata for explicit synthetic/connectivity registry injection in invariant-graph library tests.",
    reasoning={
        "summary": "Invariant-graph library tests that depend on declared workstream context now pass synthetic or connectivity registry tuples directly instead of monkeypatching provider functions, and the completed cleanup is recorded as closed touchpoint state.",
        "control": "runtime_context_injection.synthetic_registry_tests",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-TP-004 is recorded as landed metadata for repo-root and invariant-registry injection in policy_check programmatic callers.",
    reasoning={
        "summary": "policy_check.main now accepts injected repo root and invariant declared registries for tests and synthetic runtime construction without changing CLI shape, and the completed seam addition is recorded as closed touchpoint state.",
        "control": "runtime_context_injection.policy_check_main",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-TP-005 is recorded as landed metadata for declared-registry injection in tooling.runtime.invariant_graph programmatic callers.",
    reasoning={
        "summary": "tooling.runtime.invariant_graph.main now accepts injected declared registries for tests and synthetic graph builds while preserving the CLI contract, and the completed seam addition is recorded as closed touchpoint state.",
        "control": "runtime_context_injection.runtime_invariant_graph_main",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-TP-006 is recorded as landed metadata for the split between live-repo invariant-graph sentinels and synthetic-root code tests.",
    reasoning={
        "summary": "Live repository state assertions now live in a dedicated sentinel module so repo-state signals and synthetic code-targeted tests do not share the same file, and the completed split is recorded as closed touchpoint state.",
        "control": "runtime_context_injection.live_repo_test_split",
        "blocking_dependencies": (),
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


@landed_todo_decorator(
    reason="RCI-TP-007 is recorded as landed metadata for removing the deprecated live-repo PSF snapshot from the deterministic invariant-graph test surface.",
    reasoning={
        "summary": "The deterministic invariant-graph test surface no longer carries the deprecated live-repo PSF snapshot, and the remaining repo-state sentinel coverage is recorded as closed touchpoint state in the dedicated live-repo module.",
        "control": "runtime_context_injection.live_repo_snapshot_cleanup",
        "blocking_dependencies": (),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="RCI closure",
    links=[
        {"kind": "object_id", "value": "RCI"},
        {"kind": "object_id", "value": "RCI-SQ-004"},
        {"kind": "object_id", "value": "RCI-TP-007"},
    ],
)
def _rci_tp_live_repo_snapshot_cleanup() -> None:
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
                touchpoint_ids=("RCI-TP-006", "RCI-TP-007"),
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
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="RCI-SQ-004",
                touchpoint_id="RCI-TP-007",
                title="Deterministic invariant-graph test cleanup for deprecated live-repo snapshot residue",
                symbol=_rci_tp_live_repo_snapshot_cleanup,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="RCI-TS-007-A",
                        rel_path="tests/gabion/tooling/runtime_policy/test_invariant_graph.py",
                        qualname="test_invariant_graph",
                    ),
                    _module_touchsite(
                        touchsite_id="RCI-TS-007-B",
                        rel_path="tests/gabion/tooling/runtime_policy/test_invariant_graph_live_repo.py",
                        qualname="test_invariant_graph_live_repo",
                    ),
                ),
            ),
        ),
        tags=("runtime_context_injection",),
    )


__all__ = ["runtime_context_injection_workstream_registry"]
