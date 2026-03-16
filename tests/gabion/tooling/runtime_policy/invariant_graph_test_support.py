from __future__ import annotations

from pathlib import Path

from gabion.invariants import todo_decorator
from gabion.tooling.policy_substrate.connectivity_synergy_registry import (
    connectivity_synergy_workstream_registries,
)
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    declared_touchsite_definition,
    registry_marker_metadata,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_minimal_invariant_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "src" / "gabion" / "__init__.py", "")
    _write(
        tmp_path / "src" / "gabion" / "synthetic_boundaries.py",
        "\n".join(
            [
                "def idr_boundary() -> None:",
                "    return None",
                "",
                "def igm_boundary() -> None:",
                "    return None",
                "",
                "def rgc_boundary() -> None:",
                "    return None",
                "",
                "def ivl_boundary() -> None:",
                "    return None",
                "",
                "def scc_boundary() -> None:",
                "    return None",
                "",
            ]
        ),
    )
    return tmp_path


@todo_decorator(
    reason="Synthetic CSA-IDR root for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic identity/rendering root for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_idr.root",
        "blocking_dependencies": ["CSA-IDR-SQ-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IDR"}],
)
def _synthetic_csa_idr_root() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-IDR subqueue for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic identity/rendering subqueue for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_idr.subqueue",
        "blocking_dependencies": ["CSA-IDR-TP-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IDR-SQ-T01"}],
)
def _synthetic_csa_idr_subqueue() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-IDR touchpoint for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic identity/rendering touchpoint for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_idr.touchpoint",
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IDR-TP-T01"}],
)
def _synthetic_csa_idr_touchpoint() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-IGM root for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic ingress/merge root for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_igm.root",
        "blocking_dependencies": ["CSA-IGM-SQ-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IGM"}],
)
def _synthetic_csa_igm_root() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-IGM subqueue for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic ingress/merge subqueue for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_igm.subqueue",
        "blocking_dependencies": ["CSA-IGM-TP-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IGM-SQ-T01"}],
)
def _synthetic_csa_igm_subqueue() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-IGM touchpoint for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic ingress/merge touchpoint for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_igm.touchpoint",
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IGM-TP-T01"}],
)
def _synthetic_csa_igm_touchpoint() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-RGC root for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic registry-convergence root for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_rgc.root",
        "blocking_dependencies": ["CSA-RGC-SQ-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-RGC"}],
)
def _synthetic_csa_rgc_root() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-RGC subqueue for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic registry-convergence subqueue for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_rgc.subqueue",
        "blocking_dependencies": ["CSA-RGC-TP-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-RGC-SQ-T01"}],
)
def _synthetic_csa_rgc_subqueue() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-RGC touchpoint for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic registry-convergence touchpoint for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_rgc.touchpoint",
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-RGC-TP-T01"}],
)
def _synthetic_csa_rgc_touchpoint() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-IVL root for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic impact-velocity root for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_ivl.root",
        "blocking_dependencies": ["CSA-IVL-SQ-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IVL"}],
)
def _synthetic_csa_ivl_root() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-IVL subqueue for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic impact-velocity subqueue for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_ivl.subqueue",
        "blocking_dependencies": ["CSA-IVL-TP-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IVL-SQ-T01"}],
)
def _synthetic_csa_ivl_subqueue() -> None:
    return None


@todo_decorator(
    reason="Synthetic CSA-IVL touchpoint for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic impact-velocity touchpoint for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.csa_ivl.touchpoint",
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "CSA-IVL-TP-T01"}],
)
def _synthetic_csa_ivl_touchpoint() -> None:
    return None


@todo_decorator(
    reason="Synthetic SCC root for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic surface-contract-convergence root for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.scc.root",
        "blocking_dependencies": ["SCC-SQ-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "SCC"}],
)
def _synthetic_scc_root() -> None:
    return None


@todo_decorator(
    reason="Synthetic SCC subqueue for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic surface-contract-convergence subqueue for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.scc.subqueue",
        "blocking_dependencies": ["SCC-TP-T01"],
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "SCC-SQ-T01"}],
)
def _synthetic_scc_subqueue() -> None:
    return None


@todo_decorator(
    reason="Synthetic SCC touchpoint for injected invariant-graph tests.",
    reasoning={
        "summary": "Synthetic surface-contract-convergence touchpoint for dependency-injected planning tests.",
        "control": "tests.runtime_policy.synthetic.scc.touchpoint",
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "SCC-TP-T01"}],
)
def _synthetic_scc_touchpoint() -> None:
    return None


@todo_decorator(
    reason="Synthetic PSF-007 root for tmp-root invariant-graph dependency injection.",
    reasoning={
        "summary": "Synthetic PSF-007 root resolves declared connectivity dependencies without scanning live Phase-5 touchsites.",
        "control": "tests.runtime_policy.synthetic.psf007.root",
    },
    owner="tests.gabion.tooling.runtime_policy",
    expiry="2099-01-01",
    links=[{"kind": "object_id", "value": "PSF-007"}],
)
def _synthetic_psf_007_root() -> None:
    return None


def _root_definition(
    *,
    root_id: str,
    title: str,
    symbol,
    subqueue_ids: tuple[str, ...],
    status_hint: str = "",
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="tests_runtime_policy_root",
        structural_path=f"tests.runtime_policy.root::{root_id}",
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
    symbol,
    touchpoint_ids: tuple[str, ...],
    status_hint: str = "",
) -> RegisteredSubqueueDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="tests_runtime_policy_subqueue",
        structural_path=f"tests.runtime_policy.subqueue::{subqueue_id}",
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
    touchsite_id: str,
    rel_path: str,
    qualname: str,
    line: int,
    status_hint: str = "",
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="tests_runtime_policy_touchpoint",
        structural_path=f"tests.runtime_policy.touchpoint::{touchpoint_id}",
    )
    return RegisteredTouchpointDefinition(
        root_id=root_id,
        subqueue_id=subqueue_id,
        touchpoint_id=touchpoint_id,
        title=title,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        declared_touchsites=(
            declared_touchsite_definition(
                touchsite_id=touchsite_id,
                rel_path=rel_path,
                qualname=qualname,
                boundary_name=title,
                line=line,
                surface="tests_runtime_policy_touchsite",
                structural_path=f"tests.runtime_policy.touchsite::{touchsite_id}",
                object_ids=(touchsite_id,),
            ),
        ),
        status_hint=status_hint,
    )


def synthetic_connectivity_workstream_registries() -> tuple[WorkstreamRegistry, ...]:
    touchsite_rel_path = "src/gabion/synthetic_boundaries.py"
    return (
        WorkstreamRegistry(
            root=_root_definition(
                root_id="CSA-IDR",
                title="Synthetic identity/rendering root",
                symbol=_synthetic_csa_idr_root,
                subqueue_ids=("CSA-IDR-SQ-T01",),
            ),
            subqueues=(
                _subqueue_definition(
                    root_id="CSA-IDR",
                    subqueue_id="CSA-IDR-SQ-T01",
                    title="Synthetic identity/rendering subqueue",
                    symbol=_synthetic_csa_idr_subqueue,
                    touchpoint_ids=("CSA-IDR-TP-T01",),
                ),
            ),
            touchpoints=(
                _touchpoint_definition(
                    root_id="CSA-IDR",
                    subqueue_id="CSA-IDR-SQ-T01",
                    touchpoint_id="CSA-IDR-TP-T01",
                    title="Synthetic IDR touchpoint",
                    symbol=_synthetic_csa_idr_touchpoint,
                    touchsite_id="CSA-IDR-TS-T01",
                    rel_path=touchsite_rel_path,
                    qualname="idr_boundary",
                    line=1,
                ),
            ),
            tags=("identity_rendering",),
        ),
        WorkstreamRegistry(
            root=_root_definition(
                root_id="CSA-IGM",
                title="Synthetic ingress/merge root",
                symbol=_synthetic_csa_igm_root,
                subqueue_ids=("CSA-IGM-SQ-T01",),
            ),
            subqueues=(
                _subqueue_definition(
                    root_id="CSA-IGM",
                    subqueue_id="CSA-IGM-SQ-T01",
                    title="Synthetic ingress/merge subqueue",
                    symbol=_synthetic_csa_igm_subqueue,
                    touchpoint_ids=("CSA-IGM-TP-T01",),
                ),
            ),
            touchpoints=(
                _touchpoint_definition(
                    root_id="CSA-IGM",
                    subqueue_id="CSA-IGM-SQ-T01",
                    touchpoint_id="CSA-IGM-TP-T01",
                    title="Synthetic IGM touchpoint",
                    symbol=_synthetic_csa_igm_touchpoint,
                    touchsite_id="CSA-IGM-TS-T01",
                    rel_path=touchsite_rel_path,
                    qualname="igm_boundary",
                    line=4,
                ),
            ),
            tags=("ingress_merge",),
        ),
        WorkstreamRegistry(
            root=_root_definition(
                root_id="CSA-RGC",
                title="Synthetic registry-convergence root",
                symbol=_synthetic_csa_rgc_root,
                subqueue_ids=("CSA-RGC-SQ-T01",),
            ),
            subqueues=(
                _subqueue_definition(
                    root_id="CSA-RGC",
                    subqueue_id="CSA-RGC-SQ-T01",
                    title="Synthetic registry-convergence subqueue",
                    symbol=_synthetic_csa_rgc_subqueue,
                    touchpoint_ids=("CSA-RGC-TP-T01",),
                ),
            ),
            touchpoints=(
                _touchpoint_definition(
                    root_id="CSA-RGC",
                    subqueue_id="CSA-RGC-SQ-T01",
                    touchpoint_id="CSA-RGC-TP-T01",
                    title="Synthetic RGC touchpoint",
                    symbol=_synthetic_csa_rgc_touchpoint,
                    touchsite_id="CSA-RGC-TS-T01",
                    rel_path=touchsite_rel_path,
                    qualname="rgc_boundary",
                    line=7,
                ),
            ),
            tags=("registry_convergence",),
        ),
        WorkstreamRegistry(
            root=_root_definition(
                root_id="CSA-IVL",
                title="Synthetic impact-velocity root",
                symbol=_synthetic_csa_ivl_root,
                subqueue_ids=("CSA-IVL-SQ-T01",),
            ),
            subqueues=(
                _subqueue_definition(
                    root_id="CSA-IVL",
                    subqueue_id="CSA-IVL-SQ-T01",
                    title="Synthetic impact-velocity subqueue",
                    symbol=_synthetic_csa_ivl_subqueue,
                    touchpoint_ids=("CSA-IVL-TP-T01",),
                ),
            ),
            touchpoints=(
                _touchpoint_definition(
                    root_id="CSA-IVL",
                    subqueue_id="CSA-IVL-SQ-T01",
                    touchpoint_id="CSA-IVL-TP-T01",
                    title="Synthetic IVL touchpoint",
                    symbol=_synthetic_csa_ivl_touchpoint,
                    touchsite_id="CSA-IVL-TS-T01",
                    rel_path=touchsite_rel_path,
                    qualname="ivl_boundary",
                    line=10,
                ),
            ),
            tags=("impact_velocity",),
        ),
        WorkstreamRegistry(
            root=_root_definition(
                root_id="SCC",
                title="Synthetic surface-contract-convergence root",
                symbol=_synthetic_scc_root,
                subqueue_ids=("SCC-SQ-T01",),
            ),
            subqueues=(
                _subqueue_definition(
                    root_id="SCC",
                    subqueue_id="SCC-SQ-T01",
                    title="Synthetic surface-contract-convergence subqueue",
                    symbol=_synthetic_scc_subqueue,
                    touchpoint_ids=("SCC-TP-T01",),
                ),
            ),
            touchpoints=(
                _touchpoint_definition(
                    root_id="SCC",
                    subqueue_id="SCC-SQ-T01",
                    touchpoint_id="SCC-TP-T01",
                    title="Synthetic SCC touchpoint",
                    symbol=_synthetic_scc_touchpoint,
                    touchsite_id="SCC-TS-T01",
                    rel_path=touchsite_rel_path,
                    qualname="scc_boundary",
                    line=13,
                ),
            ),
            tags=("contract_convergence",),
        ),
    )


def connectivity_synergy_with_psf_stub_workstream_registries() -> tuple[WorkstreamRegistry, ...]:
    return connectivity_synergy_workstream_registries() + (
        WorkstreamRegistry(
            root=_root_definition(
                root_id="PSF-007",
                title="Synthetic PSF-007 root",
                symbol=_synthetic_psf_007_root,
                subqueue_ids=(),
            ),
            subqueues=(),
            touchpoints=(),
            tags=("projection_semantic_fragment",),
        ),
    )


def install_synthetic_connectivity_registries(monkeypatch, invariant_graph_module) -> None:
    monkeypatch.setattr(invariant_graph_module, "phase5_workstream_registry", lambda: None)
    monkeypatch.setattr(invariant_graph_module, "prf_workstream_registry", lambda: None)
    monkeypatch.setattr(
        invariant_graph_module,
        "surface_contract_convergence_workstream_registry",
        lambda: None,
    )
    monkeypatch.setattr(
        invariant_graph_module,
        "connectivity_synergy_workstream_registries",
        synthetic_connectivity_workstream_registries,
    )


__all__ = [
    "connectivity_synergy_with_psf_stub_workstream_registries",
    "install_synthetic_connectivity_registries",
    "synthetic_connectivity_workstream_registries",
    "write_minimal_invariant_repo",
]
