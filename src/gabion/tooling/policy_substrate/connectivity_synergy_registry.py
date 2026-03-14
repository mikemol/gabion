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
    reason="CSA-IDR remains active until typed identity carriers, boundary renderers, and the hierarchical identity grammar converge across the open rollout family.",
    reasoning={
        "summary": "Identity/rendering separation remains partially landed and still needs the call-cluster rollout, aggregate-test cleanup, and hierarchical identity grammar completion tranche.",
        "control": "connectivity_synergy.identity_rendering.root",
        "blocking_dependencies": (
            "CSA-IDR-SQ-001",
            "CSA-IDR-SQ-002",
            "CSA-IDR-SQ-003",
            "CSA-IDR-SQ-004",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IDR closure",
    links=[{"kind": "object_id", "value": "CSA-IDR"}],
)
def _csa_idr_root() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM remains active until cross-origin witness/remap carriers and non-Python ingress adapters converge.",
    reasoning={
        "summary": "Ingress/merge convergence remains open across witness schema, markdown/frontmatter adapters, structured XML/JSON artifact adapters, and overlap parity artifacts.",
        "control": "connectivity_synergy.ingress_merge.root",
        "blocking_dependencies": (
            "CSA-IGM-SQ-001",
            "CSA-IGM-SQ-002",
            "CSA-IGM-SQ-003",
            "CSA-IGM-SQ-004",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[{"kind": "object_id", "value": "CSA-IGM"}],
)
def _csa_igm_root() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC remains active until hardcoded governance inventories, shared selector/interner surfaces, denotational-kernel VM alignment, and wrapper/package inversions converge into declarative carriers.",
    reasoning={
        "summary": "Registry convergence remains open across governance inventory externalization, shared doc/code/query substrate convergence, TTL-kernel VM alignment over runtime semantics, scanner/package execution cleanup, wrapper manifest collapse, and governance/control-loop artifact graph convergence.",
        "control": "connectivity_synergy.registry_convergence.root",
        "blocking_dependencies": (
            "PRF",
            "CSA-RGC-SQ-001",
            "CSA-RGC-SQ-002",
            "CSA-RGC-SQ-003",
            "CSA-RGC-SQ-004",
            "CSA-RGC-SQ-005",
            "CSA-RGC-SQ-006",
            "CSA-RGC-SQ-007",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-RGC closure",
    links=[{"kind": "object_id", "value": "CSA-RGC"}],
)
def _csa_rgc_root() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL remains active until policy/workstream artifact emission and queue rendering stop dominating correction-unit turnaround.",
    reasoning={
        "summary": "Impact velocity remains constrained by invariant-workstream materialization, policy-check artifact fanout, and queue-render tail cost.",
        "control": "connectivity_synergy.impact_velocity.root",
        "blocking_dependencies": (
            "CSA-IVL-SQ-001",
            "CSA-IVL-SQ-002",
            "CSA-IVL-SQ-003",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IVL"},
    ],
)
def _csa_ivl_root() -> None:
    return None


@todo_decorator(
    reason="CSA-IDR-SQ-001 keeps the landed policy/workstream tranche visible in the planning substrate while the broader family remains open.",
    reasoning={
        "summary": "The policy queue identity and artifact-stream tranche is the kept first slice for typed identity plus boundary rendering.",
        "control": "connectivity_synergy.identity_rendering.policy_tranche",
        "blocking_dependencies": ("CSA-IDR-TP-001",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "object_id", "value": "CSA-IDR"},
        {"kind": "object_id", "value": "CSA-IDR-SQ-001"},
    ],
)
def _csa_idr_sq_policy_tranche() -> None:
    return None


@todo_decorator(
    reason="CSA-IDR-SQ-002 remains active until call-cluster carriers stop depending on rendered-string identity in core logic.",
    reasoning={
        "summary": "Call-cluster summary and consolidation surfaces still need typed carrier and boundary renderer convergence.",
        "control": "connectivity_synergy.identity_rendering.call_cluster_rollout",
        "blocking_dependencies": (
            "PSF-007",
            "CSA-IDR-TP-002",
        ),
    },
    owner="gabion.analysis.call_cluster",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "object_id", "value": "CSA-IDR"},
        {"kind": "object_id", "value": "CSA-IDR-SQ-002"},
    ],
)
def _csa_idr_sq_call_cluster_rollout() -> None:
    return None


@todo_decorator(
    reason="CSA-IDR-SQ-003 remains active until aggregate tests stop asserting presentation leaves inline.",
    reasoning={
        "summary": "Aggregate test cleanup remains open so rendering assertions move to leaf-level tests while composition stays typed.",
        "control": "connectivity_synergy.identity_rendering.aggregate_test_cleanup",
        "blocking_dependencies": ("CSA-IDR-TP-003",),
    },
    owner="gabion.analysis.call_cluster",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "object_id", "value": "CSA-IDR"},
        {"kind": "object_id", "value": "CSA-IDR-SQ-003"},
    ],
)
def _csa_idr_sq_aggregate_test_cleanup() -> None:
    return None


@todo_decorator(
    reason="CSA-IDR-SQ-004 remains active until scanner, hotspot queue, and planning chart complete the hierarchical identity grammar rollout.",
    reasoning={
        "summary": "The shared identity-zone substrate landed, but hotspot quotienting, planning-chart activation, and coherence emission still need planner-visible completion.",
        "control": "connectivity_synergy.identity_rendering.identity_grammar_completion",
        "blocking_dependencies": ("CSA-IDR-SQ-001", "CSA-IDR-TP-004"),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IDR"},
        {"kind": "object_id", "value": "CSA-IDR-SQ-004"},
    ],
)
def _csa_idr_sq_identity_grammar_completion() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM-SQ-001 remains active until a typed witness/remap contract exists for cross-origin ASPF merge.",
    reasoning={
        "summary": "Cross-origin identity witness and overlap/remap carriers are still implicit across current ingestion surfaces.",
        "control": "connectivity_synergy.ingress_merge.witness_contract",
        "blocking_dependencies": ("CSA-IGM-TP-001",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[
        {"kind": "object_id", "value": "CSA-IGM"},
        {"kind": "object_id", "value": "CSA-IGM-SQ-001"},
    ],
)
def _csa_igm_sq_witness_contract() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM-SQ-002 remains active until markdown/frontmatter/yaml ingress adapters emit the shared witness envelope.",
    reasoning={
        "summary": "Frontmatter and markdown ingress still parse locally instead of emitting one shared merge-ready carrier contract.",
        "control": "connectivity_synergy.ingress_merge.frontmatter_yaml_adapters",
        "blocking_dependencies": (
            "CSA-IGM-SQ-001",
            "CSA-IGM-TP-002",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[
        {"kind": "object_id", "value": "CSA-IGM"},
        {"kind": "object_id", "value": "CSA-IGM-SQ-002"},
    ],
)
def _csa_igm_sq_adapter_rollout() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM-SQ-003 remains active until overlap/remap parity artifacts and determinism checks are published from the shared merge substrate.",
    reasoning={
        "summary": "Merged overlap parity and determinism artifacts are not yet first-class outputs of the current ingress substrate.",
        "control": "connectivity_synergy.ingress_merge.parity_artifacts",
        "blocking_dependencies": (
            "CSA-IGM-SQ-002",
            "CSA-IGM-TP-003",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[
        {"kind": "object_id", "value": "CSA-IGM"},
        {"kind": "object_id", "value": "CSA-IGM-SQ-003"},
    ],
)
def _csa_igm_sq_parity_artifacts() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM-SQ-004 remains active until XML/JSON sensor artifacts enter the substrate through the same typed witness envelope used by the markdown/frontmatter lanes.",
    reasoning={
        "summary": "JUnit, coverage, perf, deadline, and adjacent planner/control-loop sensor artifacts still ingress through local parser shapes instead of one merge-ready carrier contract.",
        "control": "connectivity_synergy.ingress_merge.structured_artifact_ingress",
        "blocking_dependencies": (
            "CSA-IGM-SQ-001",
            "CSA-IGM-TP-004",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IGM"},
        {"kind": "object_id", "value": "CSA-IGM-SQ-004"},
    ],
)
def _csa_igm_sq_structured_artifact_ingress() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-SQ-001 remains active until governance inventory constants move into declarative carriers.",
    reasoning={
        "summary": "Governance document inventories and required field registries remain hardcoded in Python constants.",
        "control": "connectivity_synergy.registry_convergence.governance_inventory",
        "blocking_dependencies": (
            "CSA-IGM-SQ-002",
            "CSA-RGC-TP-001",
        ),
    },
    owner="gabion_governance",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-001"},
    ],
)
def _csa_rgc_sq_governance_inventory() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-SQ-002 remains active until scanner and governance probes resolve package-native implementations from declarative registries.",
    reasoning={
        "summary": "normative_symdiff still imports script surfaces directly instead of consuming a package-native declarative registry.",
        "control": "connectivity_synergy.registry_convergence.package_native_execution",
        "blocking_dependencies": (
            "CSA-RGC-SQ-001",
            "CSA-RGC-TP-002",
        ),
    },
    owner="gabion.tooling.governance",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-002"},
    ],
)
def _csa_rgc_sq_package_native_execution() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-SQ-003 remains active until remaining wrapper exports collapse to a manifest-driven launcher surface.",
    reasoning={
        "summary": "Governance command wrapper surfaces are still explicit modules rather than a manifest-driven launcher layer.",
        "control": "connectivity_synergy.registry_convergence.wrapper_manifest",
        "blocking_dependencies": (
            "CSA-RGC-SQ-002",
            "CSA-RGC-TP-003",
        ),
    },
    owner="gabion_governance",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-003"},
    ],
)
def _csa_rgc_sq_wrapper_manifest() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-SQ-004 remains active until doc-target selectors, symbol interning, and perf-query overlays converge onto one declarative query substrate.",
    reasoning={
        "summary": "The repo still splits frontmatter-driven doc selection, symbol-universe interning, and perf-query overlay construction across adjacent but non-identical carriers.",
        "control": "connectivity_synergy.registry_convergence.shared_query_substrate",
        "blocking_dependencies": (
            "CSA-IGM-SQ-002",
            "CSA-RGC-TP-004",
        ),
    },
    owner="gabion.analysis.semantics",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-004"},
    ],
)
def _csa_rgc_sq_shared_query_substrate() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-SQ-005 remains active until SPPF checklist state, influence-index adoption state, and in/ governance action carriers converge onto one common graph substrate.",
    reasoning={
        "summary": "Governance convergence remains incomplete while SPPF doc/issue dependencies, influence-index status rows, and inbox action items are still split across adjacent but only partially connected carriers.",
        "control": "connectivity_synergy.registry_convergence.governance_graph_substrate",
        "blocking_dependencies": (
            "CSA-RGC-SQ-004",
            "CSA-RGC-TP-005",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "doc_id", "value": "sppf_checklist"},
        {"kind": "doc_id", "value": "influence_index"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-005"},
    ],
)
def _csa_rgc_sq_governance_graph_substrate() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-SQ-006 remains active until planner, docflow, CI, drift, and policy-suite control-loop artifacts converge onto the same governance/query graph substrate.",
    reasoning={
        "summary": "Control-loop artifacts and git/GH provenance signals are still emitted and consumed as adjacent sidecars rather than one connected graph of governance evidence, failure bundles, planner outputs, issue linkage, and remediation state.",
        "control": "connectivity_synergy.registry_convergence.control_loop_artifact_graph",
        "blocking_dependencies": (
            "CSA-IGM-SQ-004",
            "CSA-RGC-SQ-004",
            "CSA-RGC-SQ-005",
            "CSA-RGC-TP-006",
            "CSA-RGC-TP-007",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-006"},
    ],
)
def _csa_rgc_sq_control_loop_artifact_graph() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-SQ-007 remains active until the TTL ontology, ASPF witness layer, semantic fragment, lowering stack, and planning residues converge into one denotational-kernel VM contract.",
    reasoning={
        "summary": "The repo still treats the TTL kernel, ASPF fibers, semantic fragment carriers, lowering plans, and planning residues as adjacent semantic surfaces instead of one kernel IR plus total runtime realization path.",
        "control": "connectivity_synergy.registry_convergence.kernel_vm_alignment",
        "blocking_dependencies": (
            "CSA-RGC-SQ-004",
            "CSA-RGC-TP-008",
        ),
    },
    owner="gabion.analysis.projection",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "doc_id", "value": "ttl_kernel_semantics"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-007"},
    ],
)
def _csa_rgc_sq_kernel_vm_alignment() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-SQ-001 remains active until workflow convergence and invariant-workstream materialization cost are small enough to stop dominating planner refresh loops.",
    reasoning={
        "summary": "Workflow convergence, lattice witness construction, and invariant workstream projection still form a measurable tail in correction-unit validation.",
        "control": "connectivity_synergy.impact_velocity.workstream_materialization",
        "blocking_dependencies": ("CSA-IVL-TP-001",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-001"},
    ],
)
def _csa_ivl_sq_workstream_materialization() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-SQ-002 remains active until policy_check artifact fanout stops doing redundant late-stage work after artifacts already converge.",
    reasoning={
        "summary": "The workflow validation path still pays a long tail while invariant and queue artifacts are already materially settled.",
        "control": "connectivity_synergy.impact_velocity.policy_check_tail",
        "blocking_dependencies": (
            "CSA-IVL-SQ-001",
            "CSA-IVL-TP-002",
        ),
    },
    owner="scripts.policy",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-002"},
    ],
)
def _csa_ivl_sq_policy_check_tail() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-SQ-003 remains active until projection-semantic-fragment queue rendering stops dominating the refresh tail after planner state is already available.",
    reasoning={
        "summary": "Queue rendering still burns CPU after the planner can already name the next correction cut from invariant artifacts.",
        "control": "connectivity_synergy.impact_velocity.queue_render_tail",
        "blocking_dependencies": (
            "CSA-IVL-SQ-001",
            "CSA-IVL-SQ-002",
            "CSA-IVL-TP-003",
        ),
    },
    owner="scripts.policy",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-003"},
    ],
)
def _csa_ivl_sq_queue_render_tail() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-SQ-004 remains active until planner-facing artifact emission and summary surfaces follow a least-surprise contract for freshness and operator feedback.",
    reasoning={
        "summary": "The current workflow/output path still leaves room for surprising artifact freshness and planner summary behavior unless the boundary contract is made explicit and mechanically checked.",
        "control": "connectivity_synergy.impact_velocity.least_surprise_contract",
        "blocking_dependencies": (
            "CSA-IVL-SQ-002",
            "CSA-IVL-TP-004",
        ),
    },
    owner="scripts.policy",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-004"},
    ],
)
def _csa_ivl_sq_least_surprise_contract() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-SQ-005 remains active until profiler artifacts can be fibered onto invariant-graph structure as a reusable performance heat map.",
    reasoning={
        "summary": "The live cProfile producer is now present, but impact velocity still needs DSL-rooted perf queries, cross-origin root overlay onto the libcst/pyast ASPF spine, and broader multi-profiler parity across py-spy, pyinstrument, and memray inputs.",
        "control": "connectivity_synergy.impact_velocity.perf_heat_fiber",
        "blocking_dependencies": (
            "CSA-IVL-SQ-001",
            "CSA-IGM-SQ-001",
            "CSA-RGC-SQ-004",
            "CSA-IVL-TP-005",
        ),
    },
    owner="gabion.tooling.runtime",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-005"},
    ],
)
def _csa_ivl_sq_perf_heat_fiber() -> None:
    return None


@todo_decorator(
    reason="CSA-IDR-TP-001 records the kept policy/workstream identity-rendering tranche as a planning-visible touchpoint.",
    reasoning={
        "summary": "The queue identity and artifact stream surfaces remain the baseline typed carrier plus boundary renderer tranche.",
        "control": "connectivity_synergy.identity_rendering.policy_touchpoint",
        "blocking_dependencies": ("CSA-IDR-SQ-001",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "object_id", "value": "CSA-IDR"},
        {"kind": "object_id", "value": "CSA-IDR-SQ-001"},
        {"kind": "object_id", "value": "CSA-IDR-TP-001"},
    ],
)
def _csa_idr_tp_policy_tranche() -> None:
    return None


@todo_decorator(
    reason="CSA-IDR-TP-002 tracks the open call-cluster carrier/rendering cleanup surfaces.",
    reasoning={
        "summary": "Call-cluster summary and consolidation still expose string identity and rendering seams that should move to typed carriers plus boundary renderers.",
        "control": "connectivity_synergy.identity_rendering.call_cluster_touchpoint",
        "blocking_dependencies": ("CSA-IDR-SQ-002",),
    },
    owner="gabion.analysis.call_cluster",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "object_id", "value": "CSA-IDR"},
        {"kind": "object_id", "value": "CSA-IDR-SQ-002"},
        {"kind": "object_id", "value": "CSA-IDR-TP-002"},
    ],
)
def _csa_idr_tp_call_cluster_rollout() -> None:
    return None


@todo_decorator(
    reason="CSA-IDR-TP-003 tracks aggregate-test cleanup for rendering-sensitive call-cluster assertions.",
    reasoning={
        "summary": "Aggregate tests still mix composition assertions with presentation assertions in call-cluster surfaces.",
        "control": "connectivity_synergy.identity_rendering.aggregate_test_touchpoint",
        "blocking_dependencies": ("CSA-IDR-SQ-003",),
    },
    owner="gabion.analysis.call_cluster",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "object_id", "value": "CSA-IDR"},
        {"kind": "object_id", "value": "CSA-IDR-SQ-003"},
        {"kind": "object_id", "value": "CSA-IDR-TP-003"},
    ],
)
def _csa_idr_tp_aggregate_test_cleanup() -> None:
    return None


@todo_decorator(
    reason="CSA-IDR-TP-004 tracks the remaining hierarchical identity grammar completion surfaces across scanner, hotspot queue, planning chart, and coherence witnesses.",
    reasoning={
        "summary": "The first identity-zone tranche landed, but hotspot internals still rely on raw string grouping, quotient witnesses remain representative-only, planning-chart grammar is not yet active in production, and 2-cell coherence witnesses are not emitted.",
        "control": "connectivity_synergy.identity_rendering.identity_grammar_touchpoint",
        "blocking_dependencies": ("CSA-IDR-SQ-001", "CSA-IDR-SQ-004"),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IDR"},
        {"kind": "object_id", "value": "CSA-IDR-SQ-004"},
        {"kind": "object_id", "value": "CSA-IDR-TP-004"},
    ],
)
def _csa_idr_tp_identity_grammar_completion() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM-TP-001 tracks the current witness/remap-related surfaces that still lack one typed merge contract.",
    reasoning={
        "summary": "Current witness digest, union-view, and overlap surfaces still stop short of one cross-origin remap carrier.",
        "control": "connectivity_synergy.ingress_merge.witness_touchpoint",
        "blocking_dependencies": ("CSA-IGM-SQ-001",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[
        {"kind": "object_id", "value": "CSA-IGM"},
        {"kind": "object_id", "value": "CSA-IGM-SQ-001"},
        {"kind": "object_id", "value": "CSA-IGM-TP-001"},
    ],
)
def _csa_igm_tp_witness_contract() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM-TP-002 tracks the open markdown/frontmatter/yaml ingress adapter surfaces.",
    reasoning={
        "summary": "Frontmatter and YAML parsing surfaces still need a merge-ready ASPF adapter contract.",
        "control": "connectivity_synergy.ingress_merge.adapter_touchpoint",
        "blocking_dependencies": ("CSA-IGM-SQ-002",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[
        {"kind": "object_id", "value": "CSA-IGM"},
        {"kind": "object_id", "value": "CSA-IGM-SQ-002"},
        {"kind": "object_id", "value": "CSA-IGM-TP-002"},
    ],
)
def _csa_igm_tp_adapter_rollout() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM-TP-003 tracks overlap/remap determinism and parity evidence surfaces.",
    reasoning={
        "summary": "Parity and determinism artifacts for merged overlap classes remain missing from the current substrate outputs.",
        "control": "connectivity_synergy.ingress_merge.parity_touchpoint",
        "blocking_dependencies": ("CSA-IGM-SQ-003",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[
        {"kind": "object_id", "value": "CSA-IGM"},
        {"kind": "object_id", "value": "CSA-IGM-SQ-003"},
        {"kind": "object_id", "value": "CSA-IGM-TP-003"},
    ],
)
def _csa_igm_tp_parity_artifacts() -> None:
    return None


@todo_decorator(
    reason="CSA-IGM-TP-004 tracks the open XML/JSON sensor artifact ingress surfaces that still need one typed witness envelope.",
    reasoning={
        "summary": "Structured test, planner/control-loop, profiler, and deadline carriers now have a shared ingress foothold, but the remaining live parser surfaces still need one merge-ready artifact contract.",
        "control": "connectivity_synergy.ingress_merge.structured_artifact_touchpoint",
        "blocking_dependencies": ("CSA-IGM-SQ-004",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IGM closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IGM"},
        {"kind": "object_id", "value": "CSA-IGM-SQ-004"},
        {"kind": "object_id", "value": "CSA-IGM-TP-004"},
    ],
)
def _csa_igm_tp_structured_artifact_ingress() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-TP-001 tracks the hardcoded governance inventory constants that still need declarative carriers.",
    reasoning={
        "summary": "governance_audit_impl still owns inventory-like policy constants that should move into declarative source carriers.",
        "control": "connectivity_synergy.registry_convergence.governance_constants_touchpoint",
        "blocking_dependencies": ("CSA-RGC-SQ-001",),
    },
    owner="gabion_governance",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-001"},
        {"kind": "object_id", "value": "CSA-RGC-TP-001"},
    ],
)
def _csa_rgc_tp_governance_inventory() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-TP-002 tracks package-to-scripts inversion in normative_symdiff and related governance probes.",
    reasoning={
        "summary": "normative_symdiff still reaches into script modules directly while collecting governance scanner evidence.",
        "control": "connectivity_synergy.registry_convergence.normative_symdiff_touchpoint",
        "blocking_dependencies": ("CSA-RGC-SQ-002",),
    },
    owner="gabion.tooling.governance",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-002"},
        {"kind": "object_id", "value": "CSA-RGC-TP-002"},
    ],
)
def _csa_rgc_tp_package_native_execution() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-TP-003 tracks the remaining governance wrapper export surfaces that should collapse behind a manifest-driven launcher.",
    reasoning={
        "summary": "Thin governance wrapper modules remain explicit surfaces instead of one manifest-driven command layer.",
        "control": "connectivity_synergy.registry_convergence.wrapper_manifest_touchpoint",
        "blocking_dependencies": ("CSA-RGC-SQ-003",),
    },
    owner="gabion_governance",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-003"},
        {"kind": "object_id", "value": "CSA-RGC-TP-003"},
    ],
)
def _csa_rgc_tp_wrapper_manifest() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-TP-004 tracks the split selector/interner surfaces that should converge into one shared doc/code/perf query substrate.",
    reasoning={
        "summary": "impact_index currently owns the frontmatter-driven doc-target selector while runtime perf heat still reconstructs a local boundary overlay instead of consuming one shared selector/interner carrier.",
        "control": "connectivity_synergy.registry_convergence.shared_query_substrate_touchpoint",
        "blocking_dependencies": ("CSA-RGC-SQ-004",),
    },
    owner="gabion.analysis.semantics",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-004"},
        {"kind": "object_id", "value": "CSA-RGC-TP-004"},
    ],
)
def _csa_rgc_tp_shared_query_substrate() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-TP-005 tracks the common governance graph substrate spanning SPPF checklist state, influence-index adoption state, and inbox action items.",
    reasoning={
        "summary": "The repo now has enough structure to project SPPF doc/issue links plus in/ governance goals and action items into one graph, but the substrate still spans old status-audit parsing and new invariant-graph joins.",
        "control": "connectivity_synergy.registry_convergence.governance_graph_substrate_touchpoint",
        "blocking_dependencies": ("CSA-RGC-SQ-005",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "doc_id", "value": "sppf_checklist"},
        {"kind": "doc_id", "value": "influence_index"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-005"},
        {"kind": "object_id", "value": "CSA-RGC-TP-005"},
    ],
)
def _csa_rgc_tp_governance_graph_substrate() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-TP-006 tracks the planner, docflow, CI, drift, and policy-suite artifact surfaces that should converge onto the same governance/control-loop graph substrate.",
    reasoning={
        "summary": "Planner outputs, packetized docflow results, controller-drift audits, CI-watch bundles, and policy suite sidecars still live as adjacent artifacts instead of one graph-native control loop.",
        "control": "connectivity_synergy.registry_convergence.control_loop_artifact_touchpoint",
        "blocking_dependencies": ("CSA-RGC-SQ-006",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-006"},
        {"kind": "object_id", "value": "CSA-RGC-TP-006"},
    ],
)
def _csa_rgc_tp_control_loop_artifact_graph() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-TP-007 tracks the git-range provenance, GH lifecycle, and docflow obligation surfaces that must become graph-native planning carriers rather than an out-of-band gate.",
    reasoning={
        "summary": "The graph can already see SPPF checklist issue nodes and governance status rows, but it still cannot see the current correction-unit rev-range, GH-reference validation state, or issue lifecycle evidence as first-class planning objects.",
        "control": "connectivity_synergy.registry_convergence.git_issue_provenance_touchpoint",
        "blocking_dependencies": (
            "CSA-IGM-SQ-004",
            "CSA-IVL-SQ-004",
            "CSA-RGC-SQ-005",
            "CSA-RGC-SQ-006",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "doc_id", "value": "sppf_checklist"},
        {"kind": "doc_id", "value": "influence_index"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-006"},
        {"kind": "object_id", "value": "CSA-RGC-TP-007"},
    ],
)
def _csa_rgc_tp_git_issue_provenance() -> None:
    return None


@todo_decorator(
    reason="CSA-RGC-TP-008 tracks the law-side TTL kernel, ASPF fibers, semantic fragment, lowering, and planning-surrogate surfaces that should collapse into one kernel-derived VM contract.",
    reasoning={
        "summary": "The TTL kernel already models augmented rules, polarity, quotient recovery, and reflective SHACL boundaries, but runtime semantics still realize those ideas through partially parallel ASPF, semantic-fragment, lowering, and planner-facing carriers rather than a small kernel interpreter plus residue report.",
        "control": "connectivity_synergy.registry_convergence.kernel_vm_alignment_touchpoint",
        "blocking_dependencies": (
            "CSA-IVL-SQ-001",
            "CSA-RGC-SQ-004",
            "CSA-RGC-SQ-007",
        ),
    },
    owner="gabion.analysis.projection",
    expiry="CSA-RGC closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "doc_id", "value": "ttl_kernel_semantics"},
        {"kind": "object_id", "value": "CSA-RGC"},
        {"kind": "object_id", "value": "CSA-RGC-SQ-007"},
        {"kind": "object_id", "value": "CSA-RGC-TP-008"},
    ],
)
def _csa_rgc_tp_kernel_vm_alignment() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-TP-001 tracks workflow convergence, lattice witness construction, and invariant-workstream projection surfaces that currently govern refresh cost.",
    reasoning={
        "summary": "The live workflow profile is dominated by convergence collection, semantic lattice materialization, and ASPF witness construction before invariant workstream artifacts are written.",
        "control": "connectivity_synergy.impact_velocity.workstream_materialization_touchpoint",
        "blocking_dependencies": ("CSA-IVL-SQ-001",),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-001"},
        {"kind": "object_id", "value": "CSA-IVL-TP-001"},
    ],
)
def _csa_ivl_tp_workstream_materialization() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-TP-002 tracks policy_check artifact fanout surfaces that continue running after planner artifacts are already settled.",
    reasoning={
        "summary": "Workflow artifact fanout still does redundant late-stage work around invariant and queue artifact emission.",
        "control": "connectivity_synergy.impact_velocity.policy_check_tail_touchpoint",
        "blocking_dependencies": ("CSA-IVL-SQ-002",),
    },
    owner="scripts.policy",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-002"},
        {"kind": "object_id", "value": "CSA-IVL-TP-002"},
    ],
)
def _csa_ivl_tp_policy_check_tail() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-TP-003 tracks queue-render surfaces that remain expensive after the invariant planner can already identify the next slice.",
    reasoning={
        "summary": "Projection-semantic-fragment queue structure and render paths still dominate the tail after planner truth is already available.",
        "control": "connectivity_synergy.impact_velocity.queue_render_touchpoint",
        "blocking_dependencies": ("CSA-IVL-SQ-003",),
    },
    owner="scripts.policy",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-003"},
        {"kind": "object_id", "value": "CSA-IVL-TP-003"},
    ],
)
def _csa_ivl_tp_queue_render_tail() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-TP-004 tracks planner summary and artifact-boundary surfaces that should make freshness semantics unsurprising to operators.",
    reasoning={
        "summary": "Planner summary and workstream artifact lookup surfaces should disclose and preserve one predictable artifact freshness contract.",
        "control": "connectivity_synergy.impact_velocity.least_surprise_touchpoint",
        "blocking_dependencies": ("CSA-IVL-SQ-004",),
    },
    owner="gabion.tooling.runtime",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-004"},
        {"kind": "object_id", "value": "CSA-IVL-TP-004"},
    ],
)
def _csa_ivl_tp_least_surprise_contract() -> None:
    return None


@todo_decorator(
    reason="CSA-IVL-TP-005 tracks runtime invariant-graph surfaces that normalize profiler artifacts and map them onto structural hotspots.",
    reasoning={
        "summary": "Profiler artifact production, DSL-rooted query construction, root overlay against the libcst/pyast ASPF spine, and structural hotspot mapping should remain explicit so live performance heat can be ranked against invariant-graph structure while the shared selector/interner substrate is still converging.",
        "control": "connectivity_synergy.impact_velocity.perf_heat_touchpoint",
        "blocking_dependencies": (
            "CSA-IVL-SQ-005",
            "CSA-IGM-SQ-001",
            "CSA-RGC-SQ-004",
        ),
    },
    owner="gabion.tooling.runtime",
    expiry="CSA-IVL closure",
    links=[
        {"kind": "doc_id", "value": "connectivity_synergy_audit"},
        {"kind": "object_id", "value": "CSA-IVL"},
        {"kind": "object_id", "value": "CSA-IVL-SQ-005"},
        {"kind": "object_id", "value": "CSA-IVL-TP-005"},
    ],
)
def _csa_ivl_tp_perf_heat_fiber() -> None:
    return None


def _root_definition(
    *,
    root_id: str,
    title: str,
    symbol,
    status_hint: str = "",
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="connectivity_synergy_root",
        structural_path=f"connectivity_synergy.root::{root_id}",
    )
    blocking_dependencies = tuple(
        link.value
        for link in metadata.marker_payload.links
        if False
    )
    _ = blocking_dependencies
    subqueue_ids_by_root = {
        "CSA-IDR": (
            "CSA-IDR-SQ-001",
            "CSA-IDR-SQ-002",
            "CSA-IDR-SQ-003",
            "CSA-IDR-SQ-004",
        ),
        "CSA-IGM": (
            "CSA-IGM-SQ-001",
            "CSA-IGM-SQ-002",
            "CSA-IGM-SQ-003",
            "CSA-IGM-SQ-004",
        ),
        "CSA-IVL": (
            "CSA-IVL-SQ-001",
            "CSA-IVL-SQ-002",
            "CSA-IVL-SQ-003",
            "CSA-IVL-SQ-004",
            "CSA-IVL-SQ-005",
        ),
        "CSA-RGC": (
            "CSA-RGC-SQ-001",
            "CSA-RGC-SQ-002",
            "CSA-RGC-SQ-003",
            "CSA-RGC-SQ-004",
            "CSA-RGC-SQ-005",
            "CSA-RGC-SQ-006",
            "CSA-RGC-SQ-007",
        ),
    }
    subqueue_ids = subqueue_ids_by_root[root_id]
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
        surface="connectivity_synergy_subqueue",
        structural_path=f"connectivity_synergy.subqueue::{subqueue_id}",
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
        surface="connectivity_synergy_touchpoint",
        structural_path=f"connectivity_synergy.touchpoint::{touchpoint_id}",
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


def _function_touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
    line: int,
    surface: str,
    structural_path: str,
    seam_class: str = "surviving_carrier_seam",
    status_hint: str = "",
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=line,
        surface=surface,
        structural_path=structural_path,
        seam_class=seam_class,
        status_hint=status_hint,
    )


def _governance_constant_touchsite(
    *,
    touchsite_id: str,
    qualname: str,
    line: int,
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path="src/gabion_governance/governance_audit_impl.py",
        qualname=qualname,
        boundary_name=qualname,
        line=line,
        node_kind="assign",
        surface="governance_registry_constant",
        structural_path=f"governance_registry_constant::{qualname}",
        seam_class="surviving_carrier_seam",
    )


def _wrapper_touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
    line: int,
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=line,
        node_kind="assign",
        surface="wrapper_export",
        structural_path=f"wrapper_export::{qualname}",
        seam_class="surviving_carrier_seam",
    )


def _static_touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
    line: int,
    node_kind: str,
    surface: str,
    structural_path: str,
    seam_class: str = "surviving_carrier_seam",
    status_hint: str = "",
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=line,
        node_kind=node_kind,
        surface=surface,
        structural_path=structural_path,
        seam_class=seam_class,
        status_hint=status_hint,
    )


def connectivity_synergy_workstream_registries() -> tuple[WorkstreamRegistry, ...]:
    idr_registry = WorkstreamRegistry(
        root=_root_definition(
            root_id="CSA-IDR",
            title="Identity/rendering separation and structural interning rollout",
            symbol=_csa_idr_root,
        ),
        subqueues=(
            _subqueue_definition(
                root_id="CSA-IDR",
                subqueue_id="CSA-IDR-SQ-001",
                title="Landed policy/workstream carrier-render separation tranche",
                touchpoint_ids=("CSA-IDR-TP-001",),
                symbol=_csa_idr_sq_policy_tranche,
                status_hint="landed",
            ),
            _subqueue_definition(
                root_id="CSA-IDR",
                subqueue_id="CSA-IDR-SQ-002",
                title="Call-cluster carrier/rendering rollout",
                touchpoint_ids=("CSA-IDR-TP-002",),
                symbol=_csa_idr_sq_call_cluster_rollout,
            ),
            _subqueue_definition(
                root_id="CSA-IDR",
                subqueue_id="CSA-IDR-SQ-003",
                title="Aggregate-test cleanup for rendering-sensitive assertions",
                touchpoint_ids=("CSA-IDR-TP-003",),
                symbol=_csa_idr_sq_aggregate_test_cleanup,
            ),
            _subqueue_definition(
                root_id="CSA-IDR",
                subqueue_id="CSA-IDR-SQ-004",
                title="Hierarchical identity grammar completion across scanner, hotspot queue, and planning chart",
                touchpoint_ids=("CSA-IDR-TP-004",),
                symbol=_csa_idr_sq_identity_grammar_completion,
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id="CSA-IDR",
                subqueue_id="CSA-IDR-SQ-001",
                touchpoint_id="CSA-IDR-TP-001",
                title="Policy/workstream typed identity and boundary renderer evidence",
                symbol=_csa_idr_tp_policy_tranche,
                status_hint="landed",
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-001",
                        rel_path="src/gabion/tooling/policy_substrate/policy_queue_identity.py",
                        qualname="PolicyQueueDecompositionIdentity.__str__",
                        line=69,
                        surface="identity_renderer",
                        structural_path="PolicyQueueDecompositionIdentity.__str__",
                        status_hint="landed",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-002",
                        rel_path="src/gabion/tooling/policy_substrate/policy_artifact_stream.py",
                        qualname="ArtifactSourceRef.__str__",
                        line=34,
                        surface="identity_renderer",
                        structural_path="ArtifactSourceRef.__str__",
                        status_hint="landed",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-003",
                        rel_path="src/gabion/tooling/policy_substrate/policy_artifact_stream.py",
                        qualname="ArtifactUnit.__str__",
                        line=71,
                        surface="identity_renderer",
                        structural_path="ArtifactUnit.__str__",
                        status_hint="landed",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IDR",
                subqueue_id="CSA-IDR-SQ-002",
                touchpoint_id="CSA-IDR-TP-002",
                title="Call-cluster summary and consolidation carrier/rendering surfaces",
                symbol=_csa_idr_tp_call_cluster_rollout,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-004",
                        rel_path="src/gabion/analysis/call_cluster/call_cluster_shared.py",
                        qualname="cluster_identity_from_key",
                        line=19,
                        surface="call_cluster_identity",
                        structural_path="cluster_identity_from_key",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-005",
                        rel_path="src/gabion/analysis/call_cluster/call_clusters.py",
                        qualname="build_call_clusters_payload",
                        line=70,
                        surface="call_cluster_boundary",
                        structural_path="build_call_clusters_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-006",
                        rel_path="src/gabion/analysis/call_cluster/call_clusters.py",
                        qualname="render_json_payload",
                        line=163,
                        surface="call_cluster_boundary",
                        structural_path="render_json_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-007",
                        rel_path="src/gabion/analysis/call_cluster/call_clusters.py",
                        qualname="render_markdown",
                        line=185,
                        surface="call_cluster_boundary",
                        structural_path="render_markdown",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-008",
                        rel_path="src/gabion/analysis/call_cluster/call_cluster_consolidation.py",
                        qualname="build_call_cluster_consolidation_payload",
                        line=94,
                        surface="call_cluster_boundary",
                        structural_path="build_call_cluster_consolidation_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-009",
                        rel_path="src/gabion/analysis/call_cluster/call_cluster_consolidation.py",
                        qualname="render_json_payload",
                        line=277,
                        surface="call_cluster_boundary",
                        structural_path="render_json_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-010",
                        rel_path="src/gabion/analysis/call_cluster/call_cluster_consolidation.py",
                        qualname="render_markdown",
                        line=323,
                        surface="call_cluster_boundary",
                        structural_path="render_markdown",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IDR",
                subqueue_id="CSA-IDR-SQ-003",
                touchpoint_id="CSA-IDR-TP-003",
                title="Aggregate tests that still mix composition and presentation assertions",
                symbol=_csa_idr_tp_aggregate_test_cleanup,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-011",
                        rel_path="tests/gabion/analysis/call_cluster/test_call_clusters.py",
                        qualname="test_call_clusters_emitted_payload_shape",
                        line=139,
                        surface="aggregate_test",
                        structural_path="test_call_clusters_emitted_payload_shape",
                        seam_class="surviving_test_seam",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-012",
                        rel_path="tests/gabion/analysis/call_cluster/test_call_cluster_consolidation.py",
                        qualname="test_call_cluster_consolidation_payload_and_render",
                        line=74,
                        surface="aggregate_test",
                        structural_path="test_call_cluster_consolidation_payload_and_render",
                        seam_class="surviving_test_seam",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IDR",
                subqueue_id="CSA-IDR-SQ-004",
                touchpoint_id="CSA-IDR-TP-004",
                title="Hierarchical identity grammar completion surfaces",
                symbol=_csa_idr_tp_identity_grammar_completion,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-013",
                        rel_path="scripts/policy/hotspot_neighborhood_queue.py",
                        qualname="_file_family_counts",
                        line=115,
                        surface="identity_grammar_completion",
                        structural_path="_file_family_counts",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-014",
                        rel_path="scripts/policy/hotspot_neighborhood_queue.py",
                        qualname="_file_ref",
                        line=319,
                        surface="identity_grammar_completion",
                        structural_path="_file_ref",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-015",
                        rel_path="scripts/policy/hotspot_neighborhood_queue.py",
                        qualname="_scope_ref",
                        line=205,
                        surface="identity_grammar_completion",
                        structural_path="_scope_ref",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-016",
                        rel_path=(
                            "src/gabion/tooling/policy_substrate/"
                            "planning_chart_identity.py"
                        ),
                        qualname="build_planning_chart_identity_grammar",
                        line=330,
                        surface="identity_grammar_completion",
                        structural_path="build_planning_chart_identity_grammar",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IDR-TS-017",
                        rel_path=(
                            "src/gabion/tooling/policy_substrate/"
                            "identity_zone/grammar.py"
                        ),
                        qualname="HierarchicalIdentityGrammar.add_two_cell",
                        line=453,
                        surface="identity_grammar_completion",
                        structural_path="HierarchicalIdentityGrammar.add_two_cell",
                    ),
                ),
            ),
        ),
        tags=("identity_rendering",),
    )
    igm_registry = WorkstreamRegistry(
        root=_root_definition(
            root_id="CSA-IGM",
            title="Ingress/merge witness, remap, and non-Python ASPF adapter rollout",
            symbol=_csa_igm_root,
        ),
        subqueues=(
            _subqueue_definition(
                root_id="CSA-IGM",
                subqueue_id="CSA-IGM-SQ-001",
                title="Cross-origin witness/remap contract and overlap ledger schema",
                touchpoint_ids=("CSA-IGM-TP-001",),
                symbol=_csa_igm_sq_witness_contract,
            ),
            _subqueue_definition(
                root_id="CSA-IGM",
                subqueue_id="CSA-IGM-SQ-002",
                title="Markdown/frontmatter/yaml ASPF ingress adapters",
                touchpoint_ids=("CSA-IGM-TP-002",),
                symbol=_csa_igm_sq_adapter_rollout,
            ),
            _subqueue_definition(
                root_id="CSA-IGM",
                subqueue_id="CSA-IGM-SQ-003",
                title="Merged remap/overlap parity artifacts and determinism checks",
                touchpoint_ids=("CSA-IGM-TP-003",),
                symbol=_csa_igm_sq_parity_artifacts,
            ),
            _subqueue_definition(
                root_id="CSA-IGM",
                subqueue_id="CSA-IGM-SQ-004",
                title="Structured XML/JSON artifact ingress adapters for test, perf, and planner sensors",
                touchpoint_ids=("CSA-IGM-TP-004",),
                symbol=_csa_igm_sq_structured_artifact_ingress,
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id="CSA-IGM",
                subqueue_id="CSA-IGM-SQ-001",
                touchpoint_id="CSA-IGM-TP-001",
                title="Cross-origin witness and overlap contract surfaces",
                symbol=_csa_igm_tp_witness_contract,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-001",
                        rel_path="src/gabion/server.py",
                        qualname="_analysis_manifest_digest_from_witness",
                        line=254,
                        surface="cross_origin_witness",
                        structural_path="_analysis_manifest_digest_from_witness",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-002",
                        rel_path="src/gabion/server.py",
                        qualname="_analysis_input_witness",
                        line=265,
                        surface="cross_origin_witness",
                        structural_path="_analysis_input_witness",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-003",
                        rel_path="src/gabion/tooling/policy_substrate/aspf_union_view.py",
                        qualname="build_aspf_union_view",
                        line=86,
                        surface="cross_origin_witness",
                        structural_path="build_aspf_union_view",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-004",
                        rel_path="src/gabion/tooling/policy_substrate/overlap_eval.py",
                        qualname="evaluate_condition_overlaps",
                        line=29,
                        surface="cross_origin_witness",
                        structural_path="evaluate_condition_overlaps",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-026",
                        rel_path="src/gabion/tooling/runtime/cross_origin_witness_artifact.py",
                        qualname="build_cross_origin_witness_contract_artifact_payload",
                        line=385,
                        surface="cross_origin_witness",
                        structural_path="build_cross_origin_witness_contract_artifact_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-027",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_cross_origin_witness_contract_artifact",
                        line=1805,
                        surface="cross_origin_witness",
                        structural_path="load_cross_origin_witness_contract_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-028",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="_write_cross_origin_witness_contract_artifact",
                        line=1819,
                        surface="cross_origin_witness",
                        structural_path="_write_cross_origin_witness_contract_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-029",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_cross_origin_witness_contract_artifact",
                        line=8934,
                        surface="cross_origin_witness",
                        structural_path="_join_cross_origin_witness_contract_artifact",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IGM",
                subqueue_id="CSA-IGM-SQ-002",
                touchpoint_id="CSA-IGM-TP-002",
                title="Markdown/frontmatter/yaml ingress adapter surfaces",
                symbol=_csa_igm_tp_adapter_rollout,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-023",
                        rel_path="src/gabion/frontmatter_ingress.py",
                        qualname="parse_frontmatter_document",
                        line=444,
                        surface="frontmatter_ingress",
                        structural_path="parse_frontmatter_document",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-005",
                        rel_path="src/gabion/frontmatter.py",
                        qualname="parse_strict_yaml_frontmatter",
                        line=11,
                        surface="frontmatter_ingress",
                        structural_path="parse_strict_yaml_frontmatter",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-006",
                        rel_path="src/gabion_governance/governance_audit_impl.py",
                        qualname="_frontmatter_block_from_text",
                        line=1821,
                        surface="frontmatter_ingress",
                        structural_path="_frontmatter_block_from_text",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-007",
                        rel_path="src/gabion_governance/governance_audit_impl.py",
                        qualname="_parse_frontmatter_with_mode",
                        line=1840,
                        surface="frontmatter_ingress",
                        structural_path="_parse_frontmatter_with_mode",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IGM",
                subqueue_id="CSA-IGM-SQ-003",
                touchpoint_id="CSA-IGM-TP-003",
                title="Overlap parity and deterministic merge evidence surfaces",
                symbol=_csa_igm_tp_parity_artifacts,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-019",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="build_ingress_merge_parity_artifact",
                        line=1117,
                        surface="merge_parity_carrier",
                        structural_path="build_ingress_merge_parity_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-020",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_ingress_merge_parity_artifact",
                        line=1182,
                        surface="merge_parity_carrier",
                        structural_path="load_ingress_merge_parity_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-021",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="_write_ingress_merge_parity_artifact",
                        line=1791,
                        surface="merge_parity_artifact_emitter",
                        structural_path="_write_ingress_merge_parity_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-022",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_ingress_merge_parity_artifact",
                        line=8932,
                        surface="merge_parity_graph_join",
                        structural_path="_join_ingress_merge_parity_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-008",
                        rel_path="tests/gabion/ingest/test_adapter_contract.py",
                        qualname="test_adapter_parity_on_overlapping_decision_surfaces",
                        line=68,
                        surface="merge_parity_test",
                        structural_path="test_adapter_parity_on_overlapping_decision_surfaces",
                        seam_class="surviving_test_seam",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-009",
                        rel_path="tests/test_policy_dsl.py",
                        qualname="test_registry_rejects_duplicate_rule_ids_across_yaml_and_markdown",
                        line=182,
                        surface="merge_parity_test",
                        structural_path="test_registry_rejects_duplicate_rule_ids_across_yaml_and_markdown",
                        seam_class="surviving_test_seam",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IGM",
                subqueue_id="CSA-IGM-SQ-004",
                touchpoint_id="CSA-IGM-TP-004",
                title="Structured XML/JSON sensor artifact ingress surfaces",
                symbol=_csa_igm_tp_structured_artifact_ingress,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-010",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_test_coverage",
                        line=8152,
                        surface="structured_artifact_ingress",
                        structural_path="_join_test_coverage",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-011",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_test_failures",
                        line=8283,
                        surface="structured_artifact_ingress",
                        structural_path="_join_test_failures",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-012",
                        rel_path="src/gabion/tooling/runtime/perf_artifact.py",
                        qualname="build_cprofile_perf_artifact_payload",
                        line=217,
                        surface="structured_artifact_ingress",
                        structural_path="build_cprofile_perf_artifact_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-013",
                        rel_path="scripts/ci/ci_observability_guard.py",
                        qualname="_parse_deadline_profile",
                        line=181,
                        surface="structured_artifact_ingress",
                        structural_path="_parse_deadline_profile",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-014",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_test_evidence_artifact",
                        line=667,
                        surface="structured_artifact_ingress",
                        structural_path="load_test_evidence_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-015",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_junit_failure_artifact",
                        line=740,
                        surface="structured_artifact_ingress",
                        structural_path="load_junit_failure_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-016",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_docflow_packet_enforcement_artifact",
                        line=811,
                        surface="structured_artifact_ingress",
                        structural_path="load_docflow_packet_enforcement_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-017",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_controller_drift_artifact",
                        line=922,
                        surface="structured_artifact_ingress",
                        structural_path="load_controller_drift_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-018",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_local_repro_closure_ledger_artifact",
                        line=987,
                        surface="structured_artifact_ingress",
                        structural_path="load_local_repro_closure_ledger_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-024",
                        rel_path="src/gabion/tooling/runtime/git_state_artifact.py",
                        qualname="build_git_state_artifact_payload",
                        line=117,
                        surface="structured_artifact_ingress",
                        structural_path="build_git_state_artifact_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IGM-TS-025",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_git_state_artifact",
                        line=1747,
                        surface="structured_artifact_ingress",
                        structural_path="load_git_state_artifact",
                    ),
                ),
            ),
        ),
        tags=("ingress_merge",),
    )
    rgc_registry = WorkstreamRegistry(
        root=_root_definition(
            root_id="CSA-RGC",
            title="Declarative registry convergence and wrapper/package inversion cleanup",
            symbol=_csa_rgc_root,
        ),
        subqueues=(
            _subqueue_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-001",
                title="Governance document inventory externalization from hardcoded Python constants",
                touchpoint_ids=("CSA-RGC-TP-001",),
                symbol=_csa_rgc_sq_governance_inventory,
            ),
            _subqueue_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-002",
                title="Declarative scanner registry plus package-native execution cleanup",
                touchpoint_ids=("CSA-RGC-TP-002",),
                symbol=_csa_rgc_sq_package_native_execution,
            ),
            _subqueue_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-003",
                title="Wrapper manifest and remaining package-to-scripts inversion cleanup",
                touchpoint_ids=("CSA-RGC-TP-003",),
                symbol=_csa_rgc_sq_wrapper_manifest,
            ),
            _subqueue_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-004",
                title="Shared doc/code/perf selector substrate convergence",
                touchpoint_ids=("CSA-RGC-TP-004",),
                symbol=_csa_rgc_sq_shared_query_substrate,
            ),
            _subqueue_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-005",
                title="Common governance graph substrate over SPPF, influence index, and inbox actions",
                touchpoint_ids=("CSA-RGC-TP-005",),
                symbol=_csa_rgc_sq_governance_graph_substrate,
            ),
            _subqueue_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-006",
                title="Governance/control-loop artifact graph convergence over planner, docflow, CI, and drift carriers",
                touchpoint_ids=("CSA-RGC-TP-006", "CSA-RGC-TP-007"),
                symbol=_csa_rgc_sq_control_loop_artifact_graph,
            ),
            _subqueue_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-007",
                title="TTL denotational-kernel VM alignment over ASPF, semantic fragment, lowering, and planning residues",
                touchpoint_ids=("CSA-RGC-TP-008",),
                symbol=_csa_rgc_sq_kernel_vm_alignment,
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-001",
                touchpoint_id="CSA-RGC-TP-001",
                title="Hardcoded governance inventory constants",
                symbol=_csa_rgc_tp_governance_inventory,
                declared_touchsites=(
                    _governance_constant_touchsite(
                        touchsite_id="CSA-RGC-TS-001",
                        qualname="CORE_GOVERNANCE_DOCS",
                        line=104,
                    ),
                    _governance_constant_touchsite(
                        touchsite_id="CSA-RGC-TS-002",
                        qualname="GOVERNANCE_DOCS",
                        line=112,
                    ),
                    _governance_constant_touchsite(
                        touchsite_id="CSA-RGC-TS-003",
                        qualname="REQUIRED_FIELDS",
                        line=171,
                    ),
                    _governance_constant_touchsite(
                        touchsite_id="CSA-RGC-TS-004",
                        qualname="LIST_FIELDS",
                        line=181,
                    ),
                    _governance_constant_touchsite(
                        touchsite_id="CSA-RGC-TS-005",
                        qualname="MAP_FIELDS",
                        line=188,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-002",
                touchpoint_id="CSA-RGC-TP-002",
                title="normative_symdiff package-native execution cleanup",
                symbol=_csa_rgc_tp_package_native_execution,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-006",
                        rel_path="src/gabion/tooling/governance/normative_symdiff.py",
                        qualname="_capture_policy_check",
                        line=504,
                        surface="package_script_inversion",
                        structural_path="_capture_policy_check",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-007",
                        rel_path="src/gabion/tooling/governance/normative_symdiff.py",
                        qualname="_collect_default_probes",
                        line=722,
                        surface="package_script_inversion",
                        structural_path="_collect_default_probes",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-003",
                touchpoint_id="CSA-RGC-TP-003",
                title="Remaining governance wrapper exports",
                symbol=_csa_rgc_tp_wrapper_manifest,
                declared_touchsites=(
                    _wrapper_touchsite(
                        touchsite_id="CSA-RGC-TS-008",
                        rel_path="src/gabion_governance/lint_summary_command.py",
                        qualname="run_lint_summary_cli",
                        line=5,
                    ),
                    _wrapper_touchsite(
                        touchsite_id="CSA-RGC-TS-009",
                        rel_path="src/gabion_governance/decision_tiers_command.py",
                        qualname="run_decision_tiers_cli",
                        line=5,
                    ),
                    _wrapper_touchsite(
                        touchsite_id="CSA-RGC-TS-010",
                        rel_path="src/gabion_governance/status_consistency_command.py",
                        qualname="run_status_consistency_cli",
                        line=5,
                    ),
                    _wrapper_touchsite(
                        touchsite_id="CSA-RGC-TS-011",
                        rel_path="src/gabion_governance/consolidation_command.py",
                        qualname="run_consolidation_cli",
                        line=5,
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-004",
                touchpoint_id="CSA-RGC-TP-004",
                title="Shared doc/code/perf selector and interning substrate surfaces",
                symbol=_csa_rgc_tp_shared_query_substrate,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-012",
                        rel_path="src/gabion/analysis/semantics/impact_index.py",
                        qualname="build_impact_index",
                        line=165,
                        surface="shared_query_substrate",
                        structural_path="build_impact_index",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-014",
                        rel_path="src/gabion/analysis/semantics/impact_index.py",
                        qualname="_links_from_doc",
                        line=397,
                        surface="shared_query_substrate",
                        structural_path="_links_from_doc",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-015",
                        rel_path="src/gabion/analysis/semantics/impact_index.py",
                        qualname="_collect_symbol_universe",
                        line=1017,
                        surface="shared_query_substrate",
                        structural_path="_collect_symbol_universe",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-016",
                        rel_path="src/gabion/tooling/runtime/invariant_graph.py",
                        qualname="_resolve_perf_dsl_overlay",
                        line=174,
                        surface="shared_query_boundary_adapter",
                        structural_path="_resolve_perf_dsl_overlay",
                        seam_class="surviving_boundary_adapter",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-005",
                touchpoint_id="CSA-RGC-TP-005",
                title="Common governance graph substrate surfaces",
                symbol=_csa_rgc_tp_governance_graph_substrate,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-017",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_sppf_dependency_graph",
                        line=8308,
                        surface="governance_graph_substrate",
                        structural_path="_join_sppf_dependency_graph",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-018",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_inbox_governance_docs",
                        line=8418,
                        surface="governance_graph_substrate",
                        structural_path="_join_inbox_governance_docs",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-019",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_governance_convergence_sources",
                        line=8522,
                        surface="governance_graph_substrate",
                        structural_path="_join_governance_convergence_sources",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-020",
                        rel_path="scripts/sppf/sppf_status_audit.py",
                        qualname="run_audit",
                        line=108,
                        surface="governance_status_bridge",
                        structural_path="run_audit",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-006",
                touchpoint_id="CSA-RGC-TP-006",
                title="Governance/control-loop artifact producer and convergence surfaces",
                symbol=_csa_rgc_tp_control_loop_artifact_graph,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-021",
                        rel_path="scripts/policy/docflow_packet_enforce.py",
                        qualname="main",
                        line=394,
                        surface="control_loop_artifact",
                        structural_path="docflow_packet_enforce::main",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-022",
                        rel_path="scripts/governance/governance_controller_audit.py",
                        qualname="main",
                        line=303,
                        surface="control_loop_artifact",
                        structural_path="governance_controller_audit::main",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-023",
                        rel_path="src/gabion/tooling/runtime/ci_watch.py",
                        qualname="run_watch",
                        line=474,
                        surface="control_loop_artifact",
                        structural_path="ci_watch::run_watch",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-024",
                        rel_path="scripts/policy/policy_scanner_suite.py",
                        qualname="main",
                        line=116,
                        surface="control_loop_artifact",
                        structural_path="policy_scanner_suite::main",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-025",
                        rel_path="scripts/policy/symbol_activity_audit.py",
                        qualname="main",
                        line=992,
                        surface="control_loop_artifact",
                        structural_path="symbol_activity_audit::main",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-026",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="_write_invariant_graph_artifact",
                        line=1791,
                        surface="control_loop_artifact",
                        structural_path="policy_check::_write_invariant_graph_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-027",
                        rel_path="src/gabion/plan.py",
                        qualname="write_execution_plan_artifact",
                        line=108,
                        surface="control_loop_artifact",
                        structural_path="plan::write_execution_plan_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-028",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="_write_git_state_artifact",
                        line=1811,
                        surface="control_loop_artifact",
                        structural_path="policy_check::_write_git_state_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-029",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_git_state_artifact",
                        line=9923,
                        surface="control_loop_artifact",
                        structural_path="invariant_graph::_join_git_state_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-038",
                        rel_path="src/gabion_governance/governance_audit_impl.py",
                        qualname="_emit_docflow_compliance",
                        line=2654,
                        surface="control_loop_artifact",
                        structural_path="governance_audit_impl::_emit_docflow_compliance",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-039",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_docflow_compliance_artifact",
                        line=1739,
                        surface="control_loop_artifact",
                        structural_path="structured_artifact_ingress::load_docflow_compliance_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-040",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_docflow_compliance_artifact",
                        line=9439,
                        surface="control_loop_artifact",
                        structural_path="invariant_graph::_join_docflow_compliance_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-044",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="_write_local_ci_repro_contract_artifact",
                        line=621,
                        surface="control_loop_artifact",
                        structural_path="policy_check::_write_local_ci_repro_contract_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-045",
                        rel_path="src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
                        qualname="load_local_ci_repro_contract_artifact",
                        line=2502,
                        surface="control_loop_artifact",
                        structural_path="structured_artifact_ingress::load_local_ci_repro_contract_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-046",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_local_ci_repro_contract_artifact",
                        line=10289,
                        surface="control_loop_artifact",
                        structural_path="invariant_graph::_join_local_ci_repro_contract_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-047",
                        rel_path="src/gabion/tooling/runtime/ci_local_repro.py",
                        qualname="main",
                        line=1127,
                        surface="control_loop_artifact",
                        structural_path="ci_local_repro::main",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-006",
                touchpoint_id="CSA-RGC-TP-007",
                title="Git/GH provenance and second-order governance-loop surfaces",
                symbol=_csa_rgc_tp_git_issue_provenance,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-030",
                        rel_path="src/gabion_governance/governance_audit_impl.py",
                        qualname="_sppf_sync_check",
                        line=4296,
                        surface="control_loop_provenance",
                        structural_path="_sppf_sync_check",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-031",
                        rel_path="src/gabion_governance/governance_audit_impl.py",
                        qualname="_evaluate_docflow_obligations",
                        line=4372,
                        surface="control_loop_provenance",
                        structural_path="_evaluate_docflow_obligations",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-032",
                        rel_path="src/gabion/tooling/sppf/sync_core.py",
                        qualname="_collect_commits",
                        line=91,
                        surface="control_loop_provenance",
                        structural_path="_collect_commits",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-033",
                        rel_path="src/gabion/tooling/sppf/sync_core.py",
                        qualname="_issue_ids_from_commits",
                        line=144,
                        surface="control_loop_provenance",
                        structural_path="_issue_ids_from_commits",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-041",
                        rel_path="src/gabion/tooling/sppf/sync_core.py",
                        qualname="_build_issue_link_facet",
                        line=154,
                        surface="control_loop_provenance",
                        structural_path="_build_issue_link_facet",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-043",
                        rel_path="src/gabion/tooling/sppf/sync_core.py",
                        qualname="_fetch_issue",
                        line=178,
                        surface="control_loop_provenance",
                        structural_path="_fetch_issue",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-034",
                        rel_path="src/gabion/tooling/sppf/sync_core.py",
                        qualname="_validate_issue_lifecycle",
                        line=240,
                        surface="control_loop_provenance",
                        structural_path="_validate_issue_lifecycle",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-035",
                        rel_path="src/gabion/tooling/sppf/sync_core.py",
                        qualname="_run_validate_mode",
                        line=337,
                        surface="control_loop_provenance",
                        structural_path="_run_validate_mode",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-036",
                        rel_path="src/gabion/analysis/semantics/obligation_registry.py",
                        qualname="evaluate_obligations",
                        line=64,
                        surface="control_loop_provenance",
                        structural_path="evaluate_obligations",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-037",
                        rel_path="src/gabion/execution_plan.py",
                        qualname="ExecutionPlan.with_issue_link",
                        line=62,
                        surface="control_loop_provenance",
                        structural_path="ExecutionPlan.with_issue_link",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-042",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_docflow_provenance_artifact",
                        line=9602,
                        surface="control_loop_provenance",
                        structural_path="_join_docflow_provenance_artifact",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-RGC",
                subqueue_id="CSA-RGC-SQ-007",
                touchpoint_id="CSA-RGC-TP-008",
                title="TTL kernel VM alignment and runtime realization surfaces",
                symbol=_csa_rgc_tp_kernel_vm_alignment,
                declared_touchsites=(
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-048",
                        rel_path="docs/ttl_kernel_semantics.md",
                        qualname="ttl_kernel_semantics",
                        line=1,
                        node_kind="document",
                        surface="kernel_vm_law_source",
                        structural_path="ttl_kernel_semantics",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-049",
                        rel_path="in/lg_kernel_ontology_cut_elim-1.ttl",
                        qualname="lg:AugmentedRule",
                        line=60,
                        node_kind="ttl_term",
                        surface="kernel_vm_law_source",
                        structural_path="lg:AugmentedRule",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-050",
                        rel_path="in/lg_kernel_ontology_cut_elim-1.ttl",
                        qualname="lg:RulePolarity",
                        line=351,
                        node_kind="ttl_term",
                        surface="kernel_vm_law_source",
                        structural_path="lg:RulePolarity",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-051",
                        rel_path="in/lg_kernel_ontology_cut_elim-1.ttl",
                        qualname="lg:ClosedRuleCell",
                        line=401,
                        node_kind="ttl_term",
                        surface="kernel_vm_law_source",
                        structural_path="lg:ClosedRuleCell",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-052",
                        rel_path="src/gabion/analysis/aspf/aspf_lattice_algebra.py",
                        qualname="NaturalityWitness",
                        line=273,
                        node_kind="class_def",
                        surface="kernel_vm_runtime_surface",
                        structural_path="NaturalityWitness",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-053",
                        rel_path="src/gabion/analysis/aspf/aspf_lattice_algebra.py",
                        qualname="FrontierWitness",
                        line=454,
                        node_kind="class_def",
                        surface="kernel_vm_runtime_surface",
                        structural_path="FrontierWitness",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-054",
                        rel_path="src/gabion/analysis/projection/semantic_fragment.py",
                        qualname="SemanticOpKind",
                        line=20,
                        node_kind="class_def",
                        surface="kernel_vm_runtime_surface",
                        structural_path="SemanticOpKind",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-055",
                        rel_path="src/gabion/analysis/projection/semantic_fragment.py",
                        qualname="CanonicalWitnessedSemanticRow",
                        line=46,
                        node_kind="class_def",
                        surface="kernel_vm_runtime_surface",
                        structural_path="CanonicalWitnessedSemanticRow",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-056",
                        rel_path="src/gabion/analysis/projection/semantic_fragment.py",
                        qualname="reflect_projection_fiber_witness",
                        line=70,
                        surface="kernel_vm_runtime_surface",
                        structural_path="reflect_projection_fiber_witness",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-057",
                        rel_path="src/gabion/analysis/projection/projection_semantic_lowering.py",
                        qualname="ProjectionSemanticLoweringPlan",
                        line=62,
                        node_kind="class_def",
                        surface="kernel_vm_runtime_surface",
                        structural_path="ProjectionSemanticLoweringPlan",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-058",
                        rel_path="src/gabion/analysis/projection/semantic_fragment_compile.py",
                        qualname="CompiledShaclPlan",
                        line=24,
                        node_kind="class_def",
                        surface="kernel_vm_runtime_surface",
                        structural_path="CompiledShaclPlan",
                    ),
                    _static_touchsite(
                        touchsite_id="CSA-RGC-TS-059",
                        rel_path="src/gabion/analysis/projection/semantic_fragment_compile.py",
                        qualname="CompiledSparqlPlan",
                        line=42,
                        node_kind="class_def",
                        surface="kernel_vm_runtime_surface",
                        structural_path="CompiledSparqlPlan",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-060",
                        rel_path=(
                            "src/gabion/tooling/policy_substrate/"
                            "lattice_convergence_semantic.py"
                        ),
                        qualname="materialize_semantic_lattice_convergence",
                        line=497,
                        surface="kernel_vm_runtime_surface",
                        structural_path="materialize_semantic_lattice_convergence",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-061",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="collect_aspf_lattice_convergence_result",
                        line=384,
                        surface="kernel_vm_runtime_surface",
                        structural_path="collect_aspf_lattice_convergence_result",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-062",
                        rel_path=(
                            "src/gabion/tooling/runtime/"
                            "kernel_vm_alignment_artifact.py"
                        ),
                        qualname="build_kernel_vm_alignment_artifact_payload",
                        line=749,
                        surface="kernel_vm_runtime_surface",
                        structural_path="build_kernel_vm_alignment_artifact_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-RGC-TS-063",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="_join_kernel_vm_alignment_artifact",
                        line=10520,
                        surface="kernel_vm_runtime_surface",
                        structural_path="_join_kernel_vm_alignment_artifact",
                    ),
                ),
            ),
        ),
        tags=("registry_convergence",),
    )
    ivl_registry = WorkstreamRegistry(
        root=_root_definition(
            root_id="CSA-IVL",
            title="Impact-velocity and planner turnaround optimization",
            symbol=_csa_ivl_root,
        ),
        subqueues=(
            _subqueue_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-001",
                title="Workflow convergence and invariant workstream materialization cost",
                touchpoint_ids=("CSA-IVL-TP-001",),
                symbol=_csa_ivl_sq_workstream_materialization,
            ),
            _subqueue_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-002",
                title="policy_check workflow artifact fanout tail",
                touchpoint_ids=("CSA-IVL-TP-002",),
                symbol=_csa_ivl_sq_policy_check_tail,
            ),
            _subqueue_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-003",
                title="Projection-semantic-fragment queue render tail",
                touchpoint_ids=("CSA-IVL-TP-003",),
                symbol=_csa_ivl_sq_queue_render_tail,
            ),
            _subqueue_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-004",
                title="Least-surprise artifact freshness and planner summary contract",
                touchpoint_ids=("CSA-IVL-TP-004",),
                symbol=_csa_ivl_sq_least_surprise_contract,
            ),
            _subqueue_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-005",
                title="Profiler artifact structural heat-map fiber integration",
                touchpoint_ids=("CSA-IVL-TP-005",),
                symbol=_csa_ivl_sq_perf_heat_fiber,
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-001",
                touchpoint_id="CSA-IVL-TP-001",
                title="Workflow convergence and invariant workstream build/write surfaces",
                symbol=_csa_ivl_tp_workstream_materialization,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-001",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="collect_aspf_lattice_convergence_result",
                        line=384,
                        surface="impact_velocity",
                        structural_path="collect_aspf_lattice_convergence_result",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-002",
                        rel_path=(
                            "src/gabion/tooling/policy_substrate/"
                            "lattice_convergence_semantic.py"
                        ),
                        qualname="materialize_semantic_lattice_convergence",
                        line=497,
                        surface="impact_velocity",
                        structural_path="materialize_semantic_lattice_convergence",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-019",
                        rel_path=(
                            "src/gabion/tooling/policy_substrate/"
                            "lattice_convergence_semantic.py"
                        ),
                        qualname="iter_semantic_lattice_convergence._events",
                        line=373,
                        surface="impact_velocity",
                        structural_path="iter_semantic_lattice_convergence::_events",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-020",
                        rel_path="src/gabion/analysis/aspf/aspf_lattice_algebra.py",
                        qualname="build_dataflow_fiber_bundle_for_qualname",
                        line=631,
                        surface="impact_velocity",
                        structural_path="build_dataflow_fiber_bundle_for_qualname",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-021",
                        rel_path="src/gabion/analysis/aspf/aspf_lattice_algebra.py",
                        qualname="build_fiber_bundle_for_qualname",
                        line=686,
                        surface="impact_velocity",
                        structural_path="build_fiber_bundle_for_qualname",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-022",
                        rel_path="src/gabion/analysis/aspf/aspf_lattice_algebra.py",
                        qualname="iter_lattice_witnesses._query",
                        line=944,
                        surface="impact_velocity",
                        structural_path="iter_lattice_witnesses::_query",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-023",
                        rel_path="src/gabion/analysis/aspf/aspf_lattice_algebra.py",
                        qualname="_module_bound_symbols",
                        line=2337,
                        surface="impact_velocity",
                        structural_path="_module_bound_symbols",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-024",
                        rel_path="src/gabion/analysis/aspf/aspf_lattice_algebra.py",
                        qualname="compute_lattice_witness",
                        line=901,
                        surface="impact_velocity",
                        structural_path="compute_lattice_witness",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-025",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="build_invariant_workstreams",
                        line=8308,
                        surface="impact_velocity",
                        structural_path="build_invariant_workstreams",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-026",
                        rel_path="src/gabion/tooling/policy_substrate/invariant_graph.py",
                        qualname="write_invariant_workstreams",
                        line=8784,
                        surface="impact_velocity",
                        structural_path="write_invariant_workstreams",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-002",
                touchpoint_id="CSA-IVL-TP-002",
                title="policy_check artifact fanout surfaces",
                symbol=_csa_ivl_tp_policy_check_tail,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-003",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="_write_projection_semantic_fragment_queue_artifacts",
                        line=1775,
                        surface="impact_velocity",
                        structural_path="_write_projection_semantic_fragment_queue_artifacts",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-004",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="_write_invariant_graph_artifact",
                        line=1788,
                        surface="impact_velocity",
                        structural_path="_write_invariant_graph_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-005",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="main",
                        line=1936,
                        surface="impact_velocity",
                        structural_path="main",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-003",
                touchpoint_id="CSA-IVL-TP-003",
                title="Projection-semantic-fragment queue render surfaces",
                symbol=_csa_ivl_tp_queue_render_tail,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-006",
                        rel_path="scripts/policy/projection_semantic_fragment_queue.py",
                        qualname="_phase5_structure",
                        line=475,
                        surface="impact_velocity",
                        structural_path="_phase5_structure",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-007",
                        rel_path="scripts/policy/projection_semantic_fragment_queue.py",
                        qualname="analyze",
                        line=697,
                        surface="impact_velocity",
                        structural_path="analyze",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-008",
                        rel_path="scripts/policy/projection_semantic_fragment_queue.py",
                        qualname="run",
                        line=1022,
                        surface="impact_velocity",
                        structural_path="run",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-004",
                touchpoint_id="CSA-IVL-TP-004",
                title="Planner summary and artifact freshness contract surfaces",
                symbol=_csa_ivl_tp_least_surprise_contract,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-009",
                        rel_path="src/gabion/tooling/runtime/invariant_graph.py",
                        qualname="_print_summary",
                        line=232,
                        surface="least_surprise",
                        structural_path="_print_summary",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-010",
                        rel_path="src/gabion/tooling/runtime/invariant_graph.py",
                        qualname="_ledger_projection_item",
                        line=1342,
                        surface="least_surprise",
                        structural_path="_ledger_projection_item",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-011",
                        rel_path="src/gabion/tooling/runtime/invariant_graph.py",
                        qualname="_print_workstream",
                        line=1359,
                        surface="least_surprise",
                        structural_path="_print_workstream",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-012",
                        rel_path="tests/gabion/tooling/runtime_policy/test_policy_check_output.py",
                        qualname=(
                            "test_policy_check_workflows_requires_output_to_emit_"
                            "invariant_artifacts"
                        ),
                        line=256,
                        surface="least_surprise_contract_test",
                        structural_path=(
                            "test_policy_check_workflows_requires_output_to_emit_"
                            "invariant_artifacts"
                        ),
                        seam_class="surviving_test_seam",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id="CSA-IVL",
                subqueue_id="CSA-IVL-SQ-005",
                touchpoint_id="CSA-IVL-TP-005",
                title="Profiler artifact normalization and structural heat-map surfaces",
                symbol=_csa_ivl_tp_perf_heat_fiber,
                declared_touchsites=(
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-013",
                        rel_path="src/gabion/tooling/runtime/invariant_graph.py",
                        qualname="_load_profile_observations",
                        line=244,
                        surface="perf_heat_fiber",
                        structural_path="_load_profile_observations",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-014",
                        rel_path="src/gabion/tooling/runtime/invariant_graph.py",
                        qualname="_print_perf_heat_map",
                        line=1997,
                        surface="perf_heat_fiber",
                        structural_path="_print_perf_heat_map",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-027",
                        rel_path="src/gabion/tooling/runtime/invariant_graph.py",
                        qualname="_resolve_perf_dsl_overlay",
                        line=202,
                        surface="perf_heat_shared_infimum",
                        structural_path="_resolve_perf_dsl_overlay",
                        seam_class="surviving_boundary_adapter",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-015",
                        rel_path="src/gabion/tooling/runtime/perf_artifact.py",
                        qualname="build_cprofile_perf_artifact_payload",
                        line=217,
                        surface="perf_heat_fiber",
                        structural_path="build_cprofile_perf_artifact_payload",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-016",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="main",
                        line=1936,
                        surface="perf_heat_producer",
                        structural_path="main::perf_artifact",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-017",
                        rel_path="tests/gabion/tooling/runtime_policy/test_invariant_graph.py",
                        qualname=(
                            "test_runtime_invariant_graph_cli_perf_heat_maps_"
                            "profile_artifacts"
                        ),
                        line=2281,
                        surface="perf_heat_contract_test",
                        structural_path=(
                            "test_runtime_invariant_graph_cli_perf_heat_maps_"
                            "profile_artifacts"
                        ),
                        seam_class="surviving_test_seam",
                    ),
                    _function_touchsite(
                        touchsite_id="CSA-IVL-TS-018",
                        rel_path="tests/gabion/tooling/runtime_policy/test_perf_artifact.py",
                        qualname=(
                            "test_build_cprofile_perf_artifact_payload_"
                            "enriches_structural_identity_from_graph"
                        ),
                        line=65,
                        surface="perf_heat_contract_test",
                        structural_path=(
                            "test_build_cprofile_perf_artifact_payload_"
                            "enriches_structural_identity_from_graph"
                        ),
                        seam_class="surviving_test_seam",
                    ),
                ),
            ),
        ),
        tags=("impact_velocity",),
    )
    return (idr_registry, igm_registry, ivl_registry, rgc_registry)


__all__ = ["connectivity_synergy_workstream_registries"]
