from __future__ import annotations

import json
from dataclasses import dataclass
from functools import singledispatch
from typing import TypedDict

from gabion.analysis.kernel_vm.object_images import AugmentedRule
from gabion.invariants import grade_boundary, never
from gabion.analysis.projection.semantic_fragment import (
    CanonicalWitnessedSemanticRow,
    SemanticOpKind,
)


class CompiledShaclConstraint(TypedDict):
    constraint_id: str
    focus_path: str
    severity: str
    required: bool
    expected_value: str
    message: str


class CompiledShaclPlan(TypedDict):
    plan_id: str
    source_structural_identity: str
    source_site_identity: str
    surface: str
    semantic_op: str
    target_shape_id: str
    target_node_expr: str
    constraints: list[CompiledShaclConstraint]
    witness_trace: list[str]


class CompiledSparqlPattern(TypedDict):
    subject: str
    predicate: str
    object: str


class CompiledSparqlPlan(TypedDict):
    plan_id: str
    source_structural_identity: str
    source_site_identity: str
    surface: str
    semantic_op: str
    select_vars: list[str]
    where_patterns: list[CompiledSparqlPattern]
    anti_join_filters: list[str]
    witness_trace: list[str]


_AUGMENTED_RULE_OBJECT_IMAGE = AugmentedRule


@dataclass(frozen=True)
class _ProjectionFiberQuotientFaceFieldPlan:
    focus_path: str
    predicate: str
    expected_value: str
    select_var: str
    message: str

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_reflect_to_shacl",
)
def compile_projection_fiber_reflect_to_shacl(
    row: CanonicalWitnessedSemanticRow,
) -> CompiledShaclPlan:
    structural_identity = row["structural_identity"]
    payload = row["payload"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    return {
        "plan_id": f"{structural_identity}:shacl:reflect",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": row["surface"],
        "semantic_op": SemanticOpKind.REFLECT,
        "target_shape_id": f"shape:{structural_identity}",
        "target_node_expr": f"row:{structural_identity}",
        "constraints": [
            {
                "constraint_id": f"{structural_identity}:structural_path",
                "focus_path": "payload.structural_path",
                "severity": "Violation",
                "required": True,
                "expected_value": str(payload["structural_path"]),
                "message": "projection_fiber rows must preserve structural_path in canonical payload",
            },
            {
                "constraint_id": f"{structural_identity}:complete",
                "focus_path": "payload.complete",
                "severity": "Violation",
                "required": True,
                "expected_value": str(payload["complete"]).lower(),
                "message": "projection_fiber rows must preserve completeness state through reflection",
            },
            {
                "constraint_id": f"{structural_identity}:obligation_state",
                "focus_path": "obligation_state",
                "severity": "Violation",
                "required": True,
                "expected_value": row["obligation_state"],
                "message": "projection_fiber rows must preserve obligation state on the canonical carrier",
            },
        ],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_reflect_to_sparql",
)
def compile_projection_fiber_reflect_to_sparql(
    row: CanonicalWitnessedSemanticRow,
) -> CompiledSparqlPlan:
    structural_identity = row["structural_identity"]
    payload = row["payload"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    return {
        "plan_id": f"{structural_identity}:sparql:reflect",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": row["surface"],
        "semantic_op": SemanticOpKind.REFLECT,
        "select_vars": ["?branchSite", "?dataAnchorSite", "?execFrontierSite"],
        "where_patterns": [
            {
                "subject": "?branchSite",
                "predicate": "gabion:siteIdentity",
                "object": row["site_identity"],
            },
            {
                "subject": "?dataAnchorSite",
                "predicate": "gabion:siteIdentity",
                "object": str(payload["data_anchor_site_identity"]),
            },
            {
                "subject": "?execFrontierSite",
                "predicate": "gabion:siteIdentity",
                "object": str(payload["exec_frontier_site_identity"]),
            },
            {
                "subject": "?branchSite",
                "predicate": "gabion:structuralPath",
                "object": str(payload["structural_path"]),
            },
        ],
        "anti_join_filters": [
            f"NOT EXISTS unresolved obligation for {structural_identity}"
            if row["obligation_state"] == "discharged"
            else f"ALLOW unresolved obligation for {structural_identity}"
        ],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_support_reflect_to_shacl",
)
def compile_projection_fiber_support_reflect_to_shacl(
    row: CanonicalWitnessedSemanticRow,
) -> CompiledShaclPlan:
    structural_identity = row["structural_identity"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    input_witness_kinds = _distinct_mapping_values(row["input_witnesses"], key="kind")
    synthesized_witness_kinds = _distinct_mapping_values(
        row["synthesized_witnesses"], key="kind"
    )
    boundary_kinds = _distinct_mapping_values(
        row["boundary_trace"], key="boundary_kind"
    )
    return {
        "plan_id": f"{structural_identity}:shacl:support_reflect",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": row["surface"],
        "semantic_op": SemanticOpKind.SUPPORT_REFLECT,
        "target_shape_id": f"shape:{structural_identity}:support_reflect",
        "target_node_expr": f"row:{structural_identity}:support_reflect",
        "constraints": [
            {
                "constraint_id": f"{structural_identity}:support_reflect:input_witness_kinds",
                "focus_path": "input_witnesses.kind",
                "severity": "Violation",
                "required": True,
                "expected_value": _support_context_value(input_witness_kinds),
                "message": "projection_fiber support reflection must preserve input witness kinds",
            },
            {
                "constraint_id": f"{structural_identity}:support_reflect:synthesized_witness_kinds",
                "focus_path": "synthesized_witnesses.kind",
                "severity": "Violation",
                "required": True,
                "expected_value": _support_context_value(synthesized_witness_kinds),
                "message": "projection_fiber support reflection must preserve synthesized witness kinds",
            },
            {
                "constraint_id": f"{structural_identity}:support_reflect:boundary_kinds",
                "focus_path": "boundary_trace.boundary_kind",
                "severity": "Violation",
                "required": True,
                "expected_value": _support_context_value(boundary_kinds),
                "message": "projection_fiber support reflection must preserve boundary kinds",
            },
        ],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_support_reflect_to_sparql",
)
def compile_projection_fiber_support_reflect_to_sparql(
    row: CanonicalWitnessedSemanticRow,
) -> CompiledSparqlPlan:
    structural_identity = row["structural_identity"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    input_witness_kinds = _distinct_mapping_values(row["input_witnesses"], key="kind")
    synthesized_witness_kinds = _distinct_mapping_values(
        row["synthesized_witnesses"], key="kind"
    )
    boundary_kinds = _distinct_mapping_values(
        row["boundary_trace"], key="boundary_kind"
    )
    return {
        "plan_id": f"{structural_identity}:sparql:support_reflect",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": row["surface"],
        "semantic_op": SemanticOpKind.SUPPORT_REFLECT,
        "select_vars": [
            "?inputWitnessKinds",
            "?synthesizedWitnessKinds",
            "?boundaryKinds",
        ],
        "where_patterns": [
            {
                "subject": "?frontier",
                "predicate": "gabion:structuralIdentity",
                "object": structural_identity,
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:inputWitnessKinds",
                "object": _support_context_value(input_witness_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:synthesizedWitnessKinds",
                "object": _support_context_value(synthesized_witness_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:boundaryKinds",
                "object": _support_context_value(boundary_kinds),
            },
        ],
        "anti_join_filters": [
            f"NOT EXISTS missing support context for {structural_identity}"
        ],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_wedge_to_sparql",
)
def compile_projection_fiber_wedge_to_sparql(
    row: CanonicalWitnessedSemanticRow,
) -> CompiledSparqlPlan:
    structural_identity = row["structural_identity"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    input_witness_kinds = _distinct_mapping_values(row["input_witnesses"], key="kind")
    synthesized_witness_kinds = _distinct_mapping_values(
        row["synthesized_witnesses"], key="kind"
    )
    boundary_kinds = _distinct_mapping_values(
        row["boundary_trace"], key="boundary_kind"
    )
    transform_ops = _distinct_mapping_values(row["transform_trace"], key="op")
    return {
        "plan_id": f"{structural_identity}:sparql:wedge",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": row["surface"],
        "semantic_op": SemanticOpKind.WEDGE,
        "select_vars": [
            "?inputWitnessKinds",
            "?synthesizedWitnessKinds",
            "?boundaryKinds",
            "?transformOps",
        ],
        "where_patterns": [
            {
                "subject": "?frontier",
                "predicate": "gabion:structuralIdentity",
                "object": structural_identity,
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:inputWitnessKinds",
                "object": _support_context_value(input_witness_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:synthesizedWitnessKinds",
                "object": _support_context_value(synthesized_witness_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:boundaryKinds",
                "object": _support_context_value(boundary_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:transformOps",
                "object": _support_context_value(transform_ops),
            },
        ],
        "anti_join_filters": [
            f"NOT EXISTS missing wedge context for {structural_identity}"
        ],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_reindex_to_sparql",
)
def compile_projection_fiber_reindex_to_sparql(
    row: CanonicalWitnessedSemanticRow,
) -> CompiledSparqlPlan:
    structural_identity = row["structural_identity"]
    payload = row["payload"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    return {
        "plan_id": f"{structural_identity}:sparql:reindex",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": row["surface"],
        "semantic_op": SemanticOpKind.REINDEX,
        "select_vars": [
            "?siteIdentity",
            "?dataAnchorSiteIdentity",
            "?execFrontierSiteIdentity",
        ],
        "where_patterns": [
            {
                "subject": "?frontier",
                "predicate": "gabion:structuralIdentity",
                "object": structural_identity,
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:siteIdentity",
                "object": row["site_identity"],
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:dataAnchorSiteIdentity",
                "object": str(payload["data_anchor_site_identity"]),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:execFrontierSiteIdentity",
                "object": str(payload["exec_frontier_site_identity"]),
            },
        ],
        "anti_join_filters": [],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_existential_image_to_sparql",
)
def compile_projection_fiber_existential_image_to_sparql(
    row: CanonicalWitnessedSemanticRow,
) -> CompiledSparqlPlan:
    structural_identity = row["structural_identity"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    synthesized_witness_kinds = _distinct_mapping_values(
        row["synthesized_witnesses"], key="kind"
    )
    boundary_kinds = _distinct_mapping_values(
        row["boundary_trace"], key="boundary_kind"
    )
    return {
        "plan_id": f"{structural_identity}:sparql:existential_image",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": row["surface"],
        "semantic_op": SemanticOpKind.EXISTENTIAL_IMAGE,
        "select_vars": [
            "?synthesizedWitnessKinds",
            "?boundaryKinds",
            "?obligationState",
        ],
        "where_patterns": [
            {
                "subject": "?frontier",
                "predicate": "gabion:structuralIdentity",
                "object": structural_identity,
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:synthesizedWitnessKinds",
                "object": _support_context_value(synthesized_witness_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:boundaryKinds",
                "object": _support_context_value(boundary_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:obligationState",
                "object": row["obligation_state"],
            },
        ],
        "anti_join_filters": [],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_negate_to_sparql",
)
def compile_projection_fiber_negate_to_sparql(
    row: CanonicalWitnessedSemanticRow,
) -> CompiledSparqlPlan:
    structural_identity = row["structural_identity"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    synthesized_witness_kinds = _distinct_mapping_values(
        row["synthesized_witnesses"], key="kind"
    )
    boundary_kinds = _distinct_mapping_values(
        row["boundary_trace"], key="boundary_kind"
    )
    return {
        "plan_id": f"{structural_identity}:sparql:negate",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": row["surface"],
        "semantic_op": SemanticOpKind.NEGATE,
        "select_vars": [
            "?synthesizedWitnessKinds",
            "?boundaryKinds",
            "?obligationState",
        ],
        "where_patterns": [
            {
                "subject": "?frontier",
                "predicate": "gabion:structuralIdentity",
                "object": structural_identity,
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:synthesizedWitnessKinds",
                "object": _support_context_value(synthesized_witness_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:boundaryKinds",
                "object": _support_context_value(boundary_kinds),
            },
            {
                "subject": "?frontier",
                "predicate": "gabion:obligationState",
                "object": row["obligation_state"],
            },
        ],
        "anti_join_filters": [
            f"NOT EXISTS satisfied existential image for {structural_identity}"
        ],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_quotient_face_to_shacl",
)
def compile_projection_fiber_quotient_face_to_shacl(
    row: CanonicalWitnessedSemanticRow,
    *,
    quotient_face: str,
    fields: tuple[str, ...],
    spec_identity: str,
) -> CompiledShaclPlan:
    structural_identity = row["structural_identity"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    supported_fields = _PROJECTION_FIBER_QUOTIENT_FACE_FIELDS.get(quotient_face)
    if supported_fields is None:
        never(
            "unsupported quotient face for projection fiber compilation",
            quotient_face=quotient_face,
        )
    field_plans = tuple(
        (
            field,
            _projection_fiber_quotient_face_field_plan(
                row=row,
                quotient_face=quotient_face,
                field=field,
                supported_fields=supported_fields,
            ),
        )
        for field in fields
    )
    return {
        "plan_id": f"{spec_identity}:{structural_identity}:shacl:{quotient_face}",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": quotient_face,
        "semantic_op": SemanticOpKind.QUOTIENT_FACE,
        "target_shape_id": f"shape:{structural_identity}:{quotient_face}",
        "target_node_expr": f"row:{structural_identity}:{quotient_face}",
        "constraints": [
            {
                "constraint_id": f"{structural_identity}:{quotient_face}:{field}",
                "focus_path": plan.focus_path,
                "severity": "Violation",
                "required": True,
                "expected_value": plan.expected_value,
                "message": plan.message,
            }
            for field, plan in field_plans
        ],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.compile_projection_fiber_quotient_face_to_sparql",
)
def compile_projection_fiber_quotient_face_to_sparql(
    row: CanonicalWitnessedSemanticRow,
    *,
    quotient_face: str,
    fields: tuple[str, ...],
    spec_identity: str,
) -> CompiledSparqlPlan:
    structural_identity = row["structural_identity"]
    witness_trace = [
        *[str(item["kind"]) for item in row["input_witnesses"]],
        *[str(item["op"]) for item in row["transform_trace"]],
    ]
    supported_fields = _PROJECTION_FIBER_QUOTIENT_FACE_FIELDS.get(quotient_face)
    if supported_fields is None:
        never(
            "unsupported quotient face for projection fiber compilation",
            quotient_face=quotient_face,
        )
    field_plans = tuple(
        (
            field,
            _projection_fiber_quotient_face_field_plan(
                row=row,
                quotient_face=quotient_face,
                field=field,
                supported_fields=supported_fields,
            ),
        )
        for field in fields
    )
    return {
        "plan_id": f"{spec_identity}:{structural_identity}:sparql:{quotient_face}",
        "source_structural_identity": structural_identity,
        "source_site_identity": row["site_identity"],
        "surface": quotient_face,
        "semantic_op": SemanticOpKind.QUOTIENT_FACE,
        "select_vars": [plan.select_var for _, plan in field_plans],
        "where_patterns": [
            {
                "subject": "?frontier",
                "predicate": plan.predicate,
                "object": plan.expected_value,
            }
            for _, plan in field_plans
        ],
        "anti_join_filters": [
            f"NOT EXISTS missing quotient_face field for {structural_identity}"
        ],
        "witness_trace": witness_trace,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.distinct_mapping_values",
)
def _distinct_mapping_values(
    values: list[dict[str, object]],
    *,
    key: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                normalized
                for item in values
                if (normalized := _normalized_support_context_value(item.get(key))) != ""
            }
        )
    )

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.support_context_value",
)
def _support_context_value(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.normalized_support_context_value",
)
@singledispatch
def _normalized_support_context_value(value: object) -> str:
    return ""

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.normalized_support_context_value_from_str",
)
@_normalized_support_context_value.register(str)
def _normalized_support_context_value_from_str(value: str) -> str:
    return value.strip()


_PROJECTION_FIBER_QUOTIENT_FACE_FIELDS: dict[str, tuple[str, ...]] = {
    "projection_fiber.frontier": (
        "frontier_key",
        "projection_name",
        "structural_path",
        "complete",
        "data_anchor_site_identity",
        "exec_frontier_site_identity",
        "obligation_state",
    ),
    "projection_fiber.reflective_boundary": (
        "frontier_key",
        "structural_path",
        "data_anchor_site_identity",
        "exec_frontier_site_identity",
        "complete",
        "obligation_state",
    ),
}

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment_compile.projection_fiber_quotient_face_field_plan",
)
def _projection_fiber_quotient_face_field_plan(
    *,
    row: CanonicalWitnessedSemanticRow,
    quotient_face: str,
    field: str,
    supported_fields: tuple[str, ...],
) -> _ProjectionFiberQuotientFaceFieldPlan:
    payload = row["payload"]
    if field not in supported_fields:
        never(
            "unsupported quotient-face field for projection fiber compilation",
            quotient_face=quotient_face,
            field=field,
        )
    if field == "frontier_key":
        return _ProjectionFiberQuotientFaceFieldPlan(
            focus_path="structural_identity",
            predicate="gabion:structuralIdentity",
            expected_value=row["structural_identity"],
            select_var="?frontier_key",
            message=f"{quotient_face} must preserve structural identity as frontier key",
        )
    if field == "projection_name":
        return _ProjectionFiberQuotientFaceFieldPlan(
            focus_path="surface",
            predicate="gabion:projectionName",
            expected_value=row["surface"],
            select_var="?projection_name",
            message=f"{quotient_face} must preserve the projection surface name",
        )
    if field == "structural_path":
        return _ProjectionFiberQuotientFaceFieldPlan(
            focus_path="payload.structural_path",
            predicate="gabion:structuralPath",
            expected_value=str(payload["structural_path"]),
            select_var="?structural_path",
            message=f"{quotient_face} must preserve structural_path in quotient-face payload",
        )
    if field == "complete":
        return _ProjectionFiberQuotientFaceFieldPlan(
            focus_path="payload.complete",
            predicate="gabion:complete",
            expected_value=str(payload["complete"]).lower(),
            select_var="?complete",
            message=f"{quotient_face} must preserve completeness state",
        )
    if field == "data_anchor_site_identity":
        return _ProjectionFiberQuotientFaceFieldPlan(
            focus_path="payload.data_anchor_site_identity",
            predicate="gabion:dataAnchorSiteIdentity",
            expected_value=str(payload["data_anchor_site_identity"]),
            select_var="?data_anchor_site_identity",
            message=f"{quotient_face} must preserve data-anchor site identity",
        )
    if field == "exec_frontier_site_identity":
        return _ProjectionFiberQuotientFaceFieldPlan(
            focus_path="payload.exec_frontier_site_identity",
            predicate="gabion:execFrontierSiteIdentity",
            expected_value=str(payload["exec_frontier_site_identity"]),
            select_var="?exec_frontier_site_identity",
            message=f"{quotient_face} must preserve execution-frontier site identity",
        )
    if field == "obligation_state":
        return _ProjectionFiberQuotientFaceFieldPlan(
            focus_path="obligation_state",
            predicate="gabion:obligationState",
            expected_value=row["obligation_state"],
            select_var="?obligation_state",
            message=f"{quotient_face} must preserve obligation_state on the quotient face",
        )
    never(
        "unsupported quotient-face field for projection fiber compilation",
        quotient_face=quotient_face,
        field=field,
    )


__all__ = [
    "CompiledShaclConstraint",
    "CompiledShaclPlan",
    "CompiledSparqlPattern",
    "CompiledSparqlPlan",
    "compile_projection_fiber_quotient_face_to_shacl",
    "compile_projection_fiber_quotient_face_to_sparql",
    "compile_projection_fiber_existential_image_to_sparql",
    "compile_projection_fiber_negate_to_sparql",
    "compile_projection_fiber_reindex_to_sparql",
    "compile_projection_fiber_reflect_to_shacl",
    "compile_projection_fiber_reflect_to_sparql",
    "compile_projection_fiber_support_reflect_to_shacl",
    "compile_projection_fiber_support_reflect_to_sparql",
    "compile_projection_fiber_wedge_to_sparql",
]
