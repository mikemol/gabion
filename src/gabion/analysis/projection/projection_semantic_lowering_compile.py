# gabion:grade_boundary kind=semantic_carrier_adapter name=projection_semantic_lowering_compile
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.projection.projection_semantic_lowering import (
    ProjectionSemanticLoweringPlan,
    SemanticProjectionOp,
    SemanticProjectionKind,
)
from gabion.analysis.projection.semantic_fragment import (
    CanonicalWitnessedSemanticRow,
)
from gabion.analysis.projection.semantic_fragment_compile import (
    CompiledShaclPlan,
    CompiledSparqlPlan,
    compile_projection_fiber_existential_image_to_sparql,
    compile_projection_fiber_negate_to_sparql,
    compile_projection_fiber_reflect_to_shacl,
    compile_projection_fiber_reflect_to_sparql,
    compile_projection_fiber_quotient_face_to_shacl,
    compile_projection_fiber_quotient_face_to_sparql,
    compile_projection_fiber_reindex_to_sparql,
    compile_projection_fiber_support_reflect_to_shacl,
    compile_projection_fiber_support_reflect_to_sparql,
    compile_projection_fiber_wedge_to_sparql,
)
from gabion.invariants import never
from gabion.json_types import JSONValue
from gabion.order_contract import OrderPolicy, sort_once


@dataclass(frozen=True)
class CompiledProjectionSemanticBinding:
    source_index: int
    source_op: str
    semantic_op: str
    quotient_face: str
    source_structural_identity: str
    shacl_plan_id: str
    sparql_plan_id: str


@dataclass(frozen=True)
class ProjectionSemanticCompiledPlanBundle:
    spec_identity: str
    spec_name: str
    domain: str
    bindings: tuple[CompiledProjectionSemanticBinding, ...] = ()
    compiled_shacl_plans: tuple[CompiledShaclPlan, ...] = ()
    compiled_sparql_plans: tuple[CompiledSparqlPlan, ...] = ()

    def policy_data(self) -> dict[str, object]:
        return {
            "spec_identity": self.spec_identity,
            "spec_name": self.spec_name,
            "domain": self.domain,
            "bindings": [
                {
                    "source_index": item.source_index,
                    "source_op": item.source_op,
                    "semantic_op": item.semantic_op,
                    "quotient_face": item.quotient_face,
                    "source_structural_identity": item.source_structural_identity,
                    "shacl_plan_id": item.shacl_plan_id,
                    "sparql_plan_id": item.sparql_plan_id,
                }
                for item in self.bindings
            ],
            "compiled_shacl_plans": [item for item in self.compiled_shacl_plans],
            "compiled_sparql_plans": [item for item in self.compiled_sparql_plans],
        }


def compile_projection_semantic_lowering_plan(
    lowering_plan: ProjectionSemanticLoweringPlan,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> ProjectionSemanticCompiledPlanBundle:
    check_deadline()
    bindings: list[CompiledProjectionSemanticBinding] = []
    shacl_plans: list[CompiledShaclPlan] = []
    sparql_plans: list[CompiledSparqlPlan] = []
    for semantic_op in lowering_plan.semantic_ops:
        check_deadline()
        op_bindings, op_shacl_plans, op_sparql_plans = _compile_semantic_projection_op(
            lowering_plan=lowering_plan,
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
        bindings.extend(op_bindings)
        shacl_plans.extend(op_shacl_plans)
        sparql_plans.extend(op_sparql_plans)
    return ProjectionSemanticCompiledPlanBundle(
        spec_identity=lowering_plan.spec_identity,
        spec_name=lowering_plan.spec_name,
        domain=lowering_plan.domain,
        bindings=tuple(
            sort_once(
                bindings,
                source="compile_projection_semantic_lowering_plan.bindings",
                policy=OrderPolicy.SORT,
                key=lambda item: (
                    item.source_index,
                    item.source_structural_identity,
                    item.shacl_plan_id,
                ),
            )
        ),
        compiled_shacl_plans=tuple(
            sort_once(
                shacl_plans,
                source="compile_projection_semantic_lowering_plan.shacl_plans",
                policy=OrderPolicy.SORT,
                key=lambda item: (
                    item["source_structural_identity"],
                    item["plan_id"],
                ),
            )
        ),
        compiled_sparql_plans=tuple(
            sort_once(
                sparql_plans,
                source="compile_projection_semantic_lowering_plan.sparql_plans",
                policy=OrderPolicy.SORT,
                key=lambda item: (
                    item["source_structural_identity"],
                    item["plan_id"],
                ),
            )
        ),
    )


def _compile_semantic_projection_op(
    *,
    lowering_plan: ProjectionSemanticLoweringPlan,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    if semantic_op.semantic_op is SemanticProjectionKind.QUOTIENT_FACE:
        return _compile_quotient_face_semantic_op(
            lowering_plan=lowering_plan,
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
    if semantic_op.semantic_op is SemanticProjectionKind.REFLECT:
        return _compile_reflect_semantic_op(
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
    if semantic_op.semantic_op is SemanticProjectionKind.SYNTHESIZE_WITNESS:
        return _compile_synthesize_witness_semantic_op(
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
    if semantic_op.semantic_op is SemanticProjectionKind.SUPPORT_REFLECT:
        return _compile_support_reflect_semantic_op(
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
    if semantic_op.semantic_op is SemanticProjectionKind.WEDGE:
        return _compile_wedge_semantic_op(
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
    if semantic_op.semantic_op is SemanticProjectionKind.REINDEX:
        return _compile_reindex_semantic_op(
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
    if semantic_op.semantic_op is SemanticProjectionKind.EXISTENTIAL_IMAGE:
        return _compile_existential_image_semantic_op(
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
    if semantic_op.semantic_op is SemanticProjectionKind.NEGATE:
        return _compile_negate_semantic_op(
            semantic_op=semantic_op,
            semantic_rows=semantic_rows,
        )
    never(
        "unsupported semantic projection op for lowering compilation",
        semantic_op=semantic_op.semantic_op.value,
    )


def _compile_quotient_face_semantic_op(
    *,
    lowering_plan: ProjectionSemanticLoweringPlan,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    quotient_face = _required_quotient_face(semantic_op.params)
    fields = _required_field_tuple(semantic_op.params)
    bindings: list[CompiledProjectionSemanticBinding] = []
    shacl_plans: list[CompiledShaclPlan] = []
    sparql_plans: list[CompiledSparqlPlan] = []
    for row in _semantic_rows_for_quotient_face(
        quotient_face=quotient_face,
        semantic_rows=semantic_rows,
    ):
        check_deadline()
        shacl_plan = compile_projection_fiber_quotient_face_to_shacl(
            row,
            quotient_face=quotient_face,
            fields=fields,
            spec_identity=lowering_plan.spec_identity,
        )
        sparql_plan = compile_projection_fiber_quotient_face_to_sparql(
            row,
            quotient_face=quotient_face,
            fields=fields,
            spec_identity=lowering_plan.spec_identity,
        )
        shacl_plans.append(shacl_plan)
        sparql_plans.append(sparql_plan)
        bindings.append(
            CompiledProjectionSemanticBinding(
                source_index=semantic_op.source_index,
                source_op=semantic_op.source_op,
                semantic_op=semantic_op.semantic_op.value,
                quotient_face=quotient_face,
                source_structural_identity=row["structural_identity"],
                shacl_plan_id=shacl_plan["plan_id"],
                sparql_plan_id=sparql_plan["plan_id"],
            )
        )
    return tuple(bindings), tuple(shacl_plans), tuple(sparql_plans)


def _compile_reflect_semantic_op(
    *,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    surface = _required_surface(semantic_op.params)
    shacl_plans: list[CompiledShaclPlan] = []
    sparql_plans: list[CompiledSparqlPlan] = []
    for row in _semantic_rows_for_surface(surface=surface, semantic_rows=semantic_rows):
        check_deadline()
        shacl_plans.append(compile_projection_fiber_reflect_to_shacl(row))
        sparql_plans.append(compile_projection_fiber_reflect_to_sparql(row))
    return (), tuple(shacl_plans), tuple(sparql_plans)


def _compile_support_reflect_semantic_op(
    *,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    surface = _required_surface(semantic_op.params)
    shacl_plans: list[CompiledShaclPlan] = []
    sparql_plans: list[CompiledSparqlPlan] = []
    for row in _semantic_rows_for_surface(surface=surface, semantic_rows=semantic_rows):
        check_deadline()
        shacl_plans.append(compile_projection_fiber_support_reflect_to_shacl(row))
        sparql_plans.append(compile_projection_fiber_support_reflect_to_sparql(row))
    return (), tuple(shacl_plans), tuple(sparql_plans)


def _compile_wedge_semantic_op(
    *,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    surface = _required_surface(semantic_op.params)
    sparql_plans: list[CompiledSparqlPlan] = []
    for row in _semantic_rows_for_surface(surface=surface, semantic_rows=semantic_rows):
        check_deadline()
        sparql_plans.append(compile_projection_fiber_wedge_to_sparql(row))
    return (), (), tuple(sparql_plans)


def _compile_reindex_semantic_op(
    *,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    surface = _required_surface(semantic_op.params)
    sparql_plans: list[CompiledSparqlPlan] = []
    for row in _semantic_rows_for_surface(surface=surface, semantic_rows=semantic_rows):
        check_deadline()
        sparql_plans.append(compile_projection_fiber_reindex_to_sparql(row))
    return (), (), tuple(sparql_plans)


def _compile_existential_image_semantic_op(
    *,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    surface = _required_surface(semantic_op.params)
    sparql_plans: list[CompiledSparqlPlan] = []
    for row in _semantic_rows_for_surface(surface=surface, semantic_rows=semantic_rows):
        check_deadline()
        sparql_plans.append(
            compile_projection_fiber_existential_image_to_sparql(row)
        )
    return (), (), tuple(sparql_plans)


def _compile_synthesize_witness_semantic_op(
    *,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    surface = _required_surface(semantic_op.params)
    # Witness synthesis remains semantic-core-only in v1: the compiler
    # acknowledges the op at the typed lowering boundary but does not
    # materialize executable SHACL/SPARQL ownership for witness invention.
    _semantic_rows_for_surface(surface=surface, semantic_rows=semantic_rows)
    return (), (), ()


def _compile_negate_semantic_op(
    *,
    semantic_op: SemanticProjectionOp,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[
    tuple[CompiledProjectionSemanticBinding, ...],
    tuple[CompiledShaclPlan, ...],
    tuple[CompiledSparqlPlan, ...],
]:
    surface = _required_surface(semantic_op.params)
    sparql_plans: list[CompiledSparqlPlan] = []
    for row in _semantic_rows_for_surface(surface=surface, semantic_rows=semantic_rows):
        check_deadline()
        sparql_plans.append(compile_projection_fiber_negate_to_sparql(row))
    return (), (), tuple(sparql_plans)


def _required_quotient_face(params: dict[str, object]) -> str:
    quotient_face = params.get("quotient_face")
    if quotient_face is None:
        never("semantic projection op missing quotient_face")
    return _required_quotient_face_payload(quotient_face)


@singledispatch
def _required_quotient_face_payload(value: object) -> str:
    never(
        "unsupported quotient_face payload",
        value_type=type(value).__name__,
    )


@_required_quotient_face_payload.register(str)
def _required_quotient_face_from_str(value: str) -> str:
    if value:
        return value
    never("semantic projection op missing quotient_face")


def _required_field_tuple(params: dict[str, object]) -> tuple[str, ...]:
    fields = params.get("fields")
    if fields is None:
        never("semantic projection op missing supported field list")
    return _required_field_tuple_payload(fields)


@singledispatch
def _required_field_tuple_payload(value: object) -> tuple[str, ...]:
    never(
        "unsupported field payload",
        value_type=type(value).__name__,
    )


@_required_field_tuple_payload.register(list)
def _required_field_tuple_from_list(value: list[JSONValue]) -> tuple[str, ...]:
    return _field_tuple_from_sequence(tuple(value))


@_required_field_tuple_payload.register(tuple)
def _required_field_tuple_from_tuple(value: tuple[object, ...]) -> tuple[str, ...]:
    return _field_tuple_from_sequence(value)


def _field_tuple_from_sequence(values: tuple[object, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        check_deadline()
        normalized.append(_field_name_payload(value))
    return tuple(normalized)


@singledispatch
def _field_name_payload(value: object) -> str:
    never(
        "unsupported semantic projection field value",
        value_type=type(value).__name__,
    )


@_field_name_payload.register(str)
def _field_name_from_str(value: str) -> str:
    if value:
        return value
    never("semantic projection field value must be non-empty")


def _required_surface(params: dict[str, object]) -> str:
    surface = params.get("surface")
    if surface is None:
        never("semantic projection op missing surface")
    return _required_surface_payload(surface)


@singledispatch
def _required_surface_payload(value: object) -> str:
    never(
        "unsupported surface payload",
        value_type=type(value).__name__,
    )


@_required_surface_payload.register(str)
def _required_surface_from_str(value: str) -> str:
    if value:
        return value
    never("semantic projection op missing surface")


def _semantic_rows_for_quotient_face(
    *,
    quotient_face: str,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[CanonicalWitnessedSemanticRow, ...]:
    if quotient_face in {
        "projection_fiber.frontier",
        "projection_fiber.reflective_boundary",
    }:
        return tuple(
            sort_once(
                [
                    row
                    for row in semantic_rows
                    if row["surface"] == "projection_fiber"
                ],
                source="compile_projection_semantic_lowering_plan.projection_fiber_rows",
                policy=OrderPolicy.SORT,
                key=lambda row: row["structural_identity"],
            )
        )
    never(
        "unsupported quotient face for semantic lowering compilation",
        quotient_face=quotient_face,
    )


def _semantic_rows_for_surface(
    *,
    surface: str,
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...],
) -> tuple[CanonicalWitnessedSemanticRow, ...]:
    if surface == "projection_fiber":
        return tuple(
            sort_once(
                [
                    row
                    for row in semantic_rows
                    if row["surface"] == surface
                ],
                source="compile_projection_semantic_lowering_plan.surface_rows",
                policy=OrderPolicy.SORT,
                key=lambda row: row["structural_identity"],
            )
        )
    never(
        "unsupported semantic surface for lowering compilation",
        surface=surface,
    )


__all__ = [
    "CompiledProjectionSemanticBinding",
    "ProjectionSemanticCompiledPlanBundle",
    "compile_projection_semantic_lowering_plan",
]
