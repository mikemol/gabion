from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gabion.analysis.json_types import JSONObject
from gabion.order_contract import ordered_or_sorted


class RewritePlanKind(StrEnum):
    BUNDLE_ALIGN = "BUNDLE_ALIGN"
    CTOR_NORMALIZE = "CTOR_NORMALIZE"
    SURFACE_CANONICALIZE = "SURFACE_CANONICALIZE"
    AMBIENT_REWRITE = "AMBIENT_REWRITE"


@dataclass(frozen=True)
class RewritePlanSchema:
    kind: RewritePlanKind
    required_parameters: tuple[str, ...]
    required_evidence_refs: tuple[str, ...]
    required_predicates: tuple[str, ...]

    def as_json(self) -> JSONObject:
        return {
            "kind": self.kind.value,
            "required": {
                "parameters": list(self.required_parameters),
                "evidence_refs": list(self.required_evidence_refs),
                "verification_predicates": list(self.required_predicates),
            },
        }


_REWRITE_PLAN_SCHEMAS: dict[RewritePlanKind, RewritePlanSchema] = {
    RewritePlanKind.BUNDLE_ALIGN: RewritePlanSchema(
        kind=RewritePlanKind.BUNDLE_ALIGN,
        required_parameters=("candidates",),
        required_evidence_refs=("provenance_id", "coherence_id"),
        required_predicates=("base_conservation", "ctor_coherence", "match_strata", "remainder_non_regression"),
    ),
    RewritePlanKind.CTOR_NORMALIZE: RewritePlanSchema(
        kind=RewritePlanKind.CTOR_NORMALIZE,
        required_parameters=("target_ctor_keys", "candidates"),
        required_evidence_refs=("provenance_id", "coherence_id"),
        required_predicates=("base_conservation", "ctor_coherence", "match_strata", "remainder_non_regression"),
    ),
    RewritePlanKind.SURFACE_CANONICALIZE: RewritePlanSchema(
        kind=RewritePlanKind.SURFACE_CANONICALIZE,
        required_parameters=("canonical_candidate", "candidates"),
        required_evidence_refs=("provenance_id", "coherence_id"),
        required_predicates=("base_conservation", "match_strata", "remainder_non_regression"),
    ),
    RewritePlanKind.AMBIENT_REWRITE: RewritePlanSchema(
        kind=RewritePlanKind.AMBIENT_REWRITE,
        required_parameters=("strategy", "candidates"),
        required_evidence_refs=("provenance_id", "coherence_id"),
        required_predicates=("base_conservation", "match_strata", "remainder_non_regression"),
    ),
}


def rewrite_plan_schema(kind: str | RewritePlanKind) -> RewritePlanSchema | None:
    try:
        parsed = kind if isinstance(kind, RewritePlanKind) else RewritePlanKind(str(kind))
    except ValueError:
        return None
    return _REWRITE_PLAN_SCHEMAS.get(parsed)


def rewrite_plan_kind_sort_key(kind: str) -> int:
    order = {
        RewritePlanKind.BUNDLE_ALIGN.value: 0,
        RewritePlanKind.CTOR_NORMALIZE.value: 1,
        RewritePlanKind.SURFACE_CANONICALIZE.value: 2,
        RewritePlanKind.AMBIENT_REWRITE.value: 3,
    }
    return order.get(str(kind), 99)


def _missing_keys(container: object, required: tuple[str, ...]) -> list[str]:
    if not isinstance(container, dict):
        return list(required)
    missing: list[str] = []
    for key in required:
        value = container.get(key)
        if value in (None, "", []):
            missing.append(key)
    return missing


def validate_rewrite_plan_payload(plan: JSONObject) -> list[str]:
    rewrite = plan.get("rewrite")
    if not isinstance(rewrite, dict):
        return ["missing rewrite payload"]
    kind = str(rewrite.get("kind", ""))
    schema = rewrite_plan_schema(kind)
    if schema is None:
        return [f"unknown rewrite kind: {kind}"]

    issues: list[str] = []
    params = rewrite.get("parameters")
    missing_params = _missing_keys(params, schema.required_parameters)
    if missing_params:
        issues.append(f"missing parameters: {', '.join(missing_params)}")

    evidence = plan.get("evidence")
    missing_evidence = _missing_keys(evidence, schema.required_evidence_refs)
    if missing_evidence:
        issues.append(f"missing evidence refs: {', '.join(missing_evidence)}")

    verification = plan.get("verification")
    predicates = []
    if isinstance(verification, dict) and isinstance(verification.get("predicates"), list):
        predicates = [
            str(predicate.get("kind", ""))
            for predicate in verification.get("predicates", [])
            if isinstance(predicate, dict)
        ]
    for required in schema.required_predicates:
        if required not in predicates:
            issues.append(f"missing predicate: {required}")
    return issues


def normalize_rewrite_plan_order(plans: list[JSONObject]) -> list[JSONObject]:
    return ordered_or_sorted(
        plans,
        source="normalize_rewrite_plan_order",
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            rewrite_plan_kind_sort_key(str(entry.get("rewrite", {}).get("kind", ""))),
            str(entry.get("plan_id", "")),
        ),
    )


def attach_plan_schema(plan: JSONObject) -> JSONObject:
    rewrite = plan.get("rewrite")
    if not isinstance(rewrite, dict):
        return plan
    schema = rewrite_plan_schema(str(rewrite.get("kind", "")))
    if schema is None:
        return plan
    out = dict(plan)
    out["payload_schema"] = schema.as_json()
    return out
