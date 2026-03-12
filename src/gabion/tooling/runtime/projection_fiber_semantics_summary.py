from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectionFiberSemanticPreview:
    spec_name: str
    quotient_face: str
    source_structural_identity: str
    path: str
    qualname: str
    structural_path: str
    obligation_state: str
    complete: bool

    def as_payload(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "quotient_face": self.quotient_face,
            "source_structural_identity": self.source_structural_identity,
            "path": self.path,
            "qualname": self.qualname,
            "structural_path": self.structural_path,
            "obligation_state": self.obligation_state,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class ProjectionFiberSemanticsSummary:
    decision: dict[str, Any]
    semantic_row_count: int
    compiled_projection_semantic_bundle_count: int
    compiled_projection_semantic_spec_names: tuple[str, ...]
    semantic_previews: tuple[ProjectionFiberSemanticPreview, ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "semantic_row_count": self.semantic_row_count,
            "compiled_projection_semantic_bundle_count": (
                self.compiled_projection_semantic_bundle_count
            ),
            "compiled_projection_semantic_spec_names": list(
                self.compiled_projection_semantic_spec_names
            ),
            "semantic_previews": [
                item.as_payload() for item in self.semantic_previews
            ],
        }

def projection_fiber_semantics_summary_from_payload(
    payload: object,
) -> ProjectionFiberSemanticsSummary | None:
    payload_mapping = _mapping(payload)
    if not payload_mapping:
        return None
    semantics_mapping = _mapping(payload_mapping.get("projection_fiber_semantics"))
    if semantics_mapping:
        return _summary_from_semantics_mapping(semantics_mapping)
    policy_results_mapping = _mapping(payload_mapping.get("policy_results"))
    if policy_results_mapping:
        return projection_fiber_semantics_summary_from_payload(
            policy_results_mapping.get("policy_check")
        )
    return None


def _summary_from_semantics_mapping(
    semantics_mapping: Mapping[str, object],
) -> ProjectionFiberSemanticsSummary | None:
    decision_mapping = _mapping(semantics_mapping.get("decision"))
    report_mapping = _mapping(semantics_mapping.get("report"))
    if not decision_mapping or not report_mapping:
        return None
    semantic_rows = _list_of_mappings(report_mapping.get("semantic_rows"))
    compiled_projection_semantic_bundles = _list_of_mappings(
        report_mapping.get("compiled_projection_semantic_bundles")
    )
    spec_names = tuple(
        sorted(
            {
                item
                for item in (
                    _projection_semantic_spec_name(bundle)
                    for bundle in compiled_projection_semantic_bundles
                )
                if item is not None
            }
        )
    )
    return ProjectionFiberSemanticsSummary(
        decision=dict(decision_mapping.items()),
        semantic_row_count=len(semantic_rows),
        compiled_projection_semantic_bundle_count=len(
            compiled_projection_semantic_bundles
        ),
        compiled_projection_semantic_spec_names=spec_names,
        semantic_previews=_projection_fiber_semantic_previews(
            semantic_rows=semantic_rows,
            compiled_projection_semantic_bundles=compiled_projection_semantic_bundles,
        ),
    )


def _projection_fiber_semantic_previews(
    *,
    semantic_rows: tuple[dict[str, Any], ...],
    compiled_projection_semantic_bundles: tuple[dict[str, Any], ...],
) -> tuple[ProjectionFiberSemanticPreview, ...]:
    rows_by_structural_identity = _semantic_rows_by_structural_identity(semantic_rows)
    previews: list[ProjectionFiberSemanticPreview] = []
    seen: set[tuple[str, str, str]] = set()
    for bundle_mapping in compiled_projection_semantic_bundles:
        spec_name = _projection_semantic_spec_name(bundle_mapping)
        if spec_name is None:
            continue
        bindings = bundle_mapping.get("bindings")
        if not isinstance(bindings, list):
            continue
        for binding in bindings:
            binding_mapping = _mapping(binding)
            if not binding_mapping:
                continue
            quotient_face = _required_mapping_string(
                binding_mapping,
                key="quotient_face",
            )
            source_structural_identity = _required_mapping_string(
                binding_mapping,
                key="source_structural_identity",
            )
            if quotient_face is None or source_structural_identity is None:
                continue
            dedupe_key = (
                spec_name,
                quotient_face,
                source_structural_identity,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            row_mapping = rows_by_structural_identity.get(source_structural_identity, {})
            row_payload = _mapping(row_mapping.get("payload"))
            previews.append(
                ProjectionFiberSemanticPreview(
                    spec_name=spec_name,
                    quotient_face=quotient_face,
                    source_structural_identity=source_structural_identity,
                    path=_mapping_string(row_payload, key="path"),
                    qualname=_mapping_string(row_payload, key="qualname"),
                    structural_path=_mapping_string(row_payload, key="structural_path"),
                    obligation_state=_mapping_string(
                        row_mapping,
                        key="obligation_state",
                    ),
                    complete=_mapping_bool(row_payload, key="complete"),
                )
            )
    return tuple(
        sorted(
            previews,
            key=lambda item: (
                item.spec_name,
                item.quotient_face,
                item.path,
                item.qualname,
                item.structural_path,
                item.source_structural_identity,
            ),
        )
    )


def _semantic_rows_by_structural_identity(
    semantic_rows: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row_mapping in semantic_rows:
        structural_identity = _required_mapping_string(
            row_mapping,
            key="structural_identity",
        )
        if structural_identity is None:
            continue
        rows[structural_identity] = row_mapping
    return rows


def _projection_semantic_spec_name(value: object) -> str | None:
    mapping = _mapping(value)
    if not mapping:
        return None
    spec_name = mapping.get("spec_name")
    if not isinstance(spec_name, str):
        return None
    normalized = spec_name.strip()
    if not normalized:
        return None
    return normalized


def _required_mapping_string(
    mapping: Mapping[str, Any],
    *,
    key: str,
) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _mapping_string(
    mapping: Mapping[str, Any],
    *,
    key: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        return ""
    return value.strip()


def _mapping_bool(
    mapping: Mapping[str, Any],
    *,
    key: str,
) -> bool:
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    return False


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _list_of_mappings(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_mapping(item) for item in value if isinstance(item, Mapping))


__all__ = [
    "ProjectionFiberSemanticPreview",
    "ProjectionFiberSemanticsSummary",
    "projection_fiber_semantics_summary_from_payload",
]
