# gabion:grade_boundary kind=semantic_carrier_adapter name=projection_semantic_lowering
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Mapping
from functools import singledispatch

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.projection.projection_normalize import spec_hash
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec
from gabion.invariants import never
from gabion.json_types import JSONValue


class ProjectionOpLayer(str, Enum):
    SEMANTIC = "semantic"
    PRESENTATION = "presentation"
    BRIDGE = "bridge"


class SemanticProjectionKind(str, Enum):
    REFLECT = "reflect"
    QUOTIENT_FACE = "quotient_face"
    SYNTHESIZE_WITNESS = "synthesize_witness"
    WEDGE = "wedge"
    REINDEX = "reindex"
    EXISTENTIAL_IMAGE = "existential_image"
    SUPPORT_REFLECT = "support_reflect"
    NEGATE = "negate"


class BridgeProjectionKind(str, Enum):
    PREDICATE_FILTER = "predicate_filter"
    TRAVERSE = "traverse"
    COMPATIBILITY = "compatibility"


@dataclass(frozen=True)
class SemanticProjectionOp:
    source_index: int
    source_op: str
    semantic_op: SemanticProjectionKind
    params: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class PresentationProjectionOp:
    source_index: int
    source_op: str
    params: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgeProjectionOp:
    source_index: int
    source_op: str
    bridge_kind: BridgeProjectionKind
    params: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectionSemanticLoweringPlan:
    spec_identity: str
    spec_name: str
    domain: str
    semantic_ops: tuple[SemanticProjectionOp, ...] = ()
    presentation_ops: tuple[PresentationProjectionOp, ...] = ()
    bridge_ops: tuple[BridgeProjectionOp, ...] = ()

    def policy_data(self) -> dict[str, JSONValue]:
        return {
            "spec_identity": self.spec_identity,
            "spec_name": self.spec_name,
            "domain": self.domain,
            "semantic_ops": [
                {
                    "source_index": item.source_index,
                    "source_op": item.source_op,
                    "semantic_op": item.semantic_op.value,
                    "params": _copy_json_mapping(item.params),
                }
                for item in self.semantic_ops
            ],
            "presentation_ops": [
                {
                    "source_index": item.source_index,
                    "source_op": item.source_op,
                    "params": _copy_json_mapping(item.params),
                }
                for item in self.presentation_ops
            ],
            "bridge_ops": [
                {
                    "source_index": item.source_index,
                    "source_op": item.source_op,
                    "bridge_kind": item.bridge_kind.value,
                    "params": _copy_json_mapping(item.params),
                }
                for item in self.bridge_ops
            ],
        }


@dataclass(frozen=True)
class _NormalizedProjectionOp:
    source_index: int
    op_name: str
    params: dict[str, JSONValue]


@dataclass(frozen=True)
class _LimitCountParseResult:
    is_valid: bool
    count: int = 0


@dataclass(frozen=True)
class _LoweredProjectionBase:
    layer: ProjectionOpLayer


@dataclass(frozen=True)
class _LoweredSemanticProjectionOp(_LoweredProjectionBase):
    semantic_op: SemanticProjectionOp


@dataclass(frozen=True)
class _LoweredPresentationProjectionOp(_LoweredProjectionBase):
    presentation_op: PresentationProjectionOp


@dataclass(frozen=True)
class _LoweredBridgeProjectionOp(_LoweredProjectionBase):
    bridge_op: BridgeProjectionOp


def lower_projection_spec_to_semantic_plan(
    spec: ProjectionSpec,
) -> ProjectionSemanticLoweringPlan:
    check_deadline()
    semantic_ops: list[SemanticProjectionOp] = []
    presentation_ops: list[PresentationProjectionOp] = []
    bridge_ops: list[BridgeProjectionOp] = []
    for index, op in enumerate(spec.pipeline):
        check_deadline()
        normalized_op = _normalize_projection_op(index=index, op=op)
        if normalized_op.op_name == "":
            continue
        lowered = _lower_projection_op(normalized_op)
        match lowered:
            case _LoweredSemanticProjectionOp(semantic_op=semantic_op):
                semantic_ops.append(semantic_op)
            case _LoweredPresentationProjectionOp(presentation_op=presentation_op):
                presentation_ops.append(presentation_op)
            case _LoweredBridgeProjectionOp(bridge_op=bridge_op):
                bridge_ops.append(bridge_op)
    return ProjectionSemanticLoweringPlan(
        spec_identity=spec_hash(spec),
        spec_name=str(spec.name),
        domain=str(spec.domain),
        semantic_ops=tuple(semantic_ops),
        presentation_ops=tuple(presentation_ops),
        bridge_ops=tuple(bridge_ops),
    )


def _normalize_projection_op(
    *,
    index: int,
    op: ProjectionOp,
) -> _NormalizedProjectionOp:
    op_name = str(op.op).strip()
    if not op_name:
        return _NormalizedProjectionOp(source_index=index, op_name="", params={})
    params = _copy_json_mapping(op.params)
    if op_name == "project":
        fields = _normalized_string_values(_mapping_value(params, "fields"))
        if not fields:
            return _NormalizedProjectionOp(source_index=index, op_name="", params={})
        normalized_params: dict[str, JSONValue] = {"fields": list(fields)}
        quotient_face = _normalized_nonempty_string(_mapping_value(params, "quotient_face"))
        if quotient_face:
            normalized_params["quotient_face"] = quotient_face
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params=normalized_params,
        )
    if op_name == "count_by":
        fields = _normalized_count_fields(params)
        if not fields:
            return _NormalizedProjectionOp(source_index=index, op_name="", params={})
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"fields": list(fields)},
        )
    if op_name == "select":
        predicates = _normalized_predicates(params)
        if not predicates:
            return _NormalizedProjectionOp(source_index=index, op_name="", params={})
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"predicates": list(predicates)},
        )
    if op_name == "sort":
        by_values = _normalized_string_values(_mapping_value(params, "by"))
        if not by_values:
            return _NormalizedProjectionOp(source_index=index, op_name="", params={})
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"by": list(by_values)},
        )
    if op_name == "limit":
        count_parse = _limit_count_payload(_mapping_value(params, "count"))
        if count_parse.is_valid:
            return _NormalizedProjectionOp(
                source_index=index,
                op_name=op_name,
                params={"count": count_parse.count},
            )
        return _NormalizedProjectionOp(source_index=index, op_name="", params={})
    if op_name == "reflect":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if not surface:
            never("reflect projection op missing surface")
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"surface": surface},
        )
    if op_name == "support_reflect":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if not surface:
            never("support_reflect projection op missing surface")
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"surface": surface},
        )
    if op_name == "synthesize_witness":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if not surface:
            never("synthesize_witness projection op missing surface")
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"surface": surface},
        )
    if op_name == "wedge":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if not surface:
            never("wedge projection op missing surface")
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"surface": surface},
        )
    if op_name == "reindex":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if not surface:
            never("reindex projection op missing surface")
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"surface": surface},
        )
    if op_name == "existential_image":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if not surface:
            never("existential_image projection op missing surface")
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"surface": surface},
        )
    if op_name == "negate":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if not surface:
            never("negate projection op missing surface")
        return _NormalizedProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"surface": surface},
        )
    return _NormalizedProjectionOp(
        source_index=index,
        op_name=op_name,
        params=params,
    )


def _lower_projection_op(normalized_op: _NormalizedProjectionOp) -> _LoweredProjectionBase:
    op_name = normalized_op.op_name
    params = normalized_op.params
    if op_name == "project":
        quotient_face = _normalized_nonempty_string(_mapping_value(params, "quotient_face"))
        if quotient_face:
            return _LoweredSemanticProjectionOp(
                layer=ProjectionOpLayer.SEMANTIC,
                semantic_op=SemanticProjectionOp(
                    source_index=normalized_op.source_index,
                    source_op=op_name,
                    semantic_op=SemanticProjectionKind.QUOTIENT_FACE,
                    params={
                        "quotient_face": quotient_face,
                        "fields": _mapping_value(params, "fields"),
                    },
                ),
            )
        return _LoweredPresentationProjectionOp(
            layer=ProjectionOpLayer.PRESENTATION,
            presentation_op=PresentationProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                params=_copy_json_mapping(params),
            ),
        )
    if op_name == "reflect":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if surface != "projection_fiber":
            never(
                "unsupported reflect semantic surface",
                surface=surface or "<missing>",
            )
        return _LoweredSemanticProjectionOp(
            layer=ProjectionOpLayer.SEMANTIC,
            semantic_op=SemanticProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                semantic_op=SemanticProjectionKind.REFLECT,
                params={"surface": surface},
            ),
        )
    if op_name == "support_reflect":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if surface != "projection_fiber":
            never(
                "unsupported support_reflect semantic surface",
                surface=surface or "<missing>",
            )
        return _LoweredSemanticProjectionOp(
            layer=ProjectionOpLayer.SEMANTIC,
            semantic_op=SemanticProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                semantic_op=SemanticProjectionKind.SUPPORT_REFLECT,
                params={"surface": surface},
            ),
        )
    if op_name == "synthesize_witness":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if surface != "projection_fiber":
            never(
                "unsupported synthesize_witness semantic surface",
                surface=surface or "<missing>",
            )
        return _LoweredSemanticProjectionOp(
            layer=ProjectionOpLayer.SEMANTIC,
            semantic_op=SemanticProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                semantic_op=SemanticProjectionKind.SYNTHESIZE_WITNESS,
                params={"surface": surface},
            ),
        )
    if op_name == "wedge":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if surface != "projection_fiber":
            never(
                "unsupported wedge semantic surface",
                surface=surface or "<missing>",
            )
        return _LoweredSemanticProjectionOp(
            layer=ProjectionOpLayer.SEMANTIC,
            semantic_op=SemanticProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                semantic_op=SemanticProjectionKind.WEDGE,
                params={"surface": surface},
            ),
        )
    if op_name == "reindex":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if surface != "projection_fiber":
            never(
                "unsupported reindex semantic surface",
                surface=surface or "<missing>",
            )
        return _LoweredSemanticProjectionOp(
            layer=ProjectionOpLayer.SEMANTIC,
            semantic_op=SemanticProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                semantic_op=SemanticProjectionKind.REINDEX,
                params={"surface": surface},
            ),
        )
    if op_name == "existential_image":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if surface != "projection_fiber":
            never(
                "unsupported existential_image semantic surface",
                surface=surface or "<missing>",
            )
        return _LoweredSemanticProjectionOp(
            layer=ProjectionOpLayer.SEMANTIC,
            semantic_op=SemanticProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                semantic_op=SemanticProjectionKind.EXISTENTIAL_IMAGE,
                params={"surface": surface},
            ),
        )
    if op_name == "negate":
        surface = _normalized_nonempty_string(_mapping_value(params, "surface"))
        if surface != "projection_fiber":
            never(
                "unsupported negate semantic surface",
                surface=surface or "<missing>",
            )
        return _LoweredSemanticProjectionOp(
            layer=ProjectionOpLayer.SEMANTIC,
            semantic_op=SemanticProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                semantic_op=SemanticProjectionKind.NEGATE,
                params={"surface": surface},
            ),
        )
    if op_name in {"sort", "limit", "count_by"}:
        return _LoweredPresentationProjectionOp(
            layer=ProjectionOpLayer.PRESENTATION,
            presentation_op=PresentationProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                params=_copy_json_mapping(params),
            ),
        )
    if op_name == "select":
        return _LoweredBridgeProjectionOp(
            layer=ProjectionOpLayer.BRIDGE,
            bridge_op=BridgeProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                bridge_kind=BridgeProjectionKind.PREDICATE_FILTER,
                params=_copy_json_mapping(params),
            ),
        )
    if op_name == "traverse":
        return _LoweredBridgeProjectionOp(
            layer=ProjectionOpLayer.BRIDGE,
            bridge_op=BridgeProjectionOp(
                source_index=normalized_op.source_index,
                source_op=op_name,
                bridge_kind=BridgeProjectionKind.TRAVERSE,
                params=_copy_json_mapping(params),
            ),
        )
    return _LoweredBridgeProjectionOp(
        layer=ProjectionOpLayer.BRIDGE,
        bridge_op=BridgeProjectionOp(
            source_index=normalized_op.source_index,
            source_op=op_name,
            bridge_kind=BridgeProjectionKind.COMPATIBILITY,
            params=_copy_json_mapping(params),
        ),
    )


def _normalized_count_fields(params: Mapping[str, JSONValue]) -> tuple[str, ...]:
    if "fields" in params:
        return _normalized_string_values(params["fields"])
    if "field" in params:
        return _normalized_string_values(params["field"])
    return ()


def _normalized_predicates(params: Mapping[str, JSONValue]) -> tuple[str, ...]:
    predicates: list[str] = []
    if "predicate" in params:
        predicate = _normalized_nonempty_string(params["predicate"])
        if predicate:
            predicates.append(predicate)
    if "predicates" in params:
        predicates.extend(_normalized_string_values(params["predicates"]))
    return tuple(predicates)


def _normalized_string_values(value: JSONValue) -> tuple[str, ...]:
    check_deadline()
    return _normalized_string_values_payload(value)


def _normalized_string_sequence(sequence_value: tuple[object, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for entry in sequence_value:
        check_deadline()
        match entry:
            case str() as entry_text:
                stripped = entry_text.strip()
                if stripped:
                    normalized.append(stripped)
    return tuple(normalized)


def _normalized_nonempty_string(value: JSONValue) -> str:
    match value:
        case str() as text_value:
            return text_value.strip()
    return ""


def _copy_json_mapping(params: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return {str(key): value for key, value in params.items()}


def _mapping_value(params: Mapping[str, JSONValue], key: str) -> JSONValue:
    if key in params:
        return params[key]
    return []


@singledispatch
def _normalized_string_values_payload(value: JSONValue) -> tuple[str, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_normalized_string_values_payload.register(str)
def _normalized_string_values_from_str(value: str) -> tuple[str, ...]:
    stripped = value.strip()
    if stripped:
        return (stripped,)
    return ()


@_normalized_string_values_payload.register(list)
def _normalized_string_values_from_list(value: list[JSONValue]) -> tuple[str, ...]:
    return _normalized_string_sequence(tuple(value))


@_normalized_string_values_payload.register(tuple)
def _normalized_string_values_from_tuple(value: tuple[object, ...]) -> tuple[str, ...]:
    return _normalized_string_sequence(value)


@_normalized_string_values_payload.register(set)
def _normalized_string_values_from_set(value: set[object]) -> tuple[str, ...]:
    return _normalized_string_sequence(tuple(value))


def _empty_string_values(_value: object) -> tuple[str, ...]:
    return ()


for _runtime_type in (dict, int, float, bool, type(None)):
    _normalized_string_values_payload.register(_runtime_type)(_empty_string_values)


@singledispatch
def _limit_count_payload(value: JSONValue) -> _LimitCountParseResult:
    never("unregistered runtime type", value_type=type(value).__name__)


@_limit_count_payload.register(int)
def _limit_count_from_int(value: int) -> _LimitCountParseResult:
    if value >= 0:
        return _LimitCountParseResult(is_valid=True, count=value)
    return _LimitCountParseResult(is_valid=False)


def _invalid_limit_count(_value: object) -> _LimitCountParseResult:
    return _LimitCountParseResult(is_valid=False)


for _runtime_type in (str, list, dict, float, bool, tuple, set, type(None)):
    _limit_count_payload.register(_runtime_type)(_invalid_limit_count)
