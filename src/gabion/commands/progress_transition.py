from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Final, Literal, Mapping, cast

from gabion.invariants import never

ProgressEventKind = Literal["progress", "heartbeat", "terminal", "checkpoint"]
PROGRESS_EVENT_KINDS: Final[frozenset[str]] = frozenset(
    {"progress", "heartbeat", "terminal", "checkpoint"}
)

_POST_ANALYSIS_STATE_PREFIX = "analysis_post_"
_DEFAULT_ROOT_IDENTITY = "__root__"


@dataclass(frozen=True)
class ProgressMarkerParts:
    marker_text: str
    marker_family: str
    marker_step: str


@dataclass(frozen=True)
class ProgressNode:
    identity: str
    unit: str
    done: int
    total: int
    marker: ProgressMarkerParts
    children: tuple["ProgressNode", ...] = ()


@dataclass(frozen=True)
class NormalizedProgressTransition:
    phase: str
    analysis_state: str
    event_kind: ProgressEventKind
    root: ProgressNode
    active_path: tuple[str, ...]
    terminal_complete: bool

    @property
    def primary_unit(self) -> str:
        return self.root.unit

    @property
    def primary_done(self) -> int:
        return self.root.done

    @property
    def primary_total(self) -> int:
        return self.root.total

    @property
    def active_node(self) -> ProgressNode:
        resolved = _resolve_node_by_path(self.root, self.active_path)
        if resolved is None:
            return self.root
        return resolved

    @property
    def marker(self) -> ProgressMarkerParts:
        return self.active_node.marker


@dataclass(frozen=True)
class ProgressTransitionDecision:
    valid: bool
    reason: str
    effective_event_kind: ProgressEventKind
    suppress_emit: bool = False


def _none_optional_int(value: object) -> int | None:
    _ = value
    return None


@singledispatch
def _normalize_non_negative_int(value: object) -> int | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_normalize_non_negative_int.register
def _sd_reg_1(value: int) -> int | None:
    return max(value, 0)


@_normalize_non_negative_int.register
def _sd_reg_2(value: bool) -> int | None:
    _ = value
    return None


for _runtime_type in (str, float, complex, bytes, list, tuple, dict, set, frozenset, type(None)):
    _normalize_non_negative_int.register(_runtime_type)(_none_optional_int)


def _none_optional_mapping(value: object) -> Mapping[str, object] | None:
    _ = value
    return None


@singledispatch
def _mapping_payload_optional(value: object) -> Mapping[str, object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_mapping_payload_optional.register
def _sd_reg_3(value: dict) -> Mapping[str, object] | None:
    return value


for _runtime_type in (str, int, float, bool, bytes, list, tuple, set, frozenset, type(None)):
    _mapping_payload_optional.register(_runtime_type)(_none_optional_mapping)


def _none_optional_list(value: object) -> list[object] | None:
    _ = value
    return None


@singledispatch
def _list_payload_optional(value: object) -> list[object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_list_payload_optional.register
def _sd_reg_4(value: list) -> list[object] | None:
    return value


for _runtime_type in (str, int, float, bool, bytes, dict, tuple, set, frozenset, type(None)):
    _list_payload_optional.register(_runtime_type)(_none_optional_list)


def _none_optional_str(value: object) -> str | None:
    _ = value
    return None


@singledispatch
def _string_payload_optional(value: object) -> str | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_string_payload_optional.register
def _sd_reg_5(value: str) -> str | None:
    return value


for _runtime_type in (int, float, bool, bytes, dict, list, tuple, set, frozenset, type(None)):
    _string_payload_optional.register(_runtime_type)(_none_optional_str)


def _normalize_event_kind(value: object) -> ProgressEventKind | None:
    parsed = str(value or "")
    if parsed not in PROGRESS_EVENT_KINDS:
        return None
    return cast(ProgressEventKind, parsed)


def _normalize_marker_parts(
    marker_text: object,
    *,
    marker_family: object | None = None,
    marker_step: object | None = None,
) -> ProgressMarkerParts:
    normalized_marker_text = str(marker_text or "").strip()
    normalized_marker_family = str(marker_family or "").strip()
    normalized_marker_step = str(marker_step or "").strip()
    if normalized_marker_text and not normalized_marker_family:
        if ":" in normalized_marker_text:
            normalized_marker_family, normalized_marker_step = normalized_marker_text.split(
                ":", 1
            )
        else:
            normalized_marker_family = normalized_marker_text
    return ProgressMarkerParts(
        marker_text=normalized_marker_text,
        marker_family=normalized_marker_family,
        marker_step=normalized_marker_step,
    )


def _coerce_done_total(
    done: int,
    total: int,
) -> tuple[int, int]:
    if total > 0 and done > total:
        return total, total
    return done, total


def _node_identity_from_marker(marker: ProgressMarkerParts) -> str:
    if marker.marker_step:
        return f"{marker.marker_family}:{marker.marker_step}"
    if marker.marker_family:
        return marker.marker_family
    if marker.marker_text:
        return marker.marker_text
    return "__active__"


def _fallback_primary_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> tuple[str, int | None, int | None]:
    phase_progress_v2 = _mapping_payload_optional(phase_progress.get("phase_progress_v2"))
    primary_unit = ""
    primary_done: int | None = None
    primary_total: int | None = None
    if phase_progress_v2 is not None:
        primary_unit = str(phase_progress_v2.get("primary_unit", "") or "")
        primary_done = _normalize_non_negative_int(phase_progress_v2.get("primary_done"))
        primary_total = _normalize_non_negative_int(phase_progress_v2.get("primary_total"))
    if primary_done is None or primary_total is None:
        primary_done = _normalize_non_negative_int(phase_progress.get("work_done"))
        primary_total = _normalize_non_negative_int(phase_progress.get("work_total"))
    if primary_done is None or primary_total is None:
        return primary_unit, primary_done, primary_total
    primary_done, primary_total = _coerce_done_total(primary_done, primary_total)
    return primary_unit, primary_done, primary_total


def _normalize_node_from_mapping(
    payload: Mapping[str, object],
    *,
    fallback_identity: str,
    fallback_unit: str,
    fallback_done: int,
    fallback_total: int,
    fallback_marker: ProgressMarkerParts,
) -> ProgressNode | None:
    raw_done = payload.get("done")
    raw_total = payload.get("total")
    done = _normalize_non_negative_int(raw_done)
    total = _normalize_non_negative_int(raw_total)
    if done is None:
        done = fallback_done
    if total is None:
        total = fallback_total
    done, total = _coerce_done_total(done, total)
    marker_payload = _mapping_payload_optional(payload.get("marker"))
    marker = fallback_marker
    if marker_payload is not None:
        marker = _normalize_marker_parts(
            marker_payload.get("marker_text", marker.marker_text),
            marker_family=marker_payload.get("marker_family", marker.marker_family),
            marker_step=marker_payload.get("marker_step", marker.marker_step),
        )
    else:
        marker = _normalize_marker_parts(
            payload.get("marker_text", marker.marker_text),
            marker_family=payload.get("marker_family", marker.marker_family),
            marker_step=payload.get("marker_step", marker.marker_step),
        )
    identity = str(payload.get("identity", "") or fallback_identity).strip() or fallback_identity
    unit = str(payload.get("unit", "") or fallback_unit).strip()
    raw_children = _list_payload_optional(payload.get("children"))
    child_payloads = _normalized_child_payloads(raw_children)
    if child_payloads is None:
        return None
    children = _normalized_children(
        child_payloads,
        identity=identity,
        unit=unit,
        done=done,
        total=total,
        marker=marker,
        offset=0,
    )
    if children is None:
        return None
    return ProgressNode(
        identity=identity,
        unit=unit,
        done=done,
        total=total,
        marker=marker,
        children=children,
    )


# gabion:boundary_normalization
def _resolve_node_by_path(
    root: ProgressNode,
    active_path: tuple[str, ...],
) -> ProgressNode | None:
    if not active_path:
        return None
    if active_path[0] != root.identity:
        return None
    return _resolve_child_path(root, active_path[1:])


def _resolve_child_path(
    node: ProgressNode,
    active_path: tuple[str, ...],
) -> ProgressNode | None:
    if not active_path:
        return node
    child = _child_by_identity(node.children, active_path[0])
    if child is None:
        return None
    return _resolve_child_path(child, active_path[1:])


def _child_by_identity(
    children: tuple[ProgressNode, ...],
    identity: str,
) -> ProgressNode | None:
    if not children:
        return None
    head = children[0]
    if head.identity == identity:
        return head
    return _child_by_identity(children[1:], identity)


@singledispatch
def _normalize_active_path(value: object) -> tuple[str, ...] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_normalize_active_path.register
def _sd_reg_6(value: list) -> tuple[str, ...] | None:
    nodes = _normalized_active_path_nodes(tuple(value))
    if not nodes:
        return None
    return nodes


def _none_active_path(value: object) -> tuple[str, ...] | None:
    _ = value
    return None


for _runtime_type in (str, int, float, bool, bytes, dict, tuple, set, frozenset, type(None)):
    _normalize_active_path.register(_runtime_type)(_none_active_path)


def _normalized_active_path_nodes(
    value: tuple[object, ...],
) -> tuple[str, ...] | None:
    if not value:
        return ()
    node = _string_payload_optional(value[0])
    if node is None:
        return None
    tail_nodes = _normalized_active_path_nodes(value[1:])
    if tail_nodes is None:
        return None
    return (node.strip(), *tail_nodes)


def _normalized_child_payloads(
    raw_children: list[object] | None,
) -> tuple[Mapping[str, object], ...] | None:
    if raw_children is None:
        return ()
    return _normalized_child_payload_sequence(tuple(raw_children))


def _normalized_child_payload_sequence(
    raw_children: tuple[object, ...],
) -> tuple[Mapping[str, object], ...] | None:
    if not raw_children:
        return ()
    child_payload = _mapping_payload_optional(raw_children[0])
    if child_payload is None:
        return None
    tail_payloads = _normalized_child_payload_sequence(raw_children[1:])
    if tail_payloads is None:
        return None
    return (child_payload, *tail_payloads)


def _normalized_children(
    child_payloads: tuple[Mapping[str, object], ...],
    *,
    identity: str,
    unit: str,
    done: int,
    total: int,
    marker: ProgressMarkerParts,
    offset: int,
) -> tuple[ProgressNode, ...] | None:
    if not child_payloads:
        return ()
    child = _normalize_node_from_mapping(
        child_payloads[0],
        fallback_identity=f"{identity}:{offset}",
        fallback_unit=unit,
        fallback_done=done,
        fallback_total=total,
        fallback_marker=marker,
    )
    if child is None:
        return None
    tail_children = _normalized_children(
        child_payloads[1:],
        identity=identity,
        unit=unit,
        done=done,
        total=total,
        marker=marker,
        offset=offset + 1,
    )
    if tail_children is None:
        return None
    return (child, *tail_children)


def _transition_payload_mapping(
    phase_progress: Mapping[str, object],
) -> Mapping[str, object] | None:
    return _mapping_payload_optional(phase_progress.get("progress_transition_v2"))


def _normalize_transition_from_v2(
    *,
    phase_progress: Mapping[str, object],
    transition: Mapping[str, object],
) -> NormalizedProgressTransition | None:
    phase = str(transition.get("phase", "") or phase_progress.get("phase", "") or "")
    if not phase:
        return None
    analysis_state = str(
        transition.get("analysis_state", "")
        or phase_progress.get("analysis_state", "")
        or ""
    )
    event_kind = _normalize_event_kind(transition.get("event_kind"))
    if event_kind is None:
        event_kind = _normalize_event_kind(phase_progress.get("event_kind"))
    if event_kind is None:
        return None
    fallback_unit, fallback_done, fallback_total = _fallback_primary_from_phase_progress(
        phase_progress
    )
    if fallback_done is None or fallback_total is None:
        return None
    marker_fallback = _normalize_marker_parts(phase_progress.get("progress_marker", ""))
    root_payload = _mapping_payload_optional(transition.get("root"))
    if root_payload is None:
        return None
    root = _normalize_node_from_mapping(
        root_payload,
        fallback_identity=f"{phase}:{analysis_state}" if analysis_state else _DEFAULT_ROOT_IDENTITY,
        fallback_unit=fallback_unit,
        fallback_done=fallback_done,
        fallback_total=fallback_total,
        fallback_marker=marker_fallback,
    )
    if root is None:
        return None
    active_path = _normalize_active_path(transition.get("active_path"))
    if active_path is None:
        active_path = (root.identity,)
    active_node = _resolve_node_by_path(root, active_path)
    if active_node is None:
        return None
    terminal_complete = bool(
        (phase == "post")
        and (root.total > 0)
        and (root.done == root.total)
        and (active_node.marker.marker_text == "complete")
    )
    return NormalizedProgressTransition(
        phase=phase,
        analysis_state=analysis_state,
        event_kind=event_kind,
        root=root,
        active_path=active_path,
        terminal_complete=terminal_complete,
    )


def normalize_progress_transition_boundary(
    *,
    phase: str,
    analysis_state: str | None,
    event_kind: ProgressEventKind,
    primary_unit: str,
    primary_done: int,
    primary_total: int,
    progress_marker: str | None,
) -> NormalizedProgressTransition:
    marker = _normalize_marker_parts(progress_marker or "")
    primary_done, primary_total = _coerce_done_total(primary_done, primary_total)
    root_identity = f"{phase}:{str(analysis_state or '').strip()}" if analysis_state else f"{phase}:{_DEFAULT_ROOT_IDENTITY}"
    child_identity = _node_identity_from_marker(marker)
    child = ProgressNode(
        identity=child_identity,
        unit=primary_unit,
        done=primary_done,
        total=primary_total,
        marker=marker,
        children=(),
    )
    root = ProgressNode(
        identity=root_identity,
        unit=primary_unit,
        done=primary_done,
        total=primary_total,
        marker=_normalize_marker_parts(
            marker.marker_text if marker.marker_text else "in_progress",
            marker_family="parent",
        ),
        children=(child,),
    )
    active_path = (root.identity, child.identity)
    terminal_complete = bool(
        (phase == "post")
        and (primary_total > 0)
        and (primary_done == primary_total)
        and (marker.marker_text == "complete")
    )
    return NormalizedProgressTransition(
        phase=phase,
        analysis_state=str(analysis_state or ""),
        event_kind=event_kind,
        root=root,
        active_path=active_path,
        terminal_complete=terminal_complete,
    )


def normalize_progress_transition_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> NormalizedProgressTransition | None:
    transition_v2 = _transition_payload_mapping(phase_progress)
    if transition_v2 is None:
        return None
    return _normalize_transition_from_v2(
        phase_progress=phase_progress,
        transition=transition_v2,
    )


# gabion:boundary_normalization
def transition_marker_from_phase_progress(phase_progress: Mapping[str, object]) -> str | None:
    transition = normalize_progress_transition_from_phase_progress(phase_progress)
    if transition is None:
        return None
    return transition.marker.marker_text


def transition_primary_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> tuple[str, int | None, int | None]:
    transition = normalize_progress_transition_from_phase_progress(phase_progress)
    if transition is None:
        return "", None, None
    return transition.primary_unit, transition.primary_done, transition.primary_total


# gabion:boundary_normalization
def transition_event_kind_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> ProgressEventKind | None:
    transition = normalize_progress_transition_from_phase_progress(phase_progress)
    if transition is None:
        return None
    return transition.event_kind


# gabion:boundary_normalization
def transition_reason_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> str | None:
    transition = _transition_payload_mapping(phase_progress)
    if transition is None:
        return None
    return _string_payload_optional(transition.get("reason"))


def _is_post_in_progress_analysis_state(analysis_state: str) -> bool:
    return analysis_state.startswith(_POST_ANALYSIS_STATE_PREFIX)


def _parent_index_changed(
    previous: NormalizedProgressTransition,
    current: NormalizedProgressTransition,
) -> bool:
    return bool(
        (current.primary_unit != previous.primary_unit)
        or (current.primary_done != previous.primary_done)
        or (current.primary_total != previous.primary_total)
    )


def _node_signature(node: ProgressNode) -> tuple[object, ...]:
    return (
        node.identity,
        node.unit,
        node.done,
        node.total,
        node.marker.marker_text,
        node.marker.marker_family,
        node.marker.marker_step,
        tuple(_node_signature(child) for child in node.children),
    )


def _same_terminal_state(
    previous: NormalizedProgressTransition,
    current: NormalizedProgressTransition,
) -> bool:
    return bool(
        (current.phase == previous.phase)
        and (current.analysis_state == previous.analysis_state)
        and (current.active_path == previous.active_path)
        and (_node_signature(current.root) == _node_signature(previous.root))
    )


def _validate_tree_structure(node: ProgressNode) -> str | None:
    if not node.identity:
        return "invalid_empty_node_identity"
    if node.total > 0 and node.done > node.total:
        return "invalid_node_done_exceeds_total"
    return _validate_child_structures(node.children, seen=frozenset())


def _validate_child_structures(
    children: tuple[ProgressNode, ...],
    *,
    seen: frozenset[str],
) -> str | None:
    if not children:
        return None
    head = children[0]
    if head.identity in seen:
        return "invalid_duplicate_sibling_identity"
    child_reason = _validate_tree_structure(head)
    if child_reason is not None:
        return child_reason
    return _validate_child_structures(
        children[1:],
        seen=(seen | frozenset((head.identity,))),
    )


def _validate_tree_progress_transition(
    previous: ProgressNode,
    current: ProgressNode,
) -> str | None:
    if previous.identity != current.identity:
        return "invalid_node_identity_drift"
    if current.total < previous.total:
        return "invalid_node_total_regressed"
    if current.done < previous.done:
        return "invalid_node_done_regressed"
    return _validate_shared_child_progress(previous.children, current.children)


def _validate_shared_child_progress(
    previous_children: tuple[ProgressNode, ...],
    current_children: tuple[ProgressNode, ...],
) -> str | None:
    if not previous_children:
        return None
    previous_child = previous_children[0]
    current_child = _child_by_identity(current_children, previous_child.identity)
    if current_child is None:
        return _validate_shared_child_progress(previous_children[1:], current_children)
    child_reason = _validate_tree_progress_transition(previous_child, current_child)
    if child_reason is not None:
        return child_reason
    return _validate_shared_child_progress(previous_children[1:], current_children)


def validate_progress_transition(
    *,
    previous: NormalizedProgressTransition | None,
    current: NormalizedProgressTransition,
) -> ProgressTransitionDecision:
    effective_event_kind: ProgressEventKind = current.event_kind
    if current.terminal_complete and current.event_kind == "progress":
        effective_event_kind = "terminal"
    structure_reason = _validate_tree_structure(current.root)
    if structure_reason is not None:
        return ProgressTransitionDecision(
            valid=False,
            reason=structure_reason,
            effective_event_kind=effective_event_kind,
        )
    if _resolve_node_by_path(current.root, current.active_path) is None:
        return ProgressTransitionDecision(
            valid=False,
            reason="invalid_active_path",
            effective_event_kind=effective_event_kind,
        )
    if (
        (current.phase == "post")
        and (current.marker.marker_text == "complete")
        and not current.terminal_complete
    ):
        return ProgressTransitionDecision(
            valid=False,
            reason="invalid_post_complete_without_parent_completion",
            effective_event_kind=effective_event_kind,
        )
    if previous is None:
        return ProgressTransitionDecision(
            valid=True,
            reason="initial_transition",
            effective_event_kind=effective_event_kind,
        )
    previous_structure_reason = _validate_tree_structure(previous.root)
    if previous_structure_reason is not None:
        return ProgressTransitionDecision(
            valid=False,
            reason="invalid_previous_transition_state",
            effective_event_kind=effective_event_kind,
        )
    if _resolve_node_by_path(previous.root, previous.active_path) is None:
        return ProgressTransitionDecision(
            valid=False,
            reason="invalid_previous_active_path",
            effective_event_kind=effective_event_kind,
        )
    if previous.root.identity == current.root.identity:
        tree_reason = _validate_tree_progress_transition(previous.root, current.root)
        if tree_reason is not None:
            return ProgressTransitionDecision(
                valid=False,
                reason=tree_reason,
                effective_event_kind=effective_event_kind,
            )
    if previous.terminal_complete and current.terminal_complete:
        same_terminal_state = _same_terminal_state(previous, current)
        if not same_terminal_state:
            return ProgressTransitionDecision(
                valid=False,
                reason="invalid_terminal_replay_mutated_state",
                effective_event_kind=effective_event_kind,
            )
        if effective_event_kind == "heartbeat":
            return ProgressTransitionDecision(
                valid=True,
                reason="terminal_keepalive",
                effective_event_kind=effective_event_kind,
            )
        if effective_event_kind in {"progress", "terminal"}:
            return ProgressTransitionDecision(
                valid=True,
                reason="terminal_replay_suppressed",
                effective_event_kind=effective_event_kind,
                suppress_emit=True,
            )
    same_phase = current.phase == previous.phase
    parent_identity_changed = current.analysis_state != previous.analysis_state
    parent_index_changed = _parent_index_changed(previous, current)
    child_changed = bool(
        (current.active_path != previous.active_path)
        or (current.marker.marker_text != previous.marker.marker_text)
    )
    if (
        (current.phase == "post")
        and same_phase
        and _is_post_in_progress_analysis_state(previous.analysis_state)
        and parent_identity_changed
        and not parent_index_changed
    ):
        return ProgressTransitionDecision(
            valid=False,
            reason="invalid_post_parent_identity_changed_without_index",
            effective_event_kind=effective_event_kind,
        )
    if (
        (current.phase == "post")
        and same_phase
        and _is_post_in_progress_analysis_state(current.analysis_state)
        and parent_index_changed
        and not child_changed
    ):
        return ProgressTransitionDecision(
            valid=False,
            reason="invalid_post_parent_index_changed_without_child_change",
            effective_event_kind=effective_event_kind,
        )
    if (
        (current.phase == "post")
        and same_phase
        and _is_post_in_progress_analysis_state(current.analysis_state)
        and parent_index_changed
        and (current.marker.marker_step != "done")
    ):
        return ProgressTransitionDecision(
            valid=False,
            reason="invalid_post_parent_index_changed_without_done_boundary",
            effective_event_kind=effective_event_kind,
        )
    if (
        previous.terminal_complete
        and same_phase
        and (effective_event_kind == "heartbeat")
        and not current.terminal_complete
    ):
        return ProgressTransitionDecision(
            valid=False,
            reason="invalid_terminal_keepalive_without_terminal_state",
            effective_event_kind=effective_event_kind,
        )
    if effective_event_kind == "terminal":
        return ProgressTransitionDecision(
            valid=True,
            reason="terminal_transition",
            effective_event_kind=effective_event_kind,
        )
    return ProgressTransitionDecision(
        valid=True,
        reason="parent_advanced" if parent_index_changed else "parent_held",
        effective_event_kind=effective_event_kind,
    )


def _node_payload(node: ProgressNode) -> dict[str, object]:
    return {
        "identity": node.identity,
        "unit": node.unit,
        "done": node.done,
        "total": node.total,
        "marker_text": node.marker.marker_text,
        "marker_family": node.marker.marker_family,
        "marker_step": node.marker.marker_step,
        "children": list(_node_payload_children(node.children)),
    }


def _node_payload_children(children: tuple[ProgressNode, ...]) -> tuple[dict[str, object], ...]:
    if not children:
        return ()
    return (_node_payload(children[0]), *_node_payload_children(children[1:]))


def progress_transition_v2_payload(
    *,
    transition: NormalizedProgressTransition,
    reason: str,
    effective_event_kind: ProgressEventKind,
) -> dict[str, object]:
    return {
        "format_version": 2,
        "reason": reason,
        "phase": transition.phase,
        "analysis_state": transition.analysis_state,
        "event_kind": effective_event_kind,
        "root": _node_payload(transition.root),
        "active_path": list(transition.active_path),
        "terminal_complete": transition.terminal_complete,
    }
