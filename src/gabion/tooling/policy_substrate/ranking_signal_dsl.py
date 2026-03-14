from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _is_expandable_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _resolve_path_values(root: object, path: tuple[str, ...]) -> tuple[object, ...]:
    values: tuple[object, ...] = (root,)
    for segment in path:
        next_values: list[object] = []
        for value in values:
            if segment == "*":
                if isinstance(value, Mapping):
                    next_values.extend(value.values())
                    continue
                if _is_expandable_sequence(value):
                    next_values.extend(value)
                continue
            candidate: object | None
            if isinstance(value, Mapping):
                candidate = value.get(segment)
            else:
                candidate = getattr(value, segment, None)
            if candidate is None:
                continue
            next_values.append(candidate)
        values = tuple(next_values)
        if not values:
            break
    return values


def _flatten_scalars(values: Sequence[object]) -> tuple[str, ...]:
    flattened: list[str] = []
    for value in values:
        if _is_expandable_sequence(value):
            flattened.extend(str(item) for item in value)
            continue
        flattened.append(str(value))
    return tuple(item for item in flattened if item)


@dataclass(frozen=True)
class RankingSignalCapture:
    name: str
    path: tuple[str, ...]
    render_as: str = "first"


@dataclass(frozen=True)
class RankingSignalPredicate:
    path: tuple[str, ...]
    op: str
    expected: tuple[str, ...] = ()
    bind_name: str = ""


@dataclass(frozen=True)
class RankingSignalRule:
    rule_id: str
    entry_path: tuple[str, ...]
    diagnostic_code: str
    severity: str
    score: int
    message_template: str
    captures: tuple[RankingSignalCapture, ...] = ()
    predicates: tuple[RankingSignalPredicate, ...] = ()


@dataclass(frozen=True)
class RankingSignalMatch:
    rule_id: str
    diagnostic_code: str
    severity: str
    score: int
    message: str
    entry: object
    captures: tuple[tuple[str, object], ...]

    def capture_map(self) -> dict[str, object]:
        return dict(self.captures)


def _render_capture(
    entry: object,
    capture: RankingSignalCapture,
) -> object:
    values = _resolve_path_values(entry, capture.path)
    flattened = _flatten_scalars(values)
    if capture.render_as == "count":
        if len(values) == 1 and _is_expandable_sequence(values[0]):
            return len(values[0])
        return len(flattened)
    if capture.render_as == "csv":
        return ", ".join(flattened)
    if capture.render_as == "all":
        return flattened
    return flattened[0] if flattened else ""


def _evaluate_predicate(
    entry: object,
    predicate: RankingSignalPredicate,
) -> tuple[bool, tuple[tuple[str, object], ...]]:
    values = _resolve_path_values(entry, predicate.path)
    flattened = _flatten_scalars(values)
    if predicate.op == "in":
        return (
            any(value in predicate.expected for value in flattened),
            (),
        )
    if predicate.op == "not_in":
        return (
            bool(flattened) and all(value not in predicate.expected for value in flattened),
            (),
        )
    if predicate.op == "missing_any":
        actual = set(flattened)
        missing = tuple(value for value in predicate.expected if value not in actual)
        bindings: tuple[tuple[str, object], ...] = ()
        if predicate.bind_name:
            bindings = ((predicate.bind_name, ", ".join(missing)),)
        return (bool(missing), bindings)
    if predicate.op == "nonempty":
        return (bool(flattened), ())
    raise ValueError(f"unsupported ranking predicate op: {predicate.op}")


def evaluate_ranking_signal_rules(
    *,
    carrier: object,
    rules: Sequence[RankingSignalRule],
) -> tuple[RankingSignalMatch, ...]:
    matches: list[RankingSignalMatch] = []
    for rule in rules:
        entries = _resolve_path_values(carrier, rule.entry_path) if rule.entry_path else (carrier,)
        for entry in entries:
            bound_values: list[tuple[str, object]] = []
            matched = True
            for predicate in rule.predicates:
                predicate_match, bindings = _evaluate_predicate(entry, predicate)
                if not predicate_match:
                    matched = False
                    break
                bound_values.extend(bindings)
            if not matched:
                continue
            for capture in rule.captures:
                bound_values.append((capture.name, _render_capture(entry, capture)))
            message = rule.message_template.format(**dict(bound_values))
            matches.append(
                RankingSignalMatch(
                    rule_id=rule.rule_id,
                    diagnostic_code=rule.diagnostic_code,
                    severity=rule.severity,
                    score=rule.score,
                    message=message,
                    entry=entry,
                    captures=tuple(bound_values),
                )
            )
    return tuple(matches)
