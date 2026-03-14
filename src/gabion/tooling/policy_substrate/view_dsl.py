from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PathExpr:
    path: str


@dataclass(frozen=True)
class LiteralExpr:
    value: object


@dataclass(frozen=True)
class CoalesceExpr:
    items: tuple["ViewExpr", ...]


@dataclass(frozen=True)
class AddIntExpr:
    items: tuple["ViewExpr", ...]


@dataclass(frozen=True)
class WeightedTerm:
    weight: int
    expr: "ViewExpr"


@dataclass(frozen=True)
class WeightedSumExpr:
    items: tuple[WeightedTerm, ...]


type ViewExpr = PathExpr | LiteralExpr | CoalesceExpr | AddIntExpr | WeightedSumExpr


def _path_segments(path: str) -> tuple[str, ...]:
    return tuple(segment for segment in path.split(".") if segment)


def resolve_path(payload: object, path: str) -> object | None:
    current = payload
    for segment in _path_segments(path):
        if isinstance(current, Mapping):
            current = current.get(segment)
            continue
        if isinstance(current, list) and segment.isdigit():
            index = int(segment)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return None
    return current


def collection_items(payload: object, path: str) -> tuple[Mapping[str, object], ...]:
    value = resolve_path(payload, path)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def mapping_value(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def eval_expr(expr: ViewExpr, payload: object) -> object:
    match expr:
        case PathExpr(path=path):
            return resolve_path(payload, path)
        case LiteralExpr(value=value):
            return value
        case CoalesceExpr(items=items):
            for item in items:
                candidate = eval_expr(item, payload)
                if candidate not in (None, "", [], {}, ()):
                    return candidate
            return None
        case AddIntExpr(items=items):
            return sum(eval_int(item, payload) for item in items)
        case WeightedSumExpr(items=items):
            return sum(item.weight * eval_int(item.expr, payload) for item in items)
    return None


def eval_text(expr: ViewExpr, payload: object) -> str:
    value = eval_expr(expr, payload)
    return value.strip() if isinstance(value, str) else ""


def eval_int(expr: ViewExpr, payload: object) -> int:
    value = eval_expr(expr, payload)
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) else 0


def eval_mapping(expr: ViewExpr, payload: object) -> Mapping[str, object]:
    return mapping_value(eval_expr(expr, payload))


__all__ = [
    "AddIntExpr",
    "CoalesceExpr",
    "LiteralExpr",
    "PathExpr",
    "WeightedSumExpr",
    "WeightedTerm",
    "collection_items",
    "eval_expr",
    "eval_int",
    "eval_mapping",
    "eval_text",
    "mapping_value",
    "resolve_path",
]
