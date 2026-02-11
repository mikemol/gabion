from __future__ import annotations

import re
from typing import Iterable

from gabion.synthesis.model import NamingContext
from gabion.analysis.timeout_context import check_deadline


def _camelize(value: str) -> str:
    parts = [p for p in re.split(r"[^a-zA-Z0-9]+", value) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _normalize_identifier(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", value)
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        return f"{fallback}{cleaned}"
    return cleaned


def suggest_name(fields: Iterable[str], context: NamingContext | None = None) -> str:
    check_deadline()
    context = context or NamingContext()
    field_list = [f for f in fields if f]
    if not field_list:
        base = context.fallback_prefix
    else:
        frequency = context.frequency
        anchor = max(
            sorted(field_list),
            key=lambda name: (frequency.get(name, 0), len(name)),
        )
        base = _camelize(anchor) or context.fallback_prefix

    base = f"{base}Bundle"
    base = _normalize_identifier(base, context.fallback_prefix)

    name = base
    counter = 2
    existing = set(context.existing_names)
    while name in existing:
        check_deadline()
        name = f"{base}{counter}"
        counter += 1
    return name
