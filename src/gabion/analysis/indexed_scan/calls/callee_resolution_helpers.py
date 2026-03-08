from __future__ import annotations

import ast
from typing import Callable


def decorator_name(
    node: ast.AST,
    *,
    check_deadline_fn: Callable[[], None],
):
    check_deadline_fn()
    node_type = type(node)
    if node_type is ast.Name:
        return node.id
    if node_type is ast.Attribute:
        parts: list[str] = []
        current: ast.AST = node
        while type(current) is ast.Attribute:
            check_deadline_fn()
            attribute_node = current
            parts.append(attribute_node.attr)
            current = attribute_node.value
        if type(current) is ast.Name:
            parts.append(current.id)
            return ".".join(reversed(parts))
        return None
    if node_type is ast.Call:
        return decorator_name(node.func, check_deadline_fn=check_deadline_fn)
    return None


def resolve_local_method_in_hierarchy(
    class_name: str,
    method: str,
    *,
    class_bases: dict[str, list[str]],
    local_functions: set[str],
    seen: set[str],
    check_deadline_fn: Callable[[], None],
    local_class_name_fn: Callable[[str, dict[str, list[str]]], object],
):
    check_deadline_fn()
    if class_name in seen:
        return None
    seen.add(class_name)
    candidate = f"{class_name}.{method}"
    if candidate in local_functions:
        return candidate
    for base in class_bases.get(class_name, []):
        check_deadline_fn()
        base_name = local_class_name_fn(base, class_bases)
        if base_name is not None:
            resolved = resolve_local_method_in_hierarchy(
                base_name,
                method,
                class_bases=class_bases,
                local_functions=local_functions,
                seen=seen,
                check_deadline_fn=check_deadline_fn,
                local_class_name_fn=local_class_name_fn,
            )
            if resolved is not None:
                return resolved
    return None
