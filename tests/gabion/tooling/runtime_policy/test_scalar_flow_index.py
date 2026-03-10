from __future__ import annotations

import ast

from gabion.tooling.policy_substrate import build_scalar_flow_index


# gabion:behavior primary=desired
def test_scalar_flow_index_infers_string_add_from_annotation_and_return_type() -> None:
    source = "\n".join(
        [
            "def suffix() -> str:",
            "    return 'tail'",
            "",
            "def combine(prefix: str, value):",
            "    return prefix + value",
            "",
            "def combine_with_call(value):",
            "    prefix = suffix()",
            "    return prefix + value",
            "",
        ]
    )
    tree = ast.parse(source)
    index = build_scalar_flow_index(tree=tree)
    binops = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)]

    assert len([node for node in binops if index.is_string_add(node=node)]) == 2
