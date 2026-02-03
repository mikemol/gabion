from __future__ import annotations

import ast
from pathlib import Path
import sys
import textwrap


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import _analyze_function, _collect_return_aliases
    from gabion.analysis.visitors import ParentAnnotator

    return _analyze_function, _collect_return_aliases, ParentAnnotator


def test_alias_propagation_via_return() -> None:
    _analyze_function, _collect_return_aliases, ParentAnnotator = _load()
    source = textwrap.dedent(
        """
        def ident(x):
            return x

        def sink(v):
            return v

        def foo(a):
            b = ident(a)
            return sink(b)
        """
    )
    tree = ast.parse(source)
    parents = ParentAnnotator()
    parents.visit(tree)
    funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return_aliases = _collect_return_aliases(funcs, parents.parents, ignore_params=None)
    foo = next(fn for fn in funcs if fn.name == "foo")
    use_map, _ = _analyze_function(
        foo,
        parents.parents,
        is_test=False,
        ignore_params=None,
        strictness="high",
        class_name=None,
        return_aliases=return_aliases,
    )
    assert ("sink", "arg[0]") in use_map["a"].direct_forward
