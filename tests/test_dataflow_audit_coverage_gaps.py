from __future__ import annotations

import ast
from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_deadness_and_coherence_summaries_cover_edges() -> None:
    da = _load()

    assert da._summarize_deadness_witnesses([]) == []
    deadness_entries = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["a"],
            "predicate": "P",
            "environment": {},
            "result": "UNKNOWN",
            "core": [],
        }
        for _ in range(12)
    ]
    lines = da._summarize_deadness_witnesses(deadness_entries, max_entries=10)
    assert any("... 2 more" in line for line in lines)

    assert da._summarize_coherence_witnesses([]) == []
    coherence_entries = [
        {
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "result": "UNKNOWN",
            "fork_signature": "glossary-ambiguity",
            "alternatives": ["x", "y"],
        }
        for _ in range(12)
    ]
    lines = da._summarize_coherence_witnesses(coherence_entries, max_entries=10)
    assert any("... 2 more" in line for line in lines)


def test_fingerprint_coherence_and_rewrite_plans_cover_edges() -> None:
    da = _load()
    provenance_entries = [
        {
            "path": "a.py",
            "function": "f",
            "bundle": ["a", "b"],
            "provenance_id": "prov:a.py:f:a,b",
            "base_keys": ["int", "str"],
            "ctor_keys": [],
            "glossary_matches": ["ctx_a", "ctx_b"],
        }
    ]
    coherence = da._compute_fingerprint_coherence(provenance_entries, synth_version="synth@1")
    assert coherence

    plans = da._compute_fingerprint_rewrite_plans(
        provenance_entries, coherence, synth_version="synth@1"
    )
    assert plans
    assert plans[0]["evidence"]["coherence_id"] == coherence[0]["coherence_id"]

    assert da._summarize_rewrite_plans([]) == []
    rewrite_plans = [
        {
            "plan_id": f"plan:{i}",
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "rewrite": {"kind": "BUNDLE_ALIGN"},
            "status": "UNVERIFIED",
        }
        for i in range(12)
    ]
    lines = da._summarize_rewrite_plans(rewrite_plans, max_entries=10)
    assert any("... 2 more" in line for line in lines)


def test_exception_helpers_cover_edges() -> None:
    da = _load()

    # _enclosing_function_node returns None when no enclosing function is present.
    node = ast.parse("x = 1").body[0]
    assert da._enclosing_function_node(node, parents={}) is None

    assert da._exception_param_names(None, {"a"}) == []
    expr = ast.parse("a + b").body[0].value
    assert da._exception_param_names(expr, {"a"}) == ["a"]

    handler_any = ast.ExceptHandler(type=None, name=None, body=[])
    assert da._handler_is_broad(handler_any) is True
    assert da._handler_label(handler_any) == "except:"

    handler_attr = ast.ExceptHandler(
        type=ast.Attribute(
            value=ast.Name(id="builtins", ctx=ast.Load()),
            attr="Exception",
            ctx=ast.Load(),
        ),
        name=None,
        body=[],
    )
    assert da._handler_is_broad(handler_attr) is True

    handler_weird = ast.ExceptHandler(type=object(), name=None, body=[])
    assert da._handler_is_broad(handler_weird) is False
    assert da._handler_label(handler_weird) == "except <unknown>"

    # _node_in_try_body should find a nested node inside a try block.
    tree = ast.parse(
        "try:\n"
        "    foo(1)\n"
        "except Exception:\n"
        "    pass\n"
    )
    try_node = tree.body[0]
    assert isinstance(try_node, ast.Try)
    call_node = try_node.body[0].value
    assert da._node_in_try_body(call_node, try_node) is True
    other_call = ast.parse("bar()").body[0].value
    assert da._node_in_try_body(other_call, try_node) is False


def test_exception_collection_and_summaries_cover_edges(tmp_path: Path) -> None:
    da = _load()

    bad = tmp_path / "bad.py"
    bad.write_text("def bad(:\n    pass\n")

    narrow = tmp_path / "narrow.py"
    narrow.write_text(
        "def f(a):\n"
        "    try:\n"
        "        raise ValueError(a)\n"
        "    except ValueError:\n"
        "        return a\n"
    )

    module_level = tmp_path / "module_level.py"
    module_level.write_text(
        "import builtins\n"
        "try:\n"
        "    raise RuntimeError('boom')\n"
        "except builtins.Exception:\n"
        "    pass\n"
    )

    handledness = da._collect_handledness_witnesses(
        [bad, narrow, module_level],
        project_root=tmp_path,
        ignore_params=set(),
    )
    assert handledness

    obligations = da._collect_exception_obligations(
        [bad, module_level],
        project_root=tmp_path,
        ignore_params=set(),
        handledness_witnesses=handledness,
    )
    assert obligations
    assert any(entry.get("status") == "HANDLED" for entry in obligations)

    assert da._summarize_exception_obligations([]) == []
    many_obligations = [
        {
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "status": "UNKNOWN",
            "source_kind": "E0",
        }
        for _ in range(12)
    ]
    lines = da._summarize_exception_obligations(many_obligations, max_entries=10)
    assert any("... 2 more" in line for line in lines)

    assert da._summarize_handledness_witnesses([]) == []
    many_handledness = [
        {
            "site": {"path": "a.py", "function": "f", "bundle": ["a"]},
            "handler_boundary": "except:",
        }
        for _ in range(12)
    ]
    lines = da._summarize_handledness_witnesses(many_handledness, max_entries=10)
    assert any("... 2 more" in line for line in lines)
