from __future__ import annotations

import ast
from types import SimpleNamespace

from gabion.analysis.aspf.aspf import NodeId
from gabion.analysis.dataflow.engine import dataflow_adapter_contract as adapter_contract
from gabion.analysis.dataflow.engine import dataflow_analysis_index as index_owner
from gabion.analysis.dataflow.engine import dataflow_evidence_helpers as evidence_helpers
from gabion.analysis.dataflow.engine import dataflow_function_index_decision_support as decision_support
from gabion.analysis.dataflow.engine import dataflow_post_phase_analyses as post_phase

da = SimpleNamespace(
    NodeId=NodeId,
    _annotation_exception_candidates=post_phase._annotation_exception_candidates,
    _collect_module_exports=evidence_helpers._collect_module_exports,
    _decorator_name=decision_support._decorator_name,
    _keyword_links_literal=post_phase._keyword_links_literal,
    _keyword_string_literal=post_phase._keyword_string_literal,
    _phase_work_progress=index_owner._phase_work_progress,
    _refine_exception_name_from_annotations=post_phase._refine_exception_name_from_annotations,
    _split_top_level=post_phase._split_top_level,
    _stage_cache_key_aliases=index_owner._stage_cache_key_aliases,
    _type_from_const_repr=post_phase._type_from_const_repr,
    parse_adapter_capabilities=adapter_contract.parse_adapter_capabilities,
)


def _expr(source: str) -> ast.AST:
    return ast.parse(source, mode="eval").body


def _call(source: str) -> ast.Call:
    node = _expr(source)
    assert isinstance(node, ast.Call)
    return node


# gabion:behavior primary=desired
def test_phase_work_progress_and_adapter_capabilities_normalization() -> None:
    clamped = da._phase_work_progress(work_done=9, work_total=3)
    assert clamped.work_done == 3
    assert clamped.work_total == 3

    empty = da._phase_work_progress(work_done=-4, work_total=-1)
    assert empty.work_done == 0
    assert empty.work_total == 0

    defaults = da.parse_adapter_capabilities("bad-shape")
    assert defaults.bundle_inference is True
    assert defaults.decision_surfaces is True

    payload = {
        "bundle_inference": "yes",
        "decision_surfaces": False,
        "type_flow": True,
        "exception_obligations": 1,
    }
    normalized = da.parse_adapter_capabilities(payload)
    assert normalized.bundle_inference is True
    assert normalized.decision_surfaces is False
    assert normalized.type_flow is True
    assert normalized.exception_obligations is True


# gabion:behavior primary=desired
def test_decorator_name_handles_attribute_call_and_non_name_roots() -> None:
    assert da._decorator_name(_expr("simple")) == "simple"
    assert da._decorator_name(_expr("pkg.decorators.trace")) == "pkg.decorators.trace"
    assert da._decorator_name(_expr("pkg.decorators.trace()")) == "pkg.decorators.trace"
    assert da._decorator_name(_expr("factory().trace")) is None
    assert da._decorator_name(_expr("1")) is None


# gabion:behavior primary=verboten facets=exception
def test_annotation_exception_candidates_and_refinement_paths() -> None:
    assert da._annotation_exception_candidates(None) == ()
    assert da._annotation_exception_candidates("ValueError[") == ()
    assert da._annotation_exception_candidates("ValueError | errors.TypeError") == (
        "TypeError",
        "ValueError",
    )

    expr = _expr("err")
    refined = da._refine_exception_name_from_annotations(
        expr,
        param_annotations={"err": "ValueError"},
    )
    assert refined == ("ValueError", "PARAM_ANNOTATION", ("ValueError",))

    ambiguous = da._refine_exception_name_from_annotations(
        expr,
        param_annotations={"err": "ValueError | TypeError"},
    )
    assert ambiguous[1] == "PARAM_ANNOTATION_AMBIGUOUS"
    assert ambiguous[2] == ("TypeError", "ValueError")

    direct = da._refine_exception_name_from_annotations(
        _expr("raise_error()"),
        param_annotations={"err": "ValueError"},
    )
    assert direct[1] is None
    assert direct[2] == ()


# gabion:behavior primary=desired
def test_keyword_literal_helpers_filter_and_sort_link_payloads() -> None:
    call = _call(
        "never(owner='team', links=["
        "{'kind': 'run', 'value': 'https://run'}, "
        "{'kind': 'doc', 'value': 'https://doc'}, "
        "{'kind': '', 'value': 'skip'}, "
        "{'kind': 1, 'value': 'skip'}, "
        "{'kind': 'bad'}"
        "])"
    )
    assert da._keyword_string_literal(call, "owner") == "team"
    assert da._keyword_string_literal(call, "expiry") == ""
    assert da._keyword_links_literal(call) == [
        {"kind": "doc", "value": "https://doc"},
        {"kind": "run", "value": "https://run"},
    ]

    malformed_links = _call("never(links='bad')")
    assert da._keyword_links_literal(malformed_links) == []


# gabion:behavior primary=desired
def test_stage_cache_key_aliases_cover_parse_and_nodeid_forms() -> None:
    digest = "a" * 40
    identity = f"aspf:sha1:{digest}"
    parse_key = ("parse", "PARAM_ANNOTATIONS", identity, ("module.py", 1))

    parse_aliases = da._stage_cache_key_aliases(parse_key)
    assert parse_aliases[0] == parse_key
    assert ("parse", "PARAM_ANNOTATIONS", digest, ("module.py", 1)) in parse_aliases

    scoped_key = ("scope", parse_key)
    scoped_aliases = da._stage_cache_key_aliases(scoped_key)
    assert scoped_key in scoped_aliases
    assert ("scope", ("parse", "PARAM_ANNOTATIONS", digest, ("module.py", 1))) in scoped_aliases

    node_key = da.NodeId(
        kind="ParseStageCacheIdentity",
        key=("PARAM_ANNOTATIONS", identity, ("module.py", 1)),
    )
    node_aliases = da._stage_cache_key_aliases(node_key)
    assert node_aliases[0] == node_key
    assert parse_key in node_aliases


# gabion:behavior primary=desired
def test_type_from_const_repr_split_top_level_and_module_exports() -> None:
    expected = {
        "None": "None",
        "True": "bool",
        "1": "int",
        "1.5": "float",
        "1j": "complex",
        "'txt'": "str",
        "b'raw'": "bytes",
        "[1]": "list",
        "(1,)": "tuple",
        "{1}": "set",
        "{'a': 1}": "dict",
    }
    for raw, label in expected.items():
        assert da._type_from_const_repr(raw) == label
    assert da._type_from_const_repr("not_a_literal") is None

    assert da._split_top_level("A,B[C,D],E", ",") == ["A", "B[C,D]", "E"]
    assert da._split_top_level(",A,,B,", ",") == ["A", "B"]
    assert da._split_top_level("dict[str, list[int|None]]|None", "|") == [
        "dict[str, list[int|None]]",
        "None",
    ]

    tree = ast.parse(
        "__all__: list[str] = ['A']\n"
        "__all__ += ['B']\n"
        "def A():\n"
        "    return 1\n"
    )
    export_names, export_map = da._collect_module_exports(
        tree,
        module_name="pkg.mod",
        import_map={"B": "pkg.external.B", "_private": "pkg.external._private"},
    )
    assert export_names == {"A", "B"}
    assert export_map == {"A": "pkg.mod.A", "B": "pkg.external.B"}
