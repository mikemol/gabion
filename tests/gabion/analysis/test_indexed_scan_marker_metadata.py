from __future__ import annotations

import ast

from gabion.analysis.indexed_scan import marker_metadata
from gabion.analysis.marker_protocol import MarkerKind


def _call(source: str) -> ast.Call:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return None


def _check_deadline() -> None:
    return None


def _sort_once(values, *, key=None, **_kwargs):
    return sorted(values, key=key)


#
# gabion:evidence E:function_site::indexed_scan/marker_metadata.py::gabion.analysis.indexed_scan.marker_metadata.keyword_string_literal E:function_site::indexed_scan/marker_metadata.py::gabion.analysis.indexed_scan.marker_metadata.never_reason
def test_keyword_literal_and_never_reason_helpers() -> None:
    call = _call("never('boom', owner='team', expiry='2099-01-01')")
    assert marker_metadata.keyword_string_literal(call, "owner", check_deadline_fn=_check_deadline) == "team"
    assert marker_metadata.keyword_string_literal(call, "missing", check_deadline_fn=_check_deadline) == ""
    assert marker_metadata.never_reason(call, check_deadline_fn=_check_deadline) == "boom"

    kw_reason = _call("never(reason='from-kw')")
    assert marker_metadata.never_reason(kw_reason, check_deadline_fn=_check_deadline) == "from-kw"


#
# gabion:evidence E:function_site::indexed_scan/marker_metadata.py::gabion.analysis.indexed_scan.marker_metadata.keyword_links_literal
def test_keyword_links_literal_filters_non_string_and_sorts() -> None:
    call = _call(
        "never(links=["
        "{'kind': 'run', 'value': 'https://run'},"
        "{'kind': 'doc', 'value': 'https://doc'},"
        "{'kind': '', 'value': 'skip'},"
        "{'kind': 1, 'value': 'skip'}"
        "])"
    )
    links = marker_metadata.keyword_links_literal(
        call,
        check_deadline_fn=_check_deadline,
        sort_once_fn=_sort_once,
    )
    assert links == [
        {"kind": "doc", "value": "https://doc"},
        {"kind": "run", "value": "https://run"},
    ]

    malformed = _call("never(links='bad')")
    assert marker_metadata.keyword_links_literal(
        malformed,
        check_deadline_fn=_check_deadline,
        sort_once_fn=_sort_once,
    ) == []


#
# gabion:evidence E:function_site::indexed_scan/marker_metadata.py::gabion.analysis.indexed_scan.marker_metadata.marker_alias_kind_map E:function_site::indexed_scan/marker_metadata.py::gabion.analysis.indexed_scan.marker_metadata.marker_kind_for_call E:function_site::indexed_scan/marker_metadata.py::gabion.analysis.indexed_scan.marker_metadata.never_marker_metadata
def test_alias_map_kind_resolution_and_marker_metadata() -> None:
    active_aliases, alias_map = marker_metadata.marker_alias_kind_map(
        ["custom.never"],
        check_deadline_fn=_check_deadline,
    )
    assert "custom.never" in active_aliases
    assert alias_map["custom.never"] is MarkerKind.NEVER
    assert alias_map["never"] is MarkerKind.NEVER

    call = _call("custom.never(reason='boom', owner='team')")
    kind = marker_metadata.marker_kind_for_call(
        call,
        alias_map=alias_map,
        check_deadline_fn=_check_deadline,
        decorator_name_fn=_decorator_name,
    )
    assert kind is MarkerKind.NEVER

    metadata = marker_metadata.never_marker_metadata(
        call,
        "never:mod.py:f:1:1",
        "boom",
        marker_kind=kind,
        check_deadline_fn=_check_deadline,
        sort_once_fn=_sort_once,
    )
    assert metadata["marker_kind"] == MarkerKind.NEVER.value
    assert metadata["marker_site_id"] == "never:mod.py:f:1:1"
    assert metadata["owner"] == "team"
