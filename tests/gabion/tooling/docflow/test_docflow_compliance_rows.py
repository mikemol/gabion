from __future__ import annotations

import json

from gabion.tooling.governance import governance_audit as impl
from gabion_governance import governance_audit_impl as impl_core


def _matcher(name: str, predicate: str, **params: object):
    return impl._make_invariant_matcher(
        name,
        [predicate],
        params=params,
    )


# gabion:evidence E:call_footprint::tests/test_docflow_compliance_rows.py::test_docflow_compliance_rows_dispatches_cover_never_require_active_and_proposed::governance_audit_impl.py::gabion_governance.governance_audit_impl._docflow_compliance_rows
# gabion:behavior primary=verboten facets=never
def test_docflow_compliance_rows_dispatches_cover_never_require_active_and_proposed() -> None:
    rows = [
        {
            "row_kind": "evidence_key",
            "evidence_id": "E:covered",
            "evidence_kind": "call_footprint",
            "evidence_source": "covered-source",
            "evidence_display": "covered evidence",
        },
        {
            "row_kind": "evidence_key",
            "evidence_id": "E:proposed",
            "evidence_kind": "call_footprint",
            "evidence_source": "proposed-source",
            "evidence_display": "proposed evidence",
        },
        {"row_kind": "doc_missing_frontmatter", "path": "README.md", "qual": "README"},
    ]
    invariants = [
        impl.DocflowInvariant(
            name="cover-active",
            kind="cover",
            matcher=_matcher("cover-active", "evidence_source", evidence_source="covered-source"),
            status="active",
        ),
        impl.DocflowInvariant(
            name="cover-proposed",
            kind="cover",
            matcher=_matcher("cover-proposed", "evidence_source", evidence_source="proposed-source"),
            status="proposed",
        ),
        impl.DocflowInvariant(
            name="never-active",
            kind="never",
            matcher=impl._make_invariant_matcher("never-active", ["missing_frontmatter"]),
            status="active",
        ),
        impl.DocflowInvariant(
            name="never-proposed",
            kind="never",
            matcher=impl._make_invariant_matcher("never-proposed", ["missing_frontmatter"]),
            status="proposed",
        ),
        impl.DocflowInvariant(
            name="require-active",
            kind="require",
            matcher=impl._make_invariant_matcher("require-active", ["missing_frontmatter"]),
            status="active",
        ),
        impl.DocflowInvariant(
            name="require-proposed",
            kind="require",
            matcher=impl._make_invariant_matcher("require-proposed", ["missing_governance_ref"]),
            status="proposed",
        ),
    ]

    compliance = impl._docflow_compliance_rows(rows, invariants=invariants)

    cover_active = next(row for row in compliance if row.get("invariant") == "cover-active")
    assert cover_active == {
        "row_kind": "docflow_compliance",
        "invariant": "cover-active",
        "invariant_kind": "cover",
        "status": "compliant",
        "match_count": 1,
    }

    cover_proposed = next(row for row in compliance if row.get("invariant") == "cover-proposed")
    assert cover_proposed == {
        "row_kind": "docflow_compliance",
        "invariant": "cover-proposed",
        "invariant_kind": "cover",
        "status": "proposed",
        "match_count": 1,
        "detail": None,
    }

    never_active = next(row for row in compliance if row.get("invariant") == "never-active")
    assert never_active == {
        "row_kind": "docflow_compliance",
        "invariant": "never-active",
        "invariant_kind": "never",
        "status": "contradicts",
        "match_count": 1,
        "path": "README.md",
        "qual": "README",
        "source_row_kind": "doc_missing_frontmatter",
    }

    never_proposed = next(row for row in compliance if row.get("invariant") == "never-proposed")
    assert never_proposed == {
        "row_kind": "docflow_compliance",
        "invariant": "never-proposed",
        "invariant_kind": "never",
        "status": "proposed",
        "match_count": 1,
        "would_violate": True,
    }

    require_active = next(row for row in compliance if row.get("invariant") == "require-active")
    assert require_active == {
        "row_kind": "docflow_compliance",
        "invariant": "require-active",
        "invariant_kind": "require",
        "status": "compliant",
        "match_count": 1,
    }

    require_proposed = next(row for row in compliance if row.get("invariant") == "require-proposed")
    assert require_proposed == {
        "row_kind": "docflow_compliance",
        "invariant": "require-proposed",
        "invariant_kind": "require",
        "status": "proposed",
        "match_count": 0,
        "would_violate": True,
        "detail": "requirement missing",
    }

    excess = [row for row in compliance if row.get("status") == "excess"]
    assert excess == [
        {
            "row_kind": "docflow_compliance",
            "status": "excess",
            "evidence_id": "E:proposed",
            "evidence_kind": "call_footprint",
            "evidence_display": "proposed evidence",
            "evidence_source": "proposed-source",
        }
    ]


# gabion:behavior primary=desired
def test_parse_docflow_invariant_entry_normalizes_select_only_spec_json_to_matcher() -> None:
    entry = {
        "name": "docflow:custom",
        "kind": "require",
        "spec_json": json.dumps(
            {
                "spec_version": 1,
                "name": "docflow:custom",
                "domain": "docflow",
                "pipeline": [
                    {
                        "op": "select",
                        "params": {"predicates": ["missing_frontmatter"]},
                    }
                ],
                "params": {"evidence_source": "docs"},
            }
        ),
    }

    invariant = impl_core._parse_docflow_invariant_entry(entry)

    assert invariant is not None
    assert invariant.kind == "require"
    assert invariant.matcher.predicates == ("missing_frontmatter",)
    assert dict(invariant.matcher.params) == {"evidence_source": "docs"}


# gabion:behavior primary=verboten facets=projection_leakage
def test_parse_docflow_invariant_entry_rejects_nonselect_spec_json() -> None:
    entry = {
        "name": "docflow:bad",
        "kind": "never",
        "spec_json": json.dumps(
            {
                "spec_version": 1,
                "name": "docflow:bad",
                "domain": "docflow",
                "pipeline": [
                    {
                        "op": "project",
                        "params": {"fields": ["path"]},
                    }
                ],
                "params": {},
            }
        ),
    }

    assert impl_core._parse_docflow_invariant_entry(entry) is None
