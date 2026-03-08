from __future__ import annotations

from gabion.tooling.governance import governance_audit as impl


def _spec(name: str, predicate: str, **params: object):
    return impl.spec_from_dict(
        {
            "spec_version": 1,
            "name": name,
            "domain": "docflow",
            "pipeline": [
                {
                    "op": "select",
                    "params": {
                        "predicates": [predicate],
                    },
                }
            ],
            "params": params,
        }
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
            spec=_spec("cover-active", "evidence_source", evidence_source="covered-source"),
            status="active",
        ),
        impl.DocflowInvariant(
            name="cover-proposed",
            kind="cover",
            spec=_spec("cover-proposed", "evidence_source", evidence_source="proposed-source"),
            status="proposed",
        ),
        impl.DocflowInvariant(
            name="never-active",
            kind="never",
            spec=impl._make_invariant_spec("never-active", ["missing_frontmatter"]),
            status="active",
        ),
        impl.DocflowInvariant(
            name="never-proposed",
            kind="never",
            spec=impl._make_invariant_spec("never-proposed", ["missing_frontmatter"]),
            status="proposed",
        ),
        impl.DocflowInvariant(
            name="require-active",
            kind="require",
            spec=impl._make_invariant_spec("require-active", ["missing_frontmatter"]),
            status="active",
        ),
        impl.DocflowInvariant(
            name="require-proposed",
            kind="require",
            spec=impl._make_invariant_spec("require-proposed", ["missing_governance_ref"]),
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
