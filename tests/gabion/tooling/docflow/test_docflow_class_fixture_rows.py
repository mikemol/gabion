from __future__ import annotations

from collections.abc import Iterable

from gabion.tooling.governance import governance_audit as audit_tools


def _fixture_frontmatter(
    *,
    doc_id: str,
    doc_authority: str,
    doc_requires: list[str],
    doc_reviewed_as_of: object,
    doc_review_notes: object,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "doc_role": "analysis",
        "doc_scope": ["repo"],
        "doc_authority": doc_authority,
        "doc_requires": doc_requires,
        "doc_reviewed_as_of": doc_reviewed_as_of,
        "doc_review_notes": doc_review_notes,
        "doc_change_protocol": "POLICY_SEED.md#change_protocol",
    }


def _never_invariant(name: str, predicate: str) -> audit_tools.DocflowInvariant:
    return audit_tools.DocflowInvariant(
        name=name,
        kind="never",
        matcher=audit_tools._make_invariant_matcher(name, [predicate]),
        status="active",
    )


def _fixture_rows_and_compliance(
    *,
    rel: str,
    frontmatter: dict[str, object],
    body: str,
    invariant: audit_tools.DocflowInvariant,
) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    with audit_tools._audit_deadline_scope():
        rows, warnings = audit_tools._docflow_invariant_rows(
            docs={rel: audit_tools.Doc(frontmatter=frontmatter, body=body)},
            revisions={},
            core_set=set(audit_tools.CORE_GOVERNANCE_DOCS),
            missing_frontmatter=set(),
        )
        compliance = audit_tools._docflow_compliance_rows(rows, invariants=[invariant])
    return rows, warnings, compliance


def _matching_rows(
    rows: Iterable[dict[str, object]],
    *,
    row_kind: str,
    path: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("row_kind") == row_kind and row.get("path") == path
    ]


def _contradictions_for(
    compliance: Iterable[dict[str, object]],
    *,
    invariant: str,
    path: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in compliance
        if row.get("invariant") == invariant
        and row.get("status") == "contradicts"
        and row.get("path") == path
    ]


def _fixture_doc(
    *,
    doc_id: str,
    doc_revision: int,
    doc_sections: dict[str, int] | None = None,
) -> audit_tools.Doc:
    return audit_tools.Doc(
        frontmatter={
            "doc_id": doc_id,
            "doc_revision": doc_revision,
            "doc_sections": {} if doc_sections is None else doc_sections,
        },
        body="",
    )


# gabion:behavior primary=verboten facets=missing
def test_dfx_mer_001_missing_explicit_reference_minimal() -> None:
    rel = "tmp_docs/in/fx_missing_explicit_reference_minimal.md"
    req = "POLICY_SEED.md#policy_seed"
    fm = _fixture_frontmatter(
        doc_id="fx_missing_explicit_reference_minimal",
        doc_authority="informative",
        doc_requires=[req],
        doc_reviewed_as_of={},
        doc_review_notes={},
    )
    rows, _warnings, compliance = _fixture_rows_and_compliance(
        rel=rel,
        frontmatter=fm,
        body="Body intentionally omits canonical dependency path.",
        invariant=_never_invariant("docflow:missing_explicit_reference", "missing_explicit_ref"),
    )

    req_rows = _matching_rows(rows, row_kind="doc_requires_ref", path=rel)
    assert len(req_rows) == 1
    assert req_rows[0]["req"] == req
    assert req_rows[0]["explicit"] is False
    assert req_rows[0]["implicit"] is False

    contradictions = _contradictions_for(
        compliance,
        invariant="docflow:missing_explicit_reference",
        path=rel,
    )
    assert len(contradictions) == 1
    assert contradictions[0]["source_row_kind"] == "doc_requires_ref"


# gabion:behavior primary=verboten facets=missing
def test_dfx_mer_002_missing_explicit_reference_implicit_only() -> None:
    rel = "tmp_docs/in/fx_missing_explicit_reference_implicit_only.md"
    req = "in/in-54.md#in_in_54"
    fm = _fixture_frontmatter(
        doc_id="fx_missing_explicit_reference_implicit_only",
        doc_authority="informative",
        doc_requires=[req],
        doc_reviewed_as_of={},
        doc_review_notes={},
    )
    rows, warnings, compliance = _fixture_rows_and_compliance(
        rel=rel,
        frontmatter=fm,
        body="Narrative references in-54 semantics but not canonical anchor.",
        invariant=_never_invariant("docflow:missing_explicit_reference", "missing_explicit_ref"),
    )

    req_rows = _matching_rows(rows, row_kind="doc_requires_ref", path=rel)
    assert len(req_rows) == 1
    assert req_rows[0]["req"] == req
    assert req_rows[0]["explicit"] is False
    assert req_rows[0]["implicit"] is True
    assert any("implicit reference to in/in-54.md#in_in_54" in item for item in warnings)

    contradictions = _contradictions_for(
        compliance,
        invariant="docflow:missing_explicit_reference",
        path=rel,
    )
    assert len(contradictions) == 1
    assert contradictions[0]["source_row_kind"] == "doc_requires_ref"


# gabion:behavior primary=verboten facets=invalid
def test_dfx_ift_001_invalid_field_type_doc_reviewed_as_of_null() -> None:
    rel = "tmp_docs/docs/fx_invalid_field_type_reviewed_as_of_null.md"
    fm = _fixture_frontmatter(
        doc_id="fx_invalid_field_type_reviewed_as_of_null",
        doc_authority="informative",
        doc_requires=[],
        doc_reviewed_as_of=None,
        doc_review_notes={},
    )
    rows, _warnings, compliance = _fixture_rows_and_compliance(
        rel=rel,
        frontmatter=fm,
        body="",
        invariant=_never_invariant("docflow:invalid_field_type", "invalid_field_type"),
    )

    field_rows = [
        row
        for row in _matching_rows(rows, row_kind="doc_field_type", path=rel)
        if row.get("field") == "doc_reviewed_as_of"
    ]
    assert len(field_rows) == 1
    assert field_rows[0]["expected"] == "map"
    assert field_rows[0]["valid"] is False

    contradictions = _contradictions_for(
        compliance,
        invariant="docflow:invalid_field_type",
        path=rel,
    )
    assert len(contradictions) == 1
    assert contradictions[0]["source_row_kind"] == "doc_field_type"


# gabion:behavior primary=verboten facets=invalid
def test_dfx_ift_002_invalid_field_type_doc_review_notes_scalar() -> None:
    rel = "tmp_docs/docs/fx_invalid_field_type_review_notes_scalar.md"
    fm = _fixture_frontmatter(
        doc_id="fx_invalid_field_type_review_notes_scalar",
        doc_authority="informative",
        doc_requires=[],
        doc_reviewed_as_of={},
        doc_review_notes="not-a-map",
    )
    rows, _warnings, compliance = _fixture_rows_and_compliance(
        rel=rel,
        frontmatter=fm,
        body="",
        invariant=_never_invariant("docflow:invalid_field_type", "invalid_field_type"),
    )

    field_rows = [
        row
        for row in _matching_rows(rows, row_kind="doc_field_type", path=rel)
        if row.get("field") == "doc_review_notes"
    ]
    assert len(field_rows) == 1
    assert field_rows[0]["expected"] == "map"
    assert field_rows[0]["valid"] is False

    contradictions = _contradictions_for(
        compliance,
        invariant="docflow:invalid_field_type",
        path=rel,
    )
    assert len(contradictions) == 1
    assert contradictions[0]["source_row_kind"] == "doc_field_type"


# gabion:behavior primary=verboten facets=missing
def test_dfx_mgr_001_missing_governance_ref_normative_no_roots() -> None:
    rel = "tmp_docs/in/fx_missing_governance_ref_normative_no_roots.md"
    fm = _fixture_frontmatter(
        doc_id="fx_missing_governance_ref_normative_no_roots",
        doc_authority="normative",
        doc_requires=["POLICY_SEED.md#policy_seed", "glossary.md#contract"],
        doc_reviewed_as_of={},
        doc_review_notes={},
    )
    rows, _warnings, compliance = _fixture_rows_and_compliance(
        rel=rel,
        frontmatter=fm,
        body="",
        invariant=_never_invariant("docflow:missing_governance_ref", "missing_governance_ref"),
    )

    missing_rows = _matching_rows(rows, row_kind="doc_missing_governance_ref", path=rel)
    assert sorted(row["missing"] for row in missing_rows) == [
        "AGENTS.md",
        "CONTRIBUTING.md",
        "README.md",
    ]

    contradictions = _contradictions_for(
        compliance,
        invariant="docflow:missing_governance_ref",
        path=rel,
    )
    assert len(contradictions) == 3
    assert all(row["source_row_kind"] == "doc_missing_governance_ref" for row in contradictions)


# gabion:behavior primary=verboten facets=missing
def test_dfx_mgr_002_missing_governance_ref_partial_roots() -> None:
    rel = "tmp_docs/in/fx_missing_governance_ref_partial_roots.md"
    fm = _fixture_frontmatter(
        doc_id="fx_missing_governance_ref_partial_roots",
        doc_authority="normative",
        doc_requires=[
            "POLICY_SEED.md#policy_seed",
            "glossary.md#contract",
            "README.md#repo_contract",
        ],
        doc_reviewed_as_of={},
        doc_review_notes={},
    )
    rows, _warnings, compliance = _fixture_rows_and_compliance(
        rel=rel,
        frontmatter=fm,
        body="",
        invariant=_never_invariant("docflow:missing_governance_ref", "missing_governance_ref"),
    )

    missing_rows = _matching_rows(rows, row_kind="doc_missing_governance_ref", path=rel)
    assert sorted(row["missing"] for row in missing_rows) == [
        "AGENTS.md",
        "CONTRIBUTING.md",
    ]

    contradictions = _contradictions_for(
        compliance,
        invariant="docflow:missing_governance_ref",
        path=rel,
    )
    assert len(contradictions) == 2
    assert all(row["source_row_kind"] == "doc_missing_governance_ref" for row in contradictions)


# gabion:behavior primary=verboten facets=stale_metadata
def test_dfx_rnr_001_review_note_revision_lint_requires_doc_and_section_markers() -> None:
    rel = "AGENTS.md"
    req = "README.md#repo_contract"
    fm = _fixture_frontmatter(
        doc_id="agents",
        doc_authority="normative",
        doc_requires=[req],
        doc_reviewed_as_of={req: 2},
        doc_review_notes={
            req: "Reviewed README.md rev84 (repo contract still aligns with agent obligations)."
        },
    )
    docs = {
        rel: audit_tools.Doc(frontmatter=fm, body=req),
        "README.md": _fixture_doc(
            doc_id="readme",
            doc_revision=84,
            doc_sections={"repo_contract": 2},
        ),
    }

    with audit_tools._audit_deadline_scope():
        rows, warnings = audit_tools._docflow_invariant_rows(
            docs=docs,
            revisions={"README.md#repo_contract": 2},
            core_set=set(),
            missing_frontmatter=set(),
        )
        compliance = audit_tools._docflow_compliance_rows(
            rows,
            invariants=[
                _never_invariant(
                    "docflow:review_note_revision_mismatch",
                    "review_note_revision_mismatch",
                )
            ],
        )

    assert warnings == []
    review_rows = _matching_rows(rows, row_kind="doc_review_note_revision", path=rel)
    assert len(review_rows) == 1
    assert review_rows[0]["req"] == req
    assert review_rows[0]["expected_doc_revision"] == 84
    assert review_rows[0]["expected_section_revision"] == 2
    assert review_rows[0]["match"] is False

    contradictions = _contradictions_for(
        compliance,
        invariant="docflow:review_note_revision_mismatch",
        path=rel,
    )
    assert len(contradictions) == 1
    assert contradictions[0]["source_row_kind"] == "doc_review_note_revision"
