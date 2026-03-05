from __future__ import annotations

import json

from gabion.tooling.governance import governance_audit as audit_impl


def test_matrix_extraction_is_deterministic() -> None:
    doc = audit_impl.Doc(
        frontmatter={"doc_id": "matrix_doc"},
        body="Must keep contracts stable.\nMust not keep contracts stable.\n",
    )
    first = audit_impl._extract_doc_body_notions_with_anchors(path="docs/matrix_doc.md", doc=doc)
    second = audit_impl._extract_doc_body_notions_with_anchors(path="docs/matrix_doc.md", doc=doc)
    assert first == second


def test_dependency_matrix_composition_inherits_dependency_notions() -> None:
    docs = {
        "docs/base.md": audit_impl.Doc(
            frontmatter={"doc_id": "base", "doc_requires": []},
            body="Must track correction units.",
        ),
        "docs/child.md": audit_impl.Doc(
            frontmatter={"doc_id": "child", "doc_requires": ["docs/base.md"]},
            body="Must track correction units.",
        ),
    }
    per_doc = {
        rel: audit_impl._build_doc_implication_lattice(
            path=rel,
            notions=audit_impl._extract_doc_body_notions_with_anchors(path=rel, doc=payload),
        )
        for rel, payload in docs.items()
    }

    composed, warnings, conflicts = audit_impl._compose_doc_dependency_matrices(
        docs=docs,
        per_doc_lattices=per_doc,
    )

    assert not warnings
    assert not conflicts
    child_notions = composed["docs/child.md"]["first_order"]["notions"]
    assert len(child_notions) == 2


def test_implication_matrix_violation_emits_for_counterexample_file(tmp_path) -> None:
    counterexample = tmp_path / "counterexample.md"
    counterexample.write_text("must keep matrix parity\nmust not keep matrix parity\n", encoding="utf-8")
    docs = {
        "counterexample.md": audit_impl.Doc(
            frontmatter={"doc_id": "counterexample", "doc_requires": []},
            body=counterexample.read_text(encoding="utf-8"),
        )
    }

    rows, _warnings = audit_impl._docflow_invariant_rows(
        docs=docs,
        revisions={},
        core_set=set(),
        missing_frontmatter=set(),
    )

    violations = audit_impl._evaluate_docflow_invariants(rows, invariants=audit_impl.DOCFLOW_AUDIT_INVARIANTS)
    assert any("implication matrix conflict" in violation for violation in violations)

    out = tmp_path / "implication_matrix.json"
    audit_impl._emit_docflow_implication_matrices(docs=docs, json_output=out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["conflicts"] == 1
