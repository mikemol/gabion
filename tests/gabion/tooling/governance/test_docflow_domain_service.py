from __future__ import annotations

from pathlib import Path

from gabion_governance.docflow_audit import (
    Doc,
    DocflowAuditContext,
    DocflowInvariant,
    DocflowObligationResult,
    run_docflow_domain,
)
from gabion_governance.governance_audit_impl import _make_invariant_spec


# gabion:behavior primary=desired
def test_run_docflow_domain_combines_context_and_obligation_signals() -> None:
    invariant = DocflowInvariant(
        name="docflow:test",
        kind="never",
        spec=_make_invariant_spec("docflow:test", ["missing_frontmatter"]),
    )

    def _build_context(*_args, **_kwargs) -> DocflowAuditContext:
        return DocflowAuditContext(
            docs={"README.md": Doc(frontmatter={"doc_id": "readme"}, body="body")},
            revisions={"README.md": 2},
            invariant_rows=[],
            invariants=[invariant],
            warnings=["w1"],
            violations=["v1"],
        )

    def _load_docs(*_args, **_kwargs) -> dict[str, Doc]:
        return {"README.md": Doc(frontmatter={"doc_id": "readme"}, body="body")}

    def _evaluate(*_args, **_kwargs) -> DocflowObligationResult:
        return DocflowObligationResult(entries=[], summary={}, warnings=["w2"], violations=["v2"])

    result, docs = run_docflow_domain(
        root=Path("."),
        extra_paths=None,
        extra_strict=None,
        sppf_gh_ref_mode="required",
        baseline_write_emitted=False,
        delta_guard_checked=False,
        build_context=_build_context,
        load_docs=_load_docs,
        evaluate_obligations=_evaluate,
    )

    assert sorted(result.warnings) == ["w1", "w2"]
    assert sorted(result.violations) == ["v1", "v2"]
    assert list(docs) == ["README.md"]
