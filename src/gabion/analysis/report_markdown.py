from __future__ import annotations

from typing import Iterable


def render_report_markdown(
    doc_id: str,
    lines: Iterable[str],
    *,
    doc_scope: Iterable[str] | None = None,
) -> str:
    scope = list(doc_scope or ("repo", "artifacts"))
    frontmatter = [
        "---",
        "doc_revision: 1",
        "reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.",
        f"doc_id: {doc_id}",
        "doc_role: report",
        "doc_scope:",
        *[f"  - {entry}" for entry in scope],
        "doc_authority: informative",
        "doc_owner: maintainer",
        "doc_requires: []",
        "doc_reviewed_as_of: {}",
        "doc_review_notes: {}",
        "doc_change_protocol: POLICY_SEED.md#change_protocol",
        "doc_erasure:",
        "  - formatting",
        "  - typos",
        "doc_sections:",
        f"  {doc_id}: 1",
        "doc_section_requires:",
        f"  {doc_id}: []",
        "doc_section_reviews:",
        f"  {doc_id}: {{}}",
        "---",
        "",
        f'<a id="{doc_id}"></a>',
        "",
    ]
    return "\n".join(frontmatter + list(lines)) + "\n"
