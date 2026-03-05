from __future__ import annotations

from gabion.tooling.governance.frontmatter_parser import parse_frontmatter


def test_parse_frontmatter_handles_revision_requires_and_body() -> None:
    text = (
        "---\n"
        "doc_revision: 30\n"
        "doc_requires:\n"
        "  - README.md#repo_contract\n"
        "  - CONTRIBUTING.md#contributing_contract\n"
        "---\n"
        "# Body\n"
    )
    parsed = parse_frontmatter(text)
    assert parsed.mapping["doc_revision"] == 30
    assert parsed.mapping["doc_requires"] == [
        "README.md#repo_contract",
        "CONTRIBUTING.md#contributing_contract",
    ]
    assert parsed.body == "# Body"


def test_parse_frontmatter_handles_nested_lists_and_maps() -> None:
    text = (
        "---\n"
        "doc_section_requires:\n"
        "  agent_obligations:\n"
        "    - README.md#repo_contract\n"
        "doc_section_reviews:\n"
        "  agent_obligations:\n"
        "    README.md#repo_contract:\n"
        "      dep_version: 2\n"
        "---\n"
    )
    parsed = parse_frontmatter(text)
    assert parsed.mapping["doc_section_requires"] == {
        "agent_obligations": ["README.md#repo_contract"]
    }
    assert parsed.mapping["doc_section_reviews"] == {
        "agent_obligations": {"README.md#repo_contract": {"dep_version": 2}}
    }


def test_parse_frontmatter_strips_quotes_and_ignores_comments_and_blanks() -> None:
    text = (
        "---\n"
        "# comment\n"
        "\n"
        "doc_id: \"agents\"\n"
        "reader_reintern: 'Reader-only: re-intern if doc_revision changed.'\n"
        "---\n"
        "body\n"
    )
    parsed = parse_frontmatter(text)
    assert parsed.mapping["doc_id"] == "agents"
    assert parsed.mapping["reader_reintern"] == "Reader-only: re-intern if doc_revision changed."
    assert parsed.body == "body"
