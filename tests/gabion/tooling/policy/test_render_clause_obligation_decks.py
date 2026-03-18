from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_substrate import clause_obligation_decks


# gabion:behavior primary=desired
def test_render_clause_obligation_decks_use_catalog_and_clause_index() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    catalog = clause_obligation_decks.load_catalog(
        repo_root / "docs" / "clause_obligation_decks.yaml"
    )
    clause_index = clause_obligation_decks.load_clause_index(
        repo_root / "docs" / "normative_clause_index.md"
    )
    decks = {deck.deck_id: deck for deck in catalog.decks}

    rendered_agents = clause_obligation_decks.render_deck_block(
        deck=decks["agents_required_behavior_clauses"],
        clause_index=clause_index,
    )
    rendered_contributing = clause_obligation_decks.render_deck_block(
        deck=decks["contributing_architectural_invariant_clauses"],
        clause_index=clause_index,
    )

    assert "<!-- BEGIN:generated_agent_clause_obligations -->" in rendered_agents
    assert "docs/clause_obligation_decks.yaml" in rendered_agents
    assert "[`NCI-ACTIONS-PINNED`]" in rendered_agents
    assert "<!-- BEGIN:generated_contributor_clause_invariants -->" in rendered_contributing
    assert "docs/normative_clause_index.md" in rendered_contributing
    assert "[`NCI-RUNTIME-DISTINCTION-LADDER`]" in rendered_contributing


# gabion:behavior primary=desired
def test_run_rewrites_generated_clause_obligation_blocks_and_check_detects_drift(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "clause_obligation_decks.yaml"
    clause_index_path = tmp_path / "normative_clause_index.md"
    agents_doc_path = tmp_path / "AGENTS.md"
    contributing_doc_path = tmp_path / "CONTRIBUTING.md"
    catalog_path.write_text(
        "\n".join(
            (
                "version: 1",
                "decks:",
                "  - deck_id: agents_required_behavior_clauses",
                "    doc_path: AGENTS.md",
                "    begin_marker: \"<!-- BEGIN:generated_agent_clause_obligations -->\"",
                "    end_marker: \"<!-- END:generated_agent_clause_obligations -->\"",
                "    entries:",
                "      - clause_id: NCI-TEST-A",
                "        template: \"Test clause A: {clause_link}.\"",
                "  - deck_id: contributing_architectural_invariant_clauses",
                "    doc_path: CONTRIBUTING.md",
                "    begin_marker: \"<!-- BEGIN:generated_contributor_clause_invariants -->\"",
                "    end_marker: \"<!-- END:generated_contributor_clause_invariants -->\"",
                "    entries:",
                "      - clause_id: NCI-TEST-B",
                "        template: \"Test clause B: {clause_link}.\"",
                "",
            )
        ),
        encoding="utf-8",
    )
    clause_index_path.write_text(
        "\n".join(
            (
                "<a id=\"clause-test-a\"></a>",
                "### `NCI-TEST-A`",
                "",
                "<a id=\"clause-test-b\"></a>",
                "### `NCI-TEST-B`",
                "",
            )
        ),
        encoding="utf-8",
    )
    agents_doc_path.write_text(
        "\n".join(
            (
                "# AGENTS",
                "",
                "<!-- BEGIN:generated_agent_clause_obligations -->",
                "stale",
                "<!-- END:generated_agent_clause_obligations -->",
                "",
            )
        ),
        encoding="utf-8",
    )
    contributing_doc_path.write_text(
        "\n".join(
            (
                "# CONTRIBUTING",
                "",
                "<!-- BEGIN:generated_contributor_clause_invariants -->",
                "stale",
                "<!-- END:generated_contributor_clause_invariants -->",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert (
        clause_obligation_decks.run(
            repo_root=tmp_path,
            catalog_path=catalog_path,
            clause_index_path=clause_index_path,
        )
        == 0
    )
    assert (
        clause_obligation_decks.run(
            repo_root=tmp_path,
            catalog_path=catalog_path,
            clause_index_path=clause_index_path,
            check=True,
        )
        == 0
    )

    agents_doc_path.write_text(
        agents_doc_path.read_text(encoding="utf-8").replace("NCI-TEST-A", "NCI-DRIFT-A"),
        encoding="utf-8",
    )

    assert (
        clause_obligation_decks.run(
            repo_root=tmp_path,
            catalog_path=catalog_path,
            clause_index_path=clause_index_path,
            check=True,
        )
        == 1
    )
