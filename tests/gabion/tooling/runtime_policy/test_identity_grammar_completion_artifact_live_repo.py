from __future__ import annotations

import pytest

from gabion.tooling.runtime.identity_grammar_completion_artifact import (
    build_identity_grammar_completion_artifact_payload,
)
from tests.path_helpers import REPO_ROOT


pytestmark = pytest.mark.live_repo_signal


def test_build_identity_grammar_completion_artifact_payload_captures_open_residues() -> None:
    payload = build_identity_grammar_completion_artifact_payload(root=REPO_ROOT)

    assert payload["artifact_kind"] == "identity_grammar_completion"
    assert payload["summary"]["surface_count"] == 5
    assert payload["summary"]["residue_count"] >= 4
    assert payload["summary"]["highest_severity"] == "high"
    assert {item["surface_id"] for item in payload["surfaces"]} == {
        "identity_grammar.hotspot.raw_string_grouping",
        "identity_grammar.hotspot.file_quotient",
        "identity_grammar.hotspot.scope_quotient",
        "identity_grammar.planning_chart.integration",
        "identity_grammar.coherence.two_cell",
    }
    residue_kinds = {item["residue_kind"] for item in payload["residues"]}
    assert {
        "raw_string_grouping_in_core_queue_logic",
        "partial_file_quotient_reification",
        "partial_scope_quotient_reification",
        "planning_chart_identity_grammar_unintegrated",
        "coherence_witness_emission_missing",
    }.issubset(residue_kinds)
