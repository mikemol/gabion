from __future__ import annotations

from gabion.tooling.policy_substrate.ranking_signal_dsl import (
    RankingSignalCapture,
    RankingSignalPredicate,
    RankingSignalRule,
    evaluate_ranking_signal_rules,
)


# gabion:behavior primary=desired
def test_evaluate_ranking_signal_rules_supports_report_and_entry_matches() -> None:
    carrier = {
        "issue_lifecycle_fetch_status": "partial_error",
        "issue_lifecycle_errors": ["gh auth missing"],
        "issue_lifecycles": [
            {
                "issue_id": "214",
                "state": "closed",
                "labels": ["done-on-stage"],
            }
        ],
    }
    rules = (
        RankingSignalRule(
            rule_id="fetch_incomplete",
            entry_path=(),
            diagnostic_code="fetch_incomplete",
            severity="warning",
            score=5,
            message_template="fetch status {fetch_status} errors={error_count}",
            captures=(
                RankingSignalCapture(
                    name="fetch_status",
                    path=("issue_lifecycle_fetch_status",),
                ),
                RankingSignalCapture(
                    name="error_count",
                    path=("issue_lifecycle_errors",),
                    render_as="count",
                ),
            ),
            predicates=(
                RankingSignalPredicate(
                    path=("issue_lifecycle_fetch_status",),
                    op="in",
                    expected=("error", "partial_error"),
                ),
            ),
        ),
        RankingSignalRule(
            rule_id="label_gap",
            entry_path=("issue_lifecycles", "*"),
            diagnostic_code="label_gap",
            severity="warning",
            score=3,
            message_template="GH-{issue_id} missing {missing_labels}",
            captures=(RankingSignalCapture(name="issue_id", path=("issue_id",)),),
            predicates=(
                RankingSignalPredicate(
                    path=("labels",),
                    op="missing_any",
                    expected=("done-on-stage", "status/pending-release"),
                    bind_name="missing_labels",
                ),
            ),
        ),
    )

    matches = evaluate_ranking_signal_rules(carrier=carrier, rules=rules)

    assert [match.rule_id for match in matches] == ["fetch_incomplete", "label_gap"]
    assert matches[0].message == "fetch status partial_error errors=1"
    assert matches[0].score == 5
    assert matches[1].message == "GH-214 missing status/pending-release"
    assert matches[1].capture_map()["issue_id"] == "214"
