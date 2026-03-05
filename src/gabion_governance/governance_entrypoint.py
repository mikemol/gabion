from __future__ import annotations

import argparse

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.tooling.runtime.deadline_runtime import deadline_scope_from_ticks

from gabion_governance import governance_audit_impl as impl


def parse_single_command_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gabion governance audit")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    impl._add_docflow_args(subparsers)
    impl._add_decision_tier_args(subparsers)
    impl._add_consolidation_args(subparsers)
    impl._add_lint_summary_args(subparsers)
    impl._add_sppf_graph_args(subparsers)
    impl._add_status_consistency_args(subparsers)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    with deadline_scope_from_ticks(
        budget=impl._DEFAULT_AUDIT_TIMEOUT_BUDGET,
        gas_limit=impl._audit_gas_limit(),
    ):
        args = parse_single_command_args(impl._coerce_argv(argv))
        check_deadline()
        cmd = args.cmd
        if cmd == "docflow":
            return impl._docflow_command(args)
        if cmd == "decision-tiers":
            return impl._decision_tiers_command(args)
        if cmd == "consolidation":
            return impl._consolidation_command(args)
        if cmd == "lint-summary":
            return impl._lint_summary_command(args)
        if cmd == "sppf-graph":
            return impl._sppf_graph_command(args)
        if cmd == "status-consistency":
            return impl._status_consistency_command(args)
        return 2


def run_docflow_cli(argv: list[str] | None = None) -> int:
    args = ["docflow", *(argv or [])]
    return main(args)


def run_decision_tiers_cli(argv: list[str] | None = None) -> int:
    args = ["decision-tiers", *(argv or [])]
    return main(args)


def run_consolidation_cli(argv: list[str] | None = None) -> int:
    args = ["consolidation", *(argv or [])]
    return main(args)


def run_lint_summary_cli(argv: list[str] | None = None) -> int:
    args = ["lint-summary", *(argv or [])]
    return main(args)


def run_sppf_graph_cli(argv: list[str] | None = None) -> int:
    args = ["sppf-graph", *(argv or [])]
    return main(args)


def run_status_consistency_cli(argv: list[str] | None = None) -> int:
    args = ["status-consistency", *(argv or [])]
    return main(args)
