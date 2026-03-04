from __future__ import annotations

import argparse

from gabion.cli_support.shared.option_schema import SHARED_OPTION_SPECS, add_argparse_option


def dataflow_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dataflow grammar audit in raw profile mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+")
    add_argparse_option(parser, SHARED_OPTION_SPECS["root"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["config"])
    parser.add_argument("--baseline", default=None, help="Baseline file for violations.")
    parser.add_argument(
        "--baseline-write",
        action="store_true",
        help="Write current violations to baseline file.",
    )
    add_argparse_option(parser, SHARED_OPTION_SPECS["exclude"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["ignore_params"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["transparent_decorators"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["allow_external"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["strictness"])
    parser.add_argument(
        "--language",
        default=None,
        help="Explicit analysis language adapter (for example: python).",
    )
    parser.add_argument(
        "--ingest-profile",
        default=None,
        help="Optional ingest profile used by the selected language adapter.",
    )
    add_argparse_option(parser, SHARED_OPTION_SPECS["aspf_trace_json"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["aspf_import_trace"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["aspf_equivalence_against"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["aspf_opportunities_json"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["aspf_state_json"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["aspf_delta_jsonl"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["aspf_import_state"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["aspf_semantic_surface"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["no_recursive"])
    parser.add_argument("--dot", default=None, help="Write DOT graph to file or '-' for stdout.")
    parser.add_argument(
        "--emit-structure-tree",
        default=None,
        help="Write canonical structure snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-structure-metrics",
        default=None,
        help="Write structure metrics JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-synth-json",
        default=None,
        help="Write fingerprint synth registry JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-provenance-json",
        default=None,
        help="Write fingerprint provenance JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-deadness-json",
        default=None,
        help="Write fingerprint deadness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-coherence-json",
        default=None,
        help="Write fingerprint coherence JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-rewrite-plans-json",
        default=None,
        help="Write fingerprint rewrite plans JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-exception-obligations-json",
        default=None,
        help="Write fingerprint exception obligations JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-handledness-json",
        default=None,
        help="Write fingerprint handledness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-decision-snapshot",
        default=None,
        help="Write decision surface snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument("--report", default=None, help="Write Markdown report (mermaid) to file.")
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Emit lint-style lines (path:line:col: CODE message).",
    )
    parser.add_argument(
        "--lint-jsonl",
        default=None,
        help="Write lint JSONL to file or '-' for stdout.",
    )
    parser.add_argument(
        "--lint-sarif",
        default=None,
        help="Write lint SARIF to file or '-' for stdout.",
    )
    add_argparse_option(parser, SHARED_OPTION_SPECS["max_components"])
    parser.add_argument(
        "--type-audit",
        action="store_true",
        help="Emit type-tightening suggestions based on downstream annotations.",
    )
    add_argparse_option(parser, SHARED_OPTION_SPECS["type_audit_max"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["type_audit_report"])
    parser.add_argument(
        "--fail-on-type-ambiguities",
        action="store_true",
        help="Exit non-zero if type ambiguities are detected.",
    )
    add_argparse_option(parser, SHARED_OPTION_SPECS["fail_on_violations"])
    parser.add_argument(
        "--synthesis-plan",
        default=None,
        help="Write synthesis plan JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-report",
        action="store_true",
        help="Include synthesis plan summary in the markdown report.",
    )
    parser.add_argument(
        "--synthesis-protocols",
        default=None,
        help="Write protocol/dataclass stubs to file or '-' for stdout.",
    )
    add_argparse_option(parser, SHARED_OPTION_SPECS["synthesis_protocols_kind"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["synthesis_max_tier"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["synthesis_min_bundle_size"])
    add_argparse_option(parser, SHARED_OPTION_SPECS["synthesis_allow_singletons"])
    parser.add_argument(
        "--synthesis-merge-overlap",
        type=float,
        default=None,
        help="Jaccard overlap threshold for merging bundles (0.0-1.0).",
    )
    add_argparse_option(parser, SHARED_OPTION_SPECS["refactor_plan"])
    parser.add_argument(
        "--refactor-plan-json",
        default=None,
        help="Write refactoring plan JSON to file or '-' for stdout.",
    )
    return parser
