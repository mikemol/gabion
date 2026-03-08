from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import typer


class OptionType(str, Enum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    PATH = "path"
    BOOL = "bool"


class OptionMultiplicity(str, Enum):
    SINGLE = "single"
    APPEND = "append"
    FLAG = "flag"
    BOOL_OPTIONAL = "bool_optional"


@dataclass(frozen=True)
class OptionSpec:
    key: str
    flag: str
    option_type: OptionType
    default: Any
    multiplicity: OptionMultiplicity
    help_text: str | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptionBindingOverride:
    default: Any | None = None
    multiplicity: OptionMultiplicity | None = None


SHARED_OPTION_SPECS: dict[str, OptionSpec] = {
    "root": OptionSpec(
        key="root",
        flag="--root",
        option_type=OptionType.PATH,
        default=Path("."),
        multiplicity=OptionMultiplicity.SINGLE,
    ),
    "config": OptionSpec(
        key="config",
        flag="--config",
        option_type=OptionType.PATH,
        default=None,
        multiplicity=OptionMultiplicity.SINGLE,
    ),
    "exclude": OptionSpec(
        key="exclude",
        flag="--exclude",
        option_type=OptionType.STRING,
        default=None,
        multiplicity=OptionMultiplicity.APPEND,
    ),
    "ignore_params": OptionSpec(
        key="ignore_params",
        flag="--ignore-params",
        option_type=OptionType.STRING,
        default=None,
        multiplicity=OptionMultiplicity.SINGLE,
    ),
    "transparent_decorators": OptionSpec(
        key="transparent_decorators",
        flag="--transparent-decorators",
        option_type=OptionType.STRING,
        default=None,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Comma-separated decorator names treated as transparent.",
    ),
    "allow_external": OptionSpec(
        key="allow_external",
        flag="--allow-external",
        option_type=OptionType.BOOL,
        default=None,
        multiplicity=OptionMultiplicity.BOOL_OPTIONAL,
    ),
    "strictness": OptionSpec(
        key="strictness",
        flag="--strictness",
        option_type=OptionType.STRING,
        default=None,
        multiplicity=OptionMultiplicity.SINGLE,
        choices=("high", "low"),
    ),
    "no_recursive": OptionSpec(
        key="no_recursive",
        flag="--no-recursive",
        option_type=OptionType.BOOL,
        default=False,
        multiplicity=OptionMultiplicity.FLAG,
    ),
    "aspf_trace_json": OptionSpec(
        key="aspf_trace_json",
        flag="--aspf-trace-json",
        option_type=OptionType.PATH,
        default=None,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Write ASPF execution trace JSON to file or '-' for stdout.",
    ),
    "aspf_import_trace": OptionSpec(
        key="aspf_import_trace",
        flag="--aspf-import-trace",
        option_type=OptionType.PATH,
        default=None,
        multiplicity=OptionMultiplicity.APPEND,
        help_text="Import one or more prior ASPF trace JSON artifacts.",
    ),
    "aspf_equivalence_against": OptionSpec(
        key="aspf_equivalence_against",
        flag="--aspf-equivalence-against",
        option_type=OptionType.PATH,
        default=None,
        multiplicity=OptionMultiplicity.APPEND,
        help_text="One or more ASPF trace JSON artifacts used as equivalence baseline.",
    ),
    "aspf_opportunities_json": OptionSpec(
        key="aspf_opportunities_json",
        flag="--aspf-opportunities-json",
        option_type=OptionType.PATH,
        default=None,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Write ASPF simplification/fungibility opportunities JSON to file or '-' for stdout.",
    ),
    "aspf_state_json": OptionSpec(
        key="aspf_state_json",
        flag="--aspf-state-json",
        option_type=OptionType.PATH,
        default=None,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Write ASPF serialized state JSON to file or '-' for stdout.",
    ),
    "aspf_delta_jsonl": OptionSpec(
        key="aspf_delta_jsonl",
        flag="--aspf-delta-jsonl",
        option_type=OptionType.PATH,
        default=None,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Write ASPF mutation delta ledger JSONL to file or '-' for stdout.",
    ),
    "aspf_import_state": OptionSpec(
        key="aspf_import_state",
        flag="--aspf-import-state",
        option_type=OptionType.PATH,
        default=None,
        multiplicity=OptionMultiplicity.APPEND,
        help_text="Import one or more prior ASPF serialized state JSON artifacts.",
    ),
    "aspf_semantic_surface": OptionSpec(
        key="aspf_semantic_surface",
        flag="--aspf-semantic-surface",
        option_type=OptionType.STRING,
        default=None,
        multiplicity=OptionMultiplicity.APPEND,
        help_text="Semantic surface keys to project into ASPF representatives.",
    ),
    "max_components": OptionSpec(
        key="max_components",
        flag="--max-components",
        option_type=OptionType.INT,
        default=10,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Max components in report.",
    ),
    "type_audit_max": OptionSpec(
        key="type_audit_max",
        flag="--type-audit-max",
        option_type=OptionType.INT,
        default=50,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Max type-tightening entries to print.",
    ),
    "type_audit_report": OptionSpec(
        key="type_audit_report",
        flag="--type-audit-report",
        option_type=OptionType.BOOL,
        default=False,
        multiplicity=OptionMultiplicity.FLAG,
        help_text="Include type-flow audit summary in the markdown report.",
    ),
    "fail_on_violations": OptionSpec(
        key="fail_on_violations",
        flag="--fail-on-violations",
        option_type=OptionType.BOOL,
        default=False,
        multiplicity=OptionMultiplicity.FLAG,
        help_text="Exit non-zero if undocumented/undeclared bundle violations are detected.",
    ),
    "synthesis_protocols_kind": OptionSpec(
        key="synthesis_protocols_kind",
        flag="--synthesis-protocols-kind",
        option_type=OptionType.STRING,
        default="dataclass",
        multiplicity=OptionMultiplicity.SINGLE,
        choices=("dataclass", "protocol", "contextvar"),
        help_text="Emit dataclass, typing.Protocol, or ContextVar stubs (default: dataclass).",
    ),
    "synthesis_max_tier": OptionSpec(
        key="synthesis_max_tier",
        flag="--synthesis-max-tier",
        option_type=OptionType.INT,
        default=2,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Max tier to include in synthesis plan.",
    ),
    "synthesis_min_bundle_size": OptionSpec(
        key="synthesis_min_bundle_size",
        flag="--synthesis-min-bundle-size",
        option_type=OptionType.INT,
        default=2,
        multiplicity=OptionMultiplicity.SINGLE,
        help_text="Min bundle size to include in synthesis plan.",
    ),
    "synthesis_allow_singletons": OptionSpec(
        key="synthesis_allow_singletons",
        flag="--synthesis-allow-singletons",
        option_type=OptionType.BOOL,
        default=False,
        multiplicity=OptionMultiplicity.FLAG,
        help_text="Allow single-field bundles in synthesis plan.",
    ),
    "refactor_plan": OptionSpec(
        key="refactor_plan",
        flag="--refactor-plan",
        option_type=OptionType.BOOL,
        default=False,
        multiplicity=OptionMultiplicity.FLAG,
        help_text="Include refactoring plan summary in the markdown report.",
    ),
}

# dataflow-bundle: cli.option-surface-intended-overlap
INTENDED_OPTION_OVERLAP_KEYS: tuple[str, ...] = tuple(SHARED_OPTION_SPECS.keys())

# dataflow-bundle: cli.option-surface-intentional-divergences
OPTION_SURFACE_DIVERGENCES: dict[str, str] = {
    "type_audit_report": (
        "Raw argparse exposes --type-audit-report as one-way opt-in while the Typer synth "
        "surface keeps explicit --type-audit-report/--no-type-audit-report toggles and defaults to enabled."
    ),
    "fail_on_violations": (
        "Raw argparse exposes --fail-on-violations as one-way opt-in while the Typer synth "
        "surface keeps explicit --fail-on-violations/--no-fail-on-violations toggles for scripted symmetry."
    ),
    "refactor_plan": (
        "Raw argparse exposes --refactor-plan as one-way opt-in while the Typer synth "
        "surface keeps explicit --refactor-plan/--no-refactor-plan toggles and defaults to enabled."
    ),
}


def _resolved_multiplicity(
    spec: OptionSpec,
    override: OptionBindingOverride | None,
) -> OptionMultiplicity:
    return override.multiplicity if override and override.multiplicity is not None else spec.multiplicity


def _resolved_default(spec: OptionSpec, override: OptionBindingOverride | None) -> Any:
    return override.default if override and override.default is not None else spec.default


def add_argparse_option(
    parser: argparse.ArgumentParser,
    spec: OptionSpec,
    *,
    override: OptionBindingOverride | None = None,
) -> None:
    kwargs: dict[str, Any] = {}
    multiplicity = _resolved_multiplicity(spec, override)
    if spec.help_text is not None:
        kwargs["help"] = spec.help_text
    if spec.option_type is OptionType.INT:
        kwargs["type"] = int
    elif spec.option_type is OptionType.FLOAT:
        kwargs["type"] = float
    elif spec.option_type is OptionType.PATH:
        kwargs["type"] = str
    if spec.choices:
        kwargs["choices"] = spec.choices
    if multiplicity is OptionMultiplicity.APPEND:
        kwargs["action"] = "append"
    elif multiplicity is OptionMultiplicity.FLAG:
        kwargs["action"] = "store_true"
    elif multiplicity is OptionMultiplicity.BOOL_OPTIONAL:
        kwargs["action"] = argparse.BooleanOptionalAction

    default = _resolved_default(spec, override)
    if spec.option_type is OptionType.PATH and isinstance(default, Path):
        default = str(default)
    kwargs["default"] = default
    parser.add_argument(spec.flag, **kwargs)


def typer_option(
    spec: OptionSpec,
    *,
    override: OptionBindingOverride | None = None,
) -> Any:
    multiplicity = _resolved_multiplicity(spec, override)
    default = _resolved_default(spec, override)
    if multiplicity is OptionMultiplicity.BOOL_OPTIONAL:
        return typer.Option(
            default,
            f"{spec.flag}/--no-{spec.flag[2:]}",
            help=spec.help_text,
        )
    return typer.Option(default, spec.flag, help=spec.help_text)
