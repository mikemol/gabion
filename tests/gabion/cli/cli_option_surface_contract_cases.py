from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typer.main import get_command

from gabion import cli
from gabion.cli_support.shared.option_schema import (
    INTENDED_OPTION_OVERLAP_KEYS,
    OPTION_SURFACE_DIVERGENCES,
    SHARED_OPTION_SPECS,
    OptionMultiplicity,
)
from gabion.cli_support.shared.parser_builder import dataflow_cli_parser


@dataclass(frozen=True)
class OptionFingerprint:
    multiplicity: str
    default: str
    has_negative_flag: bool
    choices: tuple[str, ...]


def _normalize_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    return repr(value)


def _normalize_argparse_inventory() -> Mapping[str, OptionFingerprint]:
    parser = dataflow_cli_parser()
    inventory: dict[str, OptionFingerprint] = {}
    for key in INTENDED_OPTION_OVERLAP_KEYS:
        spec = SHARED_OPTION_SPECS[key]
        action = parser._option_string_actions[spec.flag]
        if isinstance(action, argparse._AppendAction):
            multiplicity = OptionMultiplicity.APPEND.value
        elif isinstance(action, argparse.BooleanOptionalAction):
            multiplicity = OptionMultiplicity.BOOL_OPTIONAL.value
        elif isinstance(action, argparse._StoreTrueAction):
            multiplicity = OptionMultiplicity.FLAG.value
        else:
            multiplicity = OptionMultiplicity.SINGLE.value
        inventory[key] = OptionFingerprint(
            multiplicity=multiplicity,
            default=_normalize_default(action.default),
            has_negative_flag=f"--no-{spec.flag[2:]}" in parser._option_string_actions,
            choices=tuple(action.choices or ()),
        )
    return inventory


def _normalize_typer_inventory() -> Mapping[str, OptionFingerprint]:
    app = get_command(cli.app)
    synth = app.commands["synth"]
    by_flag: dict[str, Any] = {}
    for param in synth.params:
        for flag in param.opts:
            by_flag[flag] = param

    inventory: dict[str, OptionFingerprint] = {}
    for key in INTENDED_OPTION_OVERLAP_KEYS:
        spec = SHARED_OPTION_SPECS[key]
        param = by_flag[spec.flag]
        if param.secondary_opts:
            multiplicity = OptionMultiplicity.BOOL_OPTIONAL.value
        elif param.multiple:
            multiplicity = OptionMultiplicity.APPEND.value
        elif param.is_flag:
            multiplicity = OptionMultiplicity.FLAG.value
        else:
            multiplicity = OptionMultiplicity.SINGLE.value
        choices: tuple[str, ...] = ()
        choice_values = getattr(param.type, "choices", None)
        if choice_values is not None:
            choices = tuple(choice_values)
        inventory[key] = OptionFingerprint(
            multiplicity=multiplicity,
            default=_normalize_default(param.default),
            has_negative_flag=bool(param.secondary_opts),
            choices=choices,
        )
    return inventory


# gabion:evidence E:call_footprint::tests/test_cli_option_surface_contract.py::test_raw_and_typer_option_overlap_contract::parser_builder.py::gabion.cli_support.shared.parser_builder.dataflow_cli_parser::synth_commands.py::gabion.cli_support.synth.synth_commands.register_synth_command

def test_raw_and_typer_option_overlap_contract() -> None:
    raw_inventory = _normalize_argparse_inventory()
    typer_inventory = _normalize_typer_inventory()

    observed_divergences: set[str] = set()
    for key in INTENDED_OPTION_OVERLAP_KEYS:
        assert raw_inventory[key].choices == typer_inventory[key].choices
        if raw_inventory[key] == typer_inventory[key]:
            continue
        observed_divergences.add(key)
        assert key in OPTION_SURFACE_DIVERGENCES

    assert observed_divergences == set(OPTION_SURFACE_DIVERGENCES)
