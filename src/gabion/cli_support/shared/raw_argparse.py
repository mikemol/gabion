from __future__ import annotations

import argparse
from typing import Callable

import typer

from gabion.cli_support.shared.parser_builder import dataflow_cli_parser


def parse_dataflow_args_or_exit(
    argv: list[str],
    *,
    parser_fn: Callable[[], argparse.ArgumentParser] | None = None,
) -> argparse.Namespace:
    parser = (parser_fn or dataflow_cli_parser)()
    if any(arg in {"-h", "--help"} for arg in argv):
        parser.print_help()
        raise typer.Exit(code=0)
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise typer.Exit(code=int(exc.code))


__all__ = ["parse_dataflow_args_or_exit"]
