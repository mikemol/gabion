from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterator

from gabion.invariants import never
from gabion.tooling.runtime.deadline_runtime import (
    DeadlineBudget,
    deadline_scope_from_lsp_env,
    deadline_scope_from_ticks,
)

_DEFAULT_SCRIPT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=120_000,
    tick_ns=1_000_000,
)


class ScriptOptionKind(StrEnum):
    TEXT = "text"
    INTEGER = "integer"
    PATH = "path"
    FLAG = "flag"


class ScriptOptionArity(StrEnum):
    ONE = "one"
    ZERO_OR_MORE = "zero_or_more"
    ONE_OR_MORE = "one_or_more"
    APPEND = "append"


class ScriptRuntimeMode(StrEnum):
    FIXED = "fixed"
    LSP_ENV = "lsp_env"


@dataclass(frozen=True)
class ScriptOptionSpec:
    dest: str
    flags: tuple[str, ...]
    kind: ScriptOptionKind
    default: object = None
    help: str = ""
    required: bool = False
    positional: bool = False
    arity: ScriptOptionArity = ScriptOptionArity.ONE

    def __post_init__(self) -> None:
        if not self.dest.strip():
            never("script option dest missing")
        if not self.flags:
            never("script option flags missing", dest=self.dest)
        if self.arity is not ScriptOptionArity.ONE and self.required:
            never("repeated script option cannot be required", dest=self.dest)
        if self.positional:
            if len(self.flags) != 1 or self.flags[0].startswith("-"):
                never("invalid positional script option flags", dest=self.dest)
            if self.required:
                never("positional script option cannot set required", dest=self.dest)
            if self.kind is ScriptOptionKind.FLAG:
                never("positional script flag not supported", dest=self.dest)
            if self.arity is ScriptOptionArity.APPEND:
                never("positional append script option not supported", dest=self.dest)
        else:
            if any(not flag.startswith("-") for flag in self.flags):
                never("invalid optional script option flags", dest=self.dest)
        if self.kind is ScriptOptionKind.FLAG and self.required:
            never("flag option cannot be required", dest=self.dest)
        if self.kind is ScriptOptionKind.FLAG and self.arity is not ScriptOptionArity.ONE:
            never("repeated flag option not supported", dest=self.dest)


@dataclass(frozen=True)
class ScriptRuntimeSpec:
    mode: ScriptRuntimeMode = ScriptRuntimeMode.LSP_ENV
    deadline_budget: DeadlineBudget = _DEFAULT_SCRIPT_TIMEOUT_BUDGET
    gas_limit: int | None = None

    def __post_init__(self) -> None:
        if self.gas_limit is not None and int(self.gas_limit) <= 0:
            never("invalid script gas limit", gas_limit=self.gas_limit)


_DEFAULT_SCRIPT_RUNTIME = ScriptRuntimeSpec()


@dataclass(frozen=True)
class ScriptInvocation:
    values: Mapping[str, object]

    def text(self, dest: str) -> str:
        value = self.values.get(dest)
        if isinstance(value, str):
            return value
        never("script text argument missing or invalid", dest=dest)
        return ""

    def integer(self, dest: str) -> int:
        value = self.values.get(dest)
        if isinstance(value, bool):
            never("script integer argument invalid", dest=dest)
        if isinstance(value, int):
            return value
        never("script integer argument missing or invalid", dest=dest)
        return 0

    def path(self, dest: str) -> Path:
        value = self.values.get(dest)
        if isinstance(value, Path):
            return value
        never("script path argument missing or invalid", dest=dest)
        return Path()

    def flag(self, dest: str) -> bool:
        value = self.values.get(dest)
        if isinstance(value, bool):
            return value
        never("script flag argument missing or invalid", dest=dest)
        return False

    def texts(self, dest: str) -> tuple[str, ...]:
        value = self.values.get(dest)
        if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
            return value
        never("script text sequence argument missing or invalid", dest=dest)
        return ()

    def integers(self, dest: str) -> tuple[int, ...]:
        value = self.values.get(dest)
        if isinstance(value, tuple) and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value
        ):
            return value
        never("script integer sequence argument missing or invalid", dest=dest)
        return ()

    def paths(self, dest: str) -> tuple[Path, ...]:
        value = self.values.get(dest)
        if isinstance(value, tuple) and all(isinstance(item, Path) for item in value):
            return value
        never("script path sequence argument missing or invalid", dest=dest)
        return ()


@dataclass(frozen=True)
class DeclarativeScriptSpec:
    script_id: str
    description: str
    options: tuple[ScriptOptionSpec, ...]
    handler: Callable[[ScriptInvocation], int] = field(compare=False, repr=False)
    runtime: ScriptRuntimeSpec = _DEFAULT_SCRIPT_RUNTIME


def _normalized_scalar(
    *,
    kind: ScriptOptionKind,
    value: object,
    dest: str,
) -> object:
    match kind:
        case ScriptOptionKind.TEXT:
            return str(value)
        case ScriptOptionKind.INTEGER:
            if isinstance(value, bool):
                never("invalid integer default", dest=dest)
            if isinstance(value, int):
                return value
            never("invalid integer default", dest=dest)
        case ScriptOptionKind.PATH:
            return Path(value)
        case ScriptOptionKind.FLAG:
            if isinstance(value, bool):
                return value
            never("invalid flag default", dest=dest)
    never("unreachable script option kind", kind=kind.value)
    return None


def _normalized_sequence_default(spec: ScriptOptionSpec) -> list[object]:
    if spec.default is None:
        return []
    items = spec.default if isinstance(spec.default, (list, tuple)) else (spec.default,)
    return [
        _normalized_scalar(kind=spec.kind, value=item, dest=spec.dest)
        for item in items
    ]


def _normalized_default(spec: ScriptOptionSpec) -> object:
    if spec.arity is not ScriptOptionArity.ONE:
        return _normalized_sequence_default(spec)
    if spec.default is None:
        return None
    return _normalized_scalar(
        kind=spec.kind,
        value=spec.default,
        dest=spec.dest,
    )


def _nargs_for_arity(arity: ScriptOptionArity) -> str | None:
    match arity:
        case ScriptOptionArity.ONE:
            return None
        case ScriptOptionArity.ZERO_OR_MORE:
            return "*"
        case ScriptOptionArity.ONE_OR_MORE:
            return "+"
        case ScriptOptionArity.APPEND:
            return None
    never("unreachable script option arity", arity=arity.value)
    return None


def _type_for_kind(kind: ScriptOptionKind):
    match kind:
        case ScriptOptionKind.TEXT:
            return str
        case ScriptOptionKind.INTEGER:
            return int
        case ScriptOptionKind.PATH:
            return Path
        case ScriptOptionKind.FLAG:
            return None
    never("unreachable script option kind", kind=kind.value)
    return None


def build_parser(spec: DeclarativeScriptSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=spec.script_id,
        description=spec.description,
    )
    for option in spec.options:
        default = _normalized_default(option)
        if option.kind is ScriptOptionKind.FLAG:
            parser.add_argument(
                *option.flags,
                dest=option.dest,
                action="store_true",
                default=default if isinstance(default, bool) else False,
                help=option.help,
            )
            continue
        argument_names = option.flags if not option.positional else (option.flags[0],)
        argument_kwargs: dict[str, object] = {
            "help": option.help,
            "type": _type_for_kind(option.kind),
        }
        if not option.positional:
            argument_kwargs["dest"] = option.dest
            argument_kwargs["required"] = option.required
        if option.arity is ScriptOptionArity.APPEND:
            argument_kwargs["action"] = "append"
            argument_kwargs["default"] = default
        else:
            nargs = _nargs_for_arity(option.arity)
            if nargs is not None:
                argument_kwargs["nargs"] = nargs
            if default is not None:
                argument_kwargs["default"] = default
        parser.add_argument(
            *argument_names,
            **argument_kwargs,
        )
    return parser


def _invocation_from_namespace(
    *,
    namespace: argparse.Namespace,
    spec: DeclarativeScriptSpec,
) -> ScriptInvocation:
    values: dict[str, object] = {}
    for option in spec.options:
        value = getattr(namespace, option.dest)
        if isinstance(value, list):
            values[option.dest] = tuple(value)
            continue
        values[option.dest] = value
    return ScriptInvocation(values=values)


@contextmanager
def script_runtime_scope(
    *,
    runtime: ScriptRuntimeSpec = _DEFAULT_SCRIPT_RUNTIME,
) -> Iterator[None]:
    match runtime.mode:
        case ScriptRuntimeMode.FIXED:
            with deadline_scope_from_ticks(
                budget=runtime.deadline_budget,
                gas_limit=runtime.gas_limit,
            ):
                yield
        case ScriptRuntimeMode.LSP_ENV:
            with deadline_scope_from_lsp_env(
                default_budget=runtime.deadline_budget,
                gas_limit=runtime.gas_limit,
            ):
                yield
        case _:
            never("unreachable script runtime mode", mode=runtime.mode.value)


def invoke_script(
    spec: DeclarativeScriptSpec,
    *,
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser(spec)
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    invocation = _invocation_from_namespace(namespace=namespace, spec=spec)
    with script_runtime_scope(
        runtime=spec.runtime,
    ):
        return int(spec.handler(invocation))


__all__ = [
    "DeclarativeScriptSpec",
    "ScriptInvocation",
    "ScriptOptionArity",
    "ScriptOptionKind",
    "ScriptOptionSpec",
    "ScriptRuntimeMode",
    "ScriptRuntimeSpec",
    "build_parser",
    "invoke_script",
    "script_runtime_scope",
]
