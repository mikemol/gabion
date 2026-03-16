#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from gabion.tooling.runtime.deadline_runtime import DeadlineBudget
from gabion.tooling.runtime.declarative_script_host import (
    DeclarativeScriptSpec,
    ScriptInvocation,
    ScriptOptionArity,
    ScriptOptionKind,
    ScriptOptionSpec,
    ScriptRuntimeMode,
    ScriptRuntimeSpec,
    invoke_script,
    script_runtime_scope,
)

_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)
_SCRIPT_RUNTIME = ScriptRuntimeSpec(
    mode=ScriptRuntimeMode.LSP_ENV,
    deadline_budget=_DEFAULT_TIMEOUT_BUDGET,
)


def _add_repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    return root


def _deadline_scope_from_env():
    return script_runtime_scope(
        runtime=_SCRIPT_RUNTIME,
    )


def _run_invocation(invocation: ScriptInvocation) -> int:
    root = invocation.path("root").resolve()
    _add_repo_root()
    from gabion.analysis.surfaces import test_evidence

    paths = list(invocation.paths("tests"))
    with _deadline_scope_from_env():
        payload = test_evidence.build_test_evidence_payload(
            paths,
            root=root,
            include=[path.as_posix() for path in paths],
            exclude=list(invocation.texts("exclude")),
        )
        test_evidence.write_test_evidence(payload, invocation.path("out"))
    return 0


_SCRIPT_SPEC = DeclarativeScriptSpec(
    script_id="extract_test_evidence",
    description="Extract gabion evidence tags from tests.",
    options=(
        ScriptOptionSpec(
            dest="tests",
            flags=("--tests",),
            kind=ScriptOptionKind.PATH,
            arity=ScriptOptionArity.ZERO_OR_MORE,
            default=(Path("tests"),),
            help="Test files or directories to scan (default: tests/).",
        ),
        ScriptOptionSpec(
            dest="out",
            flags=("--out",),
            kind=ScriptOptionKind.PATH,
            required=True,
            help="Write JSON output to this path.",
        ),
        ScriptOptionSpec(
            dest="root",
            flags=("--root",),
            kind=ScriptOptionKind.PATH,
            default=Path("."),
            help="Repo root for relative paths.",
        ),
        ScriptOptionSpec(
            dest="exclude",
            flags=("--exclude",),
            kind=ScriptOptionKind.TEXT,
            arity=ScriptOptionArity.APPEND,
            default=(),
            help="Exclude path prefix.",
        ),
    ),
    handler=_run_invocation,
    runtime=_SCRIPT_RUNTIME,
)


def main(argv: list[str] | None = None) -> int:
    return invoke_script(_SCRIPT_SPEC, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
