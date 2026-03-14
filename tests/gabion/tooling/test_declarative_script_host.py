from __future__ import annotations

from pathlib import Path

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.tooling.runtime.declarative_script_host import (
    DeclarativeScriptSpec,
    ScriptInvocation,
    ScriptOptionArity,
    ScriptOptionKind,
    ScriptOptionSpec,
    ScriptRuntimeMode,
    ScriptRuntimeSpec,
    invoke_script,
)


def test_invoke_script_binds_deadline_and_typed_arguments(tmp_path: Path) -> None:
    output_path = tmp_path / "out.txt"
    observed: dict[str, object] = {}

    def _handler(invocation: ScriptInvocation) -> int:
        check_deadline()
        observed["path"] = invocation.path("out")
        observed["count"] = invocation.integer("count")
        observed["name"] = invocation.text("name")
        observed["flag"] = invocation.flag("emit")
        output_path.write_text("ok\n", encoding="utf-8")
        return invocation.integer("count")

    spec = DeclarativeScriptSpec(
        script_id="test_script_host",
        description="Test declarative script host.",
        options=(
            ScriptOptionSpec(
                dest="out",
                flags=("--out",),
                kind=ScriptOptionKind.PATH,
                default=output_path,
            ),
            ScriptOptionSpec(
                dest="count",
                flags=("--count",),
                kind=ScriptOptionKind.INTEGER,
                default=3,
            ),
            ScriptOptionSpec(
                dest="name",
                flags=("--name",),
                kind=ScriptOptionKind.TEXT,
                default="demo",
            ),
            ScriptOptionSpec(
                dest="emit",
                flags=("--emit",),
                kind=ScriptOptionKind.FLAG,
                default=False,
            ),
        ),
        handler=_handler,
    )

    rc = invoke_script(
        spec,
        argv=(
            "--out",
            str(output_path),
            "--count",
            "7",
            "--name",
            "manager",
            "--emit",
        ),
    )

    assert rc == 7
    assert observed == {
        "path": output_path,
        "count": 7,
        "name": "manager",
        "flag": True,
    }
    assert output_path.read_text(encoding="utf-8") == "ok\n"


def test_invoke_script_supports_sequence_and_positional_arguments() -> None:
    observed: dict[str, object] = {}

    def _handler(invocation: ScriptInvocation) -> int:
        check_deadline()
        observed["paths"] = invocation.paths("paths")
        observed["exclude"] = invocation.texts("exclude")
        observed["tests"] = invocation.paths("tests")
        return len(invocation.paths("paths"))

    rc = invoke_script(
        DeclarativeScriptSpec(
            script_id="sequence_script_host",
            description="Test repeated and positional script arguments.",
            options=(
                ScriptOptionSpec(
                    dest="paths",
                    flags=("paths",),
                    kind=ScriptOptionKind.PATH,
                    positional=True,
                    arity=ScriptOptionArity.ONE_OR_MORE,
                ),
                ScriptOptionSpec(
                    dest="exclude",
                    flags=("--exclude",),
                    kind=ScriptOptionKind.TEXT,
                    arity=ScriptOptionArity.APPEND,
                    default=(),
                ),
                ScriptOptionSpec(
                    dest="tests",
                    flags=("--tests",),
                    kind=ScriptOptionKind.PATH,
                    arity=ScriptOptionArity.ZERO_OR_MORE,
                    default=(Path("tests"),),
                ),
            ),
            handler=_handler,
            runtime=ScriptRuntimeSpec(mode=ScriptRuntimeMode.LSP_ENV),
        ),
        argv=(
            "docs/a.md",
            "docs/b.md",
            "--exclude",
            "tmp",
            "--exclude",
            "cache",
            "--tests",
            "tests/unit",
            "tests/integration",
        ),
    )

    assert rc == 2
    assert observed == {
        "paths": (Path("docs/a.md"), Path("docs/b.md")),
        "exclude": ("tmp", "cache"),
        "tests": (Path("tests/unit"), Path("tests/integration")),
    }
