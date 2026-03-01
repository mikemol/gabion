from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path

from gabion.tooling import governance_audit as audit_tools


@contextmanager
def _swap_attr(obj: object, name: str, value: object):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def _run_docflow_with_stubbed_findings(
    *,
    root: Path,
    warnings: list[str],
    violations: list[str],
    fail_on_violations: bool,
) -> int:
    impl = audit_tools._impl
    context = impl.DocflowAuditContext(
        docs={},
        revisions={},
        invariant_rows=[],
        invariants=[],
        warnings=warnings,
        violations=violations,
    )
    obligations = audit_tools.DocflowObligationResult(
        entries=[],
        summary={},
        warnings=[],
        violations=[],
    )

    with ExitStack() as stack:
        stack.enter_context(_swap_attr(impl, "_docflow_audit_context", lambda *_args, **_kwargs: context))
        stack.enter_context(_swap_attr(impl, "_load_docflow_docs", lambda **_kwargs: {}))
        stack.enter_context(
            _swap_attr(impl, "_evaluate_docflow_obligations", lambda **_kwargs: obligations)
        )
        stack.enter_context(_swap_attr(impl, "_emit_docflow_suite_artifacts", lambda **_kwargs: None))
        stack.enter_context(_swap_attr(impl, "_emit_docflow_compliance", lambda **_kwargs: None))
        stack.enter_context(_swap_attr(impl, "_emit_docflow_canonicality", lambda **_kwargs: None))
        stack.enter_context(_swap_attr(impl, "_emit_docflow_cycles", lambda *_args, **_kwargs: None))
        stack.enter_context(_swap_attr(impl, "_emit_docflow_change_protocol", lambda *_args, **_kwargs: None))
        stack.enter_context(_swap_attr(impl, "_emit_docflow_section_reviews", lambda *_args, **_kwargs: None))
        stack.enter_context(_swap_attr(impl, "_agent_instruction_graph", lambda **_kwargs: ([], [])))

        argv = ["--root", str(root)]
        if fail_on_violations:
            argv.append("--fail-on-violations")
        return audit_tools.run_docflow_cli(argv)


# gabion:evidence E:call_footprint::tests/test_docflow_warning_failures.py::test_docflow_warnings_fail_when_fail_on_violations_enabled::governance_audit.py::gabion.tooling.governance_audit.run_docflow_cli
def test_docflow_warnings_fail_when_fail_on_violations_enabled(tmp_path: Path) -> None:
    code = _run_docflow_with_stubbed_findings(
        root=tmp_path,
        warnings=["warning"],
        violations=[],
        fail_on_violations=True,
    )

    assert code == 1


# gabion:evidence E:call_footprint::tests/test_docflow_warning_failures.py::test_docflow_warnings_pass_when_fail_on_violations_disabled::governance_audit.py::gabion.tooling.governance_audit.run_docflow_cli
def test_docflow_warnings_pass_when_fail_on_violations_disabled(tmp_path: Path) -> None:
    code = _run_docflow_with_stubbed_findings(
        root=tmp_path,
        warnings=["warning"],
        violations=[],
        fail_on_violations=False,
    )

    assert code == 0


# gabion:evidence E:call_footprint::tests/test_docflow_warning_failures.py::test_docflow_violations_fail_when_fail_on_violations_enabled::governance_audit.py::gabion.tooling.governance_audit.run_docflow_cli
def test_docflow_violations_fail_when_fail_on_violations_enabled(tmp_path: Path) -> None:
    code = _run_docflow_with_stubbed_findings(
        root=tmp_path,
        warnings=[],
        violations=["violation"],
        fail_on_violations=True,
    )

    assert code == 1


# gabion:evidence E:call_footprint::tests/test_docflow_warning_failures.py::test_docflow_warnings_and_violations_fail_when_fail_on_violations_enabled::governance_audit.py::gabion.tooling.governance_audit.run_docflow_cli
def test_docflow_warnings_and_violations_fail_when_fail_on_violations_enabled(tmp_path: Path) -> None:
    code = _run_docflow_with_stubbed_findings(
        root=tmp_path,
        warnings=["warning"],
        violations=["violation"],
        fail_on_violations=True,
    )

    assert code == 1
