from __future__ import annotations

from pathlib import Path
import json

from scripts.policy import symbol_activity_audit


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_pyproject(entrypoint: str = "gabion.entry:main") -> str:
    return "\n".join(
        [
            "[project]",
            "name = 'tmp-gabion'",
            "version = '0.0.0'",
            "[project.scripts]",
            f"tmp = '{entrypoint}'",
            "",
        ]
    )


def _finding_symbols(payload: dict[str, object], bucket: str, section: str = "findings") -> set[tuple[str, str]]:
    items = ((payload.get(section) or {}).get(bucket) or [])  # type: ignore[union-attr]
    result: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        result.add((str(item.get("module", "")), str(item.get("symbol", ""))))
    return result


# gabion:evidence E:function_site::test_symbol_activity_audit.py::tests.gabion.tooling.policy.test_symbol_activity_audit.test_analyze_reports_activity_and_duplicate_buckets
# gabion:behavior primary=desired
def test_analyze_reports_activity_and_duplicate_buckets(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", _minimal_pyproject())
    _write(tmp_path / "scripts" / "driver.py", "import gabion.entry\n")
    _write(tmp_path / "src" / "gabion" / "__init__.py", "")
    _write(
        tmp_path / "src" / "gabion" / "entry.py",
        "\n".join(
            [
                "import gabion.dup_a as dup_a",
                "import gabion.dup_b as dup_b",
                "import gabion.service as service",
                "from gabion.same_dup import dup as same_dup_func",
                "",
                "def register(callback: object) -> None:",
                "    _ = callback",
                "",
                "def main() -> None:",
                "    service.called()",
                "    _ = service.value_only",
                "    register(service.dynamic_only)",
                "    typed: service.type_only",
                "    _ = typed",
                "    dup_a.run()",
                "    dup_b.run()",
                "    same_dup_func()",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "gabion" / "service.py",
        "\n".join(
            [
                "def called() -> None:",
                "    return None",
                "",
                "def value_only() -> None:",
                "    return None",
                "",
                "def type_only() -> None:",
                "    return None",
                "",
                "def dynamic_only() -> None:",
                "    return None",
                "",
                "def never_referenced() -> None:",
                "    return None",
                "",
            ]
        ),
    )
    _write(tmp_path / "src" / "gabion" / "dup_a.py", "def run() -> None:\n    return None\n")
    _write(tmp_path / "src" / "gabion" / "dup_b.py", "def run() -> None:\n    return None\n")
    _write(
        tmp_path / "src" / "gabion" / "same_dup.py",
        "\n".join(
            [
                "def dup() -> int:",
                "    return 1",
                "",
                "def dup() -> int:",
                "    return 2",
                "",
            ]
        ),
    )
    _write(tmp_path / "src" / "gabion" / "unreachable.py", "def ghost() -> None:\n    return None\n")

    payload = symbol_activity_audit.analyze(root=tmp_path)

    never_ref = _finding_symbols(payload, "never_referenced")
    value_only = _finding_symbols(payload, "ref_not_invoked_value")
    type_only = _finding_symbols(payload, "ref_not_invoked_type_only")
    dynamic_only = _finding_symbols(payload, "dynamic_dispatch_unresolved")
    same_module_dupes = _finding_symbols(payload, "same_module")
    cross_module_dupes = _finding_symbols(payload, "cross_module_public")
    unreachable_modules = {
        str(item.get("module", ""))
        for item in (payload.get("findings") or {}).get("unreachable_modules", [])  # type: ignore[union-attr]
        if isinstance(item, dict)
    }

    assert ("gabion.service", "never_referenced") in never_ref
    assert ("gabion.service", "value_only") in value_only
    assert ("gabion.service", "type_only") in type_only
    assert ("gabion.service", "dynamic_only") in dynamic_only
    assert ("gabion.same_dup", "dup") in same_module_dupes
    assert ("gabion.dup_a", "run") in cross_module_dupes
    assert ("gabion.dup_b", "run") in cross_module_dupes
    assert "gabion.unreachable" in unreachable_modules


# gabion:evidence E:function_site::test_symbol_activity_audit.py::tests.gabion.tooling.policy.test_symbol_activity_audit.test_todo_suppression_requires_valid_active_marker_metadata
# gabion:behavior primary=desired
def test_todo_suppression_requires_valid_active_marker_metadata(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", _minimal_pyproject())
    _write(tmp_path / "src" / "gabion" / "__init__.py", "")
    _write(
        tmp_path / "src" / "gabion" / "entry.py",
        "\n".join(
            [
                "import gabion.markers as _markers",
                "",
                "def main() -> None:",
                "    _ = _markers",
                "    return None",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "gabion" / "markers.py",
        "\n".join(
            [
                "from gabion.invariants import invariant_decorator, never_decorator, todo_decorator",
                "",
                "@invariant_decorator(",
                "    'todo',",
                "    reason='blocked dynamic dispatch proof',",
                "    reasoning={",
                "        'summary': 'defer until callback registry protocol exists',",
                "        'control': 'symbol_activity.dynamic_dispatch',",
                "        'blocking_dependencies': ['registry-protocol-v1'],",
                "    },",
                "    owner='policy',",
                "    expiry='2099-01-01',",
                ")",
                "def blocked_symbol() -> None:",
                "    return None",
                "",
                "@todo_decorator(reason='missing metadata')",
                "def invalid_todo_symbol() -> None:",
                "    return None",
                "",
                "@never_decorator(",
                "    reason='not a suppression marker',",
                "    reasoning={",
                "        'summary': 'never markers do not suppress symbol debt',",
                "        'control': 'symbol_activity.never',",
                "        'blocking_dependencies': ['none'],",
                "    },",
                "    owner='policy',",
                "    expiry='2099-01-01',",
                ")",
                "def never_symbol() -> None:",
                "    return None",
                "",
            ]
        ),
    )

    payload = symbol_activity_audit.analyze(root=tmp_path)
    unsuppressed_never_referenced = _finding_symbols(payload, "never_referenced")
    blocked_never_referenced = _finding_symbols(
        payload,
        "never_referenced",
        section="blocked_by_todo",
    )

    assert ("gabion.markers", "blocked_symbol") in blocked_never_referenced
    assert ("gabion.markers", "invalid_todo_symbol") in unsuppressed_never_referenced
    assert ("gabion.markers", "never_symbol") in unsuppressed_never_referenced

    unsuppressed_items = (payload.get("findings") or {}).get("never_referenced", [])  # type: ignore[union-attr]
    invalid_marker_payload = next(
        item.get("invariant_marker")
        for item in unsuppressed_items
        if isinstance(item, dict) and item.get("symbol") == "invalid_todo_symbol"
    )
    never_marker_payload = next(
        item.get("invariant_marker")
        for item in unsuppressed_items
        if isinstance(item, dict) and item.get("symbol") == "never_symbol"
    )
    assert isinstance(invalid_marker_payload, dict)
    assert invalid_marker_payload.get("valid") is False
    assert "reasoning.summary" in set(invalid_marker_payload.get("missing_fields", []))
    assert isinstance(never_marker_payload, dict)
    assert never_marker_payload.get("marker_kind") == "never"
    assert never_marker_payload.get("valid") is True


# gabion:evidence E:function_site::test_symbol_activity_audit.py::tests.gabion.tooling.policy.test_symbol_activity_audit.test_run_check_writes_artifact_and_fails_on_unsuppressed_findings
# gabion:behavior primary=verboten facets=fail
def test_run_check_writes_artifact_and_fails_on_unsuppressed_findings(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", _minimal_pyproject())
    _write(tmp_path / "src" / "gabion" / "__init__.py", "")
    _write(tmp_path / "src" / "gabion" / "entry.py", "def main() -> None:\n    return None\n")
    _write(tmp_path / "src" / "gabion" / "unused.py", "def stale() -> None:\n    return None\n")

    out = tmp_path / "artifacts" / "out" / "symbol_activity_audit.json"
    markdown_out = tmp_path / "artifacts" / "out" / "symbol_activity_audit.md"
    rc = symbol_activity_audit.run(
        root=tmp_path,
        out_path=out,
        markdown_out=markdown_out,
        check=True,
    )

    assert rc == 1
    assert out.exists()
    assert markdown_out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    counts = payload.get("counts", {})
    assert int(counts.get("unsuppressed_total", 0)) > 0
