from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion import server

    return server

# gabion:evidence E:decision_surface/direct::server.py::gabion.server._normalize_transparent_decorators::value
def test_normalize_transparent_decorators() -> None:
    server = _load()
    assert server._normalize_transparent_decorators(None) is None
    assert server._normalize_transparent_decorators("a, b") == {"a", "b"}
    assert server._normalize_transparent_decorators(["a", "b, c"]) == {"a", "b", "c"}
    assert server._normalize_transparent_decorators([1, "a"]) == {"a"}
    assert server._normalize_transparent_decorators([]) is None

# gabion:evidence E:function_site::server.py::gabion.server._uri_to_path
def test_uri_to_path() -> None:
    server = _load()
    path = Path("/tmp/demo.txt")
    assert server._uri_to_path(path.as_uri()) == path
    assert server._uri_to_path("relative/path.py") == Path("relative/path.py")

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report
def test_diagnostics_for_path_reports_bundle(tmp_path: Path) -> None:
    server = _load()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    diagnostics = server._diagnostics_for_path(str(sample), tmp_path)
    assert diagnostics
    assert any("Implicit bundle" in diag.message for diag in diagnostics)


# gabion:evidence E:function_site::server.py::gabion.server._analysis_witness_config_payload
def test_analysis_witness_config_payload_is_stable() -> None:
    server = _load()
    config = server.AuditConfig(
        exclude_dirs={"z", "a"},
        ignore_params={"tail", "head"},
        strictness="high",
        external_filter=True,
        transparent_decorators={"pkg.wrap", "alpha.wrap"},
    )

    payload = server._analysis_witness_config_payload(config)

    assert payload["exclude_dirs"] == ["a", "z"]
    assert payload["ignore_params"] == ["head", "tail"]
    assert payload["transparent_decorators"] == ["alpha.wrap", "pkg.wrap"]

# gabion:evidence E:function_site::server.py::gabion.server.start
def test_start_uses_injected_callable() -> None:
    server = _load()
    called = {"value": False}

    def _start() -> None:
        called["value"] = True

    server.start(_start)
    assert called["value"] is True


def test_deadline_tick_budget_allows_check_non_meter_clock() -> None:
    server = _load()

    class _Clock:
        pass

    assert server._deadline_tick_budget_allows_check(_Clock()) is True


# gabion:evidence E:function_site::server.py::gabion.server._diagnostics_for_path
def test_diagnostics_for_path_is_stable_for_shuffled_bundle_insertion_order(monkeypatch) -> None:
    server = _load()

    class _Result:
        def __init__(self, span_items: list[tuple[str, tuple[int, int, int, int]]]) -> None:
            self.groups_by_path = {
                "/tmp/sample.py": {
                    "caller": [("a", "b")],
                }
            }
            self.param_spans_by_path = {
                "/tmp/sample.py": {
                    "caller": dict(span_items)
                }
            }

    monkeypatch.setattr(
        server,
        "analyze_paths",
        lambda *args, **kwargs: _Result([
            ("a", (1, 0, 1, 1)),
            ("b", (1, 2, 1, 3)),
        ]),
    )
    stable = server._diagnostics_for_path("/tmp/sample.py", None)

    monkeypatch.setattr(
        server,
        "analyze_paths",
        lambda *args, **kwargs: _Result([
            ("b", (1, 2, 1, 3)),
            ("a", (1, 0, 1, 1)),
        ]),
    )
    shuffled = server._diagnostics_for_path("/tmp/sample.py", None)

    assert [diag.message for diag in stable] == [diag.message for diag in shuffled]
