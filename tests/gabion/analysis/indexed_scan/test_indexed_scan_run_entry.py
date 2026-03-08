from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from gabion.analysis.indexed_scan.scanners import run_entry
from gabion.exceptions import NeverThrown


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "fail_on_type_ambiguities": False,
        "type_audit": False,
        "fingerprint_deadness_json": None,
        "fingerprint_coherence_json": None,
        "fingerprint_rewrite_plans_json": None,
        "fingerprint_exception_obligations_json": None,
        "fingerprint_handledness_json": None,
        "exclude": None,
        "ignore_params": None,
        "transparent_decorators": None,
        "config": None,
        "root": ".",
        "allow_external": None,
        "strictness": None,
        "baseline": None,
        "baseline_write": False,
        "paths": ["."],
        "emit_decision_snapshot": None,
        "report": None,
        "fail_on_violations": False,
        "synthesis_plan": False,
        "synthesis_report": False,
        "synthesis_property_hook_hypothesis": False,
        "lint": False,
        "no_recursive": False,
        "type_audit_report": False,
        "type_audit_max": 50,
        "wl_refinement": False,
        "dot": None,
        "emit_structure_tree": None,
        "emit_structure_metrics": None,
        "analysis_timeout_ticks": 100,
        "analysis_timeout_tick_ns": 1_000,
        "analysis_tick_limit": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _deps(overrides: dict[str, object] | None = None) -> run_entry.RunImplDeps:
    values: dict[str, object] = {
        "dataflow_defaults_fn": lambda _root, _config: {},
        "synthesis_defaults_fn": lambda _root, _config: {},
        "decision_defaults_fn": lambda _root, _config: {},
        "decision_tier_map_fn": lambda _section: {},
        "decision_require_tiers_fn": lambda _section: False,
        "decision_ignore_list_fn": lambda _section: [],
        "exception_defaults_fn": lambda _root, _config: {},
        "exception_marker_family_fn": lambda _section, _family: set(),
        "exception_never_list_fn": lambda _section: set(),
        "fingerprint_defaults_fn": lambda _root, _config: {},
        "merge_payload_fn": lambda payload, _defaults: payload,
        "dataflow_deadline_roots_fn": lambda _merged: set(),
        "dataflow_adapter_payload_fn": lambda _merged: {},
        "dataflow_required_surfaces_fn": lambda _merged: [],
        "normalize_adapter_contract_fn": lambda payload: payload,
        "resolve_baseline_path_fn": run_entry.resolve_baseline_path,
        "resolve_synth_registry_path_fn": run_entry.resolve_synth_registry_path,
        "iter_paths_fn": lambda _paths, _config: [Path("sample.py")],
        "load_json_fn": lambda _path: {},
        "build_fingerprint_registry_fn": lambda _spec, registry_seed=None: ("registry", {}),
        "build_synth_registry_from_payload_fn": lambda _payload, _registry: None,
        "type_constructor_registry_cls": lambda registry: {"registry": registry},
        "default_marker_aliases": run_entry.DEFAULT_MARKER_ALIASES,
        "audit_config_cls": lambda **kwargs: SimpleNamespace(**kwargs),
        "forest_cls": lambda: "forest",
        "run_output_context_factory": lambda **kwargs: SimpleNamespace(**kwargs),
        "finalize_run_outputs_fn": lambda **kwargs: SimpleNamespace(exit_code=0),
    }
    if overrides:
        values.update(overrides)
    return run_entry.RunImplDeps(**values)


# gabion:evidence E:function_site::indexed_scan/run_entry.py::gabion.analysis.indexed_scan.run_entry.resolve_baseline_path E:function_site::indexed_scan/run_entry.py::gabion.analysis.indexed_scan.run_entry.resolve_synth_registry_path E:function_site::indexed_scan/run_entry.py::gabion.analysis.indexed_scan.run_entry.normalize_transparent_decorators
# gabion:behavior primary=desired
def test_path_and_transparent_decorator_helpers(tmp_path: Path) -> None:
    rel = run_entry.resolve_baseline_path("baseline.json", tmp_path)
    assert rel == tmp_path / "baseline.json"
    assert run_entry.resolve_baseline_path(None, tmp_path) is None

    marker_dir = tmp_path / "artifacts"
    marker_dir.mkdir(parents=True)
    (marker_dir / "LATEST.txt").write_text("stamp", encoding="utf-8")
    expected = marker_dir / "stamp" / "fingerprint_synth.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}", encoding="utf-8")

    assert (
        run_entry.resolve_synth_registry_path("artifacts/LATEST/fingerprint_synth.json", tmp_path)
        == expected.resolve()
    )
    assert run_entry.resolve_synth_registry_path(" ", tmp_path) is None

    normalized = run_entry.normalize_transparent_decorators(["a, b", "", 3])
    assert normalized == {"a", "b"}


# gabion:evidence E:function_site::indexed_scan/run_entry.py::gabion.analysis.indexed_scan.run_entry.analysis_deadline_scope
# gabion:behavior primary=verboten facets=invalid
def test_analysis_deadline_scope_rejects_invalid_ingress() -> None:
    with pytest.raises(NeverThrown):
        with run_entry.analysis_deadline_scope(
            _args(analysis_timeout_ticks=0, analysis_timeout_tick_ns=1)
        ):
            pass

    with pytest.raises(NeverThrown):
        with run_entry.analysis_deadline_scope(
            _args(analysis_timeout_ticks=10, analysis_timeout_tick_ns=1, analysis_tick_limit=0)
        ):
            pass


# gabion:evidence E:function_site::indexed_scan/run_entry.py::gabion.analysis.indexed_scan.run_entry.run_impl
# gabion:behavior primary=desired
def test_run_impl_requires_baseline_path_for_writes(capsys: pytest.CaptureFixture[str]) -> None:
    args = _args(baseline_write=True)
    deps = _deps(
        {
            "resolve_baseline_path_fn": lambda _path, _root: None,
        }
    )

    exit_code = run_entry.run_impl(
        args,
        deps=deps,
        analyze_paths_fn=lambda *_args, **_kwargs: object(),
        emit_report_fn=lambda *_args, **_kwargs: ("", []),
        compute_violations_fn=lambda *_args, **_kwargs: [],
    )

    assert exit_code == 2
    assert "Baseline path required" in capsys.readouterr().err


# gabion:evidence E:function_site::indexed_scan/run_entry.py::gabion.analysis.indexed_scan.run_entry.run_impl
# gabion:behavior primary=desired
def test_run_impl_hydrates_fingerprint_registry_and_executes_analysis(tmp_path: Path) -> None:
    capture: dict[str, object] = {}
    seed_path = tmp_path / "seed.json"
    synth_path = tmp_path / "synth.json"
    seed_path.write_text("{}", encoding="utf-8")
    synth_path.write_text("{}", encoding="utf-8")

    args = _args(
        fail_on_type_ambiguities=True,
        exclude=["a,b"],
        ignore_params="x, y",
        transparent_decorators="trace,log",
        strictness="invalid",
        report="report.md",
        lint=True,
        fail_on_violations=True,
        synthesis_plan=True,
        baseline="baseline.json",
    )

    deps = _deps(
        {
            "fingerprint_defaults_fn": lambda _root, _config: {
                "bundle.alpha": ["x"],
                "synth_min_occurrences": "2",
                "synth_version": "synth@2",
                "synth_registry_path": "synth",
                "fingerprint_seed_path": "seed",
                "seed_revision": "rev-a",
            },
            "resolve_synth_registry_path_fn": (
                lambda value, _root: seed_path if str(value) == "seed" else synth_path
            ),
            "load_json_fn": lambda path: {"origin": path.name},
            "build_fingerprint_registry_fn": (
                lambda spec, registry_seed=None: (
                    f"registry:{spec}",
                    {"fp": {str(registry_seed)}},
                )
            ),
            "build_synth_registry_from_payload_fn": (
                lambda payload, registry: {"payload": payload, "registry": registry}
            ),
            "iter_paths_fn": lambda _paths, config: [
                Path(str(config.project_root)) / "module.py"
            ],
            "run_output_context_factory": lambda **kwargs: capture.setdefault(
                "context",
                SimpleNamespace(**kwargs),
            ),
            "finalize_run_outputs_fn": lambda **kwargs: SimpleNamespace(exit_code=17),
        }
    )

    def _analyze(paths, **kwargs):
        capture["analyze_paths"] = paths
        capture["analyze_kwargs"] = kwargs
        return object()

    exit_code = run_entry.run_impl(
        args,
        deps=deps,
        analyze_paths_fn=_analyze,
        emit_report_fn=lambda *_args, **_kwargs: ("", []),
        compute_violations_fn=lambda *_args, **_kwargs: [],
    )

    assert exit_code == 17
    assert args.type_audit is True
    analyze_kwargs = cast(dict[str, object], capture["analyze_kwargs"])
    assert analyze_kwargs["include_decision_surfaces"] is True
    config = analyze_kwargs["config"]
    assert config.strictness == "high"
    assert config.transparent_decorators == {"trace", "log"}
    assert config.fingerprint_registry is not None
    assert config.fingerprint_synth_registry is not None


# gabion:evidence E:function_site::indexed_scan/run_entry.py::gabion.analysis.indexed_scan.run_entry.run_impl
# gabion:behavior primary=verboten facets=empty,error
def test_run_impl_handles_fingerprint_io_errors_and_empty_index() -> None:
    args = _args(strictness="low")
    capture: dict[str, object] = {}

    deps = _deps(
        {
            "fingerprint_defaults_fn": lambda _root, _config: {
                "bundle.alpha": ["x"],
                "fingerprint_seed_path": "seed",
                "synth_registry_path": "synth",
            },
            "resolve_synth_registry_path_fn": lambda _value, _root: Path("missing.json"),
            "load_json_fn": lambda _path: (_ for _ in ()).throw(OSError("missing")),
            "build_fingerprint_registry_fn": lambda _spec, registry_seed=None: ("registry", {}),
            "run_output_context_factory": lambda **kwargs: capture.setdefault(
                "context",
                SimpleNamespace(**kwargs),
            ),
            "finalize_run_outputs_fn": lambda **kwargs: SimpleNamespace(exit_code=11),
        }
    )

    exit_code = run_entry.run_impl(
        args,
        deps=deps,
        analyze_paths_fn=lambda *_args, **kwargs: capture.setdefault("analyze_kwargs", kwargs),
        emit_report_fn=lambda *_args, **_kwargs: ("", []),
        compute_violations_fn=lambda *_args, **_kwargs: [],
    )

    assert exit_code == 11
    analyze_kwargs = cast(dict[str, object], capture["analyze_kwargs"])
    config = analyze_kwargs["config"]
    assert config.strictness == "low"
    assert config.fingerprint_registry is None
    assert config.constructor_registry is None
    assert config.fingerprint_synth_registry is None
