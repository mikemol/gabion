from __future__ import annotations

from pathlib import Path
import ast
import sys

import pytest


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _write(path: Path, content: str) -> None:
    path.write_text(content)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._extract_invariant_from_expr::expr E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._invariant_term::expr,params
def test_extract_invariant_from_expr_edges() -> None:
    da = _load()
    expr = ast.parse("x").body[0].value
    assert (
        da._extract_invariant_from_expr(expr, {"x"}, scope="s", source="src")
        is None
    )
    expr = ast.parse("a == b == c").body[0].value
    assert (
        da._extract_invariant_from_expr(expr, {"a", "b"}, scope="s", source="src")
        is None
    )
    expr = ast.parse("a != b").body[0].value
    assert (
        da._extract_invariant_from_expr(expr, {"a", "b"}, scope="s", source="src")
        is None
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._scope_path::root
def test_invariant_collector_skips_nested_defs_and_lambda(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(
        path,
        "def outer(a, b):\n"
        "    assert a == b\n"
        "    def inner():\n"
        "        assert a == b\n"
        "    async def inner_async():\n"
        "        assert a == b\n"
        "    x = (lambda y: y)\n",
    )
    props = da._collect_invariant_propositions(
        path,
        ignore_params=set(),
        project_root=tmp_path,
    )
    assert len(props) == 1


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._scope_path::root
def test_collect_invariant_emitters_type_error(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n")

    def bad_emitter(_fn):
        return ["bad"]

    with pytest.raises(TypeError):
        da._collect_invariant_propositions(
            path,
            ignore_params=set(),
            project_root=tmp_path,
            emitters=[bad_emitter],
        )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._format_invariant_proposition::prop
def test_format_invariant_proposition_variants() -> None:
    da = _load()
    prop = da.InvariantProposition(
        form="Equal", terms=("a", "b"), scope="mod.py:f", source="assert"
    )
    assert "mod.py:f" in da._format_invariant_proposition(prop)
    other = da.InvariantProposition(form="LessThan", terms=("a",), scope="", source="")
    assert "LessThan" in da._format_invariant_proposition(other)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._mark_param_roots::params
def test_decision_helpers_cover_paths() -> None:
    da = _load()
    assert da._decision_root_name(ast.parse("1").body[0].value) is None
    assert da._decision_root_name(ast.parse("user.id").body[0].value) == "user"
    found: set[str] = set()
    da._mark_param_roots(ast.parse("user['id']").body[0].value, {"user"}, found)
    assert "user" in found
    assert da._contains_boolish(ast.parse("not flag").body[0].value)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params
def test_decision_surface_params_match_ifexp_and_match() -> None:
    da = _load()
    tree = ast.parse(
        "def f(a, b):\n"
        "    x = a if b else b\n"
        "    match a:\n"
        "        case _ if b:\n"
        "            return a\n"
        "    return x\n"
    )
    fn = tree.body[0]
    params = da._decision_surface_params(fn, ignore_params=set())
    assert "b" in params


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_names::fn,ignore_params E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._mark_param_roots::params
def test_value_encoded_decision_params_branches() -> None:
    da = _load()
    tree = ast.parse(
        "def f(flag):\n"
        "    value = obj.max(flag, 1)\n"
        "    value2 = flag & 1\n"
        "    value3 = (flag == 1) * 2\n"
        "    return value + value2 + value3\n"
    )
    fn = tree.body[0]
    params, reasons = da._value_encoded_decision_params(fn, ignore_params=set())
    assert "flag" in params
    assert "min/max" in reasons
    assert "bitmask" in reasons
    assert "boolean arithmetic" in reasons


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_analyze_decision_surfaces_repo_warnings(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "module.py"
    _write(
        module_path,
        "def callee(a):\n"
        "    if a:\n"
        "        return 1\n"
        "\n"
        "def caller(a):\n"
        "    return callee(a)\n"
        "\n"
        "def orphan(c):\n"
        "    if c:\n"
        "        return 1\n"
        "\n"
        "def missing_call(a):\n"
        "    unknown(a)\n",
    )
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_path = test_dir / "test_sample.py"
    _write(
        test_path,
        "def callee(a):\n"
        "    if a:\n"
        "        return 1\n"
        "\n"
        "def caller(a):\n"
        "    return callee(a)\n",
    )
    surfaces, warnings, lint_lines = da.analyze_decision_surfaces_repo(
        [module_path, test_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        decision_tiers={"a": 2},
        require_tiers=True,
    )
    assert surfaces
    assert any("missing decision tier metadata" in warning for warning in warnings)
    assert any("tier-2 decision param" in warning for warning in warnings)
    assert any("GABION_DECISION_TIER" in line for line in lint_lines)
    assert any("GABION_DECISION_SURFACE" in line for line in lint_lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._is_test_path::path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._decision_tier_for::tier_map
def test_analyze_value_encoded_decisions_repo_warnings(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "module.py"
    _write(
        module_path,
        "def vcallee(flag):\n"
        "    return (flag == 1) * 2\n"
        "\n"
        "def vcaller(flag):\n"
        "    return vcallee(flag)\n"
        "\n"
        "def vmystery(x):\n"
        "    return (x == 1) * 2\n",
    )
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_path = test_dir / "test_value.py"
    _write(
        test_path,
        "def vcallee(flag):\n"
        "    return (flag == 1) * 2\n"
        "\n"
        "def vcaller(flag):\n"
        "    return vcallee(flag)\n",
    )
    surfaces, warnings, rewrites, lint_lines = da.analyze_value_encoded_decisions_repo(
        [module_path, test_path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        decision_tiers={"flag": 2},
        require_tiers=True,
    )
    assert surfaces
    assert rewrites
    assert any("missing decision tier metadata" in warning for warning in warnings)
    assert any("tier-2 value-encoded" in warning for warning in warnings)
    assert any("GABION_VALUE_DECISION_TIER" in line for line in lint_lines)
    assert any("GABION_VALUE_DECISION_SURFACE" in line for line in lint_lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._param_annotations::fn,ignore_params
def test_param_annotations_by_path_skips_parse_errors(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.py"
    _write(bad, "def f(:\n")
    result = da._param_annotations_by_path([bad], ignore_params=set())
    assert bad not in result


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_synth_registry_path::path
def test_resolve_synth_registry_path_latest(tmp_path: Path) -> None:
    da = _load()
    root = tmp_path
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stamp = "20240203_000000"
    (out_dir / "LATEST.txt").write_text(stamp)
    (out_dir / stamp).mkdir()
    expected = (out_dir / stamp / "fingerprint_synth.json").resolve()
    expected.write_text("{}")
    resolved = da._resolve_synth_registry_path(
        "out/LATEST/fingerprint_synth.json", root
    )
    assert resolved == expected
    missing = da._resolve_synth_registry_path(
        "missing/LATEST/fingerprint_synth.json", root
    )
    assert missing is None
    relative = da._resolve_synth_registry_path("other.json", root)
    assert relative == (root / "other.json").resolve()
    assert da._resolve_synth_registry_path(" ", root) is None
    assert da._resolve_synth_registry_path(None, root) is None


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint::fingerprint E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index
def test_compute_fingerprint_matches_skips_missing_types() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}], "g": [{"b"}]}}
    annotations_by_path = {
        path: {"f": {"a": None}, "g": {"b": "int"}},
    }
    registry = da.PrimeRegistry()
    fingerprint = da.bundle_fingerprint_dimensional(["str"], registry, None)
    index = {fingerprint: {"Known"}}
    matches = da._compute_fingerprint_matches(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
    )
    assert matches == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint::fingerprint E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index
def test_compute_fingerprint_matches_skips_missing_params() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {}}}
    registry = da.PrimeRegistry()
    fingerprint = da.bundle_fingerprint_dimensional(["int"], registry, None)
    index = {fingerprint: {"Known"}}
    matches = da._compute_fingerprint_matches(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
    )
    assert matches == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_compute_fingerprint_provenance_skips_none_types() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {"a": None}}}
    registry = da.PrimeRegistry()
    entries = da._compute_fingerprint_provenance(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index={},
    )
    assert entries == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_compute_fingerprint_provenance_skips_missing_params() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {}}}
    registry = da.PrimeRegistry()
    entries = da._compute_fingerprint_provenance(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index={},
    )
    assert entries == []


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples
def test_summarize_fingerprint_provenance_empty() -> None:
    da = _load()
    assert da._summarize_fingerprint_provenance([]) == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_compute_fingerprint_synth_skips_none_types() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {"a": None}}}
    registry = da.PrimeRegistry()
    ctor_registry = da.TypeConstructorRegistry(registry)
    lines, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=ctor_registry,
        min_occurrences=2,
        version="synth@1",
    )
    assert lines == []
    assert payload is None


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_compute_fingerprint_synth_skips_missing_params() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}]}}
    annotations_by_path = {path: {"f": {}}}
    registry = da.PrimeRegistry()
    ctor_registry = da.TypeConstructorRegistry(registry)
    lines, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=ctor_registry,
        min_occurrences=2,
        version="synth@1",
    )
    assert lines == []
    assert payload is None


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_compute_fingerprint_synth_includes_ctor_keys() -> None:
    da = _load()
    path = Path("mod.py")
    groups_by_path = {path: {"f": [{"a"}], "g": [{"a"}]}}
    annotations_by_path = {
        path: {"f": {"a": "list[int]"}, "g": {"a": "list[int]"}}
    }
    registry = da.PrimeRegistry()
    ctor_registry = da.TypeConstructorRegistry(registry)
    lines, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=ctor_registry,
        min_occurrences=2,
        version="synth@1",
    )
    assert payload is not None
    assert any("ctor=" in line for line in lines)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_render_structure_snapshot_skips_invalid_invariant_scope(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    invalid = da.InvariantProposition(
        form="Equal", terms=("a", "b"), scope="bad_scope", source="assert"
    )
    valid = da.InvariantProposition(
        form="Equal", terms=("a", "b"), scope="mod.py:f", source="assert"
    )
    snapshot = da.render_structure_snapshot(
        groups_by_path,
        project_root=tmp_path,
        invariant_propositions=[invalid, valid],
    )
    files = snapshot.get("files") or []
    assert files
    functions = files[0].get("functions") or []
    assert functions
    assert "invariants" in functions[0]


def test_load_decision_snapshot_errors(tmp_path: Path) -> None:
    da = _load()
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(ValueError):
        da.load_decision_snapshot(bad)
    not_obj = tmp_path / "list.json"
    not_obj.write_text("[]")
    with pytest.raises(ValueError):
        da.load_decision_snapshot(not_obj)


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._iter_paths::config E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_paths::config,include_bundle_forest,include_coherence_witnesses,include_constant_smells,include_deadness_witnesses,include_decision_surfaces,include_exception_obligations,include_handledness_witnesses,include_invariant_propositions,include_lint_lines,include_never_invariants,include_rewrite_plans,include_unused_arg_smells,include_value_decision_surfaces,type_audit,type_audit_report E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_file_internal::config,recursive E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_rewrite_plans::exception_obligations E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants::forest E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_decision_surfaces_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_value_encoded_decisions_repo::forest,require_tiers E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._populate_bundle_forest::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_exception_obligations::handledness_witnesses E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.analyze_unused_arg_flow_repo::strictness
def test_exception_obligations_enum_and_handledness(tmp_path: Path) -> None:
    da = _load()
    module_path = tmp_path / "mod.py"
    _write(
        module_path,
        "def f(a):\n"
        "    raise ValueError('bad')\n"
        "\n"
        "def g(b):\n"
        "    try:\n"
        "        raise RuntimeError('oops')\n"
        "    except Exception:\n"
        "        return b\n",
    )
    config = da.AuditConfig(
        project_root=tmp_path,
        exclude_dirs=set(),
        ignore_params=set(),
        external_filter=False,
        strictness="high",
    )
    analysis = da.analyze_paths(
        [tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_exception_obligations=True,
        include_handledness_witnesses=True,
        config=config,
    )
    obligations = analysis.exception_obligations
    assert any(entry["status"] == "UNKNOWN" for entry in obligations)
    assert any(entry["status"] == "HANDLED" for entry in obligations)
    assert analysis.handledness_witnesses
