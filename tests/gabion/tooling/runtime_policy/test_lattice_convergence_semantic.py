from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gabion.tooling.policy_substrate import lattice_convergence_semantic


@dataclass(frozen=True)
class _StubWitness:
    complete: bool
    violation: object | None


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_semantic_lattice_convergence_request_order_is_deterministic(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "a.py",
        "def z(x):\n    if x:\n        return 1\n    return 0\n\nif True:\n    pass\n",
    )
    _write(
        tmp_path / "b.py",
        "def f(items):\n    for item in items:\n        if item:\n            return item\n    return None\n",
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_CANONICAL_CORPUS",
        ("b.py", "a.py"),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_collect_linkage_diagnostics",
        lambda: (),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "build_fiber_bundle_for_qualname",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "compute_lattice_witness",
        lambda **_: _StubWitness(complete=True, violation=None),
    )

    first = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    second = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    assert first.evaluated_requests == second.evaluated_requests
    assert first.policy_data() == second.policy_data()
    assert first.error_count == 0
    assert first.policy_data()["witness_rows"] == []
    assert tuple(request.path for request in first.evaluated_requests) == tuple(
        sorted(request.path for request in first.evaluated_requests)
    )


def test_semantic_lattice_convergence_parse_and_read_failures_increment_error_count(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "bad.py",
        "def broken(:\n    pass\n",
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_CANONICAL_CORPUS",
        ("missing.py", "bad.py"),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_collect_linkage_diagnostics",
        lambda: (),
    )

    report = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    codes = tuple(item.code for item in report.diagnostics)
    assert codes == (
        "lattice_corpus_parse_failure",
        "lattice_corpus_read_failure",
    )
    assert report.error_count == 2
    witness_rows = report.policy_data()["witness_rows"]
    assert isinstance(witness_rows, list)
    assert len(witness_rows) == 2
    assert all(isinstance(row, dict) for row in witness_rows)
    assert all(row.get("witness_kind") == "unmapped_witness" for row in witness_rows)
    assert all(row.get("mapping_complete") is False for row in witness_rows)
    assert all(row.get("boundary_crossed") is True for row in witness_rows)
    assert all("obligation_state" not in row for row in witness_rows)


def test_semantic_lattice_convergence_counts_incomplete_or_violation_once_per_request(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "one.py",
        "if True:\n    pass\n",
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_CANONICAL_CORPUS",
        ("one.py",),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_collect_linkage_diagnostics",
        lambda: (),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "build_fiber_bundle_for_qualname",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "compute_lattice_witness",
        lambda **_: _StubWitness(complete=False, violation=object()),
    )

    report = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    assert report.evaluated_request_count == 1
    assert report.error_count == 1
    assert tuple(item.code for item in report.diagnostics) == (
        "lattice_witness_incomplete_or_violation",
    )
    witness_row = report.policy_data()["witness_rows"][0]
    assert witness_row["witness_incomplete"] is True
    assert witness_row["witness_violation"] is True


def test_semantic_convergence_payload_stays_pre_transform_shape() -> None:
    source = Path("src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py").read_text(
        encoding="utf-8"
    )
    assert "obligation_state" not in source


def test_semantic_lattice_convergence_linkage_checks_frontier_payload_contract() -> None:
    codes = {
        item.code for item in lattice_convergence_semantic._collect_linkage_diagnostics()
    }
    assert "lattice_linkage_missing_frontier_payload" not in codes
    assert "lattice_linkage_frontier_payload_failure" not in codes
    assert "lattice_linkage_frontier_payload_invalid" not in codes


def test_semantic_lattice_convergence_emits_canonical_semantic_rows_for_real_witnesses(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "one.py",
        "def f(x):\n    if x:\n        return 1\n    return 0\n",
    )
    monkeypatch.setattr(lattice_convergence_semantic, "_CANONICAL_CORPUS", ("one.py",))
    monkeypatch.setattr(lattice_convergence_semantic, "_collect_linkage_diagnostics", lambda: ())

    report = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )

    payload = report.policy_data()
    semantic_rows = payload["semantic_rows"]
    assert isinstance(semantic_rows, list)
    assert len(semantic_rows) == 1
    semantic_row = semantic_rows[0]
    assert semantic_row["surface"] == "projection_fiber"
    assert semantic_row["carrier_kind"] == "frontier_witness"
    assert semantic_row["obligation_state"] == "discharged"
    assert semantic_row["payload"]["structural_path"] == "f::branch[0]::branch:if::x"
    shacl_plans = payload["compiled_shacl_plans"]
    sparql_plans = payload["compiled_sparql_plans"]
    compiled_projection_bundles = payload["compiled_projection_semantic_bundles"]
    assert isinstance(shacl_plans, list)
    assert isinstance(sparql_plans, list)
    assert isinstance(compiled_projection_bundles, list)
    assert len(shacl_plans) == 1
    assert len(sparql_plans) == 1
    assert len(compiled_projection_bundles) == 5
    assert shacl_plans[0]["source_structural_identity"] == semantic_row["structural_identity"]
    assert sparql_plans[0]["source_structural_identity"] == semantic_row["structural_identity"]
    assert shacl_plans[0]["semantic_op"] == "reflect"
    assert sparql_plans[0]["semantic_op"] == "reflect"
    bundles_by_name = {bundle["spec_name"]: bundle for bundle in compiled_projection_bundles}
    assert set(bundles_by_name) == {
        "projection_fiber_frontier",
        "projection_fiber_reflection",
        "projection_fiber_reflective_boundary",
        "projection_fiber_support_reflection",
        "projection_fiber_witness_synthesis",
    }
    frontier_bundle = bundles_by_name["projection_fiber_frontier"]
    assert len(frontier_bundle["bindings"]) == 1
    assert frontier_bundle["bindings"][0]["quotient_face"] == "projection_fiber.frontier"
    assert frontier_bundle["compiled_shacl_plans"][0]["semantic_op"] == "quotient_face"
    assert frontier_bundle["compiled_sparql_plans"][0]["semantic_op"] == "quotient_face"
    reflection_bundle = bundles_by_name["projection_fiber_reflection"]
    assert reflection_bundle["bindings"] == []
    assert reflection_bundle["compiled_shacl_plans"][0]["semantic_op"] == "reflect"
    assert reflection_bundle["compiled_sparql_plans"][0]["semantic_op"] == "reflect"
    assert (
        reflection_bundle["compiled_shacl_plans"][0]["source_structural_identity"]
        == semantic_row["structural_identity"]
    )
    assert (
        reflection_bundle["compiled_sparql_plans"][0]["source_structural_identity"]
        == semantic_row["structural_identity"]
    )
    reflective_boundary_bundle = bundles_by_name["projection_fiber_reflective_boundary"]
    assert len(reflective_boundary_bundle["bindings"]) == 1
    assert (
        reflective_boundary_bundle["bindings"][0]["quotient_face"]
        == "projection_fiber.reflective_boundary"
    )
    assert (
        reflective_boundary_bundle["compiled_shacl_plans"][0]["source_structural_identity"]
        == semantic_row["structural_identity"]
    )
    assert (
        reflective_boundary_bundle["compiled_sparql_plans"][0]["source_structural_identity"]
        == semantic_row["structural_identity"]
    )
    support_reflection_bundle = bundles_by_name["projection_fiber_support_reflection"]
    assert support_reflection_bundle["bindings"] == []
    assert support_reflection_bundle["compiled_shacl_plans"][0]["semantic_op"] == "support_reflect"
    assert support_reflection_bundle["compiled_sparql_plans"][0]["semantic_op"] == "support_reflect"
    witness_synthesis_bundle = bundles_by_name["projection_fiber_witness_synthesis"]
    assert witness_synthesis_bundle["bindings"] == []
    assert witness_synthesis_bundle["compiled_shacl_plans"] == []
    assert witness_synthesis_bundle["compiled_sparql_plans"] == []
    assert payload["witness_rows"] == []


def test_semantic_lattice_convergence_is_lazy_until_first_pull(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "one.py",
        "def f(x):\n    if x:\n        return 1\n    return 0\n",
    )
    monkeypatch.setattr(lattice_convergence_semantic, "_CANONICAL_CORPUS", ("one.py",))
    monkeypatch.setattr(lattice_convergence_semantic, "_collect_linkage_diagnostics", lambda: ())
    state = {"iter_calls": 0, "yield_calls": 0}

    def _iter_lattice_witnesses(**kwargs: object):
        state["iter_calls"] += 1
        requests = kwargs.get("requests")
        assert isinstance(requests, tuple)
        for _ in requests:
            state["yield_calls"] += 1
            yield _StubWitness(complete=True, violation=None)

    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "iter_lattice_witnesses",
        _iter_lattice_witnesses,
    )
    iterator = lattice_convergence_semantic.iter_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    assert state["iter_calls"] == 0
    assert state["yield_calls"] == 0
    first = next(iterator)
    assert first.request is not None
    assert first.diagnostic is None
    assert state["iter_calls"] == 1
    assert state["yield_calls"] == 1


def test_semantic_lattice_convergence_partial_pull_only_demands_requested_slice(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "one.py",
        (
            "def f(x):\n"
            "    if x:\n"
            "        return 1\n"
            "    while x:\n"
            "        return 0\n"
            "    return 0\n"
        ),
    )
    monkeypatch.setattr(lattice_convergence_semantic, "_CANONICAL_CORPUS", ("one.py",))
    monkeypatch.setattr(lattice_convergence_semantic, "_collect_linkage_diagnostics", lambda: ())
    state = {"yield_calls": 0}

    def _iter_lattice_witnesses(**kwargs: object):
        requests = kwargs.get("requests")
        assert isinstance(requests, tuple)
        for _ in requests:
            state["yield_calls"] += 1
            yield _StubWitness(complete=True, violation=None)

    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "iter_lattice_witnesses",
        _iter_lattice_witnesses,
    )
    iterator = lattice_convergence_semantic.iter_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    event = next(iterator)
    assert event.request is not None
    assert state["yield_calls"] == 1


def test_semantic_lattice_convergence_uses_iter_not_direct_compute(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "one.py",
        "def f(x):\n    if x:\n        return 1\n    return 0\n",
    )
    monkeypatch.setattr(lattice_convergence_semantic, "_CANONICAL_CORPUS", ("one.py",))
    monkeypatch.setattr(lattice_convergence_semantic, "_collect_linkage_diagnostics", lambda: ())

    def _raise_direct_compute(**_: object):
        raise AssertionError("collector must not call compute_lattice_witness directly")

    def _iter_lattice_witnesses(**kwargs: object):
        requests = kwargs.get("requests")
        assert isinstance(requests, tuple)
        for _ in requests:
            yield _StubWitness(complete=True, violation=None)

    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "compute_lattice_witness",
        _raise_direct_compute,
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "iter_lattice_witnesses",
        _iter_lattice_witnesses,
    )
    report = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    assert report.error_count == 0
    assert report.evaluated_request_count >= 1


def test_semantic_lattice_convergence_cold_warm_cache_parity(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "one.py",
        "def f(x):\n    if x:\n        return 1\n    return 0\n",
    )
    monkeypatch.setattr(lattice_convergence_semantic, "_CANONICAL_CORPUS", ("one.py",))
    monkeypatch.setattr(lattice_convergence_semantic, "_collect_linkage_diagnostics", lambda: ())
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "_cache_root",
        lambda: cache_root,
    )

    first = lattice_convergence_semantic.collect_semantic_lattice_convergence(repo_root=tmp_path)
    second = lattice_convergence_semantic.collect_semantic_lattice_convergence(repo_root=tmp_path)
    assert first.policy_data() == second.policy_data()
