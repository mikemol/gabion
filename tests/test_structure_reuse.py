from __future__ import annotations

import json
from pathlib import Path



def _parse_stub_payload(stubs: str) -> dict[str, object]:
    start = stubs.find("{")
    return json.loads(stubs[start:])

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse._record::child_count,value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::stale_647c5a38c22b_2e7dfa65
def test_compute_structure_reuse_detects_repeated_subtrees() -> None:
    da = _load()
    snapshot = {
        "format_version": 1,
        "root": ".",
        "files": [
            {
                "path": "a.py",
                "functions": [
                    {"name": "f", "bundles": [["a", "b"], ["c"]]},
                ],
            },
            {
                "path": "b.py",
                "functions": [
                    {"name": "g", "bundles": [["b", "a"], ["c"]]},
                ],
            },
        ],
    }
    reuse = da.compute_structure_reuse(snapshot, min_count=2)
    assert reuse["forest_signature_partial"] is True
    assert reuse["forest_signature_basis"] == "missing"
    kinds = {entry.get("kind") for entry in reuse.get("reused", [])}
    assert "bundle" in kinds
    assert "function" in kinds
    bundle_entries = [
        entry for entry in reuse["reused"] if entry.get("kind") == "bundle"
    ]
    assert all("aspf_structure_class" in entry for entry in bundle_entries)
    assert any(entry.get("value") == ["a", "b"] for entry in bundle_entries)
    suggestions = reuse.get("suggested_lemmas", [])
    assert any(
        entry.get("kind") == "bundle"
        and entry.get("suggested_name")
        and entry.get("witness_obligations")
        and entry.get("rewrite_plan_artifact")
        for entry in suggestions
    )
    replacement_map = reuse.get("replacement_map", {})
    assert any(location.startswith("a.py::f") for location in replacement_map)

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.render_reuse_lemma_stubs E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_reuse_lemma_stubs::stale_fb5b482d5857_c1880e7c
def test_render_reuse_lemma_stubs_includes_names() -> None:
    da = _load()
    reuse = {
        "format_version": 1,
        "min_count": 2,
        "suggested_lemmas": [
            {
                "hash": "deadbeef",
                "kind": "bundle",
                "count": 2,
                "suggested_name": "_gabion_bundle_lemma_deadbeef",
                "locations": ["a.py::f::bundle:a,b"],
                "value": ["a", "b"],
            }
        ],
    }
    stubs = da.render_reuse_lemma_stubs(reuse)
    payload = _parse_stub_payload(stubs)
    assert payload.get("artifact_kind") == "reuse_rewrite_plan_bundle"
    assert payload.get("plans") == []

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse._record::child_count,value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::stale_001c41908b1c
def test_structure_reuse_prefers_declared_bundle_names(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class UserContext:\n"
        "    user_id: int\n"
        "    request_id: str\n"
    )
    snapshot = {
        "format_version": 1,
        "root": str(tmp_path),
        "files": [
            {
                "path": "mod.py",
                "functions": [
                    {"name": "f", "bundles": [["user_id", "request_id"]]},
                    {"name": "g", "bundles": [["request_id", "user_id"]]},
                ],
            }
        ],
    }
    reuse = da.compute_structure_reuse(snapshot, min_count=2)
    suggestions = reuse.get("suggested_lemmas", [])
    assert any(
        entry.get("kind") == "bundle"
        and entry.get("suggested_name") == "UserContext"
        and entry.get("name_source") == "declared_bundle"
        for entry in suggestions
    )
    assert reuse.get("warnings") == []


def test_compute_structure_reuse_candidate_generation_is_deterministic() -> None:
    da = _load()
    snapshot_a = {
        "root": ".",
        "files": [
            {"path": "z.py", "functions": [{"name": "z", "bundles": [["k", "v"], ["x"]]}]},
            {"path": "a.py", "functions": [{"name": "a", "bundles": [["v", "k"], ["x"]]}]},
        ],
    }
    snapshot_b = {
        "root": ".",
        "files": list(reversed(snapshot_a["files"])),
    }
    reuse_a = da.compute_structure_reuse(snapshot_a, min_count=2)
    reuse_b = da.compute_structure_reuse(snapshot_b, min_count=2)
    sig_a = [
        (entry.get("hash"), entry.get("kind"), entry.get("suggested_name"))
        for entry in reuse_a.get("suggested_lemmas", [])
    ]
    sig_b = [
        (entry.get("hash"), entry.get("kind"), entry.get("suggested_name"))
        for entry in reuse_b.get("suggested_lemmas", [])
    ]
    assert sig_a == sig_b


def test_render_reuse_lemma_stubs_emits_plan_artifacts_from_reuse_suggestions() -> None:
    da = _load()
    snapshot = {
        "root": ".",
        "files": [
            {"path": "a.py", "functions": [{"name": "f", "bundles": [["x", "y"]]}]},
            {"path": "b.py", "functions": [{"name": "g", "bundles": [["y", "x"]]}]},
        ],
    }
    reuse = da.compute_structure_reuse(snapshot, min_count=2)
    stubs = da.render_reuse_lemma_stubs(reuse)
    payload = _parse_stub_payload(stubs)
    plans = payload.get("plans", [])
    assert plans
    assert any(
        any(
            obligation.get("kind") == "aspf_structure_class_equivalence"
            for obligation in plan.get("evidence", {}).get("witness_obligations", [])
        )
        for plan in plans
    )
