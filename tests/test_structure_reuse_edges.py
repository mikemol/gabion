from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import dataflow_audit as da

    return da

def _write(path: Path, content: str) -> None:
    path.write_text(content)

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse._record::child_count,value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count
def test_compute_structure_reuse_handles_edges(tmp_path: Path) -> None:
    da = _load()
    model_path = tmp_path / "models.py"
    _write(
        model_path,
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class Alpha:\n"
        "    a: int\n"
        "    b: int\n"
        "\n"
        "@dataclass\n"
        "class Beta:\n"
        "    a: int\n"
        "    b: int\n",
    )
    snapshot = {
        "root": str(tmp_path),
        "files": [
            "bad",
            {"path": 123},
            {
                "path": "mod.py",
                "functions": [
                    "bad",
                    {"name": 123},
                    {
                        "name": "f",
                        "bundles": [["x"], ["x"], ["z"], ["z"], ["a", "b"], ["a", "b"]],
                    },
                ],
            },
        ],
    }

    def hash_fn(kind, value, child_hashes):
        if kind == "bundle" and value == ("x",):
            return ""
        return f"{kind}:{value}:{len(child_hashes)}"

    reuse = da.compute_structure_reuse(snapshot, min_count=1, hash_fn=hash_fn)
    assert reuse["format_version"] == 1
    warnings = reuse.get("warnings") or []
    assert any("Missing declared bundle name" in warning for warning in warnings)
    suggestions = reuse.get("suggested_lemmas") or []
    assert any("name_candidates" in entry for entry in suggestions if isinstance(entry, dict))

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._build_reuse_replacement_map
def test_build_reuse_replacement_map_skips_non_string_locations() -> None:
    da = _load()
    suggested = [
        {"kind": "bundle", "hash": "h", "suggested_name": "X", "locations": [123, "ok"]}
    ]
    replacement = da._build_reuse_replacement_map(suggested)
    assert "ok" in replacement
    assert 123 not in replacement

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._build_reuse_replacement_map
def test_build_reuse_replacement_map_skips_non_list_locations() -> None:
    da = _load()
    suggested = [{"kind": "bundle", "hash": "h", "suggested_name": "X", "locations": "bad"}]
    replacement = da._build_reuse_replacement_map(suggested)
    assert replacement == {}

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.render_reuse_lemma_stubs
def test_render_reuse_lemma_stubs_no_suggestions() -> None:
    da = _load()
    stubs = da.render_reuse_lemma_stubs({"suggested_lemmas": []})
    assert "No lemma suggestions available" in stubs

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.render_reuse_lemma_stubs
def test_render_reuse_lemma_stubs_with_child_count() -> None:
    da = _load()
    reuse = {
        "suggested_lemmas": [
            {
                "suggested_name": "lemma",
                "kind": "bundle",
                "count": 2,
                "value": ["a", "b"],
                "child_count": 3,
            }
        ]
    }
    stubs = da.render_reuse_lemma_stubs(reuse)
    assert "child_count" in stubs

# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.render_reuse_lemma_stubs
def test_render_reuse_lemma_stubs_skips_invalid_names() -> None:
    da = _load()
    reuse = {"suggested_lemmas": [{"suggested_name": None, "kind": "bundle"}]}
    stubs = da.render_reuse_lemma_stubs(reuse)
    assert "def " not in stubs

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse._record::child_count,value E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_reuse::min_count
def test_compute_structure_reuse_skips_non_list_bundle() -> None:
    da = _load()
    snapshot = {
        "format_version": 1,
        "root": None,
        "files": [
            {
                "path": "mod.py",
                "functions": [
                    {"name": "f", "bundles": ["bad", ["a"]]},
                ],
            }
        ],
    }
    reuse = da.compute_structure_reuse(snapshot, min_count=2)
    assert reuse["format_version"] == 1
