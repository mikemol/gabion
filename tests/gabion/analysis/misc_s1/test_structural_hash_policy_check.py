from __future__ import annotations

from pathlib import Path

from scripts.policy import structural_hash_policy_check


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# gabion:evidence E:call_footprint::tests/test_structural_hash_policy_check.py::test_structural_hash_policy_check_accepts_clean_identity_paths::structural_hash_policy_check.py::scripts.structural_hash_policy_check.collect_violations
def test_structural_hash_policy_check_accepts_clean_identity_paths(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/analysis/derivation_contract.py",
        "from __future__ import annotations\n\nVALUE = 1\n",
    )
    _write(
        tmp_path / "src/gabion/analysis/derivation_graph.py",
        "from __future__ import annotations\n\ndef f() -> int:\n    return 1\n",
    )
    _write(
        tmp_path / "src/gabion/analysis/derivation_cache.py",
        "from __future__ import annotations\n\ndef f() -> int:\n    return 1\n",
    )
    _write(
        tmp_path / "src/gabion/analysis/aspf.py",
        "from __future__ import annotations\n\ndef structural_key_atom(value, *, source):\n    return value\n",
    )

    violations = structural_hash_policy_check.collect_violations(root=tmp_path)

    assert violations == []


# gabion:evidence E:call_footprint::tests/test_structural_hash_policy_check.py::test_structural_hash_policy_check_rejects_digest_identity_calls::structural_hash_policy_check.py::scripts.structural_hash_policy_check.collect_violations
def test_structural_hash_policy_check_rejects_digest_identity_calls(tmp_path: Path) -> None:
    _write(
        tmp_path / "src/gabion/analysis/derivation_contract.py",
        "from __future__ import annotations\n\nVALUE = 1\n",
    )
    _write(
        tmp_path / "src/gabion/analysis/derivation_graph.py",
        "from __future__ import annotations\nimport hashlib\n\ndef f(value: str) -> str:\n    return hashlib.sha1(value.encode(\"utf-8\")).hexdigest()\n",
    )
    _write(
        tmp_path / "src/gabion/analysis/derivation_cache.py",
        "from __future__ import annotations\n\ndef f() -> int:\n    return 1\n",
    )

    violations = structural_hash_policy_check.collect_violations(root=tmp_path)

    assert violations
    assert any("hashlib" in entry.call for entry in violations)



# gabion:evidence E:call_footprint::tests/test_structural_hash_policy_check.py::test_structural_hash_policy_check_writes_policy_result_output::structural_hash_policy_check.py::scripts.structural_hash_policy_check.run
def test_structural_hash_policy_check_writes_policy_result_output(tmp_path: Path) -> None:
    _write(tmp_path / "src/gabion/analysis/derivation_contract.py", "VALUE = 1\n")
    _write(tmp_path / "src/gabion/analysis/derivation_graph.py", "def f():\n    return 1\n")
    _write(tmp_path / "src/gabion/analysis/derivation_cache.py", "def f():\n    return 1\n")
    out = tmp_path / "out/result.json"
    code = structural_hash_policy_check.run(root=tmp_path, output=out)
    assert code == 0
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["rule_id"] == "structural_hash"
    assert payload["status"] == "pass"
    assert payload["violations"] == []
