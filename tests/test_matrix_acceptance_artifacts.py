from __future__ import annotations

from pathlib import Path
import json
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit

    return dataflow_audit


def _write_sample_module(path: Path) -> None:
    path.write_text(
        "from __future__ import annotations\n"
        "\n"
        "def callee(flag: int, x: int) -> int:\n"
        "    return x\n"
        "\n"
        "def caller_one(a: int, b: int) -> int:\n"
        "    callee(0, a)\n"
        "    callee(0, b)\n"
        "    return 0\n"
        "\n"
        "def caller_two(a: int, b: int) -> int:\n"
        "    callee(0, a)\n"
        "    callee(0, b)\n"
        "    return 0\n"
        "\n"
        "def raises(x: int) -> int:\n"
        "    try:\n"
        "        if x:\n"
        "            raise ValueError('boom')\n"
        "    except Exception:\n"
        "        return 0\n"
        "    return 1\n"
    )


def _write_config(path: Path) -> None:
    # Two identical glossary entries (pair_a/pair_b) create an ambiguity witness,
    # which drives coherence + rewrite-plan artifacts.
    path.write_text(
        "[fingerprints]\n"
        "pair_a = [\"int\", \"int\"]\n"
        "pair_b = [\"int\", \"int\"]\n"
        "user_context = [\"int\"]\n"
        "synth_min_occurrences = 2\n"
    )


def _run_with_artifacts(
    dataflow_audit,
    *,
    module_path: Path,
    root: Path,
    config_path: Path,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "synth": out_dir / "fingerprint_synth.json",
        "provenance": out_dir / "fingerprint_provenance.json",
        "deadness": out_dir / "fingerprint_deadness.json",
        "coherence": out_dir / "fingerprint_coherence.json",
        "rewrite_plans": out_dir / "fingerprint_rewrite_plans.json",
        "exception_obligations": out_dir / "fingerprint_exception_obligations.json",
        "handledness": out_dir / "fingerprint_handledness.json",
    }
    argv = [
        str(module_path),
        "--root",
        str(root),
        "--config",
        str(config_path),
        "--no-recursive",
        "--fingerprint-synth-json",
        str(paths["synth"]),
        "--fingerprint-provenance-json",
        str(paths["provenance"]),
        "--fingerprint-deadness-json",
        str(paths["deadness"]),
        "--fingerprint-coherence-json",
        str(paths["coherence"]),
        "--fingerprint-rewrite-plans-json",
        str(paths["rewrite_plans"]),
        "--fingerprint-exception-obligations-json",
        str(paths["exception_obligations"]),
        "--fingerprint-handledness-json",
        str(paths["handledness"]),
    ]
    assert dataflow_audit.run(argv) == 0
    for path in paths.values():
        assert path.exists()
    return paths


def test_matrix_artifacts_are_deterministic_and_have_required_fields(tmp_path: Path) -> None:
    dataflow_audit = _load()
    module_path = tmp_path / "sample.py"
    _write_sample_module(module_path)
    config_path = tmp_path / "gabion.toml"
    _write_config(config_path)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    paths_a = _run_with_artifacts(
        dataflow_audit,
        module_path=module_path,
        root=tmp_path,
        config_path=config_path,
        out_dir=out_a,
    )
    paths_b = _run_with_artifacts(
        dataflow_audit,
        module_path=module_path,
        root=tmp_path,
        config_path=config_path,
        out_dir=out_b,
    )

    bytes_a = {name: path.read_bytes() for name, path in paths_a.items()}
    bytes_b = {name: path.read_bytes() for name, path in paths_b.items()}
    assert bytes_a == bytes_b

    synth = json.loads(bytes_a["synth"])
    assert isinstance(synth, dict)
    assert "version" in synth
    assert "entries" in synth

    provenance = json.loads(bytes_a["provenance"])
    assert isinstance(provenance, list) and provenance
    prov_entry = provenance[0]
    for field in ("provenance_id", "path", "function", "bundle", "base_keys", "ctor_keys", "remainder"):
        assert field in prov_entry

    deadness = json.loads(bytes_a["deadness"])
    assert isinstance(deadness, list) and deadness
    dead_entry = deadness[0]
    for field in ("path", "function", "bundle", "environment", "predicate", "core", "result"):
        assert field in dead_entry

    coherence = json.loads(bytes_a["coherence"])
    assert isinstance(coherence, list) and coherence
    coh_entry = coherence[0]
    for field in ("coherence_id", "site", "boundary", "alternatives", "fork_signature", "frack_path", "result", "remainder"):
        assert field in coh_entry

    rewrite_plans = json.loads(bytes_a["rewrite_plans"])
    assert isinstance(rewrite_plans, list) and rewrite_plans
    plan_entry = rewrite_plans[0]
    for field in ("plan_id", "status", "site", "pre", "rewrite", "evidence", "post_expectation", "verification"):
        assert field in plan_entry

    exception_obligations = json.loads(bytes_a["exception_obligations"])
    assert isinstance(exception_obligations, list) and exception_obligations
    ex_entry = exception_obligations[0]
    for field in ("exception_path_id", "site", "source_kind", "status", "witness_ref", "remainder", "environment_ref"):
        assert field in ex_entry

    handledness = json.loads(bytes_a["handledness"])
    assert isinstance(handledness, list) and handledness
    handled_entry = handledness[0]
    for field in ("handledness_id", "exception_path_id", "site", "handler_kind", "handler_boundary", "environment", "core", "result"):
        assert field in handled_entry

