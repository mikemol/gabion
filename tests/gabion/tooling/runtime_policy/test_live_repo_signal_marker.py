from __future__ import annotations

from pathlib import Path


# gabion:behavior primary=desired
def test_live_repo_signal_marker_registered(pytestconfig) -> None:
    assert any(
        marker.split(":", 1)[0].strip() == "live_repo_signal"
        for marker in pytestconfig.getini("markers")
    )


# gabion:behavior primary=desired
def test_deterministic_runtime_policy_modules_do_not_retain_moved_live_repo_roots() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    assert "REPO_ROOT" not in (
        repo_root / "tests/gabion/tooling/runtime_policy/test_kernel_vm_alignment_artifact.py"
    ).read_text(encoding="utf-8")
    assert "REPO_ROOT" not in (
        repo_root
        / "tests/gabion/tooling/runtime_policy/test_identity_grammar_completion_artifact.py"
    ).read_text(encoding="utf-8")
    assert "REPO_ROOT" not in (
        repo_root / "tests/gabion/tooling/runtime_policy/test_connectivity_synergy_registry.py"
    ).read_text(encoding="utf-8")
