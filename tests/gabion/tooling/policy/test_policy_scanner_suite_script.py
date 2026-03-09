from __future__ import annotations

from pathlib import Path

from scripts.policy import policy_scanner_suite


# gabion:evidence E:function_site::test_policy_scanner_suite_script.py::tests.gabion.tooling.policy.test_policy_scanner_suite_script.test_run_emits_hotspot_neighborhood_queue_artifacts
# gabion:behavior primary=desired
def test_run_emits_hotspot_neighborhood_queue_artifacts(tmp_path: Path) -> None:
    root = tmp_path
    out = root / "artifacts/out/policy_suite_results.json"
    source = root / "src/gabion/branch_sample.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "def branchy(flag: bool) -> int:\n    if flag:\n        return 1\n    return 0\n",
        encoding="utf-8",
    )

    rc = policy_scanner_suite.run(root=root, out=out)

    assert rc == 1
    assert out.exists()
    assert (out.parent / "hotspot_neighborhood_queue.json").exists()
    assert (out.parent / "hotspot_neighborhood_queue.md").exists()
