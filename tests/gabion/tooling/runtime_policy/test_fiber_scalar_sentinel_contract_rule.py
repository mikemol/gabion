from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_rules import fiber_scalar_sentinel_contract_rule as rule
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch_from_sources


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:behavior primary=verboten facets=none
def test_collect_violations_detects_none_scalar_compares_and_ifexp_none() -> None:
    source = "\n".join(
        [
            "def sample(value: int | None) -> int | None:",
            "    if value is None:",
            "        return None",
            "    return value if value > 0 else None",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/sample.py": source},
    )
    violations = rule.collect_violations(batch=batch)
    kinds = {item.kind for item in violations}
    assert "none_comparison" in kinds
    assert "ifexp_none_arm" in kinds


# gabion:behavior primary=desired
def test_run_writes_fiber_payload(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/sample.py",
        "\n".join(
            [
                "def sample(value):",
                "    return value if value != 0 else None",
                "",
            ]
        ),
    )
    out = root / "artifacts/out/fiber_scalar_sentinel_contract.json"
    exit_code = rule.run(root=root, out=out)
    assert exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["violation_count"] >= 1
    first = payload["violations"][0]
    assert first["fiber_trace"]
    assert first["counterfactual_boundary"]["status"] == "present"
