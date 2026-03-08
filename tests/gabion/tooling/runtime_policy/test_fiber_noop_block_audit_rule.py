from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_rules import fiber_noop_block_audit_rule as audit_rule


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:behavior primary=allowed_unwanted facets=noop
def test_collect_violations_detects_singleton_pass_and_return_none(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/sample.py",
        "\n".join(
            [
                "def branch(flag):",
                "    if flag:",
                "        pass",
                "    else:",
                "        return None",
                "    return 1",
                "",
            ]
        ),
    )
    violations = audit_rule.collect_violations(root=root)
    assert len(violations) == 2
    kinds = {item.noop_kind for item in violations}
    assert "pass" in kinds
    assert "return_none" in kinds
    assert all(item.kind == "singleton_noop_block" for item in violations)


# gabion:behavior primary=allowed_unwanted facets=noop
def test_run_writes_fiber_payload(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/sample.py",
        "\n".join(
            [
                "def sample(x):",
                "    if x:",
                "        pass",
                "    return 1",
                "",
            ]
        ),
    )
    out = root / "artifacts/out/fiber_noop_block_audit.json"
    exit_code = audit_rule.run(root=root, out=out)
    assert exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["violation_count"] == 1
    first = payload["violations"][0]
    assert first["fiber_trace"]
    assert first["counterfactual_boundary"]["status"] == "present"
    assert first["applicability_bounds"]["current_boundary_before_ordinal"] == 2

