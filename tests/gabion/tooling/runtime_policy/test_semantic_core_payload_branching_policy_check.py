from __future__ import annotations

from pathlib import Path

from scripts.policy import policy_check


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_raw_payload_branching_flags_semantic_core_isinstance_and_cast(tmp_path: Path) -> None:
    sample = tmp_path / "src" / "gabion" / "analysis" / "sample_semantic_core.py"
    _write(
        sample,
        "from typing import Mapping, cast\n\n"
        "def run(payload: object) -> int:\n"
        "    narrowed = cast(Mapping[str, object], payload)\n"
        "    if isinstance(narrowed, Mapping):\n"
        "        return 1\n"
        "    return 0\n",
    )

    violations = policy_check._raw_payload_branching_violations(sample)
    assert violations
    assert any("isinstance Mapping/list branch outside boundary decode" in item for item in violations)


def test_raw_payload_branching_allows_decode_boundary_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "src" / "gabion" / "analysis" / "sample_adapter.py"
    _write(
        adapter,
        "from typing import Mapping\n\n"
        "def _decode_payload(raw: object) -> Mapping[str, object] | None:\n"
        "    if isinstance(raw, Mapping):\n"
        "        return raw\n"
        "    return None\n",
    )

    assert policy_check._raw_payload_branching_violations(adapter) == []
