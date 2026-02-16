from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest

from gabion.analysis.timeout_context import Deadline, deadline_clock_scope, deadline_scope
from gabion.analysis.timeout_context import forest_scope
from gabion.analysis.aspf import Forest
from gabion.analysis import evidence_keys, test_obsolescence
from gabion.deadline_clock import GasMeter
from tests.env_helpers import restore_env as _restore_env
from tests.env_helpers import set_env as _set_env


@pytest.fixture(autouse=True)
def _deadline_scope_fixture():
    with forest_scope(Forest()):
        with deadline_scope(Deadline.from_timeout_ms(120_000)):
            with deadline_clock_scope(GasMeter(limit=100_000_000)):
                yield


@pytest.fixture
def write_test_evidence_payload():
    def _write(
        path: Path,
        *,
        entries: list[dict[str, object]],
        scope_root: str = ".",
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        payload = {
            "schema_version": 2,
            "scope": {
                "root": scope_root,
                "include": list(include or ["tests"]),
                "exclude": list(exclude or []),
            },
            "tests": entries,
            "evidence_index": [],
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return _write


@pytest.fixture
def test_evidence_path(tmp_path: Path) -> Path:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "test_evidence.json"


@pytest.fixture
def env_scope():
    return _set_env


@pytest.fixture
def restore_env():
    return _restore_env


@pytest.fixture
def make_obsolescence_paramset_ref():
    def _make(value: str) -> test_obsolescence.EvidenceRef:
        key = evidence_keys.make_paramset_key([value])
        identity = evidence_keys.key_identity(key)
        return test_obsolescence.EvidenceRef(
            key=key,
            identity=identity,
            display=evidence_keys.render_display(key),
            opaque=False,
        )

    return _make


@pytest.fixture
def make_obsolescence_opaque_ref():
    def _make(display: str) -> test_obsolescence.EvidenceRef:
        key = evidence_keys.make_opaque_key(display)
        identity = evidence_keys.key_identity(key)
        return test_obsolescence.EvidenceRef(
            key=key,
            identity=identity,
            display=display,
            opaque=True,
        )

    return _make


@pytest.fixture
def obsolescence_summary_counts():
    base = {
        "redundant_by_evidence": 0,
        "equivalent_witness": 0,
        "obsolete_candidate": 0,
        "unmapped": 0,
    }

    def _make(**overrides: int) -> dict[str, int]:
        payload = dict(base)
        for key, value in overrides.items():
            payload[key] = int(value)
        return payload

    return _make
