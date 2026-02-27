from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gabion.server import _normalize_dataflow_response

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "ingest_adapter"


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _normalized_primitives(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_dataflow_response(payload)
    return {
        "exit_code": normalized["exit_code"],
        "timeout": normalized["timeout"],
        "analysis_state": normalized["analysis_state"],
        "errors": normalized["errors"],
        "lint_entries": normalized["lint_entries"],
        "decision_surfaces": normalized.get("decision_surfaces", []),
        "bundle_sites_by_path": normalized.get("bundle_sites_by_path", {}),
    }


# gabion:evidence E:function_site::tests/test_ingest_adapter_contract.py::test_python_ingest_contract_fixture

def test_python_ingest_contract_fixture() -> None:
    raw = _load_fixture("python_raw.json")
    expected = _load_fixture("python_expected.json")

    assert _normalized_primitives(raw) == expected


# gabion:evidence E:function_site::tests/test_ingest_adapter_contract.py::test_synthetic_non_python_ingest_contract_fixture

def test_synthetic_non_python_ingest_contract_fixture() -> None:
    raw = _load_fixture("synthetic_raw.json")
    expected = _load_fixture("synthetic_expected.json")

    assert _normalized_primitives(raw) == expected


# gabion:evidence E:function_site::tests/test_ingest_adapter_contract.py::test_ingest_normalization_deterministic_ordering

def test_ingest_normalization_deterministic_ordering() -> None:
    raw = _load_fixture("python_raw.json")
    reversed_insertion_payload = {key: raw[key] for key in reversed(list(raw.keys()))}

    first = _normalize_dataflow_response(raw)
    second = _normalize_dataflow_response(reversed_insertion_payload)

    assert list(first.keys()) == list(second.keys())
    assert first["fingerprint_rewrite_plans"][0]["site"]["path"] == "a.py"
    assert first["fingerprint_rewrite_plans"] == second["fingerprint_rewrite_plans"]


# gabion:evidence E:function_site::tests/test_ingest_adapter_contract.py::test_adapter_parity_on_overlapping_decision_surfaces

def test_adapter_parity_on_overlapping_decision_surfaces() -> None:
    python_primitives = _normalized_primitives(_load_fixture("python_raw.json"))
    synthetic_primitives = _normalized_primitives(_load_fixture("synthetic_raw.json"))

    assert python_primitives["exit_code"] == synthetic_primitives["exit_code"]
    assert python_primitives["timeout"] == synthetic_primitives["timeout"]
    assert python_primitives["analysis_state"] == synthetic_primitives["analysis_state"]
    assert python_primitives["decision_surfaces"] == synthetic_primitives["decision_surfaces"]
    assert python_primitives["lint_entries"][0]["code"] == synthetic_primitives["lint_entries"][0]["code"]
    assert python_primitives["lint_entries"][0]["message"] == synthetic_primitives["lint_entries"][0]["message"]
