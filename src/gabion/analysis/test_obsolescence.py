from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RiskInfo:
    risk: str
    owner: str
    rationale: str

    @classmethod
    def from_payload(cls, payload: object) -> RiskInfo | None:
        if not isinstance(payload, Mapping):
            return None
        risk = str(payload.get("risk", "") or "").strip()
        owner = str(payload.get("owner", "") or "").strip()
        rationale = str(payload.get("rationale", "") or "").strip()
        if not risk:
            return None
        return cls(risk=risk, owner=owner, rationale=rationale)


def load_test_evidence(
    path: str,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"Unsupported test evidence schema_version={schema_version!r}; expected 1"
        )
    tests = payload.get("tests", [])
    if not isinstance(tests, list):
        raise ValueError("test evidence payload is missing tests list")
    entries: list[tuple[str, list[str], str]] = []
    for entry in tests:
        if not isinstance(entry, Mapping):
            continue
        test_id = str(entry.get("test_id", "") or "").strip()
        if not test_id:
            continue
        evidence = _normalize_evidence_list(entry.get("evidence", []))
        raw_status = entry.get("status")
        status = str(raw_status).strip() if raw_status is not None else ""
        if not status:
            status = "mapped" if evidence else "unmapped"
        entries.append((test_id, evidence, status))
    evidence_by_test: dict[str, list[str]] = {}
    status_by_test: dict[str, str] = {}
    for test_id, evidence, status in sorted(entries, key=lambda item: item[0]):
        evidence_by_test[test_id] = evidence
        status_by_test[test_id] = status
    return evidence_by_test, status_by_test


def load_risk_registry(path: str) -> dict[str, RiskInfo]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    version = payload.get("version", 1)
    if version != 1:
        raise ValueError(
            f"Unsupported evidence risk registry version={version!r}; expected 1"
        )
    evidence = payload.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return {}
    registry: dict[str, RiskInfo] = {}
    for evidence_id, info_payload in evidence.items():
        if not isinstance(evidence_id, str):
            continue
        info = RiskInfo.from_payload(info_payload)
        if info is None:
            continue
        registry[evidence_id] = info
    return registry


def compute_dominators(
    evidence_by_test: dict[str, list[str]],
) -> dict[str, list[str]]:
    test_ids = sorted(evidence_by_test)
    evidence_sets = {test_id: set(evidence_by_test[test_id]) for test_id in test_ids}
    evidence_sizes = {test_id: len(evidence_sets[test_id]) for test_id in test_ids}
    dominators: dict[str, list[str]] = {}
    for test_id in test_ids:
        target = evidence_sets[test_id]
        if not target:
            dominators[test_id] = []
            continue
        candidates: list[str] = []
        for other_id in test_ids:
            if other_id == test_id:
                continue
            other_set = evidence_sets[other_id]
            if target and target.issubset(other_set) and target != other_set:
                candidates.append(other_id)
        if not candidates:
            dominators[test_id] = []
            continue
        min_size = min(evidence_sizes[candidate] for candidate in candidates)
        frontier = sorted(
            candidate for candidate in candidates if evidence_sizes[candidate] == min_size
        )
        dominators[test_id] = frontier
    return dominators


def classify_candidates(
    evidence_by_test: dict[str, list[str]],
    status_by_test: dict[str, str],
    risk_registry: dict[str, RiskInfo],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    # dataflow-bundle: evidence_by_test, status_by_test, risk_registry
    normalized_evidence = {
        test_id: _normalize_evidence_list(evidence)
        for test_id, evidence in evidence_by_test.items()
    }
    mapped_evidence = {
        test_id: evidence
        for test_id, evidence in normalized_evidence.items()
        if status_by_test.get(test_id) == "mapped" and evidence
    }
    dominators = compute_dominators(mapped_evidence)
    high_risk = sorted(
        evidence_id
        for evidence_id, info in risk_registry.items()
        if info.risk.lower() == "high"
    )
    evidence_to_tests: dict[str, set[str]] = {}
    for test_id, evidence in mapped_evidence.items():
        for evidence_id in evidence:
            evidence_to_tests.setdefault(evidence_id, set()).add(test_id)
    last_witness_by_test: dict[str, list[str]] = {}
    for evidence_id in high_risk:
        tests = evidence_to_tests.get(evidence_id, set())
        if len(tests) == 1:
            test_id = next(iter(tests))
            last_witness_by_test.setdefault(test_id, []).append(evidence_id)
    for evidence_ids in last_witness_by_test.values():
        evidence_ids.sort()

    equivalence: dict[tuple[str, ...], list[str]] = {}
    for test_id, evidence in mapped_evidence.items():
        key = tuple(evidence)
        equivalence.setdefault(key, []).append(test_id)
    for peers in equivalence.values():
        peers.sort()

    summary = {
        "redundant_by_evidence": 0,
        "equivalent_witness": 0,
        "obsolete_candidate": 0,
        "unmapped": 0,
    }
    candidates: list[dict[str, object]] = []
    for test_id in sorted(normalized_evidence):
        evidence = normalized_evidence.get(test_id, [])
        status = status_by_test.get(test_id, "unmapped")
        doms = dominators.get(test_id, [])
        guardrail_evidence = last_witness_by_test.get(test_id, [])
        reason: dict[str, object] = {"evidence": evidence}
        if status != "mapped" or not evidence:
            class_name = "unmapped"
            reason["status"] = status
            doms = []
        else:
            peers = equivalence.get(tuple(evidence), [])
            has_equivalent = len(peers) > 1
            if doms:
                class_name = "redundant_by_evidence"
            elif has_equivalent:
                class_name = "equivalent_witness"
                reason["equivalence_class_size"] = len(peers)
            else:
                class_name = "obsolete_candidate"
            if doms and guardrail_evidence:
                class_name = "obsolete_candidate"
                reason["guardrail"] = "high-risk-last-witness"
                reason["guardrail_evidence"] = guardrail_evidence
        summary[class_name] += 1
        candidates.append(
            {
                "test_id": test_id,
                "class": class_name,
                "dominators": doms,
                "reason": reason,
            }
        )
    class_order = {
        "redundant_by_evidence": 0,
        "equivalent_witness": 1,
        "obsolete_candidate": 2,
        "unmapped": 3,
    }
    candidates.sort(
        key=lambda entry: (
            class_order.get(str(entry.get("class", "")), 99),
            str(entry.get("test_id", "")),
        )
    )
    return candidates, summary


def render_markdown(
    candidates: list[dict[str, object]],
    summary_counts: dict[str, int],
) -> str:
    # dataflow-bundle: candidates, summary_counts
    lines: list[str] = []
    lines.append("# Test Obsolescence Report")
    lines.append("")
    lines.append("Summary:")
    for key in [
        "redundant_by_evidence",
        "equivalent_witness",
        "obsolete_candidate",
        "unmapped",
    ]:
        lines.append(f"- {key}: {summary_counts.get(key, 0)}")
    lines.append("")

    sections = [
        ("redundant_by_evidence", "Redundant By Evidence"),
        ("equivalent_witness", "Equivalent Witnesses"),
        ("obsolete_candidate", "Obsolete Candidates"),
        ("unmapped", "Unmapped"),
    ]
    for class_key, title in sections:
        lines.append(f"## {title}")
        entries = [
            entry for entry in candidates if entry.get("class") == class_key
        ]
        if not entries:
            lines.append("- None")
            lines.append("")
            continue
        for entry in entries:
            test_id = str(entry.get("test_id", ""))
            doms = entry.get("dominators", []) or []
            reason = entry.get("reason", {}) or {}
            suffix_parts: list[str] = []
            if doms:
                dom_list = ", ".join(f"`{dom}`" for dom in doms)
                suffix_parts.append(f"dominators: {dom_list}")
            if class_key == "unmapped":
                status = str(reason.get("status", "unmapped"))
                suffix_parts.append(f"status: {status}")
            guardrail = reason.get("guardrail")
            if guardrail:
                evidence_list = reason.get("guardrail_evidence", []) or []
                evidence = ", ".join(evidence_list)
                suffix_parts.append(f"guardrail: {guardrail}")
                if evidence:
                    suffix_parts.append(f"evidence: {evidence}")
            suffix = ""
            if suffix_parts:
                suffix = " (" + "; ".join(suffix_parts) + ")"
            lines.append(f"- `{test_id}`{suffix}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _normalize_evidence_list(value: object) -> list[str]:
    if value is None:
        return []
    items: list[str] = []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [item for item in value if isinstance(item, str)]
    else:
        return []
    cleaned = [item.strip() for item in items if item.strip()]
    return sorted(set(cleaned))
