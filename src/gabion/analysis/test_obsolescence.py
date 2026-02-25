# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from gabion.analysis import evidence_keys
from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_version,
)
from gabion.analysis.projection_exec import apply_spec
from gabion.analysis.projection_spec import ProjectionSpec
from gabion.analysis.projection_registry import (
    TEST_OBSOLESCENCE_SUMMARY_SPEC,
    spec_metadata_lines_from_payload,
)
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


@dataclass(frozen=True)
class RiskInfo:
    risk: str
    owner: str
    rationale: str

    @classmethod
    # gabion:ambiguity_boundary
    def from_payload(cls, payload: object) -> RiskInfo | None:
        if not isinstance(payload, Mapping):
            return None
        risk = str(payload.get("risk", "") or "").strip()
        owner = str(payload.get("owner", "") or "").strip()
        rationale = str(payload.get("rationale", "") or "").strip()
        if not risk:
            return None
        return cls(risk=risk, owner=owner, rationale=rationale)


@dataclass(frozen=True)
class EvidenceRef:
    key: dict[str, object]
    identity: str
    display: str
    opaque: bool


@dataclass(frozen=True)
class ClassifierOptions:
    runtime_ms_by_test: Mapping[str, float] = field(default_factory=dict)
    branch_guard_by_test: Mapping[str, bool] = field(default_factory=dict)
    default_keep_for_branch_guard: bool = False
    unresolved_test_ids: frozenset[str] = field(default_factory=frozenset)
    objective_order: tuple[str, str, str, str] = (
        "evidence_novelty",
        "branch_guard",
        "runtime_ms",
        "test_id",
    )
    prefer_lower_runtime_ms: bool = True
    lexical_test_id_ascending: bool = True


@dataclass(frozen=True)
class ClassificationResult:
    stale_candidates: list[dict[str, object]]
    stale_summary: dict[str, int]
    active_tests: list[str]
    active_summary: dict[str, int]


_STALE_CLASS_ORDER = [
    "redundant_by_evidence",
    "equivalent_witness",
    "obsolete_candidate",
    "unmapped",
]
_STALE_CLASS_RANK = {name: idx for idx, name in enumerate(_STALE_CLASS_ORDER)}

# gabion:ambiguity_boundary
def load_test_evidence(
    path: str,
) -> tuple[dict[str, list[EvidenceRef]], dict[str, str]]:
    check_deadline()
    payload = load_json(path)
    parse_version(
        payload,
        expected=(1, 2),
        field="schema_version",
        error_context="test evidence",
    )
    tests = payload.get("tests", [])
    if not isinstance(tests, list):
        raise ValueError("test evidence payload is missing tests list")
    entries: list[tuple[str, list[EvidenceRef], str]] = []
    for entry in tests:
        if not isinstance(entry, Mapping):
            continue
        test_id = str(entry.get("test_id", "") or "").strip()
        if not test_id:
            continue
        evidence = _normalize_evidence_refs(entry.get("evidence", []))
        raw_status = entry.get("status")
        status = str(raw_status).strip() if raw_status is not None else ""
        if not status:
            status = "mapped" if evidence else "unmapped"
        entries.append((test_id, evidence, status))
    evidence_by_test: dict[str, list[EvidenceRef]] = {}
    status_by_test: dict[str, str] = {}
    for test_id, evidence, status in sort_once(
        entries,
        source="load_test_evidence.entries",
        key=lambda item: item[0],
    ):
        evidence_by_test[test_id] = evidence
        status_by_test[test_id] = status
    return evidence_by_test, status_by_test


def load_risk_registry(path: str) -> dict[str, RiskInfo]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {}
    payload = load_json(path)
    return _parse_risk_registry_payload(payload)

# gabion:ambiguity_boundary
def _parse_risk_registry_payload(payload: Mapping[str, object]) -> dict[str, RiskInfo]:
    check_deadline()
    parse_version(
        payload,
        expected=1,
        error_context="evidence risk registry",
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
    check_deadline()
    test_ids = sort_once(
        evidence_by_test,
        source="compute_dominators.test_ids",
    )
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
        frontier = sort_once(
            [
                candidate
                for candidate in candidates
                if evidence_sizes[candidate] == min_size
            ],
            source="compute_dominators.frontier",
        )
        dominators[test_id] = frontier
    return dominators

# gabion:ambiguity_boundary
def classify_candidates(
    evidence_by_test: dict[str, list[EvidenceRef]],
    status_by_test: dict[str, str],
    risk_registry: dict[str, RiskInfo],
    *,
    options: ClassifierOptions | None = None,
) -> ClassificationResult:
    check_deadline()
    # dataflow-bundle: evidence_by_test, status_by_test, risk_registry
    resolved_options = options or ClassifierOptions()
    all_test_ids = sort_once(
        set(evidence_by_test) | set(status_by_test),
        source="classify_candidates.all_test_ids",
    )
    normalized_evidence = {
        test_id: _normalize_evidence_refs(evidence_by_test.get(test_id, []))
        for test_id in all_test_ids
    }
    mapped_evidence = {
        test_id: evidence
        for test_id, evidence in normalized_evidence.items()
        if status_by_test.get(test_id) == "mapped" and evidence
    }
    dominators = compute_dominators(
        {
            test_id: [ref.identity for ref in evidence]
            for test_id, evidence in mapped_evidence.items()
        }
    )
    high_risk = sort_once(
        [
            evidence_id
            for evidence_id, info in risk_registry.items()
            if info.risk.lower() == "high"
        ],
        source="classify_candidates.high_risk",
    )
    evidence_to_tests: dict[str, set[str]] = {}
    for test_id, evidence in mapped_evidence.items():
        for ref in evidence:
            evidence_to_tests.setdefault(ref.display, set()).add(test_id)
    last_witness_by_test: dict[str, list[str]] = {}
    for evidence_id in high_risk:
        tests = evidence_to_tests.get(evidence_id, set())
        if len(tests) == 1:
            test_id = next(iter(tests))
            last_witness_by_test.setdefault(test_id, []).append(evidence_id)
    for test_id, evidence_ids in last_witness_by_test.items():
        last_witness_by_test[test_id] = sort_once(
            evidence_ids,
            source="classify_candidates.last_witness_by_test",
        )

    equivalence: dict[tuple[str, ...], list[str]] = {}
    for test_id, evidence in mapped_evidence.items():
        key = tuple(ref.identity for ref in evidence)
        equivalence.setdefault(key, []).append(test_id)
    for key, peers in equivalence.items():
        equivalence[key] = sort_once(
            peers,
            source="classify_candidates.equivalence",
        )

    pareto_winner_by_key: dict[tuple[str, ...], str] = {}
    for key, peers in equivalence.items():
        if len(peers) <= 1:
            continue
        pareto_winner_by_key[key] = _pareto_winner(
            peers,
            options=resolved_options,
        )

    stale_candidates: list[dict[str, object]] = []
    active_tests: set[str] = set()
    pareto_frontier_tests: set[str] = set()
    branch_guard_retained = 0
    high_risk_guardrail_retained = 0
    unresolved_obsolete = 0
    for test_id in all_test_ids:
        evidence = normalized_evidence.get(test_id, [])
        evidence_display = [ref.display for ref in evidence]
        opaque_evidence = [ref.display for ref in evidence if ref.opaque]
        status = status_by_test.get(test_id, "unmapped")
        doms = dominators.get(test_id, [])
        guardrail_evidence = last_witness_by_test.get(test_id, [])
        reason: dict[str, object] = {"evidence": evidence_display}
        unresolved_requested = test_id in resolved_options.unresolved_test_ids
        if status != "mapped" or not evidence:
            class_name = "unmapped"
            reason["status"] = status
            doms = []
            if unresolved_requested:
                class_name = "obsolete_candidate"
                reason["resolution"] = "unresolved"
                reason["stale_from"] = "unmapped"
                unresolved_obsolete += 1
            if opaque_evidence:
                reason["opaque_evidence"] = opaque_evidence
            stale_candidates.append(
                {
                    "test_id": test_id,
                    "class": class_name,
                    "dominators": doms,
                    "reason": reason,
                }
            )
            continue

        evidence_key = tuple(ref.identity for ref in evidence)
        peers = equivalence.get(evidence_key, [])
        has_equivalent = len(peers) > 1
        if doms:
            if guardrail_evidence and not unresolved_requested:
                reason["guardrail"] = "high-risk-last-witness"
                reason["guardrail_evidence"] = guardrail_evidence
                active_tests.add(test_id)
                high_risk_guardrail_retained += 1
                continue
            if _is_branch_guarded(test_id, options=resolved_options) and not unresolved_requested:
                reason["branch_guard"] = True
                branch_guard_retained += 1
                active_tests.add(test_id)
                continue
            class_name = "redundant_by_evidence"
            if unresolved_requested:
                class_name = "obsolete_candidate"
                reason["resolution"] = "unresolved"
                reason["stale_from"] = "redundant_by_evidence"
                unresolved_obsolete += 1
            if guardrail_evidence:
                reason["guardrail"] = "high-risk-last-witness"
                reason["guardrail_evidence"] = guardrail_evidence
            if opaque_evidence:
                reason["opaque_evidence"] = opaque_evidence
            stale_candidates.append(
                {
                    "test_id": test_id,
                    "class": class_name,
                    "dominators": doms,
                    "reason": reason,
                }
            )
            continue

        if has_equivalent:
            class_name = "equivalent_witness"
            reason["equivalence_class_size"] = len(peers)
            winner = pareto_winner_by_key.get(evidence_key, "")
            if winner:
                reason["pareto_winner"] = winner
            if winner == test_id and not unresolved_requested:
                active_tests.add(test_id)
                pareto_frontier_tests.add(test_id)
                continue
            if unresolved_requested:
                class_name = "obsolete_candidate"
                reason["resolution"] = "unresolved"
                reason["stale_from"] = "equivalent_witness"
                unresolved_obsolete += 1
            if opaque_evidence:
                reason["opaque_evidence"] = opaque_evidence
            stale_candidates.append(
                {
                    "test_id": test_id,
                    "class": class_name,
                    "dominators": doms,
                    "reason": reason,
                }
            )
            continue

        if unresolved_requested:
            reason["resolution"] = "unresolved"
            reason["stale_from"] = "active_candidate"
            unresolved_obsolete += 1
            if opaque_evidence:
                reason["opaque_evidence"] = opaque_evidence
            stale_candidates.append(
                {
                    "test_id": test_id,
                    "class": "obsolete_candidate",
                    "dominators": doms,
                    "reason": reason,
                }
            )
            continue

        active_tests.add(test_id)
        pareto_frontier_tests.add(test_id)

    stale_candidates = sort_once(
        stale_candidates,
        source="classify_candidates.stale_candidates",
        key=lambda entry: (
            _STALE_CLASS_RANK.get(str(entry.get("class", "")), 99),
            str(entry.get("test_id", "")),
        ),
    )
    stale_summary = _summarize_candidates(stale_candidates, _STALE_CLASS_RANK)
    active_tests_ordered = sort_once(
        active_tests,
        source="classify_candidates.active_tests",
    )
    active_summary = {
        "active_total": len(active_tests_ordered),
        "pareto_frontier_total": len(pareto_frontier_tests),
        "branch_guard_retained": branch_guard_retained,
        "high_risk_guardrail_retained": high_risk_guardrail_retained,
        "unresolved_obsolete": unresolved_obsolete,
    }
    return ClassificationResult(
        stale_candidates=stale_candidates,
        stale_summary=stale_summary,
        active_tests=active_tests_ordered,
        active_summary=active_summary,
    )


def _is_branch_guarded(test_id: str, *, options: ClassifierOptions) -> bool:
    branch_guard_by_test = options.branch_guard_by_test
    if branch_guard_by_test is None:
        return options.default_keep_for_branch_guard
    value = branch_guard_by_test.get(test_id)
    if value is None:
        return options.default_keep_for_branch_guard
    return bool(value)


def _runtime_ms(test_id: str, *, options: ClassifierOptions) -> float:
    runtime_ms_by_test = options.runtime_ms_by_test
    if runtime_ms_by_test is None:
        return float("inf")
    value = runtime_ms_by_test.get(test_id)
    if value is None:
        return float("inf")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if parsed < 0:
        return float("inf")
    return parsed


def _pareto_winner(peers: list[str], *, options: ClassifierOptions) -> str:
    novelty_by_test = {test_id: 0 for test_id in peers}
    ordered = sort_once(
        peers,
        source="_pareto_winner.peers",
        key=lambda test_id: _pareto_sort_key(
            test_id,
            novelty=novelty_by_test.get(test_id, 0),
            options=options,
        ),
    )
    if not ordered:
        return ""
    return str(ordered[0])


def _pareto_sort_key(
    test_id: str,
    *,
    novelty: int,
    options: ClassifierOptions,
) -> tuple[object, ...]:
    runtime_ms = _runtime_ms(test_id, options=options)
    runtime_key = runtime_ms if options.prefer_lower_runtime_ms else -runtime_ms
    lexical_key: object = test_id
    if not options.lexical_test_id_ascending:
        lexical_key = tuple(-ord(char) for char in test_id)
    parts: list[object] = []
    for metric in options.objective_order:
        if metric == "evidence_novelty":
            parts.append(-int(novelty))
            continue
        if metric == "branch_guard":
            parts.append(0 if _is_branch_guarded(test_id, options=options) else 1)
            continue
        if metric == "runtime_ms":
            parts.append(runtime_key)
            continue
        if metric == "test_id":
            parts.append(lexical_key)
            continue
    if not parts:
        parts = [
            0 if _is_branch_guarded(test_id, options=options) else 1,
            runtime_key,
            lexical_key,
        ]
    return tuple(parts)


def render_markdown(
    candidates: list[dict[str, object]],
    summary_counts: dict[str, int],
) -> str:
    check_deadline()
    # dataflow-bundle: candidates, summary_counts
    payload = render_json_payload(candidates, summary_counts)
    doc = ReportDoc("out_test_obsolescence_report")
    doc.lines(spec_metadata_lines_from_payload(payload))
    doc.section("Summary")
    for key in [
        "redundant_by_evidence",
        "equivalent_witness",
        "obsolete_candidate",
        "unmapped",
    ]:
        doc.line(f"- {key}: {summary_counts.get(key, 0)}")
    doc.line()

    sections = [
        ("redundant_by_evidence", "Redundant By Evidence"),
        ("equivalent_witness", "Equivalent Witnesses"),
        ("obsolete_candidate", "Obsolete Candidates"),
        ("unmapped", "Unmapped"),
    ]
    for class_key, title in sections:
        doc.line(f"## {title}")
        entries = [
            entry for entry in candidates if entry.get("class") == class_key
        ]
        if not entries:
            doc.line("- None")
            doc.line()
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
            opaque_evidence = reason.get("opaque_evidence", []) or []
            if opaque_evidence:
                suffix_parts.append(f"opaque: {len(opaque_evidence)}")
            suffix = ""
            if suffix_parts:
                suffix = " (" + "; ".join(suffix_parts) + ")"
            doc.line(f"- `{test_id}`{suffix}")
        doc.line()
    return doc.emit()


def render_json_payload(
    candidates: list[dict[str, object]],
    summary_counts: dict[str, int],
) -> dict[str, object]:
    payload = {
        "version": 3,
        "summary": summary_counts,
        "candidates": candidates,
    }
    return attach_spec_metadata(payload, spec=TEST_OBSOLESCENCE_SUMMARY_SPEC)

def _summarize_candidates(
    candidates: list[dict[str, object]],
    class_rank: dict[str, int],
    *,
    apply: Callable[[ProjectionSpec, list[dict[str, object]]], list[dict[str, object]]] = apply_spec,
) -> dict[str, int]:
    check_deadline()
    relation: list[dict[str, object]] = []
    for entry in candidates:
        class_name = str(entry.get("class", "") or "")
        relation.append(
            {
                "class": class_name,
                "class_rank": class_rank.get(class_name, 99),
            }
        )
    summary_rows = apply(TEST_OBSOLESCENCE_SUMMARY_SPEC, relation)
    summary = {
        "redundant_by_evidence": 0,
        "equivalent_witness": 0,
        "obsolete_candidate": 0,
        "unmapped": 0,
    }
    for row in summary_rows:
        class_name = str(row.get("class", "") or "")
        try:
            count = int(row.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        if class_name in summary:
            summary[class_name] = count
    return summary

# gabion:ambiguity_boundary
def _normalize_evidence_refs(value: object) -> list[EvidenceRef]:
    check_deadline()
    if value is None:
        return []
    if isinstance(value, EvidenceRef):
        return [value]
    refs: dict[str, EvidenceRef] = {}
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, EvidenceRef):
                refs[item.identity] = item
                continue
            if isinstance(item, Mapping):
                raw_key = item.get("key")
                display = item.get("display")
                key: dict[str, object] | None = None
                if isinstance(raw_key, Mapping):
                    key = evidence_keys.normalize_key(raw_key)
                if key is None and isinstance(display, str):
                    parsed = evidence_keys.parse_display(display)
                    if parsed is not None:
                        key = parsed
                if key is None:
                    if isinstance(display, str):
                        key = evidence_keys.make_opaque_key(display)
                    else:
                        continue
                key = evidence_keys.normalize_key(key)
                identity = evidence_keys.key_identity(key)
                rendered = evidence_keys.render_display(key)
                if evidence_keys.is_opaque(key) and isinstance(display, str):
                    rendered = display
                refs[identity] = EvidenceRef(
                    key=key,
                    identity=identity,
                    display=rendered,
                    opaque=evidence_keys.is_opaque(key),
                )
            elif isinstance(item, str):
                display = item.strip()
                if not display:
                    continue
                key = evidence_keys.parse_display(display)
                if key is None:
                    key = evidence_keys.make_opaque_key(display)
                key = evidence_keys.normalize_key(key)
                identity = evidence_keys.key_identity(key)
                rendered = evidence_keys.render_display(key)
                if evidence_keys.is_opaque(key):
                    rendered = display
                refs[identity] = EvidenceRef(
                    key=key,
                    identity=identity,
                    display=rendered,
                    opaque=evidence_keys.is_opaque(key),
                )
    ordered = [
        refs[key]
        for key in sort_once(
            refs,
            source="_normalize_evidence_refs.refs",
        )
    ]
    return ordered
