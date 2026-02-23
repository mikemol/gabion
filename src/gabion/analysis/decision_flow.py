# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import re

from . import evidence_keys
from .json_types import JSONObject
from .pattern_schema import pattern_schema_id
from gabion.order_contract import OrderPolicy, sort_once

_DECISION_LINE_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<qual>[^ ]+) (?P<mode>value-encoded decision|decision surface) params: (?P<params>[^()]+)"
)


@dataclass(frozen=True)
class DecisionSurfaceRecord:
    path: str
    qual: str
    mode: str
    params: tuple[str, ...]

    @property
    def decision_id(self) -> str:
        kind = "value_decision" if self.mode == "value" else "direct_decision"
        return pattern_schema_id(
            kind=kind,
            signature={
                "path": self.path,
                "qual": self.qual,
                "params": list(self.params),
            },
        )


# dataflow-bundle: entry

def parse_decision_surface_line(entry: str) -> DecisionSurfaceRecord | None:
    match = _DECISION_LINE_RE.match(entry.strip())
    if not match:
        return None
    raw_mode = match.group("mode")
    mode = "value" if raw_mode.startswith("value-encoded") else "direct"
    params = evidence_keys.normalize_params(match.group("params").split(","))
    return DecisionSurfaceRecord(
        path=match.group("path").strip(),
        qual=match.group("qual").strip(),
        mode=mode,
        params=tuple(params),
    )


# dataflow-bundle: decision_surfaces, value_decision_surfaces

def build_decision_tables(
    *,
    decision_surfaces: Iterable[str],
    value_decision_surfaces: Iterable[str],
) -> list[JSONObject]:
    tables: list[JSONObject] = []
    for raw in [*decision_surfaces, *value_decision_surfaces]:
        parsed = parse_decision_surface_line(raw)
        if parsed is None:
            continue
        evidence_refs = sort_once(
            [
                evidence_keys.render_display(
                    evidence_keys.make_decision_surface_key(
                        mode=parsed.mode,
                        path=parsed.path,
                        qual=parsed.qual,
                        param=param,
                    )
                )
                for param in parsed.params
            ],
            source="decision_flow.build_decision_tables.evidence_refs",
            policy=OrderPolicy.SORT,
        )
        tables.append(
            {
                "decision_id": parsed.decision_id,
                "tier": 3,
                "mode": parsed.mode,
                "path": parsed.path,
                "qual": parsed.qual,
                "params": list(parsed.params),
                "analysis_evidence_keys": evidence_refs,
                "checklist_nodes": [
                    "docs/sppf_checklist.md#decision-flow-tier3",
                ],
            }
        )
    return sort_once(
        tables,
        source="decision_flow.build_decision_tables.tables",
        key=lambda item: (
            str(item.get("path", "")),
            str(item.get("qual", "")),
            str(item.get("mode", "")),
            ",".join(str(v) for v in item.get("params", [])),
            str(item.get("decision_id", "")),
        ),
        policy=OrderPolicy.SORT,
    )


def detect_repeated_guard_bundles(tables: Iterable[JSONObject]) -> list[JSONObject]:
    grouped: dict[tuple[str, ...], list[JSONObject]] = defaultdict(list)
    for table in tables:
        params = evidence_keys.normalize_params(table.get("params", []))
        if not params:
            continue
        grouped[tuple(params)].append(table)
    bundles: list[JSONObject] = []
    for params, members in grouped.items():
        if len(members) < 2:
            continue
        member_ids = sort_once(
            [str(item.get("decision_id", "")) for item in members],
            source="decision_flow.detect_repeated_guard_bundles.member_ids",
            policy=OrderPolicy.SORT,
        )
        bundle_id = pattern_schema_id(
            kind="decision_bundle",
            signature={"params": list(params), "members": member_ids},
        )
        bundles.append(
            {
                "bundle_id": bundle_id,
                "tier": 2,
                "params": list(params),
                "occurrences": len(member_ids),
                "member_decision_ids": member_ids,
                "checklist_nodes": ["docs/sppf_checklist.md#decision-flow-tier2"],
            }
        )
    return sort_once(
        bundles,
        source="decision_flow.detect_repeated_guard_bundles.bundles",
        key=lambda item: (
            -int(item.get("occurrences", 0)),
            ",".join(str(v) for v in item.get("params", [])),
            str(item.get("bundle_id", "")),
        ),
        policy=OrderPolicy.SORT,
    )


def enforce_decision_protocol_contracts(
    *,
    decision_tables: Iterable[JSONObject],
    decision_bundles: Iterable[JSONObject],
) -> list[JSONObject]:
    table_map = {str(entry.get("decision_id", "")): entry for entry in decision_tables}
    violations: list[JSONObject] = []

    for bundle in decision_bundles:
        bundle_id = str(bundle.get("bundle_id", ""))
        member_ids = bundle.get("member_decision_ids", [])
        if not isinstance(member_ids, list):
            member_ids = []
        if not member_ids:
            violations.append(
                {
                    "violation_id": f"decision_protocol:empty-members:{bundle_id}",
                    "tier": 1,
                    "code": "DECISION_PROTOCOL_EMPTY_MEMBERS",
                    "message": f"Bundle {bundle_id} has no linked decision tables.",
                    "bundle_id": bundle_id,
                    "checklist_node": "docs/sppf_checklist.md#decision-flow-tier1",
                }
            )
            continue
        for decision_id in member_ids:
            table = table_map.get(str(decision_id))
            if table is None:
                violations.append(
                    {
                        "violation_id": f"decision_protocol:missing-table:{bundle_id}:{decision_id}",
                        "tier": 1,
                        "code": "DECISION_PROTOCOL_MISSING_TABLE",
                        "message": (
                            f"Bundle {bundle_id} references missing decision table {decision_id}."
                        ),
                        "bundle_id": bundle_id,
                        "decision_id": str(decision_id),
                        "checklist_node": "docs/sppf_checklist.md#decision-flow-tier1",
                    }
                )
                continue
            evidence_keys_list = table.get("analysis_evidence_keys", [])
            checklist_nodes = table.get("checklist_nodes", [])
            if not evidence_keys_list:
                violations.append(
                    {
                        "violation_id": f"decision_protocol:missing-evidence:{decision_id}",
                        "tier": 1,
                        "code": "DECISION_PROTOCOL_MISSING_EVIDENCE",
                        "message": (
                            f"Critical decision path {decision_id} is missing analysis evidence keys."
                        ),
                        "bundle_id": bundle_id,
                        "decision_id": str(decision_id),
                        "checklist_node": "docs/sppf_checklist.md#decision-flow-tier1",
                    }
                )
            if not checklist_nodes:
                violations.append(
                    {
                        "violation_id": f"decision_protocol:missing-checklist:{decision_id}",
                        "tier": 1,
                        "code": "DECISION_PROTOCOL_MISSING_CHECKLIST_LINK",
                        "message": (
                            f"Critical decision path {decision_id} is missing checklist linkage."
                        ),
                        "bundle_id": bundle_id,
                        "decision_id": str(decision_id),
                        "checklist_node": "docs/sppf_checklist.md#decision-flow-tier1",
                    }
                )

    return sort_once(
        violations,
        source="decision_flow.enforce_decision_protocol_contracts.violations",
        key=lambda item: str(item.get("violation_id", "")),
        policy=OrderPolicy.SORT,
    )
