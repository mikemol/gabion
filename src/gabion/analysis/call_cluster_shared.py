# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


@dataclass(frozen=True)
class ClusterIdentity:
    identity: str
    key: dict[str, object]
    display: str


def cluster_identity_from_key(key: Mapping[str, object]) -> ClusterIdentity:
    check_deadline()
    normalized = evidence_keys.normalize_key(key)
    return ClusterIdentity(
        identity=evidence_keys.key_identity(normalized),
        key=normalized,
        display=evidence_keys.render_display(normalized),
    )


def sorted_unique_strings(values: Iterable[object], *, source: str) -> tuple[str, ...]:
    # dataflow-bundle: source, values
    check_deadline()
    return tuple(
        sort_once(
            {str(value) for value in values},
            source=f"{source}.sorted_unique_strings",
        )
    )


def render_cluster_heading(doc: ReportDoc, *, display: object, count: object) -> None:
    # dataflow-bundle: count, display
    doc.line()
    doc.line(f"Cluster: {display} (count: {count})")


def render_string_codeblock(doc: ReportDoc, values: Iterable[object]) -> None:
    rendered: list[str] = []
    for value in values:
        check_deadline()
        rendered.append(str(value))
    if rendered:
        doc.codeblock("\n".join(rendered))
