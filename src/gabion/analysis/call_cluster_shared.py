from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from gabion.analysis import evidence_keys
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted


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
    check_deadline()
    return tuple(
        ordered_or_sorted(
            {str(value) for value in values},
            source=source,
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
