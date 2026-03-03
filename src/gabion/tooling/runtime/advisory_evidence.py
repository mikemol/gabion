from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping

from gabion.order_contract import sort_once
from gabion.runtime import json_io

AdvisoryDomain = Literal["obsolescence", "annotation_drift", "ambiguity", "docflow"]

DEFAULT_ADVISORY_AGGREGATE_PATH = Path("artifacts/out/advisory_aggregate.json")


@dataclass(frozen=True)
class AdvisoryEvidenceEntry:
    domain: AdvisoryDomain
    key: str
    baseline: int
    current: int
    delta: int
    threshold_class: str
    message: str
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": int(self.baseline),
            "current": int(self.current),
            "delta": int(self.delta),
            "domain": str(self.domain),
            "key": str(self.key),
            "message": str(self.message),
            "threshold_class": str(self.threshold_class),
            "timestamp": str(self.timestamp),
        }


@dataclass(frozen=True)
class AdvisoryEvidencePayload:
    domain: AdvisoryDomain
    source_delta_path: str
    generated_at: str
    entries: tuple[AdvisoryEvidenceEntry, ...]

    def to_dict(self) -> dict[str, object]:
        ordered_entries = sort_once(
            self.entries,
            source="advisory_evidence.AdvisoryEvidencePayload.to_dict.entries",
            key=lambda entry: entry.key,
        )
        return {
            "domain": str(self.domain),
            "entries": [entry.to_dict() for entry in ordered_entries],
            "generated_at": str(self.generated_at),
            "schema_version": 1,
            "source_delta_path": str(self.source_delta_path),
        }


@dataclass(frozen=True)
class AdvisoryAggregatePayload:
    generated_at: str
    advisories: Mapping[str, AdvisoryEvidencePayload]

    def to_dict(self) -> dict[str, object]:
        ordered_domains = sort_once(
            self.advisories.keys(),
            source="advisory_evidence.AdvisoryAggregatePayload.to_dict.domains",
        )
        return {
            "advisories": {
                domain: self.advisories[domain].to_dict()
                for domain in ordered_domains
            },
            "generated_at": str(self.generated_at),
            "schema_version": 1,
        }


def advisory_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_payload(path: Path, payload: AdvisoryEvidencePayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json_io.dump_json_pretty(payload.to_dict()) + "\n",
        encoding="utf-8",
    )


def load_aggregate(path: Path = DEFAULT_ADVISORY_AGGREGATE_PATH) -> dict[str, object]:
    return json_io.load_json_object_path(path)


def write_aggregate(
    payloads: Mapping[str, AdvisoryEvidencePayload],
    *,
    aggregate_path: Path = DEFAULT_ADVISORY_AGGREGATE_PATH,
    generated_at: str | None = None,
) -> None:
    aggregate = AdvisoryAggregatePayload(
        generated_at=generated_at or advisory_timestamp(),
        advisories=payloads,
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json_io.dump_json_pretty(aggregate.to_dict()) + "\n",
        encoding="utf-8",
    )
