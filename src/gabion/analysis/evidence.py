# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from gabion.analysis.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


def normalize_bundle_key(bundle: object) -> str:
    """Canonicalize a bundle payload into a stable join key.

    This intentionally ignores non-string entries because bundles are defined
    over symbol names.
    """
    check_deadline()
    values: set[str] = set()
    match bundle:
        case list() | tuple() | set() as bundle_values:
            for item in bundle_values:
                match item:
                    case str() as item_text if item_text.strip():
                        values.add(item_text.strip())
                    case _:
                        pass
        case _:
            return ""
    return ",".join(
        sort_once(
            values,
            source="normalize_bundle_key.values",
        )
    )


def normalize_string_list(value: object) -> list[str]:
    """Normalize a payload field that should be a list of strings.

    The intent is to accept schema-level payloads coming from JSON where these
    fields can be absent, malformed, or represented as comma-separated strings.
    """
    check_deadline()
    raw: list[str] = []
    match value:
        case None:
            return raw
        case str() as text_value:
            raw = [text_value]
        case list() | tuple() | set() as sequence_value:
            for item in sequence_value:
                match item:
                    case str() as item_text:
                        raw.append(item_text)
                    case _:
                        pass
        case _:
            return raw

    parts: list[str] = []
    for item in raw:
        check_deadline()
        parts.extend([part.strip() for part in item.split(",") if part.strip()])
    return sort_once(
        set(parts),
        source="normalize_string_list.parts",
    )


@dataclass(frozen=True)
class Site:
    path: str
    function: str
    bundle: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object) -> object:
        match payload:
            case Mapping() as payload_mapping:
                path = str(payload_mapping.get("path", "")).strip()
                function = str(payload_mapping.get("function", "")).strip()
                bundle = normalize_string_list(payload_mapping.get("bundle", []) or [])
                return cls(path=path, function=function, bundle=tuple(bundle))
            case _:
                return None

    def bundle_key(self) -> str:
        check_deadline()
        return normalize_bundle_key(list(self.bundle))

    def key(self) -> tuple[str, str, str]:
        check_deadline()
        return (self.path, self.function, self.bundle_key())


def exception_obligation_summary_for_site(
    obligations: Iterable[Mapping[str, JSONValue]],
    *,
    site: Site,
) -> dict[str, int]:
    check_deadline()
    summary = {"UNKNOWN": 0, "DEAD": 0, "HANDLED": 0, "total": 0}
    bundle_key = site.bundle_key()
    for entry in obligations:
        check_deadline()
        raw_site = entry.get("site", {}) or {}
        match raw_site:
            case Mapping() as site_payload:
                if str(site_payload.get("path", "")) != site.path:
                    continue
                if str(site_payload.get("function", "")) != site.function:
                    continue
                entry_bundle = site_payload.get("bundle", []) or []
                if normalize_bundle_key(entry_bundle) != bundle_key:
                    continue
                status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
                if status not in {"UNKNOWN", "DEAD", "HANDLED"}:
                    status = "UNKNOWN"
                summary[status] += 1
                summary["total"] += 1
            case _:
                pass
    return summary
