from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def normalize_bundle_key(bundle: object) -> str:
    """Canonicalize a bundle payload into a stable join key.

    This intentionally ignores non-string entries because bundles are defined
    over symbol names.
    """
    if not isinstance(bundle, (list, tuple, set)):
        return ""
    values = {item.strip() for item in bundle if isinstance(item, str) and item.strip()}
    return ",".join(sorted(values))


def normalize_string_list(value: object) -> list[str]:
    """Normalize a payload field that should be a list of strings.

    The intent is to accept schema-level payloads coming from JSON where these
    fields can be absent, malformed, or represented as comma-separated strings.
    """
    raw: list[str] = []
    if value is None:
        return raw
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, (list, tuple, set)):
        raw = [item for item in value if isinstance(item, str)]
    else:
        return raw

    parts: list[str] = []
    for item in raw:
        parts.extend([part.strip() for part in item.split(",") if part.strip()])
    return sorted(set(parts))


@dataclass(frozen=True)
class Site:
    path: str
    function: str
    bundle: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object) -> Site | None:
        if not isinstance(payload, Mapping):
            return None
        path = str(payload.get("path", "")).strip()
        function = str(payload.get("function", "")).strip()
        bundle = normalize_string_list(payload.get("bundle", []) or [])
        return cls(path=path, function=function, bundle=tuple(bundle))

    def bundle_key(self) -> str:
        return normalize_bundle_key(list(self.bundle))

    def key(self) -> tuple[str, str, str]:
        return (self.path, self.function, self.bundle_key())


def exception_obligation_summary_for_site(
    obligations: Iterable[Mapping[str, object]],
    *,
    site: Site,
) -> dict[str, int]:
    summary = {"UNKNOWN": 0, "DEAD": 0, "HANDLED": 0, "total": 0}
    bundle_key = site.bundle_key()
    for entry in obligations:
        raw_site = entry.get("site", {}) or {}
        if not isinstance(raw_site, Mapping):
            continue
        if str(raw_site.get("path", "")) != site.path:
            continue
        if str(raw_site.get("function", "")) != site.function:
            continue
        entry_bundle = raw_site.get("bundle", []) or []
        if normalize_bundle_key(entry_bundle) != bundle_key:
            continue
        status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
        if status not in {"UNKNOWN", "DEAD", "HANDLED"}:
            status = "UNKNOWN"
        summary[status] += 1
        summary["total"] += 1
    return summary
