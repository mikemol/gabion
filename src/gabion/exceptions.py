"""Exception protocol markers for Gabion analysis."""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gabion.analysis.foundation.marker_protocol import MarkerPayload


class NeverRaise(RuntimeError):
    """Sentinel exception that should be statically unreachable.

    Raising this exception is a signal to Gabion that the code path is expected
    to be proven unreachable by analysis. If it is reachable, Gabion should
    treat it as a violation.
    """

    def __init__(self, message: str, *, marker_payload: MarkerPayload | None = None):
        from gabion.analysis.foundation.marker_protocol import marker_identity, never_marker_payload

        super().__init__(message)
        payload = marker_payload or never_marker_payload(reason=message)
        self.marker_payload = payload
        self.marker_id = marker_identity(payload)
        self.marker_kind = payload.marker_kind.value

    @property
    def marker_payload_dict(self) -> dict[str, object]:
        payload = asdict(self.marker_payload)
        payload["marker_kind"] = str(self.marker_payload.marker_kind.value)
        payload["lifecycle_state"] = str(self.marker_payload.lifecycle_state.value)
        match payload.get("links"):
            case list() as links:
                normalized_links: list[dict[str, str]] = []
                for raw_link in links:
                    match raw_link:
                        case dict() as link:
                            normalized_links.append(
                                {
                                    "kind": str(link.get("kind", "")),
                                    "value": str(link.get("value", "")),
                                }
                            )
                        case _:
                            pass
                payload["links"] = normalized_links
            case _:
                pass
        return payload


class NeverThrown(NeverRaise):
    """Alias for NeverRaise used by the explicit never() marker."""
