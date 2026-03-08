from __future__ import annotations

from collections.abc import Mapping

from gabion.analysis.foundation.timeout_context import check_deadline

TYPE_BASE_NAMESPACE = "type_base"
TYPE_CTOR_NAMESPACE = "type_ctor"
EVIDENCE_KIND_NAMESPACE = "evidence_kind"
SITE_KIND_NAMESPACE = "site_kind"
SYNTH_NAMESPACE = "synth"

_NAMESPACE_TO_PREFIX: dict[str, str] = {
    TYPE_BASE_NAMESPACE: "",
    TYPE_CTOR_NAMESPACE: "ctor:",
    EVIDENCE_KIND_NAMESPACE: "evidence:",
    SITE_KIND_NAMESPACE: "site:",
    SYNTH_NAMESPACE: "synth:",
}
_DYNAMIC_NAMESPACE_PREFIX = "ns:"


def namespace_to_prefix(namespace: str) -> str:
    check_deadline()
    namespace_text = str(namespace)
    known = _NAMESPACE_TO_PREFIX.get(namespace_text)
    if known is not None:
        return known
    return f"{_DYNAMIC_NAMESPACE_PREFIX}{namespace_text}:"


def known_namespaces() -> tuple[str, ...]:
    check_deadline()
    return tuple(_NAMESPACE_TO_PREFIX)


def namespace_key(key: str) -> tuple[str, str]:
    """Decode a raw registry key into (namespace, local_key)."""
    check_deadline()
    for namespace, prefix in _NAMESPACE_TO_PREFIX.items():
        check_deadline()
        if prefix and key.startswith(prefix):
            return namespace, key[len(prefix) :]
    if key.startswith(_DYNAMIC_NAMESPACE_PREFIX):
        body = key[len(_DYNAMIC_NAMESPACE_PREFIX) :]
        namespace, separator, local = body.partition(":")
        if separator:
            return namespace, local
    return TYPE_BASE_NAMESPACE, key


def raw_key(namespace: str, key: str) -> str:
    """Encode a (namespace, local_key) pair to the raw registry key form."""
    check_deadline()
    return f"{namespace_to_prefix(namespace)}{key}"


def namespace_prefix_map() -> Mapping[str, str]:
    check_deadline()
    return dict(_NAMESPACE_TO_PREFIX)
