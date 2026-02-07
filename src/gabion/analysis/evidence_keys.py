from __future__ import annotations

import json
from typing import Callable, Iterable, Mapping


def normalize_params(values: Iterable[str]) -> list[str]:
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return sorted(cleaned)


def normalize_param_string(value: str) -> str:
    return ",".join(normalize_params(value.split(",")))


def normalize_reason(value: str) -> str:
    return " ".join(str(value).strip().split())


def make_paramset_key(params: Iterable[str]) -> dict[str, object]:
    return {
        "k": "paramset",
        "params": normalize_params(params),
    }


def make_decision_surface_key(
    *,
    mode: str,
    path: str,
    qual: str,
    param: str,
) -> dict[str, object]:
    # dataflow-bundle: mode, param, path, qual
    return {
        "k": "decision_surface",
        "m": str(mode).strip() or "direct",
        "site": {
            "path": str(path).strip(),
            "qual": str(qual).strip(),
        },
        "param": normalize_param_string(str(param)),
    }


def make_never_sink_key(
    *,
    path: str,
    qual: str,
    param: str,
    reason: str | None = None,
) -> dict[str, object]:
    # dataflow-bundle: param, path, qual
    key = {
        "k": "never_sink",
        "site": {
            "path": str(path).strip(),
            "qual": str(qual).strip(),
        },
        "param": normalize_param_string(str(param)),
    }
    if reason:
        normalized = normalize_reason(reason)
        if normalized:
            key["reason"] = normalized
    return key


def make_opaque_key(display: str) -> dict[str, object]:
    return {"k": "opaque", "s": str(display).strip()}


def normalize_key(key: Mapping[str, object]) -> dict[str, object]:
    kind = str(key.get("k", "") or "").strip()
    if kind == "paramset":
        params = key.get("params", [])
        if isinstance(params, str):
            params = params.split(",")
        return make_paramset_key(params if isinstance(params, Iterable) else [])
    if kind == "decision_surface":
        mode = str(key.get("m", "direct") or "direct")
        site = key.get("site", {})
        if not isinstance(site, Mapping):
            site = {}
        path = str(site.get("path", "") or "")
        qual = str(site.get("qual", "") or "")
        param = str(key.get("param", "") or "")
        return make_decision_surface_key(mode=mode, path=path, qual=qual, param=param)
    if kind == "never_sink":
        site = key.get("site", {})
        if not isinstance(site, Mapping):
            site = {}
        path = str(site.get("path", "") or "")
        qual = str(site.get("qual", "") or "")
        param = str(key.get("param", "") or "")
        reason = key.get("reason")
        return make_never_sink_key(
            path=path,
            qual=qual,
            param=param,
            reason=str(reason) if reason else None,
        )
    if kind == "opaque":
        return make_opaque_key(str(key.get("s", "") or ""))
    if not kind:
        return make_opaque_key("")
    # Preserve unknown kinds but normalize obvious fields.
    normalized = dict(key)
    normalized["k"] = kind
    return normalized


def key_identity(key: Mapping[str, object]) -> str:
    normalized = normalize_key(key)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def render_display(
    key: Mapping[str, object],
    *,
    normalize: Callable[[Mapping[str, object]], Mapping[str, object]] = normalize_key,
) -> str:
    normalized = normalize(key)
    kind = normalized.get("k")
    if kind == "opaque":
        return str(normalized.get("s", "") or "")
    if kind == "paramset":
        params = normalized.get("params", [])
        if isinstance(params, list):
            joined = ",".join(str(p) for p in params)
        else:
            joined = ""
        return f"E:paramset::{joined}" if joined else "E:paramset"
    if kind == "decision_surface":
        mode = str(normalized.get("m", "direct") or "direct")
        site = normalized.get("site", {})
        path = str(site.get("path", "") or "")
        qual = str(site.get("qual", "") or "")
        param = str(normalized.get("param", "") or "")
        return f"E:decision_surface/{mode}::{path}::{qual}::{param}"
    if kind == "never_sink":
        site = normalized.get("site", {})
        path = str(site.get("path", "") or "")
        qual = str(site.get("qual", "") or "")
        param = str(normalized.get("param", "") or "")
        return f"E:never/sink::{path}::{qual}::{param}"
    return f"E:{kind}"


def parse_display(display: str) -> dict[str, object] | None:
    value = str(display).strip()
    if not value.startswith("E:"):
        return None
    body = value[2:]
    if not body:
        return None
    parts = body.split("::")
    prefix = parts[0]
    rest = parts[1:]
    if prefix == "paramset":
        if not rest:
            return None
        params = normalize_params(rest[0].split(","))
        return make_paramset_key(params)
    if prefix.startswith("decision_surface/"):
        if len(rest) != 3:
            return None
        mode = prefix.split("/", 1)[1]
        path, qual, param = rest
        return make_decision_surface_key(mode=mode, path=path, qual=qual, param=param)
    if prefix == "never/sink":
        if len(rest) != 3:
            return None
        path, qual, param = rest
        return make_never_sink_key(path=path, qual=qual, param=param)
    return None


def is_opaque(key: Mapping[str, object]) -> bool:
    return normalize_key(key).get("k") == "opaque"
