from __future__ import annotations

import json
from typing import Callable, Iterable, Mapping, Sequence


def normalize_params(values: Iterable[str]) -> list[str]:
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return sorted(cleaned)


def normalize_param_string(value: str) -> str:
    return ",".join(normalize_params(value.split(",")))


def normalize_reason(value: str) -> str:
    return " ".join(str(value).strip().split())


def _normalize_target(target: object) -> tuple[str, str] | None:
    if isinstance(target, Mapping):
        path = str(target.get("path", "") or "").strip()
        qual = str(target.get("qual", "") or "").strip()
    elif isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
        if len(target) < 2:
            return None
        path = str(target[0]).strip()
        qual = str(target[1]).strip()
    else:
        return None
    if not path or not qual:
        return None
    return path, qual


def _normalize_span(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        parts = [
            value.get("line"),
            value.get("col"),
            value.get("end_line"),
            value.get("end_col"),
        ]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
    else:
        return None
    if len(parts) != 4:
        return None
    normalized: list[int] = []
    for part in parts:
        try:
            item = int(part)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if item < 0:
            return None
        normalized.append(item)
    return normalized


def _normalize_site(site: object) -> dict[str, object]:
    if isinstance(site, Mapping):
        path = str(site.get("path", "") or "").strip()
        qual = str(site.get("qual", "") or "").strip()
        span = _normalize_span(site.get("span"))
    elif isinstance(site, Sequence) and not isinstance(site, (str, bytes)):
        if len(site) < 2:
            return {"path": "", "qual": ""}
        path = str(site[0]).strip()
        qual = str(site[1]).strip()
        span = None
    else:
        return {"path": "", "qual": ""}
    payload: dict[str, object] = {"path": path, "qual": qual}
    if span:
        payload["span"] = span
    return payload


def normalize_targets(targets: Iterable[object]) -> list[dict[str, str]]:
    cleaned: dict[tuple[str, str], dict[str, str]] = {}
    for target in targets:
        parts = _normalize_target(target)
        if not parts:
            continue
        path, qual = parts
        cleaned[(path, qual)] = {"path": path, "qual": qual}
    return [cleaned[key] for key in sorted(cleaned)]


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


def make_function_site_key(*, path: str, qual: str) -> dict[str, object]:
    # dataflow-bundle: path, qual
    return {
        "k": "function_site",
        "site": {
            "path": str(path).strip(),
            "qual": str(qual).strip(),
        },
    }


def make_call_footprint_key(
    *,
    path: str,
    qual: str,
    targets: Iterable[object],
) -> dict[str, object]:
    # dataflow-bundle: path, qual, targets
    return {
        "k": "call_footprint",
        "site": {
            "path": str(path).strip(),
            "qual": str(qual).strip(),
        },
        "targets": normalize_targets(targets),
    }


def make_ambiguity_set_key(
    *,
    path: str,
    qual: str,
    span: Iterable[object] | None = None,
    candidates: Iterable[object],
) -> dict[str, object]:
    # dataflow-bundle: candidates, path, qual
    site: dict[str, object] = {
        "path": str(path).strip(),
        "qual": str(qual).strip(),
    }
    normalized_span = _normalize_span(span)
    if normalized_span:
        site["span"] = normalized_span
    return {
        "k": "ambiguity_set",
        "site": site,
        "candidates": normalize_targets(candidates),
    }


def make_partition_witness_key(
    *,
    kind: str,
    site: Mapping[str, object],
    ambiguity: Mapping[str, object],
    support: Mapping[str, object] | None = None,
    collapse: Mapping[str, object] | None = None,
) -> dict[str, object]:
    # dataflow-bundle: ambiguity, kind, site
    payload: dict[str, object] = {
        "k": "partition_witness",
        "kind": str(kind).strip(),
        "site": _normalize_site(site),
        "ambiguity": normalize_key(ambiguity),
    }
    if support:
        payload["support"] = {
            str(key): str(value).strip()
            for key, value in support.items()
            if str(value).strip()
        }
    if collapse:
        payload["collapse"] = {
            str(key): str(value).strip()
            for key, value in collapse.items()
            if str(value).strip()
        }
    return payload


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
    if kind == "function_site":
        site = key.get("site", {})
        if not isinstance(site, Mapping):
            site = {}
        path = str(site.get("path", "") or "")
        qual = str(site.get("qual", "") or "")
        return make_function_site_key(path=path, qual=qual)
    if kind == "call_footprint":
        site = key.get("site", {})
        path = str(site.get("path", "") or "") if isinstance(site, Mapping) else ""
        qual = str(site.get("qual", "") or "") if isinstance(site, Mapping) else ""
        targets = key.get("targets", [])
        if isinstance(targets, str):
            targets = []
        return make_call_footprint_key(
            path=path,
            qual=qual,
            targets=targets if isinstance(targets, Iterable) else [],
        )
    if kind == "ambiguity_set":
        site = key.get("site", {})
        normalized_site = _normalize_site(site)
        path = str(normalized_site.get("path", "") or "")
        qual = str(normalized_site.get("qual", "") or "")
        span = normalized_site.get("span")
        candidates = key.get("candidates", [])
        if isinstance(candidates, str):
            candidates = []
        return make_ambiguity_set_key(
            path=path,
            qual=qual,
            span=span if isinstance(span, Iterable) else None,
            candidates=candidates if isinstance(candidates, Iterable) else [],
        )
    if kind == "partition_witness":
        kind_value = str(key.get("kind", "") or "")
        site = key.get("site", {})
        ambiguity = key.get("ambiguity", {})
        support = key.get("support")
        collapse = key.get("collapse")
        support_map = support if isinstance(support, Mapping) else None
        collapse_map = collapse if isinstance(collapse, Mapping) else None
        return make_partition_witness_key(
            kind=kind_value,
            site=site if isinstance(site, Mapping) else {},
            ambiguity=ambiguity if isinstance(ambiguity, Mapping) else {},
            support=support_map,
            collapse=collapse_map,
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
    if kind == "function_site":
        site = normalized.get("site", {})
        path = str(site.get("path", "") or "")
        qual = str(site.get("qual", "") or "")
        return f"E:function_site::{path}::{qual}"
    if kind == "call_footprint":
        site = normalized.get("site", {})
        path = str(site.get("path", "") or "")
        qual = str(site.get("qual", "") or "")
        parts = [path, qual]
        targets = normalized.get("targets", [])
        if isinstance(targets, list):
            for target in targets:
                if not isinstance(target, Mapping):
                    continue
                target_path = str(target.get("path", "") or "")
                target_qual = str(target.get("qual", "") or "")
                if not target_path or not target_qual:
                    continue
                parts.extend([target_path, target_qual])
        return "E:call_footprint::" + "::".join(parts)
    if kind == "ambiguity_set":
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return f"E:ambiguity_set::{payload}"
    if kind == "partition_witness":
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return f"E:partition_witness::{payload}"
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
    if prefix == "function_site":
        if len(rest) != 2:
            return None
        path, qual = rest
        return make_function_site_key(path=path, qual=qual)
    if prefix == "call_footprint":
        if len(rest) < 2:
            return None
        path, qual, *targets = rest
        if targets and len(targets) % 2 != 0:
            return None
        target_pairs = []
        for idx in range(0, len(targets), 2):
            target_pairs.append((targets[idx], targets[idx + 1]))
        return make_call_footprint_key(path=path, qual=qual, targets=target_pairs)
    if prefix == "ambiguity_set":
        if not rest:
            return None
        try:
            payload = json.loads("::".join(rest))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, Mapping):
            return normalize_key(payload)
        return None
    if prefix == "partition_witness":
        if not rest:
            return None
        try:
            payload = json.loads("::".join(rest))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, Mapping):
            return normalize_key(payload)
        return None
    return None


def is_opaque(key: Mapping[str, object]) -> bool:
    return normalize_key(key).get("k") == "opaque"
