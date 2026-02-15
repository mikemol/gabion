#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _load_profile(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"deadline profile must be a JSON object: {path}")
    return {str(key): data[key] for key in data}


def _site_rows(profile: Mapping[str, object]) -> list[dict[str, object]]:
    raw_sites = profile.get("sites")
    if not isinstance(raw_sites, list):
        return []
    rows: list[dict[str, object]] = []
    for entry in raw_sites:
        if not isinstance(entry, Mapping):
            continue
        rows.append({str(key): entry[key] for key in entry})
    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("elapsed_between_checks_ns")),
            str(row.get("path", "")),
            str(row.get("qual", "")),
        )
    )
    return rows


def _site_key(row: Mapping[str, object]) -> str:
    return f"{row.get('path', '')}:{row.get('qual', '')}"


def _top_rows(rows: list[dict[str, object]], *, top: int) -> list[dict[str, object]]:
    return rows[: max(0, top)]


def _to_ms(ns: int) -> float:
    return ns / 1_000_000


def _to_s(ns: int) -> float:
    return ns / 1_000_000_000


def _build_summary(
    *,
    ci_profile: Mapping[str, object],
    local_profile: Mapping[str, object] | None,
    top: int,
) -> dict[str, object]:
    ci_sites = _site_rows(ci_profile)
    local_sites = _site_rows(local_profile or {})
    ci_total_ns = _safe_int(ci_profile.get("total_elapsed_ns"))
    ci_checks_total = _safe_int(ci_profile.get("checks_total"))
    ci_unattributed_ns = _safe_int(ci_profile.get("unattributed_elapsed_ns"))
    ci_top_sites = _top_rows(ci_sites, top=top)

    local_total_ns = _safe_int(local_profile.get("total_elapsed_ns")) if local_profile else 0
    local_checks_total = _safe_int(local_profile.get("checks_total")) if local_profile else 0
    local_unattributed_ns = (
        _safe_int(local_profile.get("unattributed_elapsed_ns")) if local_profile else 0
    )

    local_site_map: dict[str, dict[str, object]] = {}
    for row in local_sites:
        local_site_map[_site_key(row)] = row
    regressions: list[dict[str, object]] = []
    for row in ci_sites:
        key = _site_key(row)
        local_row = local_site_map.get(key)
        ci_elapsed = _safe_int(row.get("elapsed_between_checks_ns"))
        local_elapsed = _safe_int(local_row.get("elapsed_between_checks_ns")) if local_row else 0
        delta = ci_elapsed - local_elapsed
        regressions.append(
            {
                "site": key,
                "ci_elapsed_ns": ci_elapsed,
                "local_elapsed_ns": local_elapsed,
                "delta_ns": delta,
                "ratio": (ci_elapsed / local_elapsed) if local_elapsed > 0 else None,
            }
        )
    regressions.sort(
        key=lambda row: (
            -_safe_int(row.get("delta_ns")),
            str(row.get("site", "")),
        )
    )

    comparison: dict[str, object] | None = None
    if local_profile is not None and local_total_ns > 0:
        comparison = {
            "total_elapsed_ratio": ci_total_ns / local_total_ns,
            "checks_total_ratio": (
                (ci_checks_total / local_checks_total) if local_checks_total > 0 else None
            ),
            "unattributed_elapsed_ratio": (
                (ci_unattributed_ns / local_unattributed_ns)
                if local_unattributed_ns > 0
                else None
            ),
        }

    return {
        "ci": {
            "total_elapsed_ns": ci_total_ns,
            "total_elapsed_s": _to_s(ci_total_ns),
            "checks_total": ci_checks_total,
            "unattributed_elapsed_ns": ci_unattributed_ns,
            "unattributed_elapsed_ms": _to_ms(ci_unattributed_ns),
            "top_sites": ci_top_sites,
        },
        "local": (
            {
                "total_elapsed_ns": local_total_ns,
                "total_elapsed_s": _to_s(local_total_ns),
                "checks_total": local_checks_total,
                "unattributed_elapsed_ns": local_unattributed_ns,
                "unattributed_elapsed_ms": _to_ms(local_unattributed_ns),
            }
            if local_profile is not None
            else None
        ),
        "comparison": comparison,
        "top_regressions": _top_rows(regressions, top=top),
    }


def _render_markdown(summary: Mapping[str, object]) -> str:
    ci = summary.get("ci", {})
    local = summary.get("local")
    comparison = summary.get("comparison")
    top_sites = ci.get("top_sites", []) if isinstance(ci, Mapping) else []
    top_regressions = summary.get("top_regressions", [])

    lines = [
        "# Deadline Profile Summary",
        "",
        "## CI Totals",
        f"- `total_elapsed_s`: `{_safe_int(ci.get('total_elapsed_ns')) / 1_000_000_000:.3f}`",
        f"- `checks_total`: `{_safe_int(ci.get('checks_total'))}`",
        f"- `unattributed_elapsed_ms`: `{_safe_int(ci.get('unattributed_elapsed_ns')) / 1_000_000:.3f}`",
        "",
    ]
    if isinstance(local, Mapping):
        lines.extend(
            [
                "## Local Totals",
                f"- `total_elapsed_s`: `{_safe_int(local.get('total_elapsed_ns')) / 1_000_000_000:.3f}`",
                f"- `checks_total`: `{_safe_int(local.get('checks_total'))}`",
                f"- `unattributed_elapsed_ms`: `{_safe_int(local.get('unattributed_elapsed_ns')) / 1_000_000:.3f}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Local Totals",
                "- Local profile not provided; comparison skipped.",
                "",
            ]
        )
    if isinstance(comparison, Mapping):
        ratio = comparison.get("total_elapsed_ratio")
        checks_ratio = comparison.get("checks_total_ratio")
        lines.extend(
            [
                "## CI vs Local",
                f"- `total_elapsed_ratio`: `{ratio:.3f}`" if isinstance(ratio, float) else "- `total_elapsed_ratio`: `n/a`",
                (
                    f"- `checks_total_ratio`: `{checks_ratio:.3f}`"
                    if isinstance(checks_ratio, float)
                    else "- `checks_total_ratio`: `n/a`"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Top CI Sites",
            "| site | elapsed_ms | checks | max_gap_ms |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    if isinstance(top_sites, list) and top_sites:
        for row in top_sites:
            if not isinstance(row, Mapping):
                continue
            site = _site_key(row)
            elapsed_ms = _safe_int(row.get("elapsed_between_checks_ns")) / 1_000_000
            checks = _safe_int(row.get("check_count"))
            max_gap_ms = _safe_int(row.get("max_gap_ns")) / 1_000_000
            lines.append(
                f"| `{site}` | {elapsed_ms:.3f} | {checks} | {max_gap_ms:.3f} |"
            )
    else:
        lines.append("| _none_ | 0.000 | 0 | 0.000 |")
    lines.append("")

    lines.extend(
        [
            "## Top CI Regressions vs Local",
            "| site | ci_ms | local_ms | delta_ms | ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if isinstance(top_regressions, list) and top_regressions:
        for row in top_regressions:
            if not isinstance(row, Mapping):
                continue
            site = str(row.get("site", ""))
            ci_ms = _safe_int(row.get("ci_elapsed_ns")) / 1_000_000
            local_ms = _safe_int(row.get("local_elapsed_ns")) / 1_000_000
            delta_ms = _safe_int(row.get("delta_ns")) / 1_000_000
            ratio = row.get("ratio")
            ratio_text = f"{ratio:.3f}" if isinstance(ratio, float) else "n/a"
            lines.append(
                f"| `{site}` | {ci_ms:.3f} | {local_ms:.3f} | {delta_ms:.3f} | {ratio_text} |"
            )
    else:
        lines.append("| _none_ | 0.000 | 0.000 | 0.000 | n/a |")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CI deadline profile and compare against local reference."
    )
    parser.add_argument(
        "--ci-profile",
        type=Path,
        default=Path("artifacts/out/deadline_profile.json"),
        help="Path to the CI deadline profile JSON.",
    )
    parser.add_argument(
        "--local-profile",
        type=Path,
        default=Path("artifacts/out/deadline_profile_local.json"),
        help="Optional local deadline profile JSON.",
    )
    parser.add_argument(
        "--allow-missing-local",
        action="store_true",
        help="Do not fail when local profile is absent.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/out/deadline_profile_ci_summary.json"),
        help="Path for JSON summary output.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("artifacts/out/deadline_profile_ci_summary.md"),
        help="Path for Markdown summary output.",
    )
    parser.add_argument(
        "--step-summary",
        type=Path,
        default=None,
        help="Optional GitHub step summary file to append markdown summary.",
    )
    parser.add_argument("--top", type=int, default=10, help="Top site rows to include.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ci_profile = _load_profile(args.ci_profile)
    if ci_profile is None:
        raise SystemExit(f"Missing CI deadline profile: {args.ci_profile}")
    local_profile = _load_profile(args.local_profile)
    if local_profile is None and not args.allow_missing_local:
        raise SystemExit(f"Missing local deadline profile: {args.local_profile}")
    summary = _build_summary(
        ci_profile=ci_profile,
        local_profile=local_profile,
        top=max(1, int(args.top)),
    )
    markdown = _render_markdown(summary)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(markdown + "\n")
    if args.step_summary is not None:
        args.step_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.step_summary.open("a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
