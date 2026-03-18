#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


_POLICY_DOC_PREFIXES = ("in/", "docs/")
_POLICY_DOC_FILES = {
    "AGENTS.md",
    "CONTRIBUTING.md",
    "README.md",
    "POLICY_SEED.md",
    "glossary.md",
}
_DEFAULT_SCOPE_ALLOWLIST = {
    "docs/sppf_checklist.md",
    "docs/docflow_strict_failure_analysis.md",
}


@dataclass(frozen=True)
class BaselineEntry:
    row_id: str
    path: str
    classification: str
    first_seen: date

    def as_dict(self) -> dict[str, str]:
        return {
            "row_id": self.row_id,
            "path": self.path,
            "classification": self.classification,
            "first_seen": self.first_seen.isoformat(),
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid JSON payload at {path}: expected object")
    return payload


def _load_packets(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    packets = payload.get("packets")
    if not isinstance(packets, list):
        raise SystemExit(f"invalid packet payload at {path}: missing packets list")
    selected: list[dict[str, Any]] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        rows = packet.get("rows")
        if not isinstance(rows, list):
            continue
        selected.append(packet)
    return selected


def _load_baseline(path: Path) -> dict[str, BaselineEntry]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    rows = payload.get("entries")
    if not isinstance(rows, list):
        raise SystemExit(f"invalid baseline payload at {path}: missing entries list")
    entries: dict[str, BaselineEntry] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("row_id")
        packet_path = row.get("path")
        classification = row.get("classification")
        first_seen_raw = row.get("first_seen")
        if not isinstance(row_id, str) or not row_id:
            raise SystemExit(f"invalid baseline entry in {path}: row_id required")
        if not isinstance(packet_path, str) or not packet_path:
            raise SystemExit(f"invalid baseline entry in {path}: path required")
        if not isinstance(classification, str) or not classification:
            raise SystemExit(f"invalid baseline entry in {path}: classification required")
        if not isinstance(first_seen_raw, str) or not first_seen_raw:
            raise SystemExit(f"invalid baseline entry in {path}: first_seen required")
        entries[row_id] = BaselineEntry(
            row_id=row_id,
            path=packet_path,
            classification=classification,
            first_seen=date.fromisoformat(first_seen_raw),
        )
    return entries


def _write_baseline(path: Path, entries: dict[str, BaselineEntry]) -> None:
    payload = {
        "entries": [entry.as_dict() for entry in sorted(entries.values(), key=lambda item: item.row_id)]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_policy_doc_path(path: str) -> bool:
    if path in _POLICY_DOC_FILES:
        return True
    return any(path.startswith(prefix) for prefix in _POLICY_DOC_PREFIXES)


def _changed_paths_from_git(root: Path, base_sha: str | None, head_sha: str | None) -> list[str]:
    if base_sha and head_sha:
        command = ["git", "diff", "--name-only", f"{base_sha}..{head_sha}"]
    else:
        command = ["git", "diff", "--name-only", "HEAD"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"failed to compute changed paths: {' '.join(command)}: {exc}") from exc
    return sorted(path.strip() for path in completed.stdout.splitlines() if path.strip())


def _packet_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = packet.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _packet_touch_set(packet: dict[str, Any]) -> set[str]:
    raw = packet.get("touch_set")
    if not isinstance(raw, list):
        return set()
    values: set[str] = set()
    for item in raw:
        if isinstance(item, str) and item:
            values.add(item)
    return values


def _packet_proving_tests(packet: dict[str, Any]) -> list[str]:
    raw = packet.get("proving_tests")
    if not isinstance(raw, list):
        return []
    values = [item for item in raw if isinstance(item, str) and item]
    return sorted(set(values))


def _packet_next_action(packet: dict[str, Any]) -> str:
    classification = str(packet.get("classification", ""))
    if classification == "metadata_only":
        return "apply metadata-only frontmatter pin normalization and rerun docflow packet loop"
    if classification == "needs_semantic_update":
        return "perform human semantic review, update stale anchors/attestations, then rerun proving tests"
    return "revalidate review pins and close packet when row set reaches zero"


def _run_proving_tests(test_targets: list[str]) -> tuple[int, str]:
    if not test_targets:
        return 0, "no proving tests selected"
    command = [sys.executable, "-m", "pytest", "-q", *test_targets]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output_tail = "\n".join(completed.stdout.splitlines()[-20:])
    if completed.stderr.strip():
        output_tail = (output_tail + "\n" + "\n".join(completed.stderr.splitlines()[-20:])).strip()
    return int(completed.returncode), output_tail


def run(
    *,
    root: Path,
    packets_path: Path,
    baseline_path: Path,
    out_path: Path,
    debt_out_path: Path,
    check: bool,
    write_baseline: bool,
    max_age_days: int,
    changed_paths: list[str] | None,
    base_sha: str | None,
    head_sha: str | None,
    scope_allowlist: set[str],
    run_proving_tests: bool,
) -> int:
    packets = _load_packets(packets_path)
    baseline = _load_baseline(baseline_path)
    now = date.today()

    current_rows: dict[str, tuple[str, str]] = {}
    for packet in packets:
        packet_path = str(packet.get("path", ""))
        classification = str(packet.get("classification", "materially_still_true"))
        for row in _packet_rows(packet):
            row_id = row.get("row_id")
            if isinstance(row_id, str) and row_id:
                current_rows[row_id] = (packet_path, classification)

    new_row_ids = sorted(set(current_rows) - set(baseline))
    drifted_row_ids: list[str] = []
    for row_id, baseline_entry in baseline.items():
        if row_id not in current_rows:
            continue
        age_days = (now - baseline_entry.first_seen).days
        if age_days > max_age_days:
            drifted_row_ids.append(row_id)
    drifted_row_ids = sorted(set(drifted_row_ids))

    if changed_paths is None:
        changed = _changed_paths_from_git(root, base_sha=base_sha, head_sha=head_sha)
    else:
        changed = sorted(set(changed_paths))

    active_packets = [packet for packet in packets if _packet_rows(packet)]
    active_touch_paths: set[str] = set()
    for packet in active_packets:
        active_touch_paths.update(_packet_touch_set(packet))

    changed_policy_docs = sorted(path for path in changed if _is_policy_doc_path(path))
    if active_packets:
        out_of_scope_touches = sorted(
            path
            for path in changed_policy_docs
            if path not in active_touch_paths and path not in scope_allowlist
        )
    else:
        out_of_scope_touches = []

    unresolved_touched_packets: list[str] = []
    changed_set = set(changed_policy_docs)
    for packet in active_packets:
        path = str(packet.get("path", ""))
        if path and path in changed_set:
            unresolved_touched_packets.append(path)
    unresolved_touched_packets = sorted(set(unresolved_touched_packets))

    packet_status: list[dict[str, Any]] = []
    for packet in active_packets:
        path = str(packet.get("path", ""))
        classification = str(packet.get("classification", "materially_still_true"))
        owner = packet.get("doc_owner")
        row_ids = [str(row.get("row_id")) for row in _packet_rows(packet) if isinstance(row.get("row_id"), str)]
        row_ages = []
        for row_id in row_ids:
            baseline_entry = baseline.get(row_id)
            if baseline_entry is None:
                continue
            row_ages.append((now - baseline_entry.first_seen).days)
        has_new = any(row_id in new_row_ids for row_id in row_ids)
        has_drift = any(row_id in drifted_row_ids for row_id in row_ids)
        status = "ready"
        if has_new:
            status = "blocked"
        elif has_drift:
            status = "drifted"
        packet_status.append(
            {
                "path": path,
                "classification": classification,
                "doc_owner": owner,
                "status": status,
                "row_count": len(row_ids),
                "row_ids": row_ids,
                "max_age_days": max(row_ages) if row_ages else None,
                "next_action": _packet_next_action(packet),
            }
        )

    proving_test_targets: list[str] = []
    for packet in active_packets:
        proving_test_targets.extend(_packet_proving_tests(packet))
    proving_test_targets = sorted(set(proving_test_targets))

    proving_test_result = {
        "status": "skipped",
        "returncode": 0,
        "targets": proving_test_targets,
        "output_tail": "",
    }
    if run_proving_tests:
        returncode, output_tail = _run_proving_tests(proving_test_targets)
        proving_test_result = {
            "status": "pass" if returncode == 0 else "fail",
            "returncode": returncode,
            "targets": proving_test_targets,
            "output_tail": output_tail,
        }

    status_counts = Counter(item["status"] for item in packet_status)
    ledger = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "active_rows": len(current_rows),
            "active_packets": len(active_packets),
            "new_rows": len(new_row_ids),
            "drifted_rows": len(drifted_row_ids),
            "ready": int(status_counts.get("ready", 0)),
            "blocked": int(status_counts.get("blocked", 0)),
            "drifted": int(status_counts.get("drifted", 0)),
        },
        "packets": packet_status,
    }

    enforce_report = {
        "scope": {
            "kind": "docflow_packet_enforcement",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "packets": str(packets_path),
            "baseline": str(baseline_path),
            "max_age_days": max_age_days,
            "check": check,
        },
        "summary": ledger["summary"],
        "new_rows": sorted(
            {
                "row_id": row_id,
                "path": current_rows[row_id][0],
                "classification": current_rows[row_id][1],
            }
            for row_id in new_row_ids
        ),
        "drifted_rows": drifted_row_ids,
        "out_of_scope_touches": out_of_scope_touches,
        "unresolved_touched_packets": unresolved_touched_packets,
        "changed_paths": changed,
        "packet_status": packet_status,
        "proving_tests": proving_test_result,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    debt_out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(enforce_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    debt_out_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if write_baseline:
        next_entries: dict[str, BaselineEntry] = {}
        for row_id, (path, classification) in current_rows.items():
            existing = baseline.get(row_id)
            first_seen = existing.first_seen if existing is not None else now
            next_entries[row_id] = BaselineEntry(
                row_id=row_id,
                path=path,
                classification=classification,
                first_seen=first_seen,
            )
        _write_baseline(baseline_path, next_entries)

    failed = False
    if check and new_row_ids:
        failed = True
        print("docflow-packet-enforce: new docflow finding rows detected:")
        for row_id in new_row_ids:
            path, classification = current_rows[row_id]
            print(f"  - {row_id}: {path} [{classification}]")
    if check and drifted_row_ids:
        failed = True
        print("docflow-packet-enforce: packet debt drift exceeded max-age:")
        for row_id in drifted_row_ids:
            print(f"  - {row_id}")
    if check and out_of_scope_touches:
        failed = True
        print("docflow-packet-enforce: out-of-scope policy-doc touches:")
        for path in out_of_scope_touches:
            print(f"  - {path}")
    if check and unresolved_touched_packets:
        failed = True
        print("docflow-packet-enforce: touched packet docs still contain active rows:")
        for path in unresolved_touched_packets:
            print(f"  - {path}")
    if check and run_proving_tests and proving_test_result["returncode"] != 0:
        failed = True
        print("docflow-packet-enforce: targeted proving tests failed.")

    print(
        "docflow-packet-enforce: "
        f"rows={len(current_rows)} new={len(new_row_ids)} drifted={len(drifted_row_ids)} "
        f"ready={status_counts.get('ready', 0)} blocked={status_counts.get('blocked', 0)} "
        f"drifted_packets={status_counts.get('drifted', 0)} out={out_path}"
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--packets", default="artifacts/out/docflow_warning_doc_packets.json")
    parser.add_argument("--baseline", default="docs/baselines/docflow_packet_baseline.json")
    parser.add_argument("--out", default="artifacts/out/docflow_packet_enforcement.json")
    parser.add_argument("--debt-out", default="artifacts/out/docflow_packet_debt_ledger.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--max-age-days", type=int, default=14)
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument("--changed-path", action="append", default=[])
    parser.add_argument("--allow-touch", action="append", default=[])
    parser.add_argument("--run-proving-tests", action="store_true")
    args = parser.parse_args(argv)

    explicit_changed = [path for path in args.changed_path if isinstance(path, str) and path]
    changed_paths: list[str] | None = explicit_changed or None
    scope_allowlist = set(_DEFAULT_SCOPE_ALLOWLIST)
    scope_allowlist.update(path for path in args.allow_touch if isinstance(path, str) and path)

    return run(
        root=Path(args.root).resolve(),
        packets_path=Path(args.packets).resolve(),
        baseline_path=Path(args.baseline).resolve(),
        out_path=Path(args.out).resolve(),
        debt_out_path=Path(args.debt_out).resolve(),
        check=args.check,
        write_baseline=args.write_baseline,
        max_age_days=args.max_age_days,
        changed_paths=changed_paths,
        base_sha=args.base_sha,
        head_sha=args.head_sha,
        scope_allowlist=scope_allowlist,
        run_proving_tests=args.run_proving_tests,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Removed direct script entrypoint: scripts/policy/docflow_packet_enforce.py. "
        "Use `gabion policy docflow-packet-enforce`. "
        "See docs/user_workflows.md#user_workflows and "
        "docs/normative_clause_index.md#clause-command-maturity-parity."
    )
