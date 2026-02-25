#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from gabion.analysis.deprecated_substrate import (
    DeprecatedFiber,
    build_deprecated_extraction_artifacts,
    ingest_perf_samples,
)


def _load_json(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected object payload: {path}")
    return raw


def _write_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract deterministic deprecated substrate artifacts")
    parser.add_argument("--input", type=Path, required=True, help="input JSON with perf_samples and deprecated_fibers")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = _load_json(args.input)
    perf_samples_raw = payload.get("perf_samples", [])
    fibers_raw = payload.get("deprecated_fibers", [])
    prev_cov = payload.get("branch_coverage_previous", {})
    cur_cov = payload.get("branch_coverage_current", {})
    perf_rows = perf_samples_raw if isinstance(perf_samples_raw, list) else []
    fiber_rows = fibers_raw if isinstance(fibers_raw, list) else []
    prev_cov_map = prev_cov if isinstance(prev_cov, Mapping) else {}
    cur_cov_map = cur_cov if isinstance(cur_cov, Mapping) else {}

    perf_samples = ingest_perf_samples([row for row in perf_rows if isinstance(row, Mapping)])
    fibers = tuple(DeprecatedFiber.from_payload(row) for row in fiber_rows if isinstance(row, Mapping))
    artifacts = build_deprecated_extraction_artifacts(
        perf_samples=perf_samples,
        deprecated_fibers=fibers,
        branch_coverage_previous={str(k): float(v) for k, v in prev_cov_map.items()},
        branch_coverage_current={str(k): float(v) for k, v in cur_cov_map.items()},
    )

    _write_payload(args.output_dir / "perf_fiber_groups.json", list(artifacts.perf_fiber_groups))
    _write_payload(args.output_dir / "fiber_group_rankings.json", list(artifacts.fiber_group_rankings))
    _write_payload(args.output_dir / "blocker_dag.json", artifacts.blocker_dag)
    _write_payload(args.output_dir / "informational_signals.json", list(artifacts.informational_signals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
