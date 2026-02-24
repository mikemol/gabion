#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=Path("artifacts/audit_reports"))
    parser.add_argument("--seed-checkpoint", type=Path, default=Path("baselines/dataflow_resume_checkpoint_ci.json"))
    parser.add_argument("--seed-chunks", type=Path, default=Path("baselines/dataflow_resume_checkpoint_ci.json.chunks"))
    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)
    restored = False
    target_checkpoint = args.target_dir / "dataflow_resume_checkpoint_ci.json"
    target_chunks = args.target_dir / "dataflow_resume_checkpoint_ci.json.chunks"
    if args.seed_checkpoint.is_file():
        shutil.copy2(args.seed_checkpoint, target_checkpoint)
        restored = True
    if args.seed_chunks.is_dir():
        if target_chunks.exists():
            shutil.rmtree(target_chunks)
        shutil.copytree(args.seed_chunks, target_chunks)
        restored = True
    print(
        "Seeded checkpoint from version-controlled baseline."
        if restored
        else "No version-controlled checkpoint seed found; continuing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
