from __future__ import annotations

import argparse
import subprocess


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    tag = args.tag

    if not tag.startswith("test-v"):
        raise SystemExit(f"Unexpected tag format: {tag}")

    _run(["git", "fetch", "origin", "main", "next", "--tags"])

    tag_sha = _git_output("rev-parse", f"refs/tags/{tag}^{{commit}}")
    main_sha = _git_output("rev-parse", "origin/main")
    next_sha = _git_output("rev-parse", "origin/next")

    if tag_sha != main_sha:
        raise SystemExit(f"Test tag must point at main.\ntag={tag_sha}\nmain={main_sha}")
    if tag_sha != next_sha:
        raise SystemExit(f"Test tag must point at next.\ntag={tag_sha}\nnext={next_sha}")


if __name__ == "__main__":
    main()
