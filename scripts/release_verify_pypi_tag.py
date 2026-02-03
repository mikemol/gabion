from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tomllib


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()


def _read_project_version(pyproject_path: Path) -> str:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not version:
        raise SystemExit("project.version not found in pyproject.toml")
    return version


def _read_package_version(init_path: Path) -> str:
    text = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', text, re.M)
    if not match:
        raise SystemExit("__version__ not found in src/gabion/__init__.py")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    tag = args.tag

    if not tag.startswith("v"):
        raise SystemExit(f"Unexpected tag format: {tag}")

    _run(["git", "fetch", "origin", "main", "next", "release", "--tags"])

    tag_sha = _git_output("rev-parse", f"refs/tags/{tag}^{{commit}}")
    main_sha = _git_output("rev-parse", "origin/main")
    next_sha = _git_output("rev-parse", "origin/next")
    release_sha = _git_output("rev-parse", "origin/release")

    if tag_sha != main_sha:
        raise SystemExit(f"Release tag must point at main.\ntag={tag_sha}\nmain={main_sha}")
    if tag_sha != next_sha:
        raise SystemExit(f"Release tag must point at next.\ntag={tag_sha}\nnext={next_sha}")
    if tag_sha != release_sha:
        raise SystemExit(f"Release tag must point at release.\ntag={tag_sha}\nrelease={release_sha}")

    project_version = _read_project_version(Path("pyproject.toml"))
    package_version = _read_package_version(Path("src/gabion/__init__.py"))
    tag_version = tag[1:]

    if tag_version != project_version:
        raise SystemExit(
            "Tag version must match project.version in pyproject.toml.\n"
            f"tag={tag_version}\nproject.version={project_version}"
        )
    if project_version != package_version:
        raise SystemExit(
            "project.version must match src/gabion/__init__.__version__.\n"
            f"project.version={project_version}\n__version__={package_version}"
        )


if __name__ == "__main__":
    main()
