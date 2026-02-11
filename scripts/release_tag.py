from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tomllib


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()


def _tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0


def _read_project_version(pyproject_path: Path) -> str:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not version:
        raise SystemExit("project.version not found in pyproject.toml")
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    tag = args.tag

    if not (tag.startswith("v") or tag.startswith("test-v")):
        raise SystemExit(f"Invalid tag: {tag} (expected v* or test-v*)")

    _run(["git", "fetch", "origin", "main", "next", "release", "--tags"])
    main_sha = _git_output("rev-parse", "origin/main")
    next_sha = _git_output("rev-parse", "origin/next")
    release_sha = _git_output("rev-parse", "origin/release")
    head_sha = _git_output("rev-parse", "HEAD")

    if next_sha != main_sha:
        raise SystemExit(f"Next branch must mirror main before tagging.\nnext={next_sha}\nmain={main_sha}")
    if release_sha != next_sha:
        raise SystemExit(
            f"Release branch must mirror next before tagging.\nrelease={release_sha}\nnext={next_sha}"
        )

    if tag.startswith("test-v"):
        if head_sha != next_sha:
            raise SystemExit(f"Test tags must be created from next.\nhead={head_sha}\nnext={next_sha}")
        tag_target = next_sha
    elif tag.startswith("v"):
        if head_sha != release_sha:
            raise SystemExit(
                f"Release tags must be created from release.\nhead={head_sha}\nrelease={release_sha}"
            )
        tag_target = release_sha

    if _tag_exists(tag):
        raise SystemExit(f"Tag already exists: {tag}")

    project_version = _read_project_version(Path("pyproject.toml"))
    tag_version = tag[1:] if tag.startswith("v") else tag[len("test-v") :]
    base_version = tag_version.split("+", 1)[0]
    if base_version != project_version:
        raise SystemExit(
            "Tag version must match project.version in pyproject.toml.\n"
            f"tag={tag}\nproject.version={project_version}"
        )

    _run(["git", "fetch", "origin", "main", "next", "release", "--tags"])
    if _git_output("rev-parse", "origin/main") != main_sha:
        raise SystemExit("Main moved after verification; re-run tagging workflow.")
    if _git_output("rev-parse", "origin/next") != next_sha:
        raise SystemExit("Next moved after verification; re-run tagging workflow.")
    if _git_output("rev-parse", "origin/release") != release_sha:
        raise SystemExit("Release moved after verification; re-run tagging workflow.")

    _run(["git", "config", "user.name", "github-actions[bot]"])
    _run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])
    _run(["git", "tag", "-a", tag, "-m", f"Release {tag}", tag_target])
    _run(["git", "push", "origin", f"refs/tags/{tag}"])


if __name__ == "__main__":
    main()
