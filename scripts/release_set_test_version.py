from __future__ import annotations

import argparse
from pathlib import Path
import re


def _normalize_test_version(tag: str) -> str:
    if not tag.startswith("test-v"):
        raise SystemExit(f"Unexpected tag: {tag}")
    version = tag[len("test-v") :]
    if not version:
        raise SystemExit("Missing version in tag")
    if "+" in version:
        base, local = version.split("+", 1)
        digits = "".join(ch for ch in local if ch.isdigit())
        if not digits:
            raise SystemExit("Missing timestamp digits in test tag")
        version = f"{base}.dev{digits}"
    return version


def _update_pyproject(version: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^version = ".*"$', f'version = "{version}"', text, count=1
    )
    if count != 1:
        raise SystemExit("Failed to update project.version in pyproject.toml")
    path.write_text(updated, encoding="utf-8")


def _update_init(version: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    pattern = r'(?m)^__version__\s*=\s*([\'\"]).*?\1\s*$'
    updated, count = re.subn(pattern, f'__version__ = "{version}"', text, count=1)
    if count != 1:
        print(
            "Warning: __version__ update skipped "
            f"(matches={count})"
        )
        return
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    version = _normalize_test_version(args.tag)
    _update_pyproject(version, Path("pyproject.toml"))
    _update_init(version, Path("src/gabion/__init__.py"))
    print(f"Set project.version to {version}")


if __name__ == "__main__":
    main()
