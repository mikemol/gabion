from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


def _read_project_version(pyproject_path: Path) -> str:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not version:
        raise SystemExit("project.version not found in pyproject.toml")
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Write key=value output to this path (GitHub Actions output file).",
    )
    args = parser.parse_args()

    version = _read_project_version(Path("pyproject.toml"))
    line = f"version={version}\n"
    if args.output:
        Path(args.output).write_text(line, encoding="utf-8")
    else:
        print(line, end="")


if __name__ == "__main__":
    main()
