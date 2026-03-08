#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportViolation:
    importer: str
    module_path: str
    symbol: str
    lineno: int

    def key(self) -> tuple[str, str, str]:
        return (self.importer, self.module_path, self.symbol)

    def render(self) -> str:
        module = self.module_path or "<root>"
        return (
            f"{self.importer}:{self.lineno} imports private symbol "
            f"{module}.{self.symbol}"
        )


@dataclass(frozen=True)
class AllowlistEntry:
    importer_glob: str
    module_glob: str
    symbol_glob: str

    def matches(self, violation: ImportViolation) -> bool:
        return (
            fnmatch.fnmatchcase(violation.importer, self.importer_glob)
            and fnmatch.fnmatchcase(violation.module_path, self.module_glob)
            and fnmatch.fnmatchcase(violation.symbol, self.symbol_glob)
        )


def _is_private_symbol(name: str) -> bool:
    if not name.startswith("_"):
        return False
    # Ignore dunder names; policy targets private underscore symbols.
    return not (name.startswith("__") and name.endswith("__") and len(name) > 4)


def _iter_python_files(root: Path):
    for base in (root / "src", root / "tests"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            yield path


def _module_owner_relpath(module_path: str) -> str | None:
    if not module_path.startswith("gabion."):
        return None
    relative = module_path.replace(".", "/")
    return f"src/{relative}.py"


def _is_owned_server_core_private_usage(*, importer: str, module_path: str) -> bool:
    owner_relpath = _module_owner_relpath(module_path)
    if owner_relpath is None:
        return False
    return importer == owner_relpath


def _collect_private_import_violations(*, root: Path) -> list[ImportViolation]:
    violations: list[ImportViolation] = []
    for path in _iter_python_files(root):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))

        server_core_import_aliases: dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_path = node.module or ""
                for alias in node.names:
                    symbol = alias.name
                    local_name = alias.asname or symbol
                    if module_path.startswith("gabion.server_core") and symbol != "*":
                        server_core_import_aliases[local_name] = f"{module_path}.{symbol}"
                    if symbol == "*" or not _is_private_symbol(symbol):
                        continue
                    if _is_owned_server_core_private_usage(
                        importer=rel,
                        module_path=module_path,
                    ):
                        continue
                    violations.append(
                        ImportViolation(
                            importer=rel,
                            module_path=module_path,
                            symbol=symbol,
                            lineno=int(node.lineno),
                        )
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    full_name = alias.name
                    module_path, _, symbol = full_name.rpartition(".")
                    if not symbol:
                        symbol = module_path
                        module_path = ""
                    local_name = alias.asname or full_name.split(".")[-1]
                    if full_name.startswith("gabion.server_core"):
                        server_core_import_aliases[local_name] = full_name
                    if not _is_private_symbol(symbol):
                        continue
                    if _is_owned_server_core_private_usage(
                        importer=rel,
                        module_path=module_path,
                    ):
                        continue
                    violations.append(
                        ImportViolation(
                            importer=rel,
                            module_path=module_path,
                            symbol=symbol,
                            lineno=int(node.lineno),
                        )
                    )

        for node in ast.walk(tree):
            if not rel.startswith("src/") or rel.startswith("src/gabion/server_core/"):
                continue
            if not isinstance(node, ast.Attribute):
                continue
            if not _is_private_symbol(node.attr):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            module_path = server_core_import_aliases.get(node.value.id)
            if module_path is None:
                continue
            if _is_owned_server_core_private_usage(
                importer=rel,
                module_path=module_path,
            ):
                continue
            violations.append(
                ImportViolation(
                    importer=rel,
                    module_path=module_path,
                    symbol=node.attr,
                    lineno=int(node.lineno),
                )
            )

    return sorted(
        violations,
        key=lambda item: (item.importer, item.module_path, item.symbol, item.lineno),
    )


def _load_allowlist(path: Path) -> list[AllowlistEntry]:
    entries: list[AllowlistEntry] = []
    if not path.exists():
        return entries
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3:
            raise SystemExit(
                f"invalid allowlist entry at {path}:{lineno}; expected 'importer|module|symbol'"
            )
        entries.append(
            AllowlistEntry(
                importer_glob=parts[0],
                module_glob=parts[1],
                symbol_glob=parts[2],
            )
        )
    return entries


def _load_baseline(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(raw_entries, list):
        raise SystemExit(f"invalid baseline payload at {path}; expected 'entries' list")
    baseline: set[tuple[str, str, str]] = set()
    for item in raw_entries:
        if not isinstance(item, dict):
            raise SystemExit(f"invalid baseline entry in {path}; expected object")
        importer = str(item.get("importer", "")).strip()
        module_path = str(item.get("module_path", "")).strip()
        symbol = str(item.get("symbol", "")).strip()
        if not importer or not symbol:
            raise SystemExit(f"invalid baseline entry in {path}; importer/symbol required")
        baseline.add((importer, module_path, symbol))
    return baseline


def _write_baseline(path: Path, entries: set[tuple[str, str, str]]) -> None:
    payload = {
        "entries": [
            {
                "importer": importer,
                "module_path": module_path,
                "symbol": symbol,
            }
            for importer, module_path, symbol in sorted(entries)
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_allowlisted(violation: ImportViolation, allowlist: list[AllowlistEntry]) -> bool:
    return any(entry.matches(violation) for entry in allowlist)


def run(
    *,
    root: Path,
    allowlist_path: Path,
    baseline_path: Path,
    out_path: Path,
    check: bool,
    write_baseline: bool,
) -> int:
    all_violations = _collect_private_import_violations(root=root)
    allowlist = _load_allowlist(allowlist_path)

    non_exempt = [
        violation for violation in all_violations if not _is_allowlisted(violation, allowlist)
    ]
    non_exempt_keys = {violation.key() for violation in non_exempt}

    baseline = _load_baseline(baseline_path)
    new_violations = sorted(non_exempt_keys - baseline)

    report = {
        "root": str(root),
        "allowlist": str(allowlist_path),
        "baseline": str(baseline_path),
        "totals": {
            "all_violations": len(all_violations),
            "non_exempt": len(non_exempt_keys),
            "new_violations": len(new_violations),
        },
        "non_exempt": [
            {
                "importer": importer,
                "module_path": module_path,
                "symbol": symbol,
            }
            for importer, module_path, symbol in sorted(non_exempt_keys)
        ],
        "new_violations": [
            {
                "importer": importer,
                "module_path": module_path,
                "symbol": symbol,
            }
            for importer, module_path, symbol in new_violations
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if write_baseline:
        _write_baseline(baseline_path, non_exempt_keys)

    print(
        "private-symbol-import-guard: "
        f"all={len(all_violations)} non_exempt={len(non_exempt_keys)} "
        f"new={len(new_violations)} out={out_path}"
    )

    if check and new_violations:
        print("new private-symbol imports detected (non-exempt):")
        for importer, module_path, symbol in new_violations:
            module = module_path or "<root>"
            print(f"  - {importer}: {module}.{symbol}")
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--allowlist",
        default="docs/policy/private_symbol_import_allowlist.txt",
    )
    parser.add_argument(
        "--baseline",
        default="docs/baselines/private_symbol_import_baseline.json",
    )
    parser.add_argument("--out", default="out/private_symbol_import_report.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)

    return run(
        root=Path(args.root).resolve(),
        allowlist_path=Path(args.allowlist).resolve(),
        baseline_path=Path(args.baseline).resolve(),
        out_path=Path(args.out).resolve(),
        check=args.check,
        write_baseline=args.write_baseline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
