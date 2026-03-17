from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
import json
from typing import Callable, Iterable, Mapping, Sequence

from gabion.invariants import never
from gabion.tooling.runtime.policy_scanner_identity import (
    PolicyScannerIdentity,
    PolicyScannerIdentitySpace,
    canonical_policy_scanner_site_identity,
    canonical_policy_scanner_structural_identity,
)


@dataclass(frozen=True)
class ScannedModule:
    path: Path
    rel_path: str
    source: str
    tree: ast.AST
    identity: PolicyScannerIdentity | None = None


@dataclass(frozen=True)
class ReadFailure:
    path: Path
    rel_path: str
    error_type: str
    error_message: str
    identity: PolicyScannerIdentity | None = None


@dataclass(frozen=True)
class ParseFailure:
    path: Path
    rel_path: str
    line: int
    column: int
    error_message: str
    identity: PolicyScannerIdentity | None = None


@dataclass(frozen=True)
class PolicyScanBatch:
    root: Path
    modules: tuple[ScannedModule, ...]
    read_failures: tuple[ReadFailure, ...]
    parse_failures: tuple[ParseFailure, ...]


@dataclass(frozen=True)
class ScanFailureSeed:
    path: str
    line: int
    column: int
    kind: str
    detail: str
    identity: PolicyScannerIdentity | None = None


def build_policy_scan_batch(
    *,
    root: Path,
    target_globs: Sequence[str],
    files: Sequence[Path] | None = None,
    include_path: Callable[[Path], bool] | None = None,
    identities: PolicyScannerIdentitySpace | None = None,
) -> PolicyScanBatch:
    resolved_root = root.resolve()
    candidates = (
        tuple(_iter_target_candidates(root=resolved_root, target_globs=target_globs))
        if files is None
        else tuple(files)
    )
    normalized_paths = tuple(
        _iter_normalized_paths(
            root=resolved_root,
            candidates=candidates,
            include_path=include_path,
        )
    )
    modules: list[ScannedModule] = []
    read_failures: list[ReadFailure] = []
    parse_failures: list[ParseFailure] = []
    for path in normalized_paths:
        rel_path = path.relative_to(resolved_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            detail = str(exc)
            failure_identity = (
                _scanner_item_identity(
                    identities=identities,
                    scanner_kind="read_failure",
                    rule_id="",
                    rel_path=rel_path,
                    qualname="<module>",
                    line=1,
                    column=1,
                    kind=type(exc).__name__,
                    structural_basis=detail or type(exc).__name__,
                    label=f"{rel_path}:read_failure",
                    surface="policy_scanner_input",
                )
                if identities is not None
                else None
            )
            read_failures.append(
                ReadFailure(
                    path=path,
                    rel_path=rel_path,
                    error_type=type(exc).__name__,
                    error_message=detail,
                    identity=failure_identity,
                )
            )
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            parse_message = str(getattr(exc, "msg", "") or "")
            failure_identity = (
                _scanner_item_identity(
                    identities=identities,
                    scanner_kind="parse_failure",
                    rule_id="",
                    rel_path=rel_path,
                    qualname="<module>",
                    line=int(exc.lineno or 1),
                    column=int(exc.offset or 1),
                    kind="SyntaxError",
                    structural_basis=f"{int(exc.lineno or 1)}:{int(exc.offset or 1)}:{parse_message}",
                    label=f"{rel_path}:parse_failure",
                    surface="policy_scanner_input",
                )
                if identities is not None
                else None
            )
            parse_failures.append(
                ParseFailure(
                    path=path,
                    rel_path=rel_path,
                    line=int(exc.lineno or 1),
                    column=int(exc.offset or 1),
                    error_message=parse_message,
                    identity=failure_identity,
                )
            )
            continue
        module_identity = (
            _scanner_item_identity(
                identities=identities,
                scanner_kind="module",
                rule_id="",
                rel_path=rel_path,
                qualname="<module>",
                line=1,
                column=1,
                kind="module",
                structural_basis=rel_path,
                label=rel_path,
                surface="policy_scanner_input",
            )
            if identities is not None
            else None
        )
        modules.append(
            ScannedModule(
                path=path,
                rel_path=rel_path,
                source=source,
                tree=tree,
                identity=module_identity,
            )
        )
    return PolicyScanBatch(
        root=resolved_root,
        modules=tuple(modules),
        read_failures=tuple(read_failures),
        parse_failures=tuple(parse_failures),
    )


def build_policy_scan_batch_from_sources(
    *,
    root: Path,
    source_by_rel_path: Mapping[str, str],
    identities: PolicyScannerIdentitySpace | None = None,
) -> PolicyScanBatch:
    resolved_root = root.resolve()
    modules: list[ScannedModule] = []
    parse_failures: list[ParseFailure] = []
    for rel_path in sorted(source_by_rel_path):
        source = str(source_by_rel_path[rel_path])
        path = (resolved_root / rel_path).resolve()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            parse_message = str(getattr(exc, "msg", "") or "")
            failure_identity = (
                _scanner_item_identity(
                    identities=identities,
                    scanner_kind="parse_failure",
                    rule_id="",
                    rel_path=rel_path,
                    qualname="<module>",
                    line=int(exc.lineno or 1),
                    column=int(exc.offset or 1),
                    kind="SyntaxError",
                    structural_basis=f"{int(exc.lineno or 1)}:{int(exc.offset or 1)}:{parse_message}",
                    label=f"{rel_path}:parse_failure",
                    surface="policy_scanner_input",
                )
                if identities is not None
                else None
            )
            parse_failures.append(
                ParseFailure(
                    path=path,
                    rel_path=rel_path,
                    line=int(exc.lineno or 1),
                    column=int(exc.offset or 1),
                    error_message=parse_message,
                    identity=failure_identity,
                )
            )
            continue
        module_identity = (
            _scanner_item_identity(
                identities=identities,
                scanner_kind="module",
                rule_id="",
                rel_path=rel_path,
                qualname="<module>",
                line=1,
                column=1,
                kind="module",
                structural_basis=rel_path,
                label=rel_path,
                surface="policy_scanner_input",
            )
            if identities is not None
            else None
        )
        modules.append(
            ScannedModule(
                path=path,
                rel_path=rel_path,
                source=source,
                tree=tree,
                identity=module_identity,
            )
        )
    return PolicyScanBatch(
        root=resolved_root,
        modules=tuple(modules),
        read_failures=(),
        parse_failures=tuple(parse_failures),
    )


def iter_failure_seeds(
    *,
    batch: PolicyScanBatch,
    read_kind: str = "read_error",
    syntax_kind: str = "syntax_error",
) -> Iterable[ScanFailureSeed]:
    for failure in batch.read_failures:
        detail = (
            f"{failure.error_type}: {failure.error_message}"
            if failure.error_message
            else failure.error_type
        )
        yield ScanFailureSeed(
            path=failure.rel_path,
            line=1,
            column=1,
            kind=read_kind,
            detail=detail or "read failure",
            identity=failure.identity,
        )
    for failure in batch.parse_failures:
        yield ScanFailureSeed(
            path=failure.rel_path,
            line=failure.line,
            column=failure.column,
            kind=syntax_kind,
            detail=failure.error_message or "syntax error",
            identity=failure.identity,
        )


def _scanner_item_identity(
    *,
    identities: PolicyScannerIdentitySpace,
    scanner_kind: str,
    rule_id: str,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    kind: str,
    structural_basis: str,
    label: str,
    surface: str,
) -> PolicyScannerIdentity:
    site_identity = canonical_policy_scanner_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        scanner_kind=scanner_kind,
        surface=surface,
    )
    structural_identity = canonical_policy_scanner_structural_identity(
        rel_path=rel_path,
        qualname=qualname,
        structural_path=structural_basis,
        scanner_kind=scanner_kind,
        surface=surface,
    )
    return identities.item_id(
        scanner_kind=scanner_kind,
        rule_id=rule_id,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        kind=kind,
        site_identity=site_identity,
        structural_identity=structural_identity,
        label=label,
    )


def load_path_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    allowed: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            allowed.add(line.replace("\\", "/"))
    return allowed


def load_structured_violation_baseline_keys(
    *,
    path: Path,
    migrate_hash: Callable[[str, str, str, int, str], str],
) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    keys: set[str] = set()
    for raw_item in _iter_structured_violation_items(payload):
        path_value = str(raw_item.get("path", "") or "")
        qualname = str(raw_item.get("qualname", "") or "")
        kind = str(raw_item.get("kind", "") or "")
        if not path_value or not qualname or not kind:
            continue
        structured_hash = _coerce_nonempty_text(raw_item.get("structured_hash"))
        if structured_hash:
            keys.add(f"{path_value}:{qualname}:{kind}:{structured_hash}")
            continue
        message = str(raw_item.get("message", "") or "")
        column_values = _coerce_non_bool_int(raw_item.get("column"))
        if message and column_values:
            column = column_values[0]
            migrated_hash = migrate_hash(path_value, qualname, kind, column, message)
            keys.add(f"{path_value}:{qualname}:{kind}:{migrated_hash}")
            continue
        line_values = _coerce_non_bool_int(raw_item.get("line"))
        if line_values:
            line = line_values[0]
            keys.add(f"{path_value}:{qualname}:{line}:{kind}")
    return keys


@singledispatch
def _iter_structured_violation_items(payload: object) -> tuple[Mapping[str, object], ...]:
    never("unregistered runtime type", value_type=type(payload).__name__)


@_iter_structured_violation_items.register(dict)
def _sd_reg_1(payload: dict[object, object]) -> tuple[Mapping[str, object], ...]:
    raw_items = payload.get("violations")
    return _coerce_mapping_list(raw_items)


@_iter_structured_violation_items.register(list)
@_iter_structured_violation_items.register(str)
@_iter_structured_violation_items.register(int)
@_iter_structured_violation_items.register(float)
@_iter_structured_violation_items.register(bool)
@_iter_structured_violation_items.register(type(None))
def _sd_reg_1b(payload: object) -> tuple[Mapping[str, object], ...]:
    _ = payload
    return ()


@singledispatch
def _coerce_mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_coerce_mapping_list.register(list)
def _sd_reg_2(value: list[object]) -> tuple[Mapping[str, object], ...]:
    out: list[Mapping[str, object]] = []
    for item in value:
        out.extend(_coerce_single_mapping(item))
    return tuple(out)


@_coerce_mapping_list.register(dict)
@_coerce_mapping_list.register(str)
@_coerce_mapping_list.register(int)
@_coerce_mapping_list.register(float)
@_coerce_mapping_list.register(bool)
@_coerce_mapping_list.register(type(None))
def _sd_reg_2b(value: object) -> tuple[Mapping[str, object], ...]:
    _ = value
    return ()


@singledispatch
def _coerce_single_mapping(value: object) -> tuple[Mapping[str, object], ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_coerce_single_mapping.register(dict)
def _sd_reg_3(value: dict[object, object]) -> tuple[Mapping[str, object], ...]:
    return ({str(key): item for key, item in value.items()},)


@_coerce_single_mapping.register(list)
@_coerce_single_mapping.register(str)
@_coerce_single_mapping.register(int)
@_coerce_single_mapping.register(float)
@_coerce_single_mapping.register(bool)
@_coerce_single_mapping.register(type(None))
def _sd_reg_3b(value: object) -> tuple[Mapping[str, object], ...]:
    _ = value
    return ()


@singledispatch
def _coerce_nonempty_text(value: object) -> str:
    never("unregistered runtime type", value_type=type(value).__name__)


@_coerce_nonempty_text.register(str)
def _sd_reg_4(value: str) -> str:
    return value


@_coerce_nonempty_text.register(dict)
@_coerce_nonempty_text.register(list)
@_coerce_nonempty_text.register(int)
@_coerce_nonempty_text.register(float)
@_coerce_nonempty_text.register(bool)
@_coerce_nonempty_text.register(type(None))
def _sd_reg_4b(value: object) -> str:
    _ = value
    return ""


@singledispatch
def _coerce_non_bool_int(value: object) -> tuple[int, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_coerce_non_bool_int.register(bool)
def _sd_reg_5(value: bool) -> tuple[int, ...]:
    _ = value
    return ()


@_coerce_non_bool_int.register(int)
def _sd_reg_6(value: int) -> tuple[int, ...]:
    return (value,)


@_coerce_non_bool_int.register(dict)
@_coerce_non_bool_int.register(list)
@_coerce_non_bool_int.register(str)
@_coerce_non_bool_int.register(float)
@_coerce_non_bool_int.register(type(None))
def _sd_reg_7(value: object) -> tuple[int, ...]:
    _ = value
    return ()


def _iter_target_candidates(*, root: Path, target_globs: Sequence[str]) -> Iterable[Path]:
    for pattern in target_globs:
        for path in sorted(root.glob(pattern)):
            yield path


def _iter_normalized_paths(
    *,
    root: Path,
    candidates: Sequence[Path],
    include_path: Callable[[Path], bool] | None,
) -> Iterable[Path]:
    seen: set[Path] = set()
    resolved_candidates = sorted((candidate.resolve() for candidate in candidates), key=str)
    for path in resolved_candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if not _is_relative_to(path, root):
            continue
        if include_path is not None and not include_path(path):
            continue
        yield path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
