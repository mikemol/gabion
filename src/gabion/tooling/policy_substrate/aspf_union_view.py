# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gabion.tooling.runtime.policy_scan_batch import PolicyScanBatch, ScannedModule

if TYPE_CHECKING:
    import libcst as cst


@dataclass(frozen=True)
class CSTParseFailureEvent:
    path: Path
    rel_path: str
    line: int
    column: int
    message: str
    kind: str = "cst_parse_failure"
    surface: str = "libcst"


@dataclass(frozen=True)
class LibCSTTreeUnavailable:
    reason: str


@dataclass(frozen=True)
class UnionModuleView:
    path: Path
    rel_path: str
    source: str
    tree: object
    pyast_tree: object
    libcst_tree: object


@dataclass(frozen=True)
class ASPFUnionView:
    root: Path
    modules: tuple[UnionModuleView, ...]
    cst_failures: tuple[CSTParseFailureEvent, ...]


@dataclass(frozen=True)
class _LibCSTParsed:
    tree: object


@dataclass(frozen=True)
class _LibCSTUnavailable:
    reason: str


@dataclass(frozen=True)
class _LibCSTFailure:
    line: int
    column: int
    message: str


_LibCSTParseResult = _LibCSTParsed | _LibCSTUnavailable | _LibCSTFailure


@dataclass(frozen=True)
class _ParsedModuleCST:
    module_view: UnionModuleView
    parse_result: _LibCSTParseResult


@dataclass(frozen=True)
class _CSTFailurePresent:
    event: CSTParseFailureEvent


@dataclass(frozen=True)
class _CSTFailureAbsent:
    reason: str


_CSTFailureOutcome = _CSTFailurePresent | _CSTFailureAbsent


def build_aspf_union_view(*, batch: PolicyScanBatch) -> ASPFUnionView:
    cst = _libcst_optional()
    parser_error_type = _libcst_parser_error_type(cst)
    parsed_rows = list(
        map(
            lambda module: _parse_module_cst(
                module=module,
                cst=cst,
                parser_error_type=parser_error_type,
            ),
            batch.modules,
        )
    )
    return ASPFUnionView(
        root=batch.root,
        modules=list(map(lambda row: row.module_view, parsed_rows)),
        cst_failures=list(
            map(
                lambda outcome: outcome.event,
                filter(
                    _is_cst_failure_present,
                    map(
                        lambda row: _failure_event_from_parse_result(
                            module_path=row.module_view.path,
                            module_rel_path=row.module_view.rel_path,
                            result=row.parse_result,
                        ),
                        parsed_rows,
                    ),
                ),
            )
        ),
    )


def _libcst_optional() -> object | None:
    try:
        import libcst as cst

        return cst
    except ImportError:
        return None


def _libcst_parser_error_type(cst: object | None) -> type[Exception]:
    if cst is None:
        return Exception
    parser_error = getattr(cst, "ParserSyntaxError", None)
    match parser_error:
        case type() as parser_error_type:
            return parser_error_type
        case _:
            return Exception


def _parse_module_cst(
    *,
    module: ScannedModule,
    cst: object | None,
    parser_error_type: type[Exception],
) -> _ParsedModuleCST:
    parse_result = _parse_libcst_result(
        module=module,
        cst=cst,
        parser_error_type=parser_error_type,
    )
    module_view = UnionModuleView(
        path=module.path,
        rel_path=module.rel_path,
        source=module.source,
        tree=module.tree,
        pyast_tree=module.tree,
        libcst_tree=_libcst_tree_from_parse_result(parse_result),
    )
    return _ParsedModuleCST(
        module_view=module_view,
        parse_result=parse_result,
    )


def _parse_libcst_result(
    *,
    module: ScannedModule,
    cst: object | None,
    parser_error_type: type[Exception],
) -> _LibCSTParseResult:
    if cst is None:
        return _LibCSTUnavailable(reason="libcst_unavailable")
    parse_module = getattr(cst, "parse_module", None)
    if not callable(parse_module):
        return _LibCSTUnavailable(reason="parse_module_missing")
    try:
        return _LibCSTParsed(tree=parse_module(module.source))
    except parser_error_type as exc:  # pragma: no cover - parser specifics vary
        return _LibCSTFailure(
            line=_line_value(getattr(exc, "raw_line", 1)),
            column=_line_value(getattr(exc, "raw_column", 1)),
            message=_exception_message(exc),
        )


def _libcst_tree_from_parse_result(result: _LibCSTParseResult) -> object:
    match result:
        case _LibCSTParsed(tree=tree):
            return tree
        case _LibCSTUnavailable(reason=reason):
            return LibCSTTreeUnavailable(reason=reason)
        case _LibCSTFailure():
            return LibCSTTreeUnavailable(reason="parse_failure")


def _failure_event_from_parse_result(
    *,
    module_path: Path,
    module_rel_path: str,
    result: _LibCSTParseResult,
) -> _CSTFailureOutcome:
    match result:
        case _LibCSTFailure(line=line, column=column, message=message):
            return _CSTFailurePresent(
                event=CSTParseFailureEvent(
                    path=module_path,
                    rel_path=module_rel_path,
                    line=line,
                    column=column,
                    message=message,
                )
            )
        case _:
            return _CSTFailureAbsent(reason="no_parse_failure")


def _is_cst_failure_present(outcome: _CSTFailureOutcome) -> bool:
    match outcome:
        case _CSTFailurePresent():
            return True
        case _CSTFailureAbsent():
            return False


def _line_value(value: object) -> int:
    match value:
        case int() as line if line > 0:
            return line
        case _:
            return 1


def _exception_message(value: object) -> str:
    message = getattr(value, "msg", "")
    match message:
        case str() as text if text:
            return text
        case _:
            return "cst_parse_failure"


__all__ = [
    "ASPFUnionView",
    "CSTParseFailureEvent",
    "LibCSTTreeUnavailable",
    "UnionModuleView",
    "build_aspf_union_view",
]
