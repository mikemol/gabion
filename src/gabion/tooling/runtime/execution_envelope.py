# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gabion.invariants import never

EnvelopeOperation = Literal["delta_bundle", "raw"]


@dataclass(frozen=True)
class ExecutionEnvelope:
    operation: EnvelopeOperation
    root: Path
    report_path: Path | None
    strictness: str | None
    allow_external: bool | None
    aspf_state_json: Path | None
    aspf_delta_jsonl: Path | None
    aspf_import_state: tuple[Path, ...]

    @classmethod
    def for_delta_bundle(
        cls,
        *,
        root: Path,
        report_path: Path,
        strictness: str | None,
        allow_external: bool | None,
        aspf_state_json: Path | None,
        aspf_delta_jsonl: Path | None,
        aspf_import_state: tuple[Path, ...] = (),
    ) -> "ExecutionEnvelope":
        return cls(
            operation="delta_bundle",
            root=root.resolve(),
            report_path=report_path,
            strictness=strictness,
            allow_external=allow_external,
            aspf_state_json=aspf_state_json,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_import_state=tuple(path.resolve() for path in aspf_import_state),
        ).validate()

    @classmethod
    def for_raw(
        cls,
        *,
        root: Path,
        aspf_state_json: Path | None,
        aspf_delta_jsonl: Path | None,
        aspf_import_state: tuple[Path, ...] = (),
    ) -> "ExecutionEnvelope":
        return cls(
            operation="raw",
            root=root.resolve(),
            report_path=None,
            strictness=None,
            allow_external=None,
            aspf_state_json=aspf_state_json,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_import_state=tuple(path.resolve() for path in aspf_import_state),
        ).validate()

    # gabion:decision_protocol
    def validate(self) -> "ExecutionEnvelope":
        if self.operation not in {"delta_bundle", "raw"}:
            never("invalid execution envelope operation", operation=str(self.operation))
        if self.operation == "delta_bundle" and self.report_path is None:
            never("delta_bundle requires report_path")
        if self.strictness is not None and self.strictness not in {"high", "low"}:
            never("invalid strictness in execution envelope", strictness=self.strictness)
        if (self.aspf_state_json is None) != (self.aspf_delta_jsonl is None):
            never(
                "aspf state/delta paths must be provided together",
                aspf_state_json=str(self.aspf_state_json),
                aspf_delta_jsonl=str(self.aspf_delta_jsonl),
            )
        return self


__all__ = ["EnvelopeOperation", "ExecutionEnvelope"]
