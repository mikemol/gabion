from __future__ import annotations

import pytest

from gabion.commands import command_envelope, command_ids, payload_codec
from gabion.exceptions import NeverThrown


# gabion:evidence E:call_footprint::tests/test_command_contract_matrix.py::test_command_envelope_normalization_matrix_includes_check::command_envelope.py::gabion.commands.command_envelope.command_payload_envelope
def test_command_envelope_normalization_matrix_includes_check() -> None:
    for command in command_ids.SEMANTIC_COMMAND_IDS:
        envelope = command_envelope.command_payload_envelope(
            command=command,
            arguments=[
                {
                    "analysis_timeout_ticks": 10,
                    "analysis_timeout_tick_ns": 1_000_000,
                    "paths": ["b.py", "a.py"],
                    "config": {"z": 2, "a": 1},
                }
            ],
        )
        assert envelope.command == command
        assert envelope.command_args[0] == envelope.payload
        assert envelope.payload["paths"] == ["a.py", "b.py"]
        assert envelope.payload["config"] == {"a": 1, "z": 2}


# gabion:evidence E:call_footprint::tests/test_command_contract_matrix.py::test_command_envelope_rejects_missing_or_non_mapping_payload::command_envelope.py::gabion.commands.command_envelope.command_payload_envelope
def test_command_envelope_rejects_missing_or_non_mapping_payload() -> None:
    with pytest.raises(NeverThrown):
        command_envelope.command_payload_envelope(
            command=command_ids.DATAFLOW_COMMAND,
            arguments=[],
        )
    with pytest.raises(NeverThrown):
        command_envelope.command_payload_envelope(
            command=command_ids.DATAFLOW_COMMAND,
            arguments=[123],
        )


@pytest.mark.parametrize(
    ("payload", "expected_ns"),
    [
        (
            {"analysis_timeout_ticks": 3, "analysis_timeout_tick_ns": 2_000_000},
            6_000_000,
        ),
        ({"analysis_timeout_ms": 7}, 7_000_000),
        ({"analysis_timeout_seconds": "1.25"}, 1_250_000_000),
    ],
)
# gabion:evidence E:call_footprint::tests/test_command_contract_matrix.py::test_timeout_codec_matrix::payload_codec.py::gabion.commands.payload_codec.analysis_timeout_total_ns::payload_codec.py::gabion.commands.payload_codec.has_analysis_timeout
def test_timeout_codec_matrix(payload: dict[str, object], expected_ns: int) -> None:
    assert payload_codec.has_analysis_timeout(payload) is True
    assert (
        payload_codec.analysis_timeout_total_ns(
            payload,
            source="tests.test_command_contract_matrix.timeout_codec_matrix",
            reject_sub_millisecond_seconds=False,
        )
        == expected_ns
    )


# gabion:evidence E:call_footprint::tests/test_command_contract_matrix.py::test_timeout_codec_preserves_explicit_tick_pair_requirement::payload_codec.py::gabion.commands.payload_codec.analysis_timeout_total_ns
def test_timeout_codec_preserves_explicit_tick_pair_requirement() -> None:
    with pytest.raises(NeverThrown):
        payload_codec.analysis_timeout_total_ns(
            {"analysis_timeout_ticks": 1},
            source="tests.test_command_contract_matrix.timeout_pair_missing_tick_ns",
            reject_sub_millisecond_seconds=False,
        )
