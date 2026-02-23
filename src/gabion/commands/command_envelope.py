from __future__ import annotations

from dataclasses import dataclass

from gabion.commands import payload_codec


@dataclass(frozen=True)
class CommandEnvelope:
    command: str
    command_args: list[object]
    payload: dict[str, object]


def command_payload_envelope(
    *,
    command: str,
    arguments: list[object],
) -> CommandEnvelope:
    command_args, payload = payload_codec.normalized_command_payload(
        command=command,
        arguments=arguments,
    )
    return CommandEnvelope(
        command=command,
        command_args=command_args,
        payload=payload,
    )
