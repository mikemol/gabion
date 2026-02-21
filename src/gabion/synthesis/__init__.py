"""Synthesis subpackage for Gabion."""

from gabion.synthesis.merge import merge_bundles
from gabion.synthesis.emission import render_protocol_stubs
from gabion.synthesis.model import (
    FieldSpec,
    NamingContext,
    ProtocolSpec,
    SynthesisConfig,
    SynthesisPlan,
)
from gabion.synthesis.naming import suggest_name
from gabion.synthesis.protocols import Synthesizer
from gabion.synthesis.schedule import ScheduleResult, topological_schedule

__all__ = [
    "FieldSpec",
    "NamingContext",
    "ProtocolSpec",
    "ScheduleResult",
    "SynthesisConfig",
    "SynthesisPlan",
    "Synthesizer",
    "render_protocol_stubs",
    "merge_bundles",
    "suggest_name",
    "topological_schedule",
]
