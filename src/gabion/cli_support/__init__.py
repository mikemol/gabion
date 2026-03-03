"""CLI support extraction surfaces."""

from .check_commands import (
    register_check_delta_bundle_command,
    register_check_group_callback,
    register_check_run_command,
)
from .check_command_runtime import run_check_command
from .check_execution_plan import build_check_execution_plan_request
from .check_runtime import run_check
from .dispatch_runtime import dispatch_command
from .synth_commands import register_synth_command
from .runtime_flags import register_runtime_flags_callback
from .timeout_progress import render_timeout_progress_markdown
from .tooling_commands import register_ci_watch_command
from .output_emitters import emit_dataflow_result_outputs
from .payload_builder import build_dataflow_payload
from .parser_builder import dataflow_cli_parser
from .synth_runtime import run_synth

__all__ = [
    "dispatch_command",
    "register_check_run_command",
    "register_check_group_callback",
    "register_check_delta_bundle_command",
    "register_synth_command",
    "register_runtime_flags_callback",
    "render_timeout_progress_markdown",
    "register_ci_watch_command",
    "build_check_execution_plan_request",
    "build_dataflow_payload",
    "dataflow_cli_parser",
    "emit_dataflow_result_outputs",
    "run_check",
    "run_check_command",
    "run_synth",
]
