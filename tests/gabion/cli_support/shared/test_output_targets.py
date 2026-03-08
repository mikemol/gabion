from __future__ import annotations

from pathlib import Path

from gabion.cli_support.shared import output_targets


# gabion:behavior primary=desired
def test_output_target_normalization_handles_stdout_alias_and_path() -> None:
    assert output_targets.is_stdout_target("-", stdout_alias="-", stdout_path="/dev/stdout")
    assert output_targets.is_stdout_target(
        Path("/dev/stdout"),
        stdout_alias="-",
        stdout_path="/dev/stdout",
    )
    assert (
        output_targets.normalize_output_target("-", stdout_alias="-", stdout_path="/dev/stdout")
        == "/dev/stdout"
    )


# gabion:behavior primary=desired
def test_write_text_to_target_overwrites_file_payload(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    output_targets.write_text_to_target(
        target,
        "alpha",
        stdout_alias="-",
        stdout_path="/dev/stdout",
    )
    output_targets.write_text_to_target(
        target,
        "beta",
        stdout_alias="-",
        stdout_path="/dev/stdout",
    )
    assert target.read_text(encoding="utf-8") == "beta"


# gabion:behavior primary=desired
def test_lint_parse_and_collection_adapters_delegate_shared_parser() -> None:
    parsed = output_targets.parse_lint_line("pkg/mod.py:4:5: GAB001 message")
    assert parsed is not None
    assert parsed["path"] == "pkg/mod.py"
    entries = output_targets.collect_lint_entries(
        ["pkg/mod.py:4:5: GAB001 message", "broken"],
        check_deadline_fn=lambda: None,
    )
    assert len(entries) == 1
