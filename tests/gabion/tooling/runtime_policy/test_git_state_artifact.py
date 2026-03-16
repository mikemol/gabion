from __future__ import annotations

from pathlib import Path
import subprocess

from gabion.tooling.runtime.git_state_artifact import (
    GitStateCommandOutputs,
    assemble_git_state_artifact_payload,
    build_git_state_artifact_payload,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_build_git_state_artifact_payload_captures_repo_state_classes(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Gabion Tests")
    _git(tmp_path, "config", "user.email", "tests@example.com")

    _write(tmp_path / "tracked.txt", "initial\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial commit")

    _write(tmp_path / "staged.txt", "staged\n")
    _git(tmp_path, "add", "staged.txt")
    _write(tmp_path / "tracked.txt", "modified\n")
    _write(tmp_path / "untracked.txt", "untracked\n")

    payload = build_git_state_artifact_payload(root=tmp_path)

    assert payload["artifact_kind"] == "git_state"
    assert payload["head_sha"]
    assert payload["branch"] == "main"
    assert payload["summary"]["committed_count"] >= 1
    assert payload["summary"]["staged_count"] == 1
    assert payload["summary"]["unstaged_count"] == 1
    assert payload["summary"]["untracked_count"] == 1
    assert any(
        item["state_class"] == "committed" and item["path"] == "tracked.txt"
        for item in payload["entries"]
    )
    assert any(
        item["state_class"] == "staged" and item["path"] == "staged.txt"
        for item in payload["entries"]
    )
    assert any(
        item["state_class"] == "staged"
        and item["path"] == "staged.txt"
        and item["current_line_spans"] == [{"start_line": 1, "line_count": 1}]
        for item in payload["entries"]
    )
    assert any(
        item["state_class"] == "unstaged" and item["path"] == "tracked.txt"
        for item in payload["entries"]
    )
    assert any(
        item["state_class"] == "unstaged"
        and item["path"] == "tracked.txt"
        and item["current_line_spans"] == [{"start_line": 1, "line_count": 1}]
        for item in payload["entries"]
    )
    assert any(
        item["state_class"] == "untracked" and item["path"] == "untracked.txt"
        for item in payload["entries"]
    )
    assert any(
        item["state_class"] == "untracked"
        and item["path"] == "untracked.txt"
        and item["current_line_spans"] == [{"start_line": 1, "line_count": 1}]
        for item in payload["entries"]
    )


def test_assemble_git_state_artifact_payload_supports_injected_command_outputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "scratch.txt").write_text("line one\nline two\n", encoding="utf-8")

    payload = assemble_git_state_artifact_payload(
        root=tmp_path,
        command_outputs=GitStateCommandOutputs(
            head_sha="abc123",
            branch="main",
            upstream="origin/main",
            committed_name_status="M\ttracked.txt",
            staged_name_status="A\tstaged.txt",
            unstaged_name_status="M\ttracked.txt",
            untracked_paths="scratch.txt",
            head_diff="\n".join(
                (
                    "diff --git a/tracked.txt b/tracked.txt",
                    "+++ b/tracked.txt",
                    "@@ -0,0 +3,2 @@",
                )
            ),
        ),
    )

    assert payload["head_sha"] == "abc123"
    assert payload["branch"] == "main"
    assert payload["upstream"] == "origin/main"
    assert payload["summary"] == {
        "committed_count": 1,
        "staged_count": 1,
        "unstaged_count": 1,
        "untracked_count": 1,
    }
    assert any(
        item["state_class"] == "unstaged"
        and item["path"] == "tracked.txt"
        and item["current_line_spans"] == [{"start_line": 3, "line_count": 2}]
        for item in payload["entries"]
    )
    assert any(
        item["state_class"] == "untracked"
        and item["path"] == "scratch.txt"
        and item["current_line_spans"] == [{"start_line": 1, "line_count": 2}]
        for item in payload["entries"]
    )


def test_build_git_state_artifact_payload_allows_injected_command_runner(
    tmp_path: Path,
) -> None:
    observed_commands: list[tuple[str, ...]] = []

    responses = {
        ("rev-parse", "HEAD"): "deadbeef",
        ("rev-parse", "--abbrev-ref", "HEAD"): "HEAD",
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"): "",
        ("show", "--name-status", "--find-renames", "--format=", "HEAD"): "",
        ("diff", "--cached", "--name-status", "--find-renames"): "A\tstaged.txt",
        ("diff", "--name-status", "--find-renames"): "",
        ("ls-files", "--others", "--exclude-standard"): "",
        ("diff", "HEAD", "--unified=0", "--find-renames", "--no-color"): "",
    }

    def _runner(root: Path, args: tuple[str, ...]) -> str:
        assert root == tmp_path.resolve()
        observed_commands.append(args)
        return responses[args]

    payload = build_git_state_artifact_payload(
        root=tmp_path,
        git_command_runner=_runner,
    )

    assert payload["head_sha"] == "deadbeef"
    assert payload["is_detached"] is True
    assert payload["summary"]["staged_count"] == 1
    assert tuple(observed_commands) == tuple(responses)
