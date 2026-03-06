from __future__ import annotations

import json
import urllib.request

from gabion.cli_support.shared import github_artifact_restore


# gabion:evidence E:call_footprint::tests/test_github_artifact_restore.py::test_state_requires_chunk_artifacts_detects_state_ref::github_artifact_restore.py::gabion.cli_support.shared.github_artifact_restore.state_requires_chunk_artifacts
def test_state_requires_chunk_artifacts_detects_state_ref() -> None:
    payload = {
        "collection_resume": {
            "analysis_index_resume": {
                "state_ref": "chunk://resume-state"
            }
        }
    }
    assert github_artifact_restore.state_requires_chunk_artifacts(
        checkpoint_bytes=json.dumps(payload).encode("utf-8")
    )


# gabion:evidence E:call_footprint::tests/test_github_artifact_restore.py::test_no_redirect_handler_redirect_request_returns_none::github_artifact_restore.py::gabion.cli_support.shared.github_artifact_restore.NoRedirectHandler.redirect_request
def test_no_redirect_handler_redirect_request_returns_none() -> None:
    handler = github_artifact_restore.NoRedirectHandler()
    request = urllib.request.Request("https://example.invalid/archive.zip")
    assert (
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {"Location": "https://example.invalid/redirect"},
            "https://example.invalid/redirect",
        )
        is None
    )


# gabion:evidence E:call_footprint::tests/test_github_artifact_restore.py::test_private_aliases_remain_mapped_to_public_entry_points::github_artifact_restore.py::gabion.cli_support.shared.github_artifact_restore.restore_aspf_state_from_github_artifacts
def test_private_aliases_remain_mapped_to_public_entry_points() -> None:
    assert (
        github_artifact_restore._restore_aspf_state_from_github_artifacts
        is github_artifact_restore.restore_aspf_state_from_github_artifacts
    )
    assert github_artifact_restore._NoRedirectHandler is github_artifact_restore.NoRedirectHandler
    assert (
        github_artifact_restore._download_artifact_archive_bytes
        is github_artifact_restore.download_artifact_archive_bytes
    )
    assert (
        github_artifact_restore._state_requires_chunk_artifacts
        is github_artifact_restore.state_requires_chunk_artifacts
    )
