from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable, Mapping
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import typer


def _restore_aspf_state_from_github_artifacts(
    *,
    token: str,
    repo: str,
    output_dir: Path,
    ref_name: str = "",
    current_run_id: str = "",
    artifact_name: str = "dataflow-report",
    state_name: str = "aspf_state_ci.json",
    per_page: int = 100,
    urlopen_fn: Callable[..., object] = urllib.request.urlopen,
    no_redirect_open_fn: Callable[..., object] | None = None,
    follow_redirect_open_fn: Callable[..., object] | None = None,
) -> int:
    token = token.strip()
    repo = repo.strip()
    ref_name = ref_name.strip()
    current_run_id = current_run_id.strip()
    artifact_name = artifact_name.strip() or "dataflow-report"
    state_name = state_name.strip() or "aspf_state_ci.json"
    if not token or not repo:
        typer.echo("GitHub token/repository unavailable; skipping ASPF state restore.")
        return 0

    api_url = (
        f"https://api.github.com/repos/{repo}/actions/artifacts"
        f"?name={artifact_name}&per_page={max(1, int(per_page))}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urlopen_fn(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        typer.echo(f"Unable to query prior artifacts ({exc}); skipping ASPF state restore.")
        return 0

    match payload:
        case dict() as payload_mapping:
            artifacts = payload_mapping.get("artifacts", [])
        case _:
            artifacts = []

    def _artifact_is_candidate(item: object) -> bool:
        match item:
            case dict() as artifact_item:
                pass
            case _:
                return False
        download_url = str(artifact_item.get("archive_download_url", "") or "")
        if artifact_item.get("expired", True) or not download_url:
            return False
        match artifact_item.get("workflow_run"):
            case dict() as workflow_run:
                pass
            case _:
                return False
        if current_run_id and str(workflow_run.get("id", "")) == current_run_id:
            return False
        if ref_name and str(workflow_run.get("head_branch", "")) != ref_name:
            return False
        event_name = str(workflow_run.get("event", "")).strip()
        if event_name and event_name not in {"push", "workflow_dispatch"}:
            return False
        return True

    artifact_candidates = [
        item for item in artifacts if _artifact_is_candidate(item)
    ]
    if not artifact_candidates:
        typer.echo(
            "No reusable same-branch dataflow-report artifact found; continuing without checkpoint."
        )
        return 0
    chunk_prefix = f"{state_name}.chunks/"
    output_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for artifact in artifact_candidates:
        download_url = str(artifact.get("archive_download_url", "") or "")
        try:
            archive_bytes = _download_artifact_archive_bytes(
                download_url=download_url,
                headers=headers,
                urlopen_fn=urlopen_fn,
                no_redirect_open_fn=no_redirect_open_fn,
                follow_redirect_open_fn=follow_redirect_open_fn,
            )
            checkpoint_member: str | None = None
            chunk_members: list[str] = []
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
                names = [name for name in zf.namelist() if not name.endswith("/")]
                for name in names:
                    base = name.split("/", 1)[-1]
                    if base == state_name:
                        checkpoint_member = name
                    elif base.startswith(chunk_prefix):
                        chunk_members.append(name)
                match checkpoint_member:
                    case str() as checkpoint_member_value:
                        pass
                    case _:
                        continue
                checkpoint_bytes = zf.read(checkpoint_member_value)
                if _state_requires_chunk_artifacts(
                    checkpoint_bytes=checkpoint_bytes
                ) and not chunk_members:
                    continue
                checkpoint_output = output_dir / state_name
                chunk_output_dir = output_dir / f"{state_name}.chunks"
                if checkpoint_output.exists():
                    checkpoint_output.unlink()
                if chunk_output_dir.exists():
                    for existing in chunk_output_dir.glob("*"):
                        if existing.is_file():
                            existing.unlink()
                checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_output.write_bytes(checkpoint_bytes)
                restored = 1
                for name in chunk_members:
                    base = name.split("/", 1)[-1]
                    destination = output_dir / base
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(zf.read(name))
                    restored += 1
                typer.echo(
                    f"Restored {restored} ASPF state artifact file(s) from prior run."
                )
                return 0
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        typer.echo(
            f"Unable to restore ASPF state from prior artifacts ({last_error}); continuing without checkpoint."
        )
        return 0
    typer.echo(
        "Prior artifacts did not include usable ASPF state files; continuing without restore."
    )
    return 0


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        _ = (req, fp, code, msg, headers, newurl)
        return None


def _download_artifact_archive_bytes(
    *,
    download_url: str,
    headers: Mapping[str, str],
    urlopen_fn: Callable[..., object] = urllib.request.urlopen,
    no_redirect_open_fn: Callable[..., object] | None = None,
    follow_redirect_open_fn: Callable[..., object] | None = None,
) -> bytes:
    req_zip = urllib.request.Request(download_url, headers=dict(headers))
    if urlopen_fn is not urllib.request.urlopen:
        with urlopen_fn(req_zip, timeout=60) as response:
            return response.read()
    if no_redirect_open_fn is None:
        no_redirect_open_fn = urllib.request.build_opener(_NoRedirectHandler()).open
    if follow_redirect_open_fn is None:
        follow_redirect_open_fn = urllib.request.urlopen
    try:
        with no_redirect_open_fn(req_zip, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        redirect_url = str(exc.headers.get("Location", "") or "")
        if not redirect_url:
            raise
        follow_headers: dict[str, str] = {}
        redirect_host = (urllib.parse.urlparse(redirect_url).hostname or "").lower()
        if redirect_host.endswith("github.com"):
            follow_headers = dict(headers)
        follow_req = urllib.request.Request(redirect_url, headers=follow_headers)
        with follow_redirect_open_fn(follow_req, timeout=60) as response:
            return response.read()


def _state_requires_chunk_artifacts(*, checkpoint_bytes: bytes) -> bool:
    try:
        payload = json.loads(checkpoint_bytes.decode("utf-8"))
    except Exception:
        parse_failed = True
        _ = parse_failed
        return False
    match payload:
        case dict() as payload_map:
            pass
        case _:
            return False
    match payload_map.get("collection_resume"):
        case dict() as collection_resume:
            pass
        case _:
            return False
    match collection_resume.get("analysis_index_resume"):
        case dict() as analysis_index_resume:
            pass
        case _:
            return False
    match analysis_index_resume.get("state_ref"):
        case str() as state_ref:
            return bool(state_ref.strip())
        case _:
            return False
