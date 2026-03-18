from __future__ import annotations

from pathlib import Path
import re

import pytest

from gabion.tooling.policy_substrate import generated_artifact_manifest


_ARTIFACT_PATH_RE = re.compile(
    r"(?:artifacts(?:/(?:out|audit_reports|test_runs))?|out)/[A-Za-z0-9_./*-]+"
)


def _normalize_expected_path(path: str) -> str:
    normalized = path.strip().rstrip(").,")
    if normalized == "artifacts/out/aspf_state":
        return "artifacts/out/aspf_state/**"
    if normalized == "artifacts/test_runs/htmlcov":
        return "artifacts/test_runs/htmlcov/**"
    return normalized


def _workflow_expected_paths(repo_root: Path) -> set[str]:
    text = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    paths = {
        _normalize_expected_path(match.group(0))
        for match in _ARTIFACT_PATH_RE.finditer(text)
    }
    return {
        path
        for path in paths
        if path
        and path not in {
            "artifacts/out",
            "artifacts/test_runs",
            "artifacts/out/controller_drift_gate_history.zip",
            "out/deprecated_fibers_baseline.json",
        }
    }


def _governance_audit_expected_paths(repo_root: Path) -> set[str]:
    text = (
        repo_root / "src" / "gabion_governance" / "governance_audit_impl.py"
    ).read_text(encoding="utf-8")
    return {
        match.group(1)
        for match in re.finditer(r'default=Path\("([^"]+)"\)', text)
        if match.group(1).startswith(("artifacts/", "out/"))
    }


def _policy_check_expected_paths(repo_root: Path) -> set[str]:
    text = (repo_root / "scripts" / "policy" / "policy_check.py").read_text(
        encoding="utf-8"
    )
    paths = {
        f"out/{match.group(1)}"
        for match in re.finditer(r'_WORKFLOW_POLICY_OUTPUT_ROOT / "([^"]+)"', text)
    }
    paths.update(
        f"artifacts/out/{match.group(1)}"
        for match in re.finditer(r'output_path\.parent / "([^"]+)"', text)
    )
    if "local_ci_repro_contract.json" in text:
        paths.add("artifacts/out/local_ci_repro_contract.json")
    return paths


def _runtime_invariant_graph_expected_paths(repo_root: Path) -> set[str]:
    text = (
        repo_root / "src" / "gabion" / "tooling" / "runtime" / "invariant_graph.py"
    ).read_text(encoding="utf-8")
    return {
        match.group(1)
        for match in re.finditer(
            r'_DEFAULT_(?:ARTIFACT|WORKSTREAMS_ARTIFACT|LEDGER_ARTIFACT) = Path\("([^"]+)"\)',
            text,
        )
    }


def _command_orchestrator_expected_paths(repo_root: Path) -> set[str]:
    text = (
        repo_root / "src" / "gabion" / "server_core" / "command_orchestrator.py"
    ).read_text(encoding="utf-8")
    artifact_paths = {
        f"artifacts/out/{match.group(1)}"
        for match in re.finditer(r'\(artifact_dir / "([^"]+)"\)', text)
    }
    out_paths = {
        f"out/{match.group(1)}"
        for match in re.finditer(r'\(out_dir / "([^"]+)"\)', text)
    }
    return artifact_paths | out_paths


# gabion:behavior primary=desired
def test_load_catalog_and_rendered_block_cover_real_catalog() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    catalog = generated_artifact_manifest.load_catalog(
        repo_root / "docs" / "generated_artifact_manifest.yaml"
    )

    generated_artifact_manifest.validate_catalog_against_repo(
        catalog,
        repo_root=repo_root,
    )

    rendered = generated_artifact_manifest.render_manifest_block(catalog)
    assert "<!-- BEGIN:generated_artifact_manifest -->" in rendered
    assert "docs/generated_artifact_manifest.yaml" in rendered
    assert "`docflow_compliance_bundle`" in rendered
    assert "`policy_check_projection_bundle`" in rendered


# gabion:behavior primary=desired
def test_run_rewrites_generated_artifact_manifest_block_and_check_detects_drift(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "generated_artifact_manifest.yaml"
    doc_path = tmp_path / "generated_artifact_manifest.md"
    source_ref = "scripts/policy/render_generated_artifact_manifest.py"
    catalog_path.write_text(
        "\n".join(
            (
                "version: 1",
                "families:",
                "  - family_id: sample",
                "    title: Sample family",
                "    description: Sample family description.",
                "artifacts:",
                "  - artifact_id: sample_artifact",
                "    family_id: sample",
                "    paths:",
                "      - artifacts/out/sample.json",
                "    format: json",
                "    process_domain: local_validation",
                "    emitter_kind: command",
                "    source: sample command",
                f"    source_refs:\n      - {source_ref}",
                "    conditional: false",
                "    trigger_condition: emitted on sample runs",
                "    regeneration:",
                "      - mise exec -- python -m sample.command",
                "    primary_consumers:",
                "      - sample consumer",
                "    notes: sample notes",
                "",
            )
        ),
        encoding="utf-8",
    )
    doc_path.write_text(
        "\n".join(
            (
                "# Generated Artifact Manifest",
                "",
                "<!-- BEGIN:generated_artifact_manifest -->",
                "stale",
                "<!-- END:generated_artifact_manifest -->",
                "",
            )
        ),
        encoding="utf-8",
    )

    assert (
        generated_artifact_manifest.run(
            catalog_path=catalog_path,
            doc_path=doc_path,
        )
        == 0
    )
    assert (
        generated_artifact_manifest.run(
            catalog_path=catalog_path,
            doc_path=doc_path,
            check=True,
        )
        == 0
    )

    doc_path.write_text(
        doc_path.read_text(encoding="utf-8").replace(
            "sample_artifact",
            "stale_artifact",
        ),
        encoding="utf-8",
    )
    assert (
        generated_artifact_manifest.run(
            catalog_path=catalog_path,
            doc_path=doc_path,
            check=True,
        )
        == 1
    )


# gabion:behavior primary=desired
def test_load_catalog_rejects_duplicate_artifact_ids(tmp_path: Path) -> None:
    catalog_path = tmp_path / "generated_artifact_manifest.yaml"
    catalog_path.write_text(
        "\n".join(
            (
                "version: 1",
                "families:",
                "  - family_id: dup",
                "    title: Duplicate family",
                "    description: family",
                "artifacts:",
                "  - artifact_id: dup_artifact",
                "    family_id: dup",
                "    paths: [artifacts/out/a.json]",
                "    format: json",
                "    process_domain: local_validation",
                "    emitter_kind: command",
                "    source: command a",
                "    source_refs: [scripts/policy/render_generated_artifact_manifest.py]",
                "    conditional: false",
                "    trigger_condition: emitted",
                "    regeneration: [mise exec -- python -m a]",
                "    primary_consumers: [consumer]",
                "  - artifact_id: dup_artifact",
                "    family_id: dup",
                "    paths: [artifacts/out/b.json]",
                "    format: json",
                "    process_domain: local_validation",
                "    emitter_kind: command",
                "    source: command b",
                "    source_refs: [scripts/policy/render_generated_artifact_manifest.py]",
                "    conditional: false",
                "    trigger_condition: emitted",
                "    regeneration: [mise exec -- python -m b]",
                "    primary_consumers: [consumer]",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate artifact_id"):
        generated_artifact_manifest.load_catalog(catalog_path)


# gabion:behavior primary=desired
def test_repo_manifest_covers_expected_normal_course_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    catalog = generated_artifact_manifest.load_catalog(
        repo_root / "docs" / "generated_artifact_manifest.yaml"
    )
    manifest_paths = {
        path
        for artifact in catalog.artifacts
        for path in artifact.paths
    }
    expected_paths = (
        _workflow_expected_paths(repo_root)
        | _governance_audit_expected_paths(repo_root)
        | _policy_check_expected_paths(repo_root)
        | _runtime_invariant_graph_expected_paths(repo_root)
        | _command_orchestrator_expected_paths(repo_root)
    )

    assert expected_paths <= manifest_paths


# gabion:behavior primary=desired
def test_generated_artifact_manifest_doc_is_up_to_date() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    assert (
        generated_artifact_manifest.run(
            catalog_path=repo_root / "docs" / "generated_artifact_manifest.yaml",
            doc_path=repo_root / "docs" / "generated_artifact_manifest.md",
            check=True,
        )
        == 0
    )
