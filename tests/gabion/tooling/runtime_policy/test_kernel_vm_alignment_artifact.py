from __future__ import annotations

from pathlib import Path

from gabion.tooling.runtime.kernel_vm_alignment_artifact import (
    build_kernel_vm_alignment_artifact_payload,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:behavior primary=desired
def test_build_kernel_vm_alignment_artifact_payload_counts_imported_and_assigned_runtime_symbols(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "in" / "lg_kernel_ontology_cut_elim-1.ttl",
        "\n".join(
            [
                "lg:AugmentedRule",
                "lg:hasSyntaxClause",
                "lg:hasTypingClause",
                "lg:hasCategoricalClause",
            ]
        )
        + "\n",
    )
    _write(tmp_path / "docs" / "ttl_kernel_semantics.md", "AugmentedRule\n")
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "aspf" / "aspf_lattice_algebra.py",
        "from gabion.analysis.kernel_vm.object_images import AugmentedRule\n",
    )
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "projection" / "semantic_fragment.py",
        "\n".join(
            [
                "CanonicalWitnessedSemanticRow = dict[str, object]",
                "",
                "def reflect_projection_fiber_witness():",
                "    return None",
                "",
                "AugmentedRule = object",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path
        / "src"
        / "gabion"
        / "analysis"
        / "projection"
        / "projection_semantic_lowering.py",
        "\n".join(
            [
                "from gabion.analysis.kernel_vm.object_images import AugmentedRule as ImportedAugmentedRule",
                "",
                "AugmentedRule = ImportedAugmentedRule",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path
        / "src"
        / "gabion"
        / "analysis"
        / "projection"
        / "semantic_fragment_compile.py",
        "AugmentedRule = 1\n",
    )

    payload = build_kernel_vm_alignment_artifact_payload(root=tmp_path)
    binding = next(
        item
        for item in payload["bindings"]
        if item["binding_id"] == "kernel_vm.augmented_rule_core"
    )
    capability = next(
        item
        for item in binding["capabilities"]
        if item["capability_id"] == "runtime_object_image"
    )

    assert capability["match_mode"] == "all"
    assert capability["status"] == "pass"
    assert len(capability["matched_refs"]) == 4
    assert capability["missing_refs"] == []
    assert binding["status"] == "pass"
