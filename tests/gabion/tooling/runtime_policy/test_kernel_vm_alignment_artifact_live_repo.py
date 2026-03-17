from __future__ import annotations

import pytest

from gabion.tooling.runtime.kernel_vm_alignment_artifact import (
    build_kernel_vm_alignment_artifact_payload,
)
from tests.path_helpers import REPO_ROOT


pytestmark = pytest.mark.live_repo_signal


def test_build_kernel_vm_alignment_artifact_payload_live_repo_drops_only_augmented_rule_runtime_object_gap() -> None:
    payload = build_kernel_vm_alignment_artifact_payload(root=REPO_ROOT)
    residue_ids = {item["residue_id"] for item in payload["residues"]}

    assert (
        "kernel_vm.augmented_rule_core:missing_runtime_object_image"
        not in residue_ids
    )
    assert payload["summary"]["runtime_object_image_gap_count"] == 0
    assert not {
        "kernel_vm.closed_rule_cell_quotient_recovery:missing_runtime_object_image",
        "kernel_vm.query_ast_reflective_boundary:missing_runtime_object_image",
        "kernel_vm.rule_polarity_package:missing_runtime_object_image",
    }.intersection(residue_ids)
