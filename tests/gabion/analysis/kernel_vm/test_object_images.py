from __future__ import annotations

from gabion.analysis.kernel_vm.object_images import AugmentedRule


# gabion:behavior primary=desired
def test_augmented_rule_carrier_preserves_ttl_backed_identity_structure() -> None:
    assert AugmentedRule.label == "AugmentedRule"
    assert AugmentedRule.object_id > 0
    assert AugmentedRule.zone_id > 0
    assert len(AugmentedRule.source_path_ids) == 3
    assert all(item > 0 for item in AugmentedRule.source_path_ids)
    assert AugmentedRule.class_term_id > 0
    assert len(AugmentedRule.supporting_term_ids) >= 4
    assert all(item > 0 for item in AugmentedRule.supporting_term_ids)
    assert len(AugmentedRule.ontology_term_ids) >= 4
    assert len(AugmentedRule.shape_term_ids) >= 1
    assert len(AugmentedRule.example_term_ids) >= 1
