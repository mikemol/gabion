from __future__ import annotations

from gabion.analysis.aspf import Forest


def test_paramset_packed_reuse() -> None:
    forest = Forest()
    site_a = forest.add_site("a.py", "mod.fn_a")
    site_b = forest.add_site("b.py", "mod.fn_b")
    paramset = forest.add_paramset(["alpha", "beta"])
    forest.add_alt("DecisionSurface", (site_a, paramset))
    forest.add_alt("ValueDecisionSurface", (site_b, paramset))

    paramsets = [node for node in forest.nodes.values() if node.kind == "ParamSet"]
    assert len(paramsets) == 1

    alts = [alt for alt in forest.alts if alt.kind in {"DecisionSurface", "ValueDecisionSurface"}]
    assert len(alts) == 2
