from __future__ import annotations

from gabion.analysis.aspf import Forest


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_paramset_packed_reuse::aspf.py::gabion.analysis.aspf.Forest
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


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_site_records_span::aspf.py::gabion.analysis.aspf.Forest
def test_add_site_records_span() -> None:
    forest = Forest()
    site = forest.add_site("mod.py", "mod.fn", span=(1, 2, 3, 4))
    node = forest.nodes[site]
    assert node.meta["span"] == [1, 2, 3, 4]


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_site_records_file_site::aspf.py::gabion.analysis.aspf.Forest
def test_add_site_records_file_site() -> None:
    forest = Forest()
    site = forest.add_site("mod.py", "mod.fn")
    file_nodes = [node for node in forest.nodes.values() if node.kind == "FileSite"]
    assert len(file_nodes) == 1
    assert file_nodes[0].meta["path"] == "mod.py"
    assert any(
        alt.kind == "FunctionSiteInFile" and alt.inputs[0] == site for alt in forest.alts
    )


# gabion:evidence E:call_footprint::tests/test_aspf.py::test_add_suite_site_records_file_site::aspf.py::gabion.analysis.aspf.Forest
def test_add_suite_site_records_file_site() -> None:
    forest = Forest()
    suite = forest.add_suite_site("mod.py", "mod.fn", "loop", span=(0, 1, 2, 3))
    file_nodes = [node for node in forest.nodes.values() if node.kind == "FileSite"]
    assert len(file_nodes) == 1
    assert any(
        alt.kind == "SuiteSiteInFile" and alt.inputs[0] == suite for alt in forest.alts
    )


def test_add_suite_site_parent_emits_suite_contains() -> None:
    forest = Forest()
    parent = forest.add_suite_site("mod.py", "mod.fn", "function")
    child = forest.add_suite_site(
        "mod.py",
        "mod.fn",
        "if_body",
        span=(2, 4, 3, 8),
        parent=parent,
    )

    assert any(
        alt.kind == "SuiteContains" and alt.inputs == (parent, child)
        for alt in forest.alts
    )
