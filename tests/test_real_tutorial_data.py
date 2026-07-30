"""Integrity checks for the frozen real-data tutorial conclusions."""

from pathlib import Path

import pandas as pd


DATA = Path(__file__).parents[1] / "tutorials" / "data"


def test_subq_rab8a_direction_counts():
    effects = pd.read_csv(DATA / "subq_rab8a_component_near_far.tsv", sep="\t")
    expected = {
        "Malignant-like fraction": (1, 7),
        "Hypoxia program": (1, 8),
        "IFN-response program": (-1, 7),
    }
    assert effects["clone_id"].nunique() == 8
    assert effects["section"].nunique() == 3
    for response, (sign, agreeing) in expected.items():
        rows = effects.loc[effects["response"].eq(response)]
        assert len(rows) == 8
        assert int((rows["effect"] * sign > 0).sum()) == agreeing


def test_lung_ccn1_triangulation():
    evidence = pd.read_csv(DATA / "lung_ccn1_evidence.tsv", sep="\t").iloc[0]
    boundary = pd.read_csv(
        DATA / "lung_ccn1_boundary_sensitivity.tsv", sep="\t"
    )
    assert len(boundary) == 5
    assert boundary["same_direction"].eq(1).all()
    assert evidence["raw_near_far_effect"] > 0
    assert evidence["matched_near_far_effect"] > 0
    assert evidence["raw_permutation_fdr"] < 0.10
    assert evidence["matched_permutation_fdr"] < 0.10


def test_spatiotemporal_gata3_is_not_radius_eligible():
    gate = pd.read_csv(DATA / "spatiotemporal_gata3_nk_gate.tsv", sep="\t").iloc[0]
    assert gate["guide"] == "sgGata3"
    assert gate["response"] == "fraction_NK"
    assert gate["global_fdr"] > 0.10
    assert gate["max_abs_smd"] > 0.25
    assert not bool(gate["radius_eligible"])
