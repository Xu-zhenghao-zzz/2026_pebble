import pandas as pd

import perturbradius as pr


def test_end_to_end_recovers_decay_and_passes_gate(synthetic_screen):
    result = pr.analyze_radius(
        synthetic_screen,
        perturbation="GeneA",
        ring_edges=(0, 20, 40, 80, 120, 180),
        eps_um=15,
        min_samples=4,
        min_source_bins=15,
        bin_size_um=10,
        replicate_col="mouse",
    )
    assert result.model_fit.success
    assert result.model_fit.monotonic
    assert 25 < result.model_fit.length_scale_um < 100
    assert result.model_fit.r_squared > 0.8
    assert result.radius_decision.reportable
    assert len(result.effect_table) == 5


def test_missing_ntc_is_rejected(synthetic_screen):
    without_ntc = synthetic_screen.loc[~synthetic_screen["is_ntc"]].copy()
    result = pr.analyze_radius(
        without_ntc,
        perturbation="GeneA",
        ring_edges=(0, 20, 40, 80, 120, 180),
        eps_um=15,
        min_samples=4,
        min_source_bins=15,
        bin_size_um=10,
        replicate_col="mouse",
    )
    assert not result.radius_decision.reportable
    assert any("NTC" in reason for reason in result.radius_decision.reasons)


def test_non_monotonic_curve_is_detected():
    table = pd.DataFrame(
        {
            "distance_midpoint_um": [10, 30, 60, 100],
            "effect": [1.0, 0.4, 0.8, 0.0],
        }
    )
    fit = pr.fit_exponential(table)
    assert fit.success
    assert not fit.monotonic

