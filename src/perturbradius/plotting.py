"""Minimal plotting functions."""

from __future__ import annotations

import numpy as np

from .analysis import RadiusResult, _exponential


def plot_distance_response(result: RadiusResult, *, ax=None):
    """Plot the NTC-adjusted empirical response and diagnostic exponential fit."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.6))
    table = result.effect_table
    distance = table["distance_midpoint_um"].to_numpy(dtype=float)
    effect = table["effect"].to_numpy(dtype=float)
    error = table["effect_se"].to_numpy(dtype=float)
    finite_error = np.where(np.isfinite(error), error, 0.0)
    ax.errorbar(
        distance,
        effect,
        yerr=finite_error,
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="empirical NTC-adjusted effect",
    )
    if result.model_fit.success:
        grid = np.linspace(float(distance.min()), float(distance.max()), 200)
        ax.plot(
            grid,
            _exponential(
                grid,
                result.model_fit.amplitude,
                result.model_fit.length_scale_um,
            ),
            linestyle="--",
            linewidth=1.5,
            label="diagnostic exponential fit",
        )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Distance from source boundary (µm)")
    ax.set_ylabel("Far-centred response")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    status = "radius reportable" if result.radius_decision.reportable else "descriptive only"
    ax.set_title(f"PerturbRadius — {status}")
    return ax

