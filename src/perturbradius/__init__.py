"""PerturbRadius: minimal spatial perturbation-scale analysis."""

from .analysis import (
    ModelFit,
    RadiusDecision,
    RadiusResult,
    analyze_radius,
    assess_radius_support,
    estimate_distance_response,
    fit_exponential,
)
from .data import REQUIRED_COLUMNS, validate_data
from .plotting import plot_distance_response
from .spatial import ComponentSet, assign_rings, build_distance_rings, call_components

__all__ = [
    "ComponentSet",
    "ModelFit",
    "REQUIRED_COLUMNS",
    "RadiusDecision",
    "RadiusResult",
    "analyze_radius",
    "assess_radius_support",
    "assign_rings",
    "build_distance_rings",
    "call_components",
    "estimate_distance_response",
    "fit_exponential",
    "plot_distance_response",
    "validate_data",
]

__version__ = "0.1.0"

