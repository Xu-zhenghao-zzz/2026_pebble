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
from .plotting import (
    local_response_grid,
    plot_distance_response,
    plot_local_response,
    plot_section_map,
)
from .raw import (
    empirical_source_posteriors,
    h5_schema,
    marker_lineage,
    parse_guide,
    read_dense_guide_calls,
    read_h5_axes,
    read_sparse_guide_calls,
    stream_module_scores,
)
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
    "empirical_source_posteriors",
    "h5_schema",
    "local_response_grid",
    "marker_lineage",
    "parse_guide",
    "plot_distance_response",
    "plot_local_response",
    "plot_section_map",
    "read_dense_guide_calls",
    "read_h5_axes",
    "read_sparse_guide_calls",
    "stream_module_scores",
    "validate_data",
]

__version__ = "0.2.0"
