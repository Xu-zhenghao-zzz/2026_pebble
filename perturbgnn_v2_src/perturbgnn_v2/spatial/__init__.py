"""Spatial subpackage."""
from .slx import fit_slx, SLXConfig
from .resistance import fit_resistance_slx, ResistanceConfig
from .durbin import fit_durbin, DurbinConfig

__all__ = [
    "fit_slx", "SLXConfig",
    "fit_resistance_slx", "ResistanceConfig",
    "fit_durbin", "DurbinConfig",
]
