"""Embedding subpackage for perturbgnn_v2 (v3 supervised cross-modal)."""
from .bt_encoder import (
    NeighborhoodEncoder,
    ClassificationHead,
    SupervisedCrossModalEncoder,
    BarlowTwinsSpotEncoder,  # alias
)

__all__ = [
    "NeighborhoodEncoder",
    "ClassificationHead",
    "SupervisedCrossModalEncoder",
    "BarlowTwinsSpotEncoder",
]
