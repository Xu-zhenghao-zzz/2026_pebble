"""EGAT sub-package: hand-rolled edge-aware Graph Attention Conv."""
from .egat_conv import EGATConv
from .egat_encoder import EGATEncoder, assemble_node_features

__all__ = ["EGATConv", "EGATEncoder", "assemble_node_features"]
