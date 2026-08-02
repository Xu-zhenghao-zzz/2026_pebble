"""EGAT sub-package."""
from .egat_conv import EGATConv
from .egat_encoder import EGATEncoder, assemble_node_features
from .egat_encoder_v2 import EGATEncoder as EGATEncoderV2
from .egat_encoder_v3 import EGATEncoderV3
from .egat_encoder_v4 import EGATEncoderV4

__all__ = ["EGATConv", "EGATEncoder", "EGATEncoderV2", "EGATEncoderV3",
           "EGATEncoderV4", "assemble_node_features"]
