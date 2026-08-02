"""Hand-rolled EGATConv — edge-aware Graph Attention Convolution.

v2 of this file: actually returns attention weights when requested, so
the encoder can compute an entropy regulariser.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor


class EGATConv(MessagePassing):
    """Edge-aware Graph Attention Convolution (hand-rolled).

    Returns (out, alpha) when return_attention_weights=True, where alpha is
    (E, heads) attention weights per edge.

    Parameters
    ----------
    in_channels, edge_dim, out_channels, heads, concat: standard.
    negative_slope : LeakyReLU slope in attention score.
    dropout : attention weight dropout.
    add_self_loops : if True, add self-loops with zero edge feature.
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if edge_dim <= 0:
            raise ValueError(f"edge_dim must be > 0, got {edge_dim}")

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.W_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_e = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att = nn.Parameter(torch.empty(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.W_q, self.W_k, self.W_e, self.W_v]:
            layer.reset_parameters()
        nn.init.xavier_uniform_(self.att)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        size: PairTensor = None,
        return_attention_weights: bool = False,
    ):
        N = x.size(0)
        H, C = self.heads, self.out_channels

        h_proj = self.W_k(x) + self.W_v(x)
        h_proj = h_proj.view(N, H, C)
        h_q = self.W_q(x).view(N, H, C)

        if edge_attr is None:
            raise ValueError("EGATConv requires edge_attr.")

        if self.add_self_loops and isinstance(edge_index, Tensor):
            from torch_geometric.utils import add_self_loops as add_sl
            edge_index, edge_attr = add_sl(
                edge_index, edge_attr, fill_value=0.0, num_nodes=N)

        e_proj = self.W_e(edge_attr).view(-1, H, C)

        # Stash for message() to access via self._alpha_after_softmax.
        # We compute alpha inside message and stash it on self.
        self._alpha = None

        out = self.propagate(
            edge_index=edge_index,
            x=h_proj,
            h_q=h_q,
            e=e_proj,
            size=size,
        )

        # After propagate, self._alpha holds the (E, H) attention weights.
        alpha = self._alpha

        if self.concat:
            out = out.reshape(N, H * C)
            out = out + self.W_v(x)
        else:
            out = out.mean(dim=1)
            out = out + self.W_v(x).view(N, H, C).mean(dim=1)

        if return_attention_weights:
            # Return the ORIGINAL edge_index (post-self-loop if added) and alpha.
            return out, (edge_index, alpha)
        return out

    # ------------------------------------------------------------------
    def message(
        self,
        x_j: Tensor,
        h_q_i: Tensor,
        e: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: int,
    ) -> Tensor:
        score = F.leaky_relu(
            h_q_i + x_j + e,
            negative_slope=self.negative_slope,
        )
        score = (score * self.att).sum(dim=-1)

        from torch_geometric.utils import softmax as pyg_softmax
        alpha = pyg_softmax(score, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Stash alpha for forward() to retrieve.
        self._alpha = alpha

        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out

    def __repr__(self) -> str:
        return (f"EGATConv({self.in_channels}, {self.out_channels}, "
                f"edge_dim={self.edge_dim}, heads={self.heads}, "
                f"concat={self.concat})")
