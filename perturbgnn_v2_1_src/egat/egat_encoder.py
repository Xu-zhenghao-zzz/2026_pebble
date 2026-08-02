"""EGATEncoder: 2-layer edge-aware GAT for spot embedding.

Replaces v2's `SupervisedCrossModalEncoder` (set-pooling) with true
message-passing GNN. See GNN_SELECTION.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .egat_conv import EGATConv


def assemble_node_features(pyg_data) -> torch.Tensor:
    """Build the node feature matrix from raw tissue graph fields.

    Handles -1 cell_type (Missing) by mapping to an extra "unknown" class.

    Returns (N, 32 + n_ct + 2) float32. With n_ct=9, that's 43 dims.
    """
    pca = pyg_data.x.float()                                # (N, 32)
    ct = pyg_data.cell_type.long()                          # (N,)
    # Map -1 (Missing) to an extra "unknown" class.
    ct_unknown = (ct < 0).long()
    ct_safe = torch.clamp(ct, min=0)
    n_known = int(getattr(pyg_data, "n_cell_types", 8))
    n_ct = n_known + 1   # +1 for unknown
    ct_onehot = F.one_hot(ct_safe, num_classes=n_known).float()    # (N, 8)
    ct_unknown_col = ct_unknown.float().unsqueeze(-1)              # (N, 1)
    ct_full = torch.cat([ct_onehot, ct_unknown_col], dim=-1)       # (N, 9)
    density = pyg_data.density_local.float().unsqueeze(-1)         # (N, 1)
    vessel = pyg_data.vessel_distance.float().unsqueeze(-1)        # (N, 1)
    return torch.cat([pca, ct_full, density, vessel], dim=-1)      # (N, 43)


class EGATEncoder(nn.Module):
    """2-layer EGAT encoder with cell_type + niche supervision heads."""

    def __init__(
        self,
        in_dim: int = 43,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        edge_dim: int = 8,
        heads: int = 4,
        n_cell_types: int = 8,
        n_niches: int = 12,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        assert hidden_dim % heads == 0, f"hidden_dim {hidden_dim} not divisible by heads {heads}"
        l1_out_per_head = hidden_dim // heads
        self.egat1 = EGATConv(
            in_channels=in_dim,
            edge_dim=edge_dim,
            out_channels=l1_out_per_head,
            heads=heads,
            concat=True,
            negative_slope=negative_slope,
            dropout=attn_dropout,
            add_self_loops=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.residual1 = nn.Linear(in_dim, hidden_dim, bias=False)

        self.egat2 = EGATConv(
            in_channels=hidden_dim,
            edge_dim=edge_dim,
            out_channels=embed_dim,
            heads=heads,
            concat=False,
            negative_slope=negative_slope,
            dropout=attn_dropout,
            add_self_loops=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Heads: predict over the 8 known cell types; -1 (Missing) excluded
        # from the loss via a mask in the training loop.
        self.head_ct = _ClassificationHead(embed_dim, n_cell_types, dropout)
        self.head_niche = _ClassificationHead(embed_dim, n_niches, dropout)

    def encode(self, x, edge_index, edge_attr):
        h1 = self.egat1(x, edge_index, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout1(h1)
        h1 = h1 + self.residual1(x)

        h2 = self.egat2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout2(h2)
        return h2

    def forward(self, x, edge_index, edge_attr, cell_type=None, niche_id=None):
        z = self.encode(x, edge_index, edge_attr)

        ct_loss = torch.tensor(0.0, device=z.device)
        niche_loss = torch.tensor(0.0, device=z.device)
        ct_acc = torch.tensor(0.0, device=z.device)
        niche_acc = torch.tensor(0.0, device=z.device)

        if cell_type is not None:
            # Mask out -1 (Missing) cells — they can't contribute to CE.
            valid_ct = (cell_type >= 0)
            if valid_ct.any():
                ct_logits = self.head_ct(z)
                ct_loss = F.cross_entropy(
                    ct_logits[valid_ct], cell_type[valid_ct].long())
                with torch.no_grad():
                    ct_acc = (ct_logits[valid_ct].argmax(-1)
                              == cell_type[valid_ct]).float().mean()
        if niche_id is not None:
            valid_niche = (niche_id >= 0)
            if valid_niche.any():
                niche_logits = self.head_niche(z)
                niche_loss = F.cross_entropy(
                    niche_logits[valid_niche], niche_id[valid_niche].long())
                with torch.no_grad():
                    niche_acc = (niche_logits[valid_niche].argmax(-1)
                                 == niche_id[valid_niche]).float().mean()

        loss = ct_loss + niche_loss
        with torch.no_grad():
            z_std = z.std(dim=0).mean()

        return {
            "embed": z,
            "loss": loss,
            "cls_loss": ct_loss.detach() + niche_loss.detach(),
            "ct_loss": ct_loss.detach(),
            "niche_loss": niche_loss.detach(),
            "ct_acc": ct_acc.detach(),
            "niche_acc": niche_acc.detach(),
            "std_mean": z_std.detach(),
        }


class _ClassificationHead(nn.Module):
    """Single hidden-layer classifier. Mirrors v2's head."""

    def __init__(self, embed_dim: int = 64, n_classes: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, z):
        return self.net(z)
