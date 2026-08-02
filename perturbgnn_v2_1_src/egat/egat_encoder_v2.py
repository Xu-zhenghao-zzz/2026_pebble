"""EGATEncoder v2 — fixes attention collapse from v1.

Changes from v1:
1. add_self_loops=False in both EGAT layers (was True).
   Self-loops caused attention to collapse to "self only" → equivalent
   to MLP, message passing useless.
2. Attention entropy regulariser: -H(alpha) per layer, weight 0.05.
   Forces attention to spread across multiple neighbours.
3. Loss reweighting: ct_loss * 0.3 + niche_loss * 1.0.
   cell_type (8 classes) is too easy and dominated v1 training.
4. Encoder reports per-layer attention entropy for monitoring.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .egat_conv import EGATConv


def assemble_node_features(pyg_data) -> torch.Tensor:
    """Build the node feature matrix from raw tissue graph fields.

    Handles -1 cell_type (Missing) by mapping to an extra "unknown" class.
    Returns (N, 32 + 9 + 1 + 1) = (N, 43) float32.
    """
    pca = pyg_data.x.float()
    ct = pyg_data.cell_type.long()
    ct_unknown = (ct < 0).long()
    ct_safe = torch.clamp(ct, min=0)
    n_known = int(getattr(pyg_data, "n_cell_types", 8))
    ct_onehot = F.one_hot(ct_safe, num_classes=n_known).float()
    ct_unknown_col = ct_unknown.float().unsqueeze(-1)
    ct_full = torch.cat([ct_onehot, ct_unknown_col], dim=-1)
    density = pyg_data.density_local.float().unsqueeze(-1)
    vessel = pyg_data.vessel_distance.float().unsqueeze(-1)
    return torch.cat([pca, ct_full, density, vessel], dim=-1)


class EGATEncoder(nn.Module):
    """2-layer EGAT with anti-collapse fixes."""

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
        ct_loss_weight: float = 0.3,
        niche_loss_weight: float = 1.0,
        entropy_reg_weight: float = 0.05,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.ct_loss_weight = ct_loss_weight
        self.niche_loss_weight = niche_loss_weight
        self.entropy_reg_weight = entropy_reg_weight

        assert hidden_dim % heads == 0
        l1_out_per_head = hidden_dim // heads

        # KEY CHANGE: add_self_loops=False to prevent attention collapse.
        self.egat1 = EGATConv(
            in_channels=in_dim,
            edge_dim=edge_dim,
            out_channels=l1_out_per_head,
            heads=heads,
            concat=True,
            negative_slope=negative_slope,
            dropout=attn_dropout,
            add_self_loops=False,
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
            add_self_loops=False,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.head_ct = _ClassificationHead(embed_dim, n_cell_types, dropout)
        self.head_niche = _ClassificationHead(embed_dim, n_niches, dropout)

    def encode(self, x, edge_index, edge_attr):
        # Layer 1.
        h1, attn1 = self.egat1(x, edge_index, edge_attr,
                                return_attention_weights=True)
        alpha1 = attn1[1]  # (E, H) tensor
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout1(h1)
        h1 = h1 + self.residual1(x)

        # Layer 2.
        h2, attn2 = self.egat2(h1, edge_index, edge_attr,
                                return_attention_weights=True)
        alpha2 = attn2[1]
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout2(h2)

        # Cache attention for entropy regulariser.
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        return h2

    def _attention_entropy(self, alpha, index, num_receivers):
        """Per-receiver entropy of attention distribution.

        alpha: (E, H) attention weights in [0, 1].
        index: (E,) receiver node index.
        Returns mean per-receiver entropy across heads.
        """
        if alpha is None:
            return torch.tensor(0.0, device=self.head_ct.net[0].weight.device)
        H = alpha.shape[1]
        # For each receiver, sum -alpha * log alpha over its senders.
        eps = 1e-12
        per_edge_ent = -(alpha * (alpha + eps).log())
        # Sum over senders per receiver per head.
        ent_per_receiver_head = torch.zeros(num_receivers, H,
                                            device=alpha.device)
        ent_per_receiver_head.scatter_add_(
            0, index.unsqueeze(-1).expand_as(per_edge_ent), per_edge_ent)
        return ent_per_receiver_head.mean()

    def forward(self, x, edge_index, edge_attr, cell_type=None, niche_id=None):
        z = self.encode(x, edge_index, edge_attr)
        N = z.shape[0]

        ct_loss = torch.tensor(0.0, device=z.device)
        niche_loss = torch.tensor(0.0, device=z.device)
        ct_acc = torch.tensor(0.0, device=z.device)
        niche_acc = torch.tensor(0.0, device=z.device)

        if cell_type is not None:
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

        # Attention entropy regulariser (anti-collapse).
        # We need the receiver index for entropy computation. PyG stores it
        # as edge_index[1] (target of message in PyG convention is row 0,
        # but with our scatter convention the receiver is index 1).
        # Recompute the receiver index.
        receiver_idx = edge_index[1]
        ent1 = self._attention_entropy(self._alpha1, receiver_idx, N) if self._alpha1 is not None else torch.tensor(0.0, device=z.device)
        ent2 = self._attention_entropy(self._alpha2, receiver_idx, N) if self._alpha2 is not None else torch.tensor(0.0, device=z.device)
        # Negative entropy → we want to MAXIMISE entropy, so add -H to loss.
        ent_reg = -(ent1 + ent2)

        # Reweighted classification loss + entropy regulariser.
        loss = (
            self.ct_loss_weight * ct_loss
            + self.niche_loss_weight * niche_loss
            + self.entropy_reg_weight * ent_reg
        )

        with torch.no_grad():
            z_std = z.std(dim=0).mean()

        return {
            "embed": z,
            "loss": loss,
            "cls_loss": (self.ct_loss_weight * ct_loss
                         + self.niche_loss_weight * niche_loss).detach(),
            "ct_loss": ct_loss.detach(),
            "niche_loss": niche_loss.detach(),
            "ct_acc": ct_acc.detach(),
            "niche_acc": niche_acc.detach(),
            "ent1": ent1.detach(),
            "ent2": ent2.detach(),
            "ent_reg": ent_reg.detach(),
            "std_mean": z_std.detach(),
        }


class _ClassificationHead(nn.Module):
    """Single hidden-layer classifier."""

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
