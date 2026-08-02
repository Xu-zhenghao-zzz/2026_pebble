"""EGATEncoder v4 — single layer, wider hidden.

Diagnosis chain:
- v1 (2-layer, ct+niche) → ct collapse, δ=+0.10
- v2 (2-layer, + entropy reg) → still overfits, killed
- v3 (2-layer, drop ct + InfoNCE) → δ=+0.11, doesn't recover
- ALL 2-layer variants cap at δ≈+0.10-0.40, 6× weaker than v2 set-pooling

Root cause: L=2 message-passing receptive field (~120µm) matches niche
radius → embeddings collapse within niches.

v4 hypothesis: drop to L=1 (single message-passing round) + widen
hidden to compensate. Now the receptive field is 1-hop (~60µm), same
as v2 set-pooling, but with learned attention instead of fixed
mean+max pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .egat_conv import EGATConv
from .egat_encoder_v3 import infonce_contrastive


def assemble_node_features(pyg_data) -> torch.Tensor:
    """43-dim node features."""
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


class EGATEncoderV4(nn.Module):
    """Single-layer EGAT, wider hidden, niche CE + InfoNCE."""

    def __init__(
        self,
        in_dim: int = 43,
        hidden_dim: int = 256,           # widened from 128
        embed_dim: int = 64,
        edge_dim: int = 8,
        heads: int = 8,                   # increased for wider layer
        n_niches: int = 12,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.1,
        contrastive_weight: float = 0.5,
        temperature: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        assert hidden_dim % heads == 0, f"hidden_dim {hidden_dim} not divisible by heads {heads}"
        l1_out_per_head = hidden_dim // heads

        # Input projection (bring 43-dim node features into the wider space).
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # SINGLE EGAT layer (no layer 2).
        self.egat = EGATConv(
            in_channels=hidden_dim,
            edge_dim=edge_dim,
            out_channels=l1_out_per_head,
            heads=heads,
            concat=True,
            negative_slope=negative_slope,
            dropout=attn_dropout,
            add_self_loops=False,
        )
        # After EGAT: heads * l1_out_per_head = hidden_dim
        self.post_egat_norm = nn.LayerNorm(hidden_dim)
        self.post_egat_dropout = nn.Dropout(dropout)

        # Output projection to embed_dim.
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_dropout = nn.Dropout(dropout)

        # Residual on the EGAT layer (input is already projected).
        self.residual = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.head_niche = _ClassificationHead(embed_dim, n_niches, dropout)

    def encode(self, x, edge_index, edge_attr):
        # Project input to wider space.
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)

        # Single EGAT layer.
        h_egat, attn = self.egat(h, edge_index, edge_attr,
                                  return_attention_weights=True)
        self._alpha = attn[1]
        h_egat = self.post_egat_norm(h_egat)
        h_egat = F.relu(h_egat)
        h_egat = self.post_egat_dropout(h_egat)
        # Residual.
        h = h_egat + self.residual(h)

        # Project to embed_dim.
        z = self.output_proj(h)
        z = self.output_norm(z)
        z = F.relu(z)
        z = self.output_dropout(z)
        return z

    def forward(self, x, edge_index, edge_attr, niche_id=None, **kwargs):
        z = self.encode(x, edge_index, edge_attr)

        niche_loss = torch.tensor(0.0, device=z.device)
        niche_acc = torch.tensor(0.0, device=z.device)
        if niche_id is not None:
            valid = (niche_id >= 0)
            if valid.any():
                logits = self.head_niche(z)
                niche_loss = F.cross_entropy(logits[valid], niche_id[valid].long())
                with torch.no_grad():
                    niche_acc = (logits[valid].argmax(-1) == niche_id[valid]).float().mean()

        contr_loss = infonce_contrastive(z, niche_id, temperature=self.temperature)
        loss = niche_loss + self.contrastive_weight * contr_loss

        with torch.no_grad():
            z_std = z.std(dim=0).mean()
            n = z.shape[0]
            if n >= 2:
                idx = torch.randperm(n, device=z.device)[:200]
                z_norm = F.normalize(z[idx], dim=-1)
                rand_cos = (z_norm @ z_norm.T)
                mask = ~torch.eye(len(idx), device=z.device).bool()
                avg_cos = rand_cos[mask].mean()
            else:
                avg_cos = torch.tensor(0.0, device=z.device)

        return {
            "embed": z,
            "loss": loss,
            "cls_loss": niche_loss.detach(),
            "niche_loss": niche_loss.detach(),
            "contr_loss": contr_loss.detach(),
            "niche_acc": niche_acc.detach(),
            "std_mean": z_std.detach(),
            "avg_cos_random": avg_cos.detach(),
        }


class _ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int = 64, n_classes: int = 12,
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
