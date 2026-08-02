"""Supervised cross-modal spot encoder (v2 Layer 1, design v3).

Switched from Barlow Twins (v1) and SimSiam (v2) to a supervised
multi-task design (v3, 2026-08-01). Both unsupervised approaches hit
representation collapse: Barlow Twins assumes dim-wise alignment that
doesn't exist between PCA and cell-type modalities; SimSiam collapses
to a trivial predictor-target equilibrium without negative pairs.

This design leverages the fact that SPAC-seq provides explicit labels
(cell_type, niche_id) that we already trust. We train two encoders
that each predict cell_type + niche from their own modality, AND we
add a soft alignment loss that pulls the two views' embeddings together
via cosine similarity. The supervised signal prevents collapse; the
alignment loss ensures cross-modal correspondence at the embedding level.

Architecture:
  view 1 (PCA):  neighborhood expression PCs → encoder → z1
  view 2 (CT):   neighborhood cell-type comp → encoder → z2

  Heads:
    h_ct(z1) → cell_type prediction (8-class)
    h_niche(z1) → niche_id prediction (12-class)
    h_ct(z2), h_niche(z2): same heads, shared weights

  Losses:
    L_cls = CE(h_ct(z1), ct) + CE(h_niche(z1), niche)
          + CE(h_ct(z2), ct) + CE(h_niche(z2), niche)
    L_align = 1 - cosine_sim(z1, z2).mean()   # pull views together
    L = L_cls + α * L_align   (α = 0.1)

  Test-time embedding: 0.5 * (z1 + z2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborhoodEncoder(nn.Module):
    """Maps (B, K, feat_dim) → (B, embed_dim)."""

    def __init__(self, K: int, feat_dim: int, hidden: int = 128,
                 embed_dim: int = 64, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        in_dim = 3 * feat_dim
        dims = [in_dim] + [hidden] * (n_layers - 1) + [embed_dim]
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < n_layers - 1:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, nbr_feats: torch.Tensor) -> torch.Tensor:
        mean_pool = nbr_feats.mean(dim=1)
        max_pool = nbr_feats.max(dim=1).values
        anchor = nbr_feats[:, 0, :]
        combined = torch.cat([mean_pool, max_pool, anchor], dim=1)
        return self.mlp(combined)


class ClassificationHead(nn.Module):
    """Single-layer classifier on top of embedding."""

    def __init__(self, embed_dim: int = 64, n_classes: int = 8, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SupervisedCrossModalEncoder(nn.Module):
    """Dual-view encoder with supervised classification + soft alignment."""

    def __init__(self, K: int = 15, pca_dim: int = 32, n_cell_types: int = 8,
                 n_niches: int = 12,
                 hidden: int = 128, embed_dim: int = 64,
                 n_layers: int = 3, dropout: float = 0.1,
                 alpha_align: float = 0.1):
        super().__init__()
        self.K = K
        self.alpha_align = alpha_align
        self.pca_encoder = NeighborhoodEncoder(
            K=K, feat_dim=pca_dim, hidden=hidden, embed_dim=embed_dim,
            n_layers=n_layers, dropout=dropout,
        )
        self.celltype_encoder = NeighborhoodEncoder(
            K=K, feat_dim=n_cell_types, hidden=hidden, embed_dim=embed_dim,
            n_layers=n_layers, dropout=dropout,
        )
        # shared classification heads (both views use same head)
        self.head_ct = ClassificationHead(embed_dim=embed_dim,
                                          n_classes=n_cell_types, dropout=dropout)
        self.head_niche = ClassificationHead(embed_dim=embed_dim,
                                             n_classes=n_niches, dropout=dropout)

    def encode_pca(self, nbr_pca: torch.Tensor) -> torch.Tensor:
        return self.pca_encoder(nbr_pca)

    def encode_celltype(self, nbr_ct: torch.Tensor) -> torch.Tensor:
        return self.celltype_encoder(nbr_ct)

    def forward(self, nbr_pca, nbr_ct, cell_type=None, niche_id=None):
        z1 = self.encode_pca(nbr_pca)
        z2 = self.encode_celltype(nbr_ct)
        # alignment loss (cosine)
        cos = F.cosine_similarity(z1, z2, dim=-1).mean()
        align_loss = 1.0 - cos
        # classification losses (if labels provided)
        cls_loss = torch.tensor(0.0, device=z1.device)
        ct_acc1 = ct_acc2 = niche_acc1 = niche_acc2 = torch.tensor(0.0, device=z1.device)
        if cell_type is not None and niche_id is not None:
            logit_ct1 = self.head_ct(z1)
            logit_niche1 = self.head_niche(z1)
            logit_ct2 = self.head_ct(z2)
            logit_niche2 = self.head_niche(z2)
            cls_loss = (F.cross_entropy(logit_ct1, cell_type)
                        + F.cross_entropy(logit_niche1, niche_id)
                        + F.cross_entropy(logit_ct2, cell_type)
                        + F.cross_entropy(logit_niche2, niche_id))
            with torch.no_grad():
                ct_acc1 = (logit_ct1.argmax(-1) == cell_type).float().mean()
                ct_acc2 = (logit_ct2.argmax(-1) == cell_type).float().mean()
                niche_acc1 = (logit_niche1.argmax(-1) == niche_id).float().mean()
                niche_acc2 = (logit_niche2.argmax(-1) == niche_id).float().mean()
        loss = cls_loss + self.alpha_align * align_loss
        return {
            "embed": 0.5 * (z1 + z2),
            "z1": z1, "z2": z2,
            "loss": loss,
            "cls_loss": cls_loss.detach(),
            "align_loss": align_loss.detach(),
            "cos_sim": cos.detach(),
            "ct_acc1": ct_acc1.detach(),
            "ct_acc2": ct_acc2.detach(),
            "niche_acc1": niche_acc1.detach(),
            "niche_acc2": niche_acc2.detach(),
            "std1_mean": z1.std(dim=0).mean().detach(),
            "std2_mean": z2.std(dim=0).mean().detach(),
        }


# Backward-compat aliases
BarlowTwinsSpotEncoder = SupervisedCrossModalEncoder
