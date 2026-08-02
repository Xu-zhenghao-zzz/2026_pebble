"""EGATEncoder v3 — drop cell_type supervision, add contrastive loss.

Diagnosis from v1/v2:
  cell_type CE supervision is the easy-path collapse route. cell_type is
  already a one-hot in the input features, so the model trivially learns
  "copy the one-hot" → ct_acc=1.0 in 5 epochs, gradients dominated by
  ct_loss, niche structure never learned.

v3 fix:
  1. Drop cell_type CE entirely (input contains it; predicting it teaches
     nothing about graph structure).
  2. Keep niche_id CE (this is the only signal that requires the model to
     actually use neighbour information — niche is a 12-class label
     derived from cell-type *composition* in a 40µm window).
  3. Add SimCLR-style contrastive: pull embeddings of spots in the same
     niche together, push different-niche spots apart. This is a stronger
     signal than CE alone because it shapes the geometry of the embedding
     space, not just the decision boundary.

Loss:
    L_niche  = CE(head_niche(z), niche)
    L_contr  = InfoNCE(z_i, z_j) with positive = same niche, negative = different
    L        = L_niche + 0.5 * L_contr
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .egat_conv import EGATConv


def assemble_node_features(pyg_data) -> torch.Tensor:
    """43-dim node features (32 PCA + 9 CT one-hot incl unknown + 1 density + 1 vessel)."""
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


def infonce_contrastive(z: torch.Tensor, niche_id: torch.Tensor,
                        temperature: float = 0.1,
                        n_anchor: int = 1024) -> torch.Tensor:
    """SimCLR-style InfoNCE: positive pairs are same-niche spots.

    Subsamples n_anchor spots to keep memory O(n_anchor^2) instead of O(N^2).
    For each anchor i, the positive is a randomly chosen same-niche spot j.
    Negatives are all other sampled spots.
    """
    n = z.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=z.device)

    valid = (niche_id >= 0)
    if valid.sum() < 4:
        return torch.tensor(0.0, device=z.device)

    # Subsample anchors (and use them as the candidate pool too).
    if n > n_anchor:
        idx = torch.randperm(n, device=z.device)[:n_anchor]
        z = z[idx]
        niche_id = niche_id[idx]
        n = n_anchor

    z_norm = F.normalize(z, dim=-1)

    # Group niches for positive sampling.
    niche_groups = {}
    for ni in niche_id.unique().tolist():
        if ni < 0:
            continue
        gidx = (niche_id == ni).nonzero(as_tuple=True)[0]
        if len(gidx) >= 2:
            niche_groups[ni] = gidx
    if not niche_groups:
        return torch.tensor(0.0, device=z.device)

    pos_idx = torch.arange(n, device=z.device)
    for i in range(n):
        ni = int(niche_id[i].item())
        if ni in niche_groups and len(niche_groups[ni]) >= 2:
            choices = niche_groups[ni][niche_groups[ni] != i]
            if len(choices) > 0:
                pos_idx[i] = choices[torch.randint(len(choices), (1,), device=z.device)]

    sim = z_norm @ z_norm.T / temperature   # (n, n) — at most 1024^2 = 1M floats = 4MB.
    mask_self = torch.eye(n, device=z.device).bool()
    sim = sim.masked_fill(mask_self, -1e9)
    pos_sim = sim[torch.arange(n, device=z.device), pos_idx]
    log_prob = pos_sim - torch.logsumexp(sim, dim=-1)
    loss = -log_prob.mean()
    return loss


class EGATEncoderV3(nn.Module):
    """v3: drop cell_type, keep niche, add contrastive."""

    def __init__(
        self,
        in_dim: int = 43,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        edge_dim: int = 8,
        heads: int = 4,
        n_niches: int = 12,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.1,
        contrastive_weight: float = 0.5,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        assert hidden_dim % heads == 0
        l1 = hidden_dim // heads
        self.egat1 = EGATConv(in_dim, edge_dim, l1, heads, concat=True,
                              negative_slope=negative_slope,
                              dropout=attn_dropout, add_self_loops=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.residual1 = nn.Linear(in_dim, hidden_dim, bias=False)

        self.egat2 = EGATConv(hidden_dim, edge_dim, embed_dim, heads,
                              concat=False, negative_slope=negative_slope,
                              dropout=attn_dropout, add_self_loops=False)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Only niche head — cell_type dropped intentionally.
        self.head_niche = _ClassificationHead(embed_dim, n_niches, dropout)

    def encode(self, x, edge_index, edge_attr):
        h1, attn1 = self.egat1(x, edge_index, edge_attr,
                                return_attention_weights=True)
        alpha1 = attn1[1]
        h1 = self.norm1(h1); h1 = F.relu(h1); h1 = self.dropout1(h1)
        h1 = h1 + self.residual1(x)

        h2, attn2 = self.egat2(h1, edge_index, edge_attr,
                                return_attention_weights=True)
        alpha2 = attn2[1]
        h2 = self.norm2(h2); h2 = F.relu(h2); h2 = self.dropout2(h2)

        self._alpha1 = alpha1
        self._alpha2 = alpha2
        return h2

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

        # Contrastive loss.
        contr_loss = infonce_contrastive(z, niche_id, temperature=self.temperature)

        loss = niche_loss + self.contrastive_weight * contr_loss

        with torch.no_grad():
            z_std = z.std(dim=0).mean()
            # Cosine between random pair (collapse check).
            n = z.shape[0]
            if n >= 2:
                idx = torch.randperm(n, device=z.device)[:200]
                z_norm = F.normalize(z[idx], dim=-1)
                rand_cos = (z_norm @ z_norm.T)
                # exclude diagonal
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
