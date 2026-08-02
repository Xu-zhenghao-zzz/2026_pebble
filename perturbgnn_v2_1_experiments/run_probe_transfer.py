"""Fix 3 — Probe transfer: measure true cross-modal alignment.

v2's SupervisedCrossModalEncoder shares head_ct and head_niche between
z1 (PCA view) and z2 (CT view). cos(z1, z2) = 0.9997 may therefore be
an architectural constraint, not a learned alignment quality.

Probe transfer protocol:
  1. Load v2 encoder checkpoint (bt_encoder_v3.pt).
  2. Split spots 80/20 train/test.
  3. Train head_ct and head_niche on z1 in train set ONLY.
  4. Evaluate zero-shot accuracy on:
     - z1(test) → head → upper bound (same-view)
     - z2(test) → head → cross-modal (the real measurement)
     - z2(test, permuted) → head → lower bound (chance)
  5. Acceptance: cross-modal ≥ 70% × same-view.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
from perturbgnn_v2.embedding.bt_encoder import (
    SupervisedCrossModalEncoder, NeighborhoodEncoder, ClassificationHead)


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
OUT = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/processed/sensitivity")
OUT.mkdir(parents=True, exist_ok=True)


def gather_neighbourhoods(adata, K=15):
    """Build per-spot K-neighbour PCA + CT features (same as train_bt.py)."""
    from scipy.spatial import cKDTree
    xy = adata.obsm["spatial"].astype(np.float32)
    pca = adata.obsm["X_pca"].astype(np.float32)
    ct = adata.obs["cell_type"].to_numpy()
    niche = adata.obs["niche_id"].to_numpy().astype(np.int64)
    vocab = sorted(set(ct.tolist()) - {"Missing"})
    ct_lookup = {c: i for i, c in enumerate(vocab)}

    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=K + 1)
    idx = idx[:, 1:]
    N = xy.shape[0]
    pca_dim = pca.shape[1]
    n_ct = len(vocab)

    nbr_pca = np.take(pca, idx, axis=0).astype(np.float32)
    nbr_ct = np.zeros((N, K, n_ct), dtype=np.float32)
    for k in range(K):
        ct_codes = np.array([ct_lookup.get(c, -1) for c in ct[idx[:, k]]])
        valid = ct_codes >= 0
        nbr_ct[valid, k, ct_codes[valid]] = 1.0
    miss_mask = nbr_ct.sum(axis=-1) == 0
    nbr_ct[miss_mask] = 1.0 / n_ct

    ct_self = np.array([ct_lookup.get(c, 0) for c in ct], dtype=np.int64)
    return nbr_pca, nbr_ct, ct_self, niche.astype(np.int64), vocab


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # Load v2 encoder.
    ckpt_path = PROCESSED / "bt_encoder_v3.pt"
    print(f"Loading v2 encoder: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    ckpt_args = ckpt["args"]
    print(f"  trained {ckpt_args.get('epochs', '?')} epochs, K={ckpt_args.get('K', 15)}", flush=True)

    K = ckpt_args.get("K", 15)
    pca_dim = ckpt_args.get("pca_dim", 32)
    n_cell_types = 8  # v2 default
    n_niches = ckpt_args.get("n_niches", 12)

    model = SupervisedCrossModalEncoder(
        K=K, pca_dim=pca_dim, n_cell_types=n_cell_types, n_niches=n_niches,
        hidden=ckpt_args.get("hidden", 128),
        embed_dim=ckpt_args.get("embed_dim", 64),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_results = {}
    for slice_name in ["M001", "M002", "M003"]:
        print(f"\n=== {slice_name} ===", flush=True)
        adata = ad.read_h5ad(PROCESSED / f"{slice_name}_v2.h5ad")
        print(f"  adata: {adata.shape}", flush=True)

        nbr_pca, nbr_ct, ct_self, niche, vocab = gather_neighbourhoods(adata, K=K)
        # Filter valid (cell_type >= 0).
        valid_mask = np.array([vocab.index(c) if c in vocab else -1
                                for c in adata.obs["cell_type"]])
        valid = valid_mask >= 0
        print(f"  valid cell_type: {valid.sum()}/{len(valid)}", flush=True)

        nbr_pca_t = torch.from_numpy(nbr_pca[valid]).to(device)
        nbr_ct_t = torch.from_numpy(nbr_ct[valid]).to(device)
        ct_t = torch.from_numpy(ct_self[valid]).to(device)
        niche_t = torch.from_numpy(niche[valid]).to(device)

        # Forward through frozen encoder, get z1 and z2.
        print(f"  encoding {len(nbr_pca_t)} spots...", flush=True)
        with torch.no_grad():
            z1 = model.encode_pca(nbr_pca_t)
            z2 = model.encode_celltype(nbr_ct_t)
        # z1, z2: (N, 64)

        # Split 80/20.
        N = z1.shape[0]
        idx = np.random.default_rng(7).permutation(N)
        n_train = int(0.8 * N)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        print(f"  train: {len(train_idx)}, test: {len(test_idx)}", flush=True)

        z1_train = z1[train_idx]; z1_test = z1[test_idx]
        z2_train = z2[train_idx]; z2_test = z2[test_idx]
        ct_train = ct_t[train_idx]; ct_test = ct_t[test_idx]
        niche_train = niche_t[train_idx]; niche_test = niche_t[test_idx]

        # Train fresh heads on z1 only.
        head_ct = ClassificationHead(embed_dim=64, n_classes=n_cell_types).to(device)
        head_niche = ClassificationHead(embed_dim=64, n_classes=n_niches).to(device)
        optim = torch.optim.Adam(
            list(head_ct.parameters()) + list(head_niche.parameters()),
            lr=1e-3, weight_decay=1e-4)

        # Quick training (300 steps).
        batch_size = 1024
        n_steps = 300
        for step in range(n_steps):
            bi = np.random.randint(0, len(z1_train), batch_size)
            z1_b = z1_train[bi]
            ct_b = ct_train[bi]; niche_b = niche_train[bi]
            logit_ct = head_ct(z1_b); logit_niche = head_niche(z1_b)
            loss = F.cross_entropy(logit_ct, ct_b) + F.cross_entropy(logit_niche, niche_b)
            optim.zero_grad(); loss.backward(); optim.step()
            if (step + 1) % 100 == 0:
                print(f"    step {step+1}: loss={loss.item():.4f}", flush=True)

        # Eval on test set, three conditions.
        with torch.no_grad():
            # Same-view (upper bound): z1 test → head.
            ct_pred_z1 = head_ct(z1_test).argmax(-1)
            niche_pred_z1 = head_niche(z1_test).argmax(-1)
            ct_acc_z1toz1 = (ct_pred_z1 == ct_test).float().mean().item()
            niche_acc_z1toz1 = (niche_pred_z1 == niche_test).float().mean().item()

            # Cross-modal: z2 test → head (zero-shot).
            ct_pred_z2 = head_ct(z2_test).argmax(-1)
            niche_pred_z2 = head_niche(z2_test).argmax(-1)
            ct_acc_z1toz2 = (ct_pred_z2 == ct_test).float().mean().item()
            niche_acc_z1toz2 = (niche_pred_z2 == niche_test).float().mean().item()

            # Lower bound: z2 permuted → head.
            perm = torch.randperm(len(z2_test), device=device)
            ct_pred_z2p = head_ct(z2_test[perm]).argmax(-1)
            niche_pred_z2p = head_niche(z2_test[perm]).argmax(-1)
            ct_acc_random = (ct_pred_z2p == ct_test).float().mean().item()
            niche_acc_random = (niche_pred_z2p == niche_test).float().mean().item()

        # Also compute reported cos(z1, z2) for reference.
        with torch.no_grad():
            cos = F.cosine_similarity(z1, z2, dim=-1).mean().item()

        # Acceptance.
        ct_pass = ct_acc_z1toz2 >= 0.7 * ct_acc_z1toz1
        niche_pass = niche_acc_z1toz2 >= 0.7 * niche_acc_z1toz1

        result = {
            "n_total": N,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "v2_reported_cos": round(cos, 4),
            "ct_acc_z1toz1": round(ct_acc_z1toz1, 4),
            "ct_acc_z1toz2": round(ct_acc_z1toz2, 4),
            "ct_acc_random": round(ct_acc_random, 4),
            "ct_pass_70pct": ct_pass,
            "niche_acc_z1toz1": round(niche_acc_z1toz1, 4),
            "niche_acc_z1toz2": round(niche_acc_z1toz2, 4),
            "niche_acc_random": round(niche_acc_random, 4),
            "niche_pass_70pct": niche_pass,
        }
        all_results[slice_name] = result
        print(f"\n  Results for {slice_name}:", flush=True)
        print(f"    v2 reported cos(z1, z2) = {cos:.4f}", flush=True)
        print(f"    cell_type:", flush=True)
        print(f"      z1 → z1 (same-view):   {ct_acc_z1toz1:.3f}", flush=True)
        print(f"      z1 → z2 (cross-modal): {ct_acc_z1toz2:.3f}  ({100*ct_acc_z1toz2/max(ct_acc_z1toz1,1e-6):.1f}% of same-view)", flush=True)
        print(f"      random:                {ct_acc_random:.3f}", flush=True)
        print(f"      PASS 70% threshold:    {ct_pass}", flush=True)
        print(f"    niche:", flush=True)
        print(f"      z1 → z1 (same-view):   {niche_acc_z1toz1:.3f}", flush=True)
        print(f"      z1 → z2 (cross-modal): {niche_acc_z1toz2:.3f}  ({100*niche_acc_z1toz2/max(niche_acc_z1toz1,1e-6):.1f}% of same-view)", flush=True)
        print(f"      random:                {niche_acc_random:.3f}", flush=True)
        print(f"      PASS 70% threshold:    {niche_pass}", flush=True)

    out_path = OUT / "probe_transfer.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(ckpt_path), "slices": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
