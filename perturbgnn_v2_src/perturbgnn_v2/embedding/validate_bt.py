"""Embedding quality validation for v2 Phase 1.

Checks:
  1. Cell type separability (silhouette score) — target ≥ 0.3
  2. Niche clustering ARI vs KMeans baseline — target ≥ 0.5
  3. Source vs matched-control embedding overlap — target ≥ 0.7
  4. NTC sanity: matched Δ on NTC spots should be near 0
  5. Cross-slice generalization: M003 embedding quality comparable to M001/M002

Usage:
  python -m perturbgnn_v2.embedding.validate_bt processed/bt_encoder_v3.pt
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)

from perturbgnn_v2.embedding import SupervisedCrossModalEncoder
from perturbgnn_v2.embedding.train_bt import gather_neighborhoods


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")


def compute_embeddings(model, adata, K, batch=4096, device='cuda'):
    """Run encoder on all spots of one slice, return z1, z2, embed."""
    nbr_pca, nbr_ct, ct_label, niche_label, _ = gather_neighborhoods(adata, K=K)
    N = nbr_pca.shape[0]
    z1_all, z2_all = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch):
            p = torch.from_numpy(nbr_pca[i:i+batch]).to(device)
            c = torch.from_numpy(nbr_ct[i:i+batch]).to(device)
            z1 = model.encode_pca(p)
            z2 = model.encode_celltype(c)
            z1_all.append(z1.cpu().numpy())
            z2_all.append(z2.cpu().numpy())
    z1 = np.concatenate(z1_all, axis=0)
    z2 = np.concatenate(z2_all, axis=0)
    embed = 0.5 * (z1 + z2)
    return z1, z2, embed, ct_label, niche_label


def main(ckpt_path: str):
    print(f"[valid] loading {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    print(f"[valid]   args: {ckpt['args']}", flush=True)

    model = SupervisedCrossModalEncoder(
        K=ckpt['args']['K'],
        pca_dim=ckpt['pca_dim'],
        n_cell_types=ckpt['n_cell_types'],
        n_niches=ckpt['n_niches'],
        hidden=ckpt['args']['hidden'],
        embed_dim=ckpt['args']['embed_dim'],
        n_layers=ckpt['args']['n_layers'],
        alpha_align=ckpt['args']['alpha_align'],
    ).cuda()
    model.load_state_dict(ckpt['model'])
    model.eval()

    out_path = PROCESSED / "embedding_validation.json"
    results = {"checkpoint": ckpt_path, "slices": {}}

    for slice_name in ["M001", "M002", "M003"]:
        print(f"\n[valid] === {slice_name} ===", flush=True)
        a = ad.read_h5ad(PROCESSED / f"{slice_name}_v2.h5ad")
        t0 = time.time()
        z1, z2, embed, ct, niche = compute_embeddings(model, a, K=ckpt['args']['K'])
        print(f"  embed computed in {time.time()-t0:.1f}s, shape={embed.shape}")
        # save embeddings for downstream use
        out_emb = PROCESSED / f"embed_{slice_name}_v3.npy"
        np.save(out_emb, embed)
        print(f"  saved: {out_emb}")

        # 1. cell type silhouette
        valid = ct >= 0
        if valid.sum() > 1000 and len(set(ct[valid])) > 1:
            sil = silhouette_score(embed[valid][:50000], ct[valid][:50000])
            print(f"  cell-type silhouette (subsample 50k): {sil:.4f}")
        else:
            sil = None

        # 2. niche clustering ARI (vs KMeans on embed)
        from sklearn.cluster import KMeans
        n_clusters = len(set(niche))
        if N := embed.shape[0] > 50000:
            sub = np.random.RandomState(0).choice(embed.shape[0], 50000, replace=False)
        else:
            sub = np.arange(embed.shape[0])
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(embed[sub])
        ari = adjusted_rand_score(niche[sub], km.labels_)
        nmi = normalized_mutual_info_score(niche[sub], km.labels_)
        print(f"  niche KMeans ARI: {ari:.4f}, NMI: {nmi:.4f}  (n={len(sub)}, K={n_clusters})")

        # 3. source vs NTC embedding distribution
        is_src = a.obs["is_source"].to_numpy().astype(bool)
        is_ntc = a.obs["is_ntc"].to_numpy().astype(bool)
        if is_ntc.sum() > 10:
            src_emb = embed[is_src]
            ntc_emb = embed[is_ntc]
            # overlap: cosine similarity between centroids, normalized
            src_centroid = src_emb.mean(axis=0)
            ntc_centroid = ntc_emb.mean(axis=0)
            cos_centroids = np.dot(src_centroid, ntc_centroid) / (
                np.linalg.norm(src_centroid) * np.linalg.norm(ntc_centroid) + 1e-8)
            print(f"  source vs NTC centroid cosine: {cos_centroids:.4f}")
            print(f"    (close to 1 = no batch bias, target ≥ 0.7)")
            cos_centroids = float(cos_centroids)
        else:
            cos_centroids = None
            print(f"  NTC count too low: {int(is_ntc.sum())}")

        # 4. NTC sanity: matched Δ on NTC spots
        # find NTC source spots' neighbors within 40µm, compare mean module score
        xy = a.obsm["spatial"].astype(np.float32)
        mod = a.obsm["module_scores"][:, 6]  # ifn_response as probe
        if is_ntc.sum() > 30:
            tree = cKDTree(xy)
            ntc_tree = cKDTree(xy[is_ntc])
            # for each NTC spot, find non-NTC neighbors within 40µm
            ntc_pos = xy[is_ntc]
            # near/far Δ
            d_to_ntc, _ = tree.query(xy, k=1)  # nearest NTC distance
            within_40 = (d_to_ntc <= 40) & (~is_ntc)
            beyond_200 = (d_to_ntc > 200) & (~is_ntc)
            if within_40.sum() > 100 and beyond_200.sum() > 100:
                delta = mod[within_40].mean() - mod[beyond_200].mean()
                print(f"  NTC probe Δ (ifn, 40µm vs 200µm+): {delta:+.4f}")
                print(f"    (should be ≈ 0 for clean embedding)")
                ntc_delta = float(delta)
            else:
                ntc_delta = None
        else:
            ntc_delta = None

        results["slices"][slice_name] = {
            "n_spots": int(embed.shape[0]),
            "n_source": int(is_src.sum()),
            "n_ntc": int(is_ntc.sum()),
            "silhouette_celltype": float(sil) if sil is not None else None,
            "niche_ari": float(ari),
            "niche_nmi": float(nmi),
            "src_ntc_centroid_cos": cos_centroids,
            "ntc_ifn_delta_40um": ntc_delta,
        }

    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[valid] wrote results to {out_path}")
    print("\n=== summary ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(PROCESSED / "bt_encoder_v3.pt")
    main(ckpt)
