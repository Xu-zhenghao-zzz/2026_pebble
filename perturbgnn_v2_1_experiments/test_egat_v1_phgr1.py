"""Quick L2+L3 test: 用 EGAT 坏模型 embedding 跑 Phgr1 × fibroblast 看 δ.

Compares to v2 set-pooling baseline: Δ=+0.62 (M001, Phgr1×fibroblast).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
OUT_PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/processed")

MODULES = ["malignant", "cd8_like", "macrophage", "fibroblast",
           "endothelial", "hypoxia", "ifn_response"]


def cosine_nn_match(source_emb, pool_emb):
    """Find index of closest pool entry (cosine) for each source."""
    s = source_emb / (np.linalg.norm(source_emb, axis=1, keepdims=True) + 1e-12)
    p = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-12)
    sim = s @ p.T
    return sim.argmax(axis=1)


def smd_per_dim(a, b):
    """Per-dim Standardized Mean Difference."""
    pooled_std = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2) + 1e-12
    return np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled_std


def main():
    target_gene = "Phgr1"
    response = "fibroblast"
    suffix = "egat_v1_bad"

    rows = []
    for slice_name in ["M001", "M002", "M003"]:
        print(f"\n=== {slice_name} ===", flush=True)
        adata = ad.read_h5db(PROCESSED / f"{slice_name}_v2.h5ad") if False else ad.read_h5ad(PROCESSED / f"{slice_name}_v2.h5ad")
        embed = np.load(OUT_PROCESSED / f"embed_{slice_name}_{suffix}.npy")
        print(f"  adata: {adata.shape}, embed: {embed.shape}", flush=True)

        # Module score for fibroblast.
        mod_idx = MODULES.index(response)
        y = adata.obsm["module_scores"][:, mod_idx].astype(np.float32)

        # Source / NTC masks.
        source_mask = adata.obs["guide"].astype(str).str.contains(
            target_gene, case=False, na=False).values
        ntc_mask = adata.obs["is_ntc"].values

        n_source = int(source_mask.sum())
        n_ntc = int(ntc_mask.sum())
        print(f"  source={n_source}, ntc={n_ntc}", flush=True)
        if n_source == 0 or n_ntc == 0:
            print(f"  skip (insufficient bins)", flush=True)
            continue

        # Cosine-NN matched control.
        source_emb = embed[source_mask]
        ntc_emb = embed[ntc_mask]
        match_idx = cosine_nn_match(source_emb, ntc_emb)
        matched_y = y[ntc_mask][match_idx]
        source_y = y[source_mask]

        # Direct delta + permutation p.
        delta = float(source_y.mean() - matched_y.mean())
        se = float(np.sqrt(source_y.var()/len(source_y) + matched_y.var()/len(matched_y)))
        z = delta / (se + 1e-12)

        rng = np.random.default_rng(7)
        n_perm = 200
        all_y = np.concatenate([source_y, matched_y])
        n_s = len(source_y)
        perm_deltas = np.empty(n_perm)
        for i in range(n_perm):
            idx = rng.permutation(len(all_y))
            perm_deltas[i] = all_y[idx[:n_s]].mean() - all_y[idx[n_s:]].mean()
        p = float((np.sum(np.abs(perm_deltas) >= abs(delta)) + 1) / (n_perm + 1))

        # Pure NTC baseline delta (no matching).
        ntc_y = y[ntc_mask]
        ntc_delta = float(source_y.mean() - ntc_y.mean())

        # SMD between source and matched control embeddings.
        smd = smd_per_dim(source_emb, ntc_emb[match_idx])

        rows.append({
            "slice": slice_name,
            "gene": target_gene,
            "response": response,
            "n_source": n_source,
            "n_ntc": n_ntc,
            "delta_matched": round(delta, 4),
            "delta_ntc_baseline": round(ntc_delta, 4),
            "z_stat": round(z, 2),
            "perm_p": p,
            "smd_max": round(float(smd.max()), 4),
            "smd_mean": round(float(smd.mean()), 4),
        })
        print(f"  delta_matched={delta:+.4f} (v2 baseline=+0.62)", flush=True)
        print(f"  delta_ntc_baseline={ntc_delta:+.4f}", flush=True)
        print(f"  perm_p={p:.4f}, SMD max={smd.max():.4f}, SMD mean={smd.mean():.4f}", flush=True)

    out_path = OUT_PROCESSED / f"phgr1_fibroblast_egat_v1_bad.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
