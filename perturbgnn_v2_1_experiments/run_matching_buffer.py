"""Fix 4 — Matched control with 80µm spatial buffer.

For each Phgr1 source bin, exclude NTC bins within 80µm of any source
of any gene (not just Phgr1), then cosine-match in v2 embedding.
Compare δ + SMD with vs without buffer.

Acceptance: Phgr1 δ sign holds across 5 responses; pool size drops
no more than 50%.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
OUT = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/processed/sensitivity")
OUT.mkdir(parents=True, exist_ok=True)

MODULES = ["malignant", "cd8_like", "macrophage", "fibroblast",
           "endothelial", "hypoxia", "ifn_response"]
TARGET_GENE = "Phgr1"


def cosine_nn_match(source_emb, pool_emb):
    s = source_emb / (np.linalg.norm(source_emb, axis=1, keepdims=True) + 1e-12)
    p = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-12)
    sim = s @ p.T
    return sim.argmax(axis=1)


def smd_per_dim(a, b):
    pooled_std = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2) + 1e-12
    return np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled_std


def main():
    buffer_um = 80.0
    rows = []
    for slice_name in ["M001", "M002", "M003"]:
        print(f"\n=== {slice_name} ===", flush=True)
        adata = ad.read_h5ad(PROCESSED / f"{slice_name}_v2.h5ad")
        embed = np.load(PROCESSED / f"embed_{slice_name}_v3.npy")
        xy = adata.obs[["x_um", "y_um"]].values.astype(np.float32)

        # All-source mask (any gene).
        any_source_mask = adata.obs["is_source"].values
        ntc_mask = adata.obs["is_ntc"].values
        target_source_mask = adata.obs["guide"].astype(str).str.contains(
            TARGET_GENE, case=False, na=False).values

        n_target_src = int(target_source_mask.sum())
        n_ntc_orig = int(ntc_mask.sum())
        print(f"  Phgr1 source: {n_target_src}, NTC: {n_ntc_orig}", flush=True)
        if n_target_src == 0 or n_ntc_orig == 0:
            continue

        # Build cKDTree on any-source coordinates.
        any_src_xy = xy[any_source_mask]
        src_tree = cKDTree(any_src_xy)

        # For each NTC, distance to nearest any-source bin.
        ntc_xy = xy[ntc_mask]
        dist_to_src, _ = src_tree.query(ntc_xy, k=1)
        keep = dist_to_src >= buffer_um
        n_kept = int(keep.sum())
        print(f"  NTC after buffer ≥{buffer_um}µm: {n_kept}/{n_ntc_orig} "
              f"({100*n_kept/n_ntc_orig:.1f}%)", flush=True)
        if n_kept < 10:
            print(f"  skip (insufficient buffered NTC)", flush=True)
            continue

        # Filtered NTC mask (global index).
        ntc_global_idx = np.where(ntc_mask)[0]
        kept_global_idx = ntc_global_idx[keep]
        # New filtered ntc mask (in xy/embed space).
        filtered_ntc_mask = np.zeros(len(xy), dtype=bool)
        filtered_ntc_mask[kept_global_idx] = True

        # Match: Phgr1 source → kept NTC.
        source_emb = embed[target_source_mask]
        ntc_emb_filtered = embed[filtered_ntc_mask]
        match_idx = cosine_nn_match(source_emb, ntc_emb_filtered)
        matched_global_idx = kept_global_idx[match_idx]

        # For each response: compute δ + perm p.
        for resp_idx, response in enumerate(MODULES):
            y = adata.obsm["module_scores"][:, resp_idx].astype(np.float32)
            source_y = y[target_source_mask]
            matched_y = y[matched_global_idx]
            delta = float(source_y.mean() - matched_y.mean())

            # Perm p.
            rng = np.random.default_rng(7)
            n_perm = 200
            all_y = np.concatenate([source_y, matched_y])
            n_s = len(source_y)
            perm_deltas = np.empty(n_perm)
            for i in range(n_perm):
                idx = rng.permutation(len(all_y))
                perm_deltas[i] = all_y[idx[:n_s]].mean() - all_y[idx[n_s:]].mean()
            p = float((np.sum(np.abs(perm_deltas) >= abs(delta)) + 1) / (n_perm + 1))

            smd = smd_per_dim(source_emb, ntc_emb_filtered[match_idx])

            rows.append({
                "slice": slice_name,
                "response": f"score_{response}",
                "buffer_um": buffer_um,
                "n_source": n_target_src,
                "n_ntc_orig": n_ntc_orig,
                "n_ntc_buffered": n_kept,
                "ntc_retained_pct": round(100*n_kept/n_ntc_orig, 1),
                "delta_buffered": round(delta, 4),
                "perm_p": p,
                "smd_max": round(float(smd.max()), 4),
                "smd_mean": round(float(smd.mean()), 4),
            })

    df = pd.DataFrame(rows)
    out_path = OUT / "matching_buffer_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)

    # Print summary.
    print("\n=== Summary (delta_buffered vs v2 baseline) ===")
    v2_baseline = {
        ("M001", "score_fibroblast"): +0.62,
        ("M002", "score_fibroblast"): +0.48,
    }
    for slice_name in ["M001", "M002", "M003"]:
        sdf = df[df.slice.eq(slice_name)]
        print(f"\n{slice_name}:")
        for _, r in sdf.iterrows():
            base = v2_baseline.get((slice_name, r["response"]), None)
            base_str = f" (v2 baseline δ={base:+.2f})" if base else ""
            print(f"  {r['response']:24} δ_buf={r['delta_buffered']:+.4f}"
                  f"  SMD_max={r['smd_max']:.3f}  p={r['perm_p']:.4f}"
                  f"  NTC_kept={r['ntc_retained_pct']:.1f}%{base_str}")


if __name__ == "__main__":
    main()
