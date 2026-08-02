"""Fix 1 — Run AIPW doubly robust estimator on Phgr1 × 5 responses.

For each (slice, response):
  T = source-vs-matched-control indicator (from v2 match)
  X = [v2 embedding (64d), cell_type one-hot (8d), niche one-hot (12d),
       density, vessel_dist, 7 module scores (excluding y)]  → 93d
  y = module score for response

Compare τ̂_AIPW vs v2's Durbin δ.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/src")
sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
from perturbgnn_v2_1.causal.aipw import aipw_crossfit


PROCESSED_V2 = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
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


def main():
    rows = []
    for slice_name in ["M001", "M002", "M003"]:
        print(f"\n=== {slice_name} ===", flush=True)
        adata = ad.read_h5ad(PROCESSED_V2 / f"{slice_name}_v2.h5ad")
        embed_v2 = np.load(PROCESSED_V2 / f"embed_{slice_name}_v3.npy")
        print(f"  adata: {adata.shape}, embed_v2: {embed_v2.shape}", flush=True)

        # Source / NTC masks.
        source_mask = adata.obs["guide"].astype(str).str.contains(
            TARGET_GENE, case=False, na=False).values
        ntc_mask = adata.obs["is_ntc"].values
        n_source = int(source_mask.sum())
        n_ntc = int(ntc_mask.sum())
        print(f"  source={n_source}, ntc={n_ntc}", flush=True)
        if n_source == 0 or n_ntc == 0:
            continue

        # Cosine-NN matched control via v2 embedding.
        source_emb = embed_v2[source_mask]
        ntc_emb = embed_v2[ntc_mask]
        match_idx = cosine_nn_match(source_emb, ntc_emb)
        matched_global_idx = np.where(ntc_mask)[0][match_idx]

        # Build covariate matrix X for source + matched control.
        # All 8 module scores + cell_type + niche + density + vessel + 64d embed.
        all_module = adata.obsm["module_scores"]  # (N, 7)
        ct = adata.obs["cell_type"].astype(str).values
        ct_vocab = sorted(set(ct) - {"Missing"})
        ct_lookup = {c: i for i, c in enumerate(ct_vocab)}
        niche = adata.obs["niche_id"].values.astype(int)
        n_ct = len(ct_vocab)
        n_niche = int(niche.max() + 1)

        def build_X(global_indices):
            n = len(global_indices)
            X_emb = embed_v2[global_indices]  # (n, 64)
            X_mod = all_module[global_indices]  # (n, 7)
            ct_idx = np.array([ct_lookup.get(c, -1) for c in ct[global_indices]])
            ct_onehot = np.zeros((n, n_ct), dtype=np.float32)
            valid = ct_idx >= 0
            ct_onehot[valid, ct_idx[valid]] = 1.0
            niche_onehot = np.zeros((n, n_niche), dtype=np.float32)
            niche_onehot[np.arange(n), niche[global_indices]] = 1.0
            return np.column_stack([X_emb, X_mod, ct_onehot, niche_onehot]).astype(np.float32)

        X_source = build_X(np.where(source_mask)[0])
        X_match = build_X(matched_global_idx)

        for resp_idx, response in enumerate(MODULES):
            y_source = all_module[source_mask, resp_idx].astype(np.float32)
            y_match = all_module[matched_global_idx, resp_idx].astype(np.float32)
            # IMPORTANT: drop the y column from X to avoid leakage.
            drop_col = 64 + resp_idx  # module scores start at col 64
            Xs = np.delete(X_source, drop_col, axis=1)
            Xm = np.delete(X_match, drop_col, axis=1)

            X = np.concatenate([Xs, Xm], axis=0)
            T = np.concatenate([np.ones(len(y_source)), np.zeros(len(y_match))]).astype(np.int8)
            y = np.concatenate([y_source, y_match]).astype(np.float32)

            t0 = time.time()
            try:
                out = aipw_crossfit(X, T, y, n_splits=5, seed=7)
                dt = time.time() - t0
                row = {
                    "slice": slice_name,
                    "gene": TARGET_GENE,
                    "response": f"score_{response}",
                    "n_source": n_source,
                    "n_control": n_ntc,
                    "aipw_tau": round(out["tau_hat"], 4),
                    "aipw_se": round(out["se"], 4),
                    "aipw_ci_low": round(out["ci_low"], 4),
                    "aipw_ci_high": round(out["ci_high"], 4),
                    "v2_durbin_delta_ref": None,  # to fill from v2 scan
                    "compute_time_s": round(dt, 1),
                    "mean_propensity": round(out["mean_propensity"], 4),
                }
                rows.append(row)
                print(f"  {response}: τ̂={out['tau_hat']:+.4f} "
                      f"(95% CI [{out['ci_low']:+.3f}, {out['ci_high']:+.3f}]) "
                      f"prop={out['mean_propensity']:.3f} {dt:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"  {response}: FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)
                rows.append({
                    "slice": slice_name, "gene": TARGET_GENE,
                    "response": f"score_{response}", "error": str(e)[:200],
                })

    # Cross-reference with v2 Durbin delta for comparison.
    for slice_name in ["M001", "M002", "M003"]:
        scan_path = PROCESSED_V2 / f"genome_scan_v2_{slice_name}.csv"
        if not scan_path.exists():
            continue
        scan = pd.read_csv(scan_path)
        phgr1 = scan[scan["gene"].eq(TARGET_GENE)]
        for _, r in phgr1.iterrows():
            for row in rows:
                if (row.get("slice") == slice_name
                    and row.get("response") == r["response"]):
                    row["v2_durbin_delta_ref"] = round(r["durbin_delta"], 4)
                    row["v2_durbin_p_ref"] = round(r["durbin_p"], 6) if "durbin_p" in r else None

    df = pd.DataFrame(rows)
    out_path = OUT / "aipw_phgr1.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)
    print("\n=== Summary ===")
    print(df[["slice", "response", "aipw_tau", "aipw_ci_low", "aipw_ci_high",
              "v2_durbin_delta_ref"]].to_string(index=False))


if __name__ == "__main__":
    main()
