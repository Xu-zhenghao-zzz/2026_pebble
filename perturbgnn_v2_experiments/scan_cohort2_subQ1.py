"""Quick causal scan on subQ-1 (cohort 2) using X_pca as covariate."""
import sys, time, os
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import anndata as ad
import numpy as np

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")

from perturbgnn_v2.spatial import fit_durbin, DurbinConfig
from perturbgnn_v2.causal import CausalConfig, spatial_did
from perturbgnn_v2.matching.match import _attach_module_scores

# focus genes for cohort 2 (top + cohort-1 overlap)
GENES = ["Phgr1", "Utrn", "Cttn", "Ccn1", "Blnk", "Rab8a", "Bcam"]
RESPONSES = ["score_ifn_response", "score_fibroblast", "score_hypoxia",
             "score_macrophage", "score_cd8_like", "score_endothelial",
             "score_malignant"]


def main():
    sid = "subQ-1"
    print(f"[scan-c2] loading {sid}...", flush=True)
    a = ad.read_h5ad(PROCESSED / f"{sid}_v2.h5ad")
    _attach_module_scores(a)
    # use X_pca as covariate (cohort 2 has no trained embedding yet)
    emb = a.obsm["X_pca"].astype(np.float32)
    print(f"  shape: {a.shape}, sources: {int(a.obs.is_source.sum())}")
    print(f"  embedding (X_pca) shape: {emb.shape}")

    rows = []
    for gene in GENES:
        n_src = int((a.obs.target_gene == gene).sum())
        if n_src < 30:
            print(f"\n[gene] {gene}: skipped (only {n_src} sources)"); continue
        print(f"\n[gene] {gene}: {n_src} sources")
        for resp in RESPONSES:
            row = {"slice": sid, "gene": gene, "response": resp,
                   "n_sources": n_src}
            # Durbin
            try:
                d = fit_durbin(a, emb, gene, resp, DurbinConfig())
                if "error" not in d:
                    b = d["best"]
                    row["durbin_delta"] = b["delta"]
                    row["durbin_p"] = b["delta_p"]
                    row["durbin_lambda"] = b["lambda_um"]
                    row["n_clones"] = d["n_clones"]
            except Exception as e:
                row["durbin_err"] = str(e)[:60]
            # DiD
            try:
                did = spatial_did(a, emb, gene, resp, CausalConfig())
                if "error" not in did:
                    row["did_delta"] = did["delta_did"]
                    row["did_p"] = did["p"]
            except Exception as e:
                row["did_err"] = str(e)[:60]
            rows.append(row)
            if "durbin_delta" in row:
                star = "***" if row["durbin_p"] < 0.001 else "**" if row["durbin_p"] < 0.01 else "*" if row["durbin_p"] < 0.05 else ""
                print(f"  {resp:25s} δ={row['durbin_delta']:+.3f} (p={row['durbin_p']:.3g}) {star}")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = PROCESSED / f"cohort2_{sid}_scan.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote: {out}")
    print(f"\n=== significant (q<0.05 after BH) ===")
    if "durbin_p" in df.columns:
        valid = df.dropna(subset=["durbin_p"]).copy()
        if len(valid) > 0:
            from scipy.stats import false_discovery_control
            try:
                valid["q"] = false_discovery_control(valid["durbin_p"].values, method="bh")
            except Exception:
                # manual BH
                ranks = valid["durbin_p"].rank(method="first")
                valid["q"] = (valid["durbin_p"] * len(valid) / ranks).clip(upper=1)
            sig = valid[valid["q"] < 0.05].sort_values("q")
            print(sig[["gene", "response", "durbin_delta", "durbin_p", "q"]].to_string(index=False))


if __name__ == "__main__":
    main()
