"""v1 vs-NTC FDR framework on cohort 2 subQ-1, for direct comparison with v2.

Simplified v1 logic (adapted for cohort 2 which has no cell_type annotation):
  For each gene G + response Y:
    1. Compute Δ_gene = mean(Y near G source) - mean(Y far from any source)
    2. Build NTC null: sample size-matched NTC subsets, compute Δ_NTC
    3. z = (Δ_gene - mean_NTC) / std_NTC
    4. Stouffer + BH FDR across genes

This is the v1 framework's core: "is this gene different from NTC?"
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import json
import time
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import norm, false_discovery_control

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def vs_ntc_screen(adata, responses, near_um=40, far_um=200, n_null=200, min_sources=30):
    """Run v1-style vs-NTC FDR screen on one slice."""
    xy = adata.obsm["spatial"].astype(np.float32)
    obs = adata.obs

    # build cKDTree per target gene + NTC source set
    is_ntc_src = (obs["guide_total"] >= 2) & (obs["top_fraction"] >= 0.7) & obs["is_ntc"]
    ntc_xy = xy[is_ntc_src]
    print(f"  NTC source bins: {len(ntc_xy)}", flush=True)

    # global "far" mask: far from ANY source (use NTC + all sources as ref)
    is_any_source = (obs["guide_total"] >= 2) & (obs["top_fraction"] >= 0.7) & (~obs["is_ntc"])
    all_src_xy = xy[is_any_source | is_ntc_src]
    if len(all_src_xy) == 0:
        return pd.DataFrame()
    src_tree = cKDTree(all_src_xy)
    d_to_src, _ = src_tree.query(xy, k=1)
    far_mask = d_to_src > far_um

    rows = []
    # NTC null: for each gene's source count, sample N subsets of NTC of same size
    FOCUS_GENES = {"Phgr1", "Utrn", "Cttn", "Ccn1", "Blnk", "Rab8a", "Bcam",
                  "Bcam", "Iqgap1", "H2-Ea", "Abcc3", "Piezo1", "Gdf15", "Cd99",
                  "Cttn", "Gnas", "Phgr1", "H2-M2", "Cd74", "Abcc4", "Itgb2",
                  "Nfatc3", "Pgap6", "Mcoln1", "Unc93b1", "Sema4c", "Actg1",
                  "Stat1", "Atp13a2", "Rnd3", "Abi1", "Sdcbp2", "Muc20",
                  "Erbin", "Clcn2", "Ifitm3", "Spint1", "Abca2"}
    all_genes = obs.loc[(obs["guide_total"] >= 2) & (obs["top_fraction"] >= 0.7) & (~obs["is_ntc"]), "target_gene"].unique()
    gene_list = [g for g in all_genes if g in FOCUS_GENES]
    print(f"  focus genes: {len(gene_list)} of {len(all_genes)}", flush=True)
    for gene in gene_list:
        gene_mask = (obs["guide_total"] >= 2) & (obs["top_fraction"] >= 0.7) & (obs["target_gene"] == gene)
        n_src = int(gene_mask.sum())
        if n_src < min_sources:
            continue
        gene_xy = xy[gene_mask]
        gene_tree = cKDTree(gene_xy)
        d_gene, _ = gene_tree.query(xy, k=1)
        near_mask_gene = (d_gene <= near_um) & (~gene_mask)

        # for each response
        for resp in responses:
            Y = obs[resp].to_numpy() if resp in obs.columns else None
            if Y is None:
                continue
            valid = ~pd.isna(Y)
            Y = Y.astype(np.float32)

            if near_mask_gene.sum() < 30 or far_mask.sum() < 30:
                continue
            delta_gene = Y[near_mask_gene & valid].mean() - Y[far_mask & valid].mean()

            ntc_xy_arr = xy[is_ntc_src]
            if len(ntc_xy_arr) >= 30:
                ntc_tree_local = cKDTree(ntc_xy_arr)
                d_ntc_local, _ = ntc_tree_local.query(xy, k=1)
                near_ntc = (d_ntc_local <= near_um) & (~is_ntc_src)
                if near_ntc.sum() >= 30 and far_mask.sum() >= 30:
                    mu = float(Y[near_ntc & valid].mean() - Y[far_mask & valid].mean())
                    sd = float(np.sqrt(Y[near_ntc & valid].var() / max(near_ntc.sum(), 1) +
                                       Y[far_mask & valid].var() / max(far_mask.sum(), 1))) + 1e-8
                    z = (delta_gene - mu) / sd
                    p = float(2 * (1 - norm.cdf(abs(z))))
                else:
                    mu = np.nan; sd = np.nan; z = np.nan; p = np.nan
            else:
                mu = np.nan; sd = np.nan; z = np.nan; p = np.nan

            rows.append({
                "gene": gene, "response": resp,
                "n_sources": n_src,
                "delta_gene": delta_gene,
                "delta_ntc_mean": mu,
                "delta_ntc_std": sd,
                "excess": delta_gene - mu,
                "z": z, "p": p,
            })
    df = pd.DataFrame(rows)
    # BH FDR per response
    if "p" in df.columns and df["p"].notna().any():
        valid = df.dropna(subset=["p"]).copy()
        try:
            valid["q"] = false_discovery_control(valid["p"].values, method="bh")
        except Exception:
            ranks = valid["p"].rank(method="first")
            valid["q"] = (valid["p"] * len(valid) / ranks).clip(upper=1)
        df = df.merge(valid[["gene", "response", "q"]], on=["gene", "response"], how="left")
    return df


def main():
    sid = "subQ-1"
    print(f"[v1-v2] loading {sid}...", flush=True)
    a = ad.read_h5ad(PROCESSED / f"{sid}_v2.h5ad")
    # attach module scores as obs columns
    if "module_scores" in a.obsm:
        names = a.uns.get("module_names",
            ["malignant", "cd8_like", "macrophage", "fibroblast",
             "endothelial", "hypoxia", "ifn_response"])
        for i, n in enumerate(names):
            a.obs[f"score_{n}"] = a.obsm["module_scores"][:, i]
    print(f"  shape: {a.shape}, sources: {int(a.obs.is_source.sum())}", flush=True)

    responses = ["score_ifn_response", "score_fibroblast", "score_hypoxia",
                 "score_macrophage", "score_cd8_like", "score_endothelial",
                 "score_malignant"]

    t0 = time.time()
    df = vs_ntc_screen(a, responses, n_null=30)
    print(f"\nvs-NTC screen: {len(df)} rows in {time.time()-t0:.0f}s")

    out = PROCESSED / f"v1_vsntc_{sid}.csv"
    df.to_csv(out, index=False)
    print(f"wrote: {out}")

    # v2 cohort2 subQ-1 scan for comparison
    v2 = pd.read_csv(PROCESSED / f"cohort2_{sid}_scan.csv")
    # merge on (gene, response)
    merged = df.merge(v2[["gene", "response", "durbin_delta", "durbin_p"]],
                       on=["gene", "response"], how="outer",
                       suffixes=("_v1", "_v2"))
    merged.to_csv(PROCESSED / "v1_vs_v2_subQ1.csv", index=False)

    # categorize
    v1_sig = merged["q"] < 0.05
    v2_sig = merged["durbin_p"] < 0.05
    n_v1_only = (v1_sig & ~v2_sig.fillna(False)).sum()
    n_v2_only = (~v1_sig.fillna(True) & v2_sig).sum()
    n_both = (v1_sig & v2_sig).sum()
    print(f"\n=== v1 vs v2 on {sid} ===")
    print(f"  v1 only (q<0.05, v2 NS): {n_v1_only}")
    print(f"  v2 only (v2 p<0.05, v1 NS): {n_v2_only}")
    print(f"  both significant: {n_both}")

    # figure: scatter v1 z vs v2 δ
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    valid = merged.dropna(subset=["z", "durbin_delta"])
    # color by significance
    colors = []
    for _, r in valid.iterrows():
        v1 = pd.notna(r["q"]) and r["q"] < 0.05
        v2 = pd.notna(r["durbin_p"]) and r["durbin_p"] < 0.05
        if v1 and v2: colors.append("#2ca02c")  # both — true positives
        elif v1: colors.append("#d62728")        # v1 only — likely false positive
        elif v2: colors.append("#1f77b4")        # v2 only — v2 finds more
        else: colors.append("#7f7f7f")
    ax.scatter(valid["z"], valid["durbin_delta"], c=colors, alpha=0.7, s=60,
               edgecolors="black", linewidth=0.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    # highlight key genes
    for gene in ["Phgr1", "Bcam", "Rab8a", "Blnk", "Utrn", "Ccn1", "Cttn"]:
        sub = valid[valid["gene"] == gene]
        if len(sub) > 0:
            top = sub.iloc[0]
            ax.annotate(f"{gene}\n({top['response'].replace('score_','')})",
                        (top["z"], top["durbin_delta"]),
                        fontsize=9, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#2ca02c", label="Both significant"),
        Patch(facecolor="#d62728", label="v1 only (likely false positive)"),
        Patch(facecolor="#1f77b4", label="v2 only (more sensitive)"),
        Patch(facecolor="#7f7f7f", label="Neither"),
    ]
    ax.legend(handles=legend, loc="best")
    ax.set_xlabel("v1 vs-NTC z-score")
    ax.set_ylabel("v2 Durbin δ")
    ax.set_title(f"v1 (vs-NTC FDR) vs v2 (Durbin) on {sid}\n"
                 f"v1 only = {n_v1_only}, v2 only = {n_v2_only}, both = {n_both}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_fig = FIGS / "F17_v1_vs_v2.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"wrote: {out_fig}")


if __name__ == "__main__":
    main()
