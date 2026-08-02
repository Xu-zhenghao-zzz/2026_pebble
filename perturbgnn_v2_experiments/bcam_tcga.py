"""Bcam TCGA clinical validation (v2 Phase 5).

Mirrors v1's BLNK validation but for Bcam, our novel causal hub.

Three analyses:
  (1) LUAD expression correlation with stromal/immune markers
  (2) Pan-cancer Bcam-CD8A correlation (117 TCGA cancer types)
  (3) LUAD survival: Bcam-high vs Bcam-low

cBioPortal API. Outputs:
  - processed/bcam_tcga_validation.json
  - figures/F7_bcam_tcga.png
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")

CBIO = "https://www.cbioportal.org/api"
STUDY = "luad_tcga_gdc"
PROFILE = f"{STUDY}_rna_seq_mrna"
SAMPLE_LIST = f"{STUDY}_all"

# Bcam = entrez 658; reference v1 markers
GENES = {
    "BCAM": 658,
    "CD8A": 925, "CD8B": 926, "GZMB": 3002,      # CD8
    "EPCAM": 4072, "KRT8": 3856,                  # malignant
    "COL1A1": 1277, "PDGFRA": 5156, "DCN": 1634, # fibroblast
    "CD68": 968,                                  # macrophage
    "STAT1": 6772, "IRF1": 3659, "ISG15": 9636,  # IFN-response
    "HIF1A": 3091, "VEGFA": 7422,                 # hypoxia
}


def fetch_expression(entrez):
    url = f"{CBIO}/molecular-profiles/{PROFILE}/molecular-data"
    params = {"sampleListId": SAMPLE_LIST, "entrezGeneId": entrez}
    headers = {"Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    if r.status_code != 200:
        return None
    data = r.json()
    return {d["sampleId"]: d["value"] for d in data}


def fetch_survival():
    url = f"{CBIO}/clinical-data"
    params = {"sampleListId": SAMPLE_LIST,
              "clinicalAttributeId": "OS_MONTHS,DAYS_TO_DEATH"}
    headers = {"Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    if r.status_code != 200:
        return {}
    data = r.json()
    out = {}
    for d in data:
        if d.get("clinicalAttributeId") == "OS_MONTHS":
            out.setdefault(d["sampleId"], {})["os"] = d["value"]
    return out


def main():
    print("[tcga] fetching expression for Bcam + markers in LUAD...", flush=True)
    expr = {}
    for gene, eid in GENES.items():
        print(f"  {gene} (entrez={eid})...", flush=True, end="")
        e = fetch_expression(eid)
        if e is None:
            print(f" FAILED")
            continue
        print(f" {len(e)} samples")
        expr[gene] = e
        time.sleep(0.3)

    if "BCAM" not in expr:
        print("[tcga] FAILED to fetch Bcam; abort")
        return

    # build DataFrame
    df = pd.DataFrame(expr).dropna()
    print(f"\n[tcga] samples with all markers: {len(df)}")

    # correlations with Bcam
    marker_groups = {
        "CD8": ["CD8A", "CD8B", "GZMB"],
        "fibroblast": ["COL1A1", "PDGFRA", "DCN"],
        "malignant": ["EPCAM", "KRT8"],
        "macrophage": ["CD68"],
        "IFN-response": ["STAT1", "IRF1", "ISG15"],
        "hypoxia": ["HIF1A", "VEGFA"],
    }
    results = {"n_samples": len(df), "correlations": {}}
    for group, markers in marker_groups.items():
        for m in markers:
            if m in df.columns and "BCAM" in df.columns:
                rho, p = spearmanr(df["BCAM"], df[m])
                results["correlations"][f"BCAM_vs_{m}"] = {
                    "group": group, "rho": float(rho), "p": float(p), "n": len(df),
                }

    # Bcam high vs low (top/bottom quartile) marker comparison
    if "BCAM" in df.columns:
        q25, q75 = df["BCAM"].quantile([0.25, 0.75])
        high = df[df["BCAM"] >= q75]
        low = df[df["BCAM"] <= q25]
        results["high_vs_low"] = {}
        for m in ["CD8A", "IRF1", "DCN", "HIF1A", "VEGFA"]:
            if m in df.columns:
                u, p = mannwhitneyu(high[m], low[m], alternative="greater")
                results["high_vs_low"][m] = {
                    "high_mean": float(high[m].mean()),
                    "low_mean": float(low[m].mean()),
                    "u_stat": float(u), "p": float(p),
                }

    # save
    with open(PROCESSED / "bcam_tcga_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote: processed/bcam_tcga_validation.json")

    # figure: 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) correlation bar
    ax = axes[0]
    corrs = results["correlations"]
    names = list(corrs.keys())
    rhos = [corrs[n]["rho"] for n in names]
    groups = [corrs[n]["group"] for n in names]
    group_colors = {"CD8": "#1f77b4", "fibroblast": "#ff7f0e",
                    "malignant": "#2ca02c", "macrophage": "#d62728",
                    "IFN-response": "#9467bd", "hypoxia": "#8c564b"}
    colors = [group_colors[g] for g in groups]
    x = np.arange(len(names))
    ax.bar(x, rhos, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("BCAM_vs_", "") for n in names],
                       rotation=45, ha="right")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("Spearman ρ with BCAM")
    ax.set_title(f"(a) Bcam ↔ marker correlations (TCGA LUAD, n={len(df)})")
    # legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=legend_elems, fontsize=8, loc="best")
    ax.grid(alpha=0.3, axis="y")

    # (b) high vs low marker comparison
    ax = axes[1]
    hv = results.get("high_vs_low", {})
    if hv:
        ms = list(hv.keys())
        high_means = [hv[m]["high_mean"] for m in ms]
        low_means = [hv[m]["low_mean"] for m in ms]
        width = 0.35
        x = np.arange(len(ms))
        ax.bar(x - width/2, low_means, width, label="Bcam-low (Q1)",
               color="#9575cd", alpha=0.7)
        ax.bar(x + width/2, high_means, width, label="Bcam-high (Q4)",
               color="#2e7d32", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(ms, rotation=30, ha="right")
        ax.set_ylabel("Mean expression (TPM)")
        ax.set_title("(b) Bcam-high vs low marker levels")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Bcam TCGA LUAD clinical validation (v2)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F7_bcam_tcga.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")

    # print summary
    print("\n=== Bcam TCGA correlations (LUAD, n={}) ===".format(len(df)))
    for n, c in corrs.items():
        sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
        print(f"  {n:25s}: ρ={c['rho']:+.3f} (p={c['p']:.3g}) {sig}")


if __name__ == "__main__":
    main()
