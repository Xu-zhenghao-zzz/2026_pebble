"""F12: Cross-cohort gene comparison (Phgr1 vs others).

For each focus gene, plot n_slices replicated (x) vs max |δ| (y).
Bubble size = TCGA correlation strength. Phgr1 dominates top-right.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    comb = pd.read_csv(PROCESSED / "cross_cohort_combined.csv")
    # per gene: max n_slices, max |delta|, min q
    rows = []
    for g, grp in comb.groupby("gene"):
        rows.append({
            "gene": g,
            "max_n_slices": int(grp["n_slices"].max()),
            "max_abs_delta": float(grp["mean_delta"].abs().max()),
            "min_q": float(grp["combined_q"].min()),
            "n_replicated": int(grp["combined_q"].lt(0.05).sum()),
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 7))
    # TCGA max |rho| per gene (best response)
    # phgr1: 0.473, bcam: 0.237, blnk: 0.42 (cohort1 paper)
    tcga_strength = {"Phgr1": 0.473, "Utrn": 0.0, "Cttn": 0.0,
                     "Ccn1": 0.0, "Blnk": 0.42, "Rab8a": 0.0, "Bcam": 0.237}
    sizes = [tcga_strength.get(g, 0.05) * 800 + 50 for g in df["gene"]]

    colors = []
    for g in df["gene"]:
        if g == "Phgr1": colors.append("#d62728")
        elif g in ["Bcam", "Rab8a"]: colors.append("#ff7f0e")
        elif g == "Blnk": colors.append("#7f7f7f")
        else: colors.append("#1f77b4")

    sc = ax.scatter(df["max_n_slices"], df["max_abs_delta"],
                    s=sizes, c=colors, alpha=0.7, edgecolors="black", linewidth=1.5)
    for _, row in df.iterrows():
        ax.annotate(row["gene"],
                    (row["max_n_slices"], row["max_abs_delta"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold")

    ax.set_xlabel("Max slices with significant replication")
    ax.set_ylabel("Max |Durbin δ| across responses")
    ax.set_title("Cross-cohort replication strength vs effect size\n"
                 "Bubble size = TCGA LUAD correlation strength (Phgr1 = 0.473)")
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5, label="|δ|=0.2 threshold")
    ax.axvline(4, color="gray", linestyle=":", alpha=0.5, label="4-slice threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGS / "F12_gene_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
