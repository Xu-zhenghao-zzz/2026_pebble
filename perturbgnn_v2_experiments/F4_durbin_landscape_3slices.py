"""F4 cross-slice combined: gene × response heatmap with replication status.

Replaces the M002-only F4. Shows mean Durbin δ across slices + highlights
(gene, response) pairs that replicate (≥2 slices, direction consistent).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    df = pd.read_csv(PROCESSED / "genome_scan_v2_combined_3slices.csv")
    replicated = pd.read_csv(PROCESSED / "genome_scan_v2_replicated.csv")
    rep_keys = set(zip(replicated["gene"], replicated["response"]))

    # pivot: gene × response -> mean_durbin_delta
    pivot_delta = df.pivot_table(index="gene", columns="response",
                                  values="mean_durbin_delta", aggfunc="first")
    pivot_q = df.pivot_table(index="gene", columns="response",
                              values="combined_q", aggfunc="first")
    pivot_n = df.pivot_table(index="gene", columns="response",
                              values="n_slices", aggfunc="first")

    # sort genes by max abs delta
    max_abs = pivot_delta.abs().max(axis=1).sort_values(ascending=False)
    pivot_delta = pivot_delta.loc[max_abs.index]
    pivot_q = pivot_q.loc[max_abs.index]
    pivot_n = pivot_n.loc[max_abs.index]

    n_genes = len(pivot_delta)
    n_resps = len(pivot_delta.columns)

    fig, ax = plt.subplots(figsize=(11, max(7, n_genes * 0.35)))

    # mask NaN
    delta_masked = np.ma.masked_invalid(pivot_delta.values)

    im = ax.imshow(delta_masked, cmap="RdBu_r", aspect="auto",
                    vmin=-0.7, vmax=0.7)
    ax.set_xticks(range(n_resps))
    ax.set_xticklabels([c.replace("score_", "") for c in pivot_delta.columns],
                       rotation=45, ha="right")
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(pivot_delta.index, fontsize=8)

    # annotate: significance stars + replication highlight
    for i in range(n_genes):
        for j in range(n_resps):
            q = pivot_q.iloc[i, j]
            delta = pivot_delta.iloc[i, j]
            n_sl = pivot_n.iloc[i, j]
            gene = pivot_delta.index[i]
            resp = pivot_delta.columns[j]
            if pd.notna(q):
                if q < 0.001: star = "***"
                elif q < 0.01: star = "**"
                elif q < 0.05: star = "*"
                else: star = ""
                if star:
                    ax.text(j, i, star, ha="center", va="center",
                            fontsize=7, color="black")
            # replication box
            if (gene, resp) in rep_keys:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                          linewidth=2.5, edgecolor="gold",
                                          facecolor="none")
                ax.add_patch(rect)
            # n_slices label (small, top-right)
            if pd.notna(n_sl) and n_sl >= 2:
                ax.text(j+0.4, i-0.4, f"×{int(n_sl)}",
                        ha="right", va="top", fontsize=6, color="darkgreen",
                        fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.025, label="Mean Durbin δ (propagation slope)")
    ax.set_title("Cross-slice Durbin δ landscape — 3 slices combined\n"
                 "(*** p<0.001, ** p<0.01, * p<0.05 | gold=replicated ≥2 slices | "
                 "green ×N = N slices present)",
                 fontsize=11)
    ax.set_xlabel("Response")
    ax.set_ylabel("Target gene")

    fig.tight_layout()
    out = FIGS / "F4_durbin_landscape_3slices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
