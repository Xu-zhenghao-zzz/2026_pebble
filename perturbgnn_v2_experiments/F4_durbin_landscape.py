"""Phase 5 Figure 4: Durbin δ landscape across genome.

Heatmap of Durbin δ (best λ) per (gene, response) for one slice.
Reveals the multi-response perturbation landscape.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")
FIGS.mkdir(exist_ok=True)


def main(slice_name="M002"):
    df = pd.read_csv(PROCESSED / "genome_scan_v2.csv")
    # filter to this slice
    if "slice" in df.columns:
        df = df[df["slice"] == slice_name]
    # pivot: gene × response → durbin_delta
    if "durbin_delta" not in df.columns:
        print(f"no durbin_delta column; cols={list(df.columns)}")
        return
    pivot_delta = df.pivot_table(index="gene", columns="response",
                                  values="durbin_delta", aggfunc="first")
    pivot_p = df.pivot_table(index="gene", columns="response",
                              values="durbin_p", aggfunc="first")

    # sort genes by max |delta| (most impactful first)
    max_abs = pivot_delta.abs().max(axis=1).sort_values(ascending=False)
    pivot_delta = pivot_delta.loc[max_abs.index]
    pivot_p = pivot_p.loc[max_abs.index]

    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, len(pivot_delta) * 0.4)),
                              gridspec_kw={"width_ratios": [3, 1]})

    # left: δ heatmap with significance stars
    ax = axes[0]
    im = ax.imshow(pivot_delta.values, cmap="RdBu_r", aspect="auto",
                    vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(pivot_delta.columns)))
    ax.set_xticklabels([c.replace("score_", "") for c in pivot_delta.columns],
                       rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_delta.index)))
    ax.set_yticklabels(pivot_delta.index, fontsize=9)
    # annotate with significance
    for i in range(len(pivot_delta.index)):
        for j in range(len(pivot_delta.columns)):
            p = pivot_p.iloc[i, j]
            if pd.notna(p):
                if p < 0.001:
                    star = "***"
                elif p < 0.01:
                    star = "**"
                elif p < 0.05:
                    star = "*"
                else:
                    star = ""
                if star:
                    ax.text(j, i, star, ha="center", va="center",
                            fontsize=8, color="black")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Durbin δ (propagation slope)")
    ax.set_title(f"{slice_name} — Durbin δ per (gene, response)\n"
                 f"(*** p<0.001, ** p<0.01, * p<0.05)")
    ax.set_xlabel("Response")
    ax.set_ylabel("Target gene")

    # right: combined_q bar (ranked)
    ax2 = axes[1]
    combined = pd.read_csv(PROCESSED / "genome_scan_v2_combined.csv")
    # use min q per gene
    gene_q = combined.groupby("gene")["combined_q"].min()
    gene_q = gene_q.reindex(pivot_delta.index)
    log_q = -np.log10(np.clip(gene_q.values.astype(float), 1e-300, None))
    log_q = np.nan_to_num(log_q, nan=0.0)
    colors = ["#d62728" if q < 0.05 else "#7f7f7f" for q in gene_q.values]
    ax2.barh(range(len(gene_q)), log_q, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(gene_q)))
    ax2.set_yticklabels([""] * len(gene_q))  # already on left
    ax2.axvline(-np.log10(0.05), color="red", linestyle="--", lw=1,
                label="q=0.05")
    ax2.set_xlabel("-log10(min combined q)")
    ax2.set_title("Significance")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.invert_yaxis()

    fig.tight_layout()
    out = FIGS / f"F4_durbin_landscape_{slice_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "M002"
    main(s)
