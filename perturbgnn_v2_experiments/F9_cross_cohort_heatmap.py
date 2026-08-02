"""F9 simplified: cross-cohort replication heatmap, no fancy sorting."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    csvs = []
    for s in ["M001", "M002", "M003"]:
        p = PROCESSED / f"genome_scan_v2_{s}.csv" if s != "M002" else PROCESSED / "genome_scan_v2.csv"
        if p.exists():
            df = pd.read_csv(p); df["slice"] = s; csvs.append(df)
    for s in ["subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]:
        p = PROCESSED / f"cohort2_{s}_scan.csv"
        if p.exists():
            df = pd.read_csv(p); df["slice"] = s; csvs.append(df)
    merged = pd.concat(csvs, ignore_index=True)
    FOCUS = ["Phgr1", "Utrn", "Cttn", "Ccn1", "Blnk", "Rab8a", "Bcam"]
    merged = merged[merged["gene"].isin(FOCUS)].copy()

    repl = pd.read_csv(PROCESSED / "cross_cohort_replicated.csv")
    repl_keys = set(zip(repl["gene"], repl["response"]))

    SLICE_ORDER = ["M001", "M002", "M003", "subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]
    # build matrix manually
    gene_resp_pairs = []
    for g in FOCUS:
        sub = merged[merged["gene"] == g]
        for r in sorted(sub["response"].unique()):
            gene_resp_pairs.append((g, r))

    # reorder: replicated first
    pairs_repl = [p for p in gene_resp_pairs if p in repl_keys]
    pairs_other = [p for p in gene_resp_pairs if p not in repl_keys]
    # sort others by max |delta|
    def max_abs(pair):
        g, r = pair
        sub = merged[(merged["gene"] == g) & (merged["response"] == r)]
        return sub["durbin_delta"].abs().max() if len(sub) > 0 else 0
    pairs_other.sort(key=max_abs, reverse=True)
    ordered = pairs_repl + pairs_other

    nrows = len(ordered)
    ncols = len(SLICE_ORDER)
    delta_mat = np.full((nrows, ncols), np.nan)
    p_mat = np.full((nrows, ncols), np.nan)
    for i, (g, r) in enumerate(ordered):
        sub = merged[(merged["gene"] == g) & (merged["response"] == r)]
        for j, s in enumerate(SLICE_ORDER):
            row = sub[sub["slice"] == s]
            if len(row) > 0:
                delta_mat[i, j] = row["durbin_delta"].iloc[0]
                p_mat[i, j] = row["durbin_p"].iloc[0]

    fig, ax = plt.subplots(figsize=(13, max(7, nrows * 0.35)))
    masked = np.ma.masked_invalid(delta_mat)
    im = ax.imshow(masked, cmap="RdBu_r", aspect="auto", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(ncols))
    cohort_labels = []
    for c in SLICE_ORDER:
        if c.startswith("subQ"):
            cohort_labels.append(f"{c}\n(c2)")
        else:
            cohort_labels.append(f"{c}\n(c1)")
    ax.set_xticklabels(cohort_labels, fontsize=9)
    ax.set_yticks(range(nrows))
    labels = [f"{g} | {r.replace('score_', '')}{'  ★' if (g,r) in repl_keys else ''}"
              for g, r in ordered]
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(delta_mat[i, j]):
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                facecolor="#f0f0f0", edgecolor="none"))
                continue
            p = p_mat[i, j]
            if p < 0.001: star = "***"
            elif p < 0.01: star = "**"
            elif p < 0.05: star = "*"
            else: star = ""
            if star:
                ax.text(j, i, star, ha="center", va="center", fontsize=7, color="black")
        if ordered[i] in repl_keys:
            ax.add_patch(patches.Rectangle((-0.5, i-0.5), ncols, 1,
                                           fill=False, edgecolor="gold", linewidth=2.5))

    plt.colorbar(im, ax=ax, fraction=0.025, label="Durbin δ (propagation slope)")
    ax.set_title("Cross-cohort Durbin δ replication heatmap (★ = cross-cohort replicated)\n"
                 "*** p<0.001, ** p<0.01, * p<0.05",
                 fontsize=11)
    ax.set_xlabel("Slice")
    ax.set_ylabel("Gene | Response")
    fig.tight_layout()
    out = FIGS / "F9_cross_cohort_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
