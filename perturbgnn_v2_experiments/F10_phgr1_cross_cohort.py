"""F10: Phgr1 cross-cohort direction consistency.

For each Phgr1 response, plot δ across all 8 slices with cohort color.
Shows that 5/7 responses have consistent direction across both cohorts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    csvs = []
    for s in ["M001", "M002", "M003"]:
        p = PROCESSED / f"genome_scan_v2_{s}.csv" if s != "M002" else PROCESSED / "genome_scan_v2.csv"
        if p.exists():
            df = pd.read_csv(p); df["slice"] = s; df["cohort"] = "cohort1"; csvs.append(df)
    for s in ["subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]:
        p = PROCESSED / f"cohort2_{s}_scan.csv"
        if p.exists():
            df = pd.read_csv(p); df["slice"] = s; df["cohort"] = "cohort2"; csvs.append(df)
    merged = pd.concat(csvs, ignore_index=True)
    phgr1 = merged[merged["gene"] == "Phgr1"].copy()

    responses = ["score_ifn_response", "score_fibroblast", "score_hypoxia",
                 "score_macrophage", "score_cd8_like", "score_endothelial",
                 "score_malignant"]

    SLICE_ORDER = ["M001", "M002", "M003", "subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]
    phgr1["slice_order"] = phgr1["slice"].apply(lambda x: SLICE_ORDER.index(x) if x in SLICE_ORDER else 99)
    phgr1 = phgr1.sort_values("slice_order")

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, resp in enumerate(responses):
        sub = phgr1[phgr1["response"] == resp]
        if len(sub) == 0: continue
        x = np.arange(len(sub)) + i * 0.1  # slight offset per response
        # color by cohort
        colors = ["#1f77b4" if c == "cohort1" else "#d62728" for c in sub["cohort"]]
        ax.scatter(x, sub["durbin_delta"], c=colors, s=60, alpha=0.8,
                   edgecolors="black", linewidth=0.5)
        # connect with line
        ax.plot(x, sub["durbin_delta"], color="gray", alpha=0.3, lw=0.5)
        # label response at right
        ax.text(len(sub) + i * 0.1 + 0.3, sub["durbin_delta"].iloc[-1],
                resp.replace("score_", ""), fontsize=9, va="center")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Slice (M001-M003 = cohort1, subQ-1~5 = cohort2)")
    ax.set_ylabel("Durbin δ (Phgr1 propagation slope)")
    ax.set_title("Phgr1 cross-cohort direction consistency\n"
                 "Blue = cohort 1 (lung metastasis), Red = cohort 2 (subcutaneous primary)",
                 fontsize=12, fontweight="bold")
    # custom legend
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                     markersize=10, label='cohort 1 (M001-M003)'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                     markersize=10, label='cohort 2 (subQ-1~5)')]
    ax.legend(handles=legend, loc="best")
    ax.grid(alpha=0.3, axis="y")

    # x-tick labels
    n_slices_per_resp = phgr1.groupby("response").size().max()
    ax.set_xticks(range(n_slices_per_resp))
    ax.set_xticklabels(SLICE_ORDER[:n_slices_per_resp], rotation=30, ha="right")

    fig.tight_layout()
    out = FIGS / "F10_phgr1_cross_cohort.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
