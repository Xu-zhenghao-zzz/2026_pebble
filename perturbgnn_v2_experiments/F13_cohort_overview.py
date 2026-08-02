"""F13: Cohort data overview — 8 slices × sample counts.

Bar chart of bins / sources / NTC per slice, color-coded by cohort.
"""
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    slices = ["M001", "M002", "M003", "subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]
    bins = []; sources = []; ntcs = []
    for s in slices:
        p = PROCESSED / f"{s}_v2.h5ad"
        try:
            a = ad.read_h5ad(p, backed="r")
            bins.append(a.n_obs)
            sources.append(int(a.obs.is_source.sum()))
            ntcs.append(int(a.obs.is_ntc.sum()))
            a.file.close()
        except Exception as e:
            print(f"{s}: {e}")
            bins.append(0); sources.append(0); ntcs.append(0)

    x = np.arange(len(slices))
    width = 0.27
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, bins, width, label="total bins",
           color=["#1f77b4" if s.startswith("M") else "#d62728" for s in slices], alpha=0.4)
    ax.bar(x, sources, width, label="source bins (guide+)", color="#2ca02c", alpha=0.8)
    ax.bar(x + width, ntcs, width, label="NTC bins", color="#ff7f0e", alpha=0.9)

    ax.set_yscale("log")
    ax.set_xticks(x)
    cohort_labels = [f"{s}\n(c1)" if s.startswith("M") else f"{s}\n(c2)" for s in slices]
    ax.set_xticklabels(cohort_labels)
    ax.set_ylabel("Count (log scale)")
    ax.set_title("SPAC-seq cohort overview: 8 slices, 4.13M bins total\n"
                 "cohort 1 (M001-M003, lung metastasis) + cohort 2 (subQ-1~5, subcutaneous primary)",
                 fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    # annotate totals
    total_bins = sum(bins); total_src = sum(sources); total_ntc = sum(ntcs)
    ax.text(0.02, 0.95, f"Totals: {total_bins:,} bins | {total_src:,} sources | {total_ntc:,} NTC",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    out = FIGS / "F13_cohort_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
