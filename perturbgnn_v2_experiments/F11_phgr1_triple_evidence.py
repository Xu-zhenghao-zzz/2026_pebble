"""F11: Phgr1 triple-evidence summary.

Three-panel figure showing:
  (a) SPAC-seq: Phgr1 Durbin δ per response across 2 cohorts
  (b) TCGA LUAD: PHGR1 ↔ marker Spearman ρ per compartment
  (c) Survival: KM curve Phgr1-high vs low

Demonstrates mouse perturbation → human cancer concordance.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    # gather SPAC-seq Phgr1 data
    csvs = []
    for s in ["M001", "M002", "M003"]:
        p = PROCESSED / f"genome_scan_v2_{s}.csv" if s != "M002" else PROCESSED / "genome_scan_v2.csv"
        if p.exists():
            df = pd.read_csv(p); df["slice"] = s; df["cohort"] = "c1"; csvs.append(df)
    for s in ["subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]:
        p = PROCESSED / f"cohort2_{s}_scan.csv"
        if p.exists():
            df = pd.read_csv(p); df["slice"] = s; df["cohort"] = "c2"; csvs.append(df)
    merged = pd.concat(csvs, ignore_index=True)
    phgr1 = merged[merged["gene"] == "Phgr1"].copy()

    tcga = json.load(open(PROCESSED / "phgr1_tcga_validation.json"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # (a) SPAC-seq Phgr1 δ per response, both cohorts
    ax = axes[0]
    responses = ["score_ifn_response", "score_fibroblast", "score_hypoxia",
                 "score_macrophage", "score_cd8_like", "score_endothelial"]
    resps_short = [r.replace("score_", "") for r in responses]
    x = np.arange(len(responses))
    width = 0.35
    c1_deltas = []; c2_deltas = []
    for r in responses:
        sub_c1 = phgr1[(phgr1["response"] == r) & (phgr1["cohort"] == "c1")]
        sub_c2 = phgr1[(phgr1["response"] == r) & (phgr1["cohort"] == "c2")]
        c1_deltas.append(sub_c1["durbin_delta"].mean() if len(sub_c1) > 0 else 0)
        c2_deltas.append(sub_c2["durbin_delta"].mean() if len(sub_c2) > 0 else 0)
    ax.bar(x - width/2, c1_deltas, width, label="cohort 1 (lung mets)",
           color="#1f77b4", alpha=0.8)
    ax.bar(x + width/2, c2_deltas, width, label="cohort 2 (subQ primary)",
           color="#d62728", alpha=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(resps_short, rotation=30, ha="right")
    ax.set_ylabel("Mean Durbin δ (Phgr1 propagation slope)")
    ax.set_title("(a) SPAC-seq: Phgr1 δ per response\n"
                 "(cross-cohort direction consistency)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (b) TCGA marker correlations
    ax = axes[1]
    corrs = tcga["correlations"]
    markers = ["IRF1", "STAT1", "ISG15", "CD68", "CD8A", "CD8B",
               "DCN", "PDGFRA", "COL1A1", "HIF1A", "VEGFA", "GZMB"]
    comp_map = {"IRF1": "IFN", "STAT1": "IFN", "ISG15": "IFN",
                "CD68": "Macrophage", "CD8A": "CD8", "CD8B": "CD8", "GZMB": "CD8",
                "DCN": "Fibroblast", "PDGFRA": "Fibroblast", "COL1A1": "Fibroblast",
                "HIF1A": "Hypoxia", "VEGFA": "Hypoxia"}
    comp_colors = {"IFN": "#9467bd", "Macrophage": "#d62728",
                   "CD8": "#1f77b4", "Fibroblast": "#ff7f0e",
                   "Hypoxia": "#2ca02c"}
    rhos = [corrs[f"PHGR1_vs_{m}"]["rho"] for m in markers]
    ps = [corrs[f"PHGR1_vs_{m}"]["p"] for m in markers]
    colors = [comp_colors[comp_map[m]] for m in markers]
    y = np.arange(len(markers))
    ax.barh(y, rhos, color=colors, alpha=0.8)
    for i, (r, p) in enumerate(zip(rhos, ps)):
        if p < 0.001: star = "***"
        elif p < 0.01: star = "**"
        elif p < 0.05: star = "*"
        else: star = ""
        ax.text(r + 0.01, i, f"{r:+.2f} {star}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(markers)
    ax.axvline(0, color="gray", lw=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ with PHGR1 (TCGA LUAD)")
    ax.set_title("(b) TCGA LUAD (n=516): PHGR1 marker correlations")
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=g) for g, c in comp_colors.items()]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    # (c) Survival KM curve (simplified from F8)
    ax = axes[2]
    surv = json.load(open(PROCESSED / "phgr1_survival.json"))
    # need original KM data — use survival json for boxplot of OS by group instead
    groups = ["Phgr1-low\n(Q1)", "Phgr1-high\n(Q4)"]
    med_os = [surv["median_os_low"], surv["median_os_high"]]
    n_groups = [surv["n_low"], surv["n_high"]]
    bars = ax.bar(groups, med_os, color=["#9575cd", "#2e7d32"], alpha=0.8)
    for i, (b, n, m) in enumerate(zip(bars, n_groups, med_os)):
        ax.text(b.get_x() + b.get_width()/2, m + 0.4, f"{m:.1f} mo\n(n={n})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Median Overall Survival (months)")
    ax.set_ylim(0, max(med_os) * 1.3)
    ax.set_title(f"(c) Survival: Phgr1 high vs low\n"
                 f"log-rank p={surv['logrank_p']:.4g}, "
                 f"Cox HR={surv['cox_uni_HR']:.2f}")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Phgr1: triple-evidence for multi-response NCA causal hub",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F11_phgr1_triple_evidence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
