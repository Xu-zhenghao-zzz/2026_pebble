"""F15: Phgr1 Durbin δ vs kernel λ curves (dose-response decay).

How does Phgr1's effect decay with spatial scale? Steep decay = local effect,
slow decay = long-range signaling.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    # use Phgr1 full diagnostics on M001 + M002 (has durbin_by_lambda)
    diag = json.load(open(PROCESSED / "phgr1_full_diagnostics.json"))

    responses = ["score_ifn_response", "score_fibroblast", "score_hypoxia",
                 "score_macrophage", "score_endothelial"]
    colors = ["#9467bd", "#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax_idx, (slice_name, ax) in enumerate(zip(["M001", "M002"], axes)):
        for r, color in zip(responses, colors):
            entry = diag.get(slice_name, {}).get(r, {})
            if "durbin_by_lambda" not in entry:
                continue
            by_lam = entry["durbin_by_lambda"]
            lams = sorted([int(k) for k in by_lam.keys()])
            deltas = [by_lam[str(l)]["delta"] if str(l) in by_lam else by_lam[l]["delta"]
                      for l in lams]
            ps = [by_lam[str(l)]["p"] if str(l) in by_lam else by_lam[l]["p"]
                  for l in lams]
            ax.plot(lams, deltas, marker="o", color=color, linewidth=2,
                    label=r.replace("score_", ""))
            # mark significance with star
            for lam, d, p in zip(lams, deltas, ps):
                if p < 0.001: star = "***"
                elif p < 0.01: star = "**"
                elif p < 0.05: star = "*"
                else: star = ""
                if star:
                    ax.text(lam, d, star, fontsize=9, ha="center", va="bottom")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Kernel λ (µm)")
        if ax_idx == 0:
            ax.set_ylabel("Durbin δ (propagation slope)")
        ax.set_title(f"{slice_name} (cohort 1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Phgr1 Durbin δ vs kernel length scale (dose-response decay)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F15_phgr1_decay_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
