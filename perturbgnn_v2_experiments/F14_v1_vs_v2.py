"""F14: v1 (Blnk) vs v2 (Phgr1) — direct head-to-head.

Side-by-side: Blnk evidence strength vs Phgr1 evidence strength
across 4 dimensions (slices, response breadth, TCGA effect, Cox HR).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    dims = ["Cross-cohort\nslices", "Significant\nresponses",
            "TCGA LUAD\nmax ρ", "Cox HR\n(inverse)"]
    # Phgr1: 6 slices, 5 responses sig, max ρ=0.473, HR=0.556 (inv=1.80)
    # Blnk:  1 slice (M001 only), 1 response, max ρ=0.42, HR=0.744 (inv=1.34)
    phgr1 = [6, 5, 0.473, 1/0.556]
    blnk = [1, 1, 0.420, 1/0.744]
    # normalize to max
    maxes = [max(p, b) for p, b in zip(phgr1, blnk)]
    phgr1_norm = [p/m for p, m in zip(phgr1, maxes)]
    blnk_norm = [b/m for b, m in zip(blnk, maxes)]

    x = np.arange(len(dims))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, phgr1_norm, width, label="Phgr1 (v2 finding)",
                   color="#d62728", alpha=0.8)
    bars2 = ax.bar(x + width/2, blnk_norm, width, label="Blnk (v1 finding)",
                   color="#7f7f7f", alpha=0.8)

    # annotate actual values
    actual = [
        ["6 slices", "1 slice"],
        ["5 responses", "1 response"],
        ["ρ=0.473", "ρ=0.420"],
        ["HR=0.556", "HR=0.744"],
    ]
    for i, (p, b) in enumerate(zip(phgr1_norm, blnk_norm)):
        ax.text(i - width/2, p + 0.02, actual[i][0], ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, b + 0.02, actual[i][1], ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Normalized strength (relative to max)")
    ax.set_ylim(0, 1.3)
    ax.set_title("v1 Blnk vs v2 Phgr1: head-to-head evidence comparison\n"
                 "Phgr1 dominates on 3/4 dimensions; Blnk lacks cross-cohort replication",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = FIGS / "F14_v1_vs_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
