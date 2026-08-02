"""F6: v2 causal landscape — gene × response significance vs effect size."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    df = pd.read_csv(PROCESSED / "genome_scan_v2_combined_3slices.csv")
    valid = df.dropna(subset=["mean_durbin_delta", "combined_q"]).copy()
    valid["log_q"] = -np.log10(valid["combined_q"].clip(lower=1e-300))

    highlight = ["Phgr1", "Utrn", "Cttn", "Ccn1", "Blnk"]
    colors = ["#d62728" if g in highlight else "#1f77b4" for g in valid["gene"]]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(valid["mean_durbin_delta"], valid["log_q"],
               c=colors, alpha=0.6, s=50)

    for g in highlight:
        sub = valid[valid["gene"] == g]
        if len(sub) > 0:
            top = sub.sort_values("combined_q").iloc[0]
            resp = top["response"].replace("score_", "")
            ax.annotate(f"{g}\n({resp})",
                        (top["mean_durbin_delta"], top["log_q"]),
                        fontsize=9, fontweight="bold",
                        xytext=(6, 5), textcoords="offset points")

    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", label="q=0.05")
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Mean Durbin δ across slices")
    ax.set_ylabel("-log10(combined q)")
    ax.set_title("v2 causal landscape — 3-slice combined\n"
                 "Red = highlighted (Phgr1/Utrn/Cttn replicate; Ccn1/Blnk single-slice)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGS / "F6_v2_landscape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
