"""F3: Matched control covariate balance + NTC sanity.

3 panels:
  (a) Covariate SMD before vs after matching
  (b) NTC matched Δ per response (should center near 0)
  (c) Effective sample size per analysis spot
"""
import sys
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")

from perturbgnn_v2.matching import MatchConfig, matched_control_for_gene, ntc_sanity_check
from perturbgnn_v2.matching.match import _attach_module_scores


def main():
    print("loading M002...", flush=True)
    a = ad.read_h5ad(PROCESSED / "M002_v2.h5ad")
    emb = np.load(PROCESSED / "embed_M002_v3.npy")
    _attach_module_scores(a)
    cfg = MatchConfig()

    print("running Ccn1 matched control + balance...", flush=True)
    df, bal = matched_control_for_gene(a, emb, "Ccn1", cfg)

    print("running NTC sanity...", flush=True)
    ntc_df = ntc_sanity_check(a, emb, cfg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) SMD before/after — we only have after; show as bar with threshold
    ax = axes[0]
    if len(bal) > 0:
        smds = bal["smd"].values
        names = bal["covariate"].values
        colors = ["#d62728" if abs(s) > 0.1 else "#2ca02c" for s in smds]
        x = np.arange(len(smds))
        ax.bar(x, smds, color=colors, alpha=0.7)
        ax.axhline(0.1, color="orange", linestyle="--", label="|SMD|=0.1")
        ax.axhline(-0.1, color="orange", linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_ylabel("Standardized mean difference (SMD)")
        ax.set_title(f"(a) Covariate balance after matching\n"
                     f"{(np.abs(smds)<0.1).sum()}/{len(smds)} within target")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    # (b) NTC Δ distribution per response
    ax = axes[1]
    if len(ntc_df) > 0:
        delta_cols = [c for c in ntc_df.columns if c.endswith("_delta")]
        resps = [c.replace("_delta", "").replace("score_", "") for c in delta_cols]
        means = [ntc_df[c].dropna().mean() for c in delta_cols]
        ses = [ntc_df[c].dropna().sem() for c in delta_cols]
        ax.bar(resps, means, yerr=[1.96 * s for s in ses], capsize=4,
               color="#9467bd", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticklabels(resps, rotation=30, ha="right")
        ax.set_ylabel("NTC matched Δ (95% CI)")
        ax.set_title(f"(b) NTC sanity check (should be ≈ 0)\n"
                     f"n={len(ntc_df)} NTC analysis spots")
        ax.grid(alpha=0.3, axis="y")

    # (c) matched control count distribution
    ax = axes[2]
    if len(df) > 0 and "n_matched" in df.columns:
        ax.hist(df["n_matched"], bins=20, color="#17becf", alpha=0.7)
        ax.axvline(10, color="red", linestyle="--", label="target K=10")
        ax.set_xlabel("# matched controls per analysis spot")
        ax.set_ylabel("count")
        ax.set_title(f"(c) Effective match count (Ccn1)\n"
                     f"median={int(df['n_matched'].median())}, "
                     f"mean={df['n_matched'].mean():.1f}")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Matched control quality (v2 Phase 2, M002 Ccn1)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F3_matched_control.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
