"""Cohort 3: T cell perturbation time-course analysis.

For each timepoint (Day4/7/10) and each guide, compute:
  - cd8_like module score in source bins (guide+) vs NTC bins
  - ifn_response module score (secondary response)
  - Track how perturbation effect evolves Day4 → Day7 → Day10

This is descriptive (no Durbin), focusing on time dynamics.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import re
import json
import time
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")
SRC = Path("/mnt/data/xuzh/spac_seq/downloads/spatiotemporal")


MODULES = {
    "cd8_like": ["Cd3d", "Cd3e", "Cd247", "Cd8a", "Cd8b1", "Trac"],
    "ifn_response": ["Stat1", "Irf1", "Isg15", "Ifit1", "Ifit3", "Cxcl10"],
    "malignant": ["Epcam", "Krt8", "Krt18", "Krt19", "Krt20", "Krt7"],
}
NTC_PATTERN = re.compile(r"(?i)^sg(non[-_]?target|ntc)")


def module_scores(rna, modules):
    var = rna.var_names
    X = rna.X
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32)
        sums = np.asarray(X.mean(axis=0)).ravel()
        sumsq = np.asarray((X.multiply(X)).mean(axis=0)).ravel()
    else:
        X = X.astype(np.float32)
        sums = X.mean(axis=0); sumsq = (X * X).mean(axis=0)
    std = np.sqrt(np.clip(sumsq - sums**2, 0, None)) + 1e-8
    scores = {}
    for m, genes in modules.items():
        present = [g for g in genes if g in var]
        if not present:
            scores[m] = np.zeros(rna.n_obs, dtype=np.float32); continue
        idx = np.array([var.get_loc(g) for g in present])
        sub = X[:, idx].toarray() if sp.issparse(X) else X[:, idx]
        z = (sub - sums[idx]) / std[idx]
        scores[m] = z.mean(axis=1).astype(np.float32)
    return scores


def analyze_one(timepoint, rep):
    """Returns DataFrame: per-guide mean score in source vs NTC + p-value."""
    sid = f"{timepoint}_{rep}"
    print(f"\n[c3] === {sid} ===", flush=True)
    rna_path = SRC / "Transcriptome" / "Transcriptome" / f"{timepoint}_{rep}.h5"
    guide_path = SRC / "Perturbation" / "Perturbation" / f"{timepoint}_{rep}.guide.h5"
    if not rna_path.exists() or not guide_path.exists():
        return None

    rna = sc.read_h5ad(str(rna_path))
    rna.var_names_make_unique()
    guide = sc.read_h5ad(str(guide_path))
    guide.var_names_make_unique()
    print(f"  RNA {rna.shape}, guide {guide.shape}", flush=True)

    # align by barcode (they should match exactly)
    common = rna.obs_names.intersection(guide.obs_names)
    rna = rna[common].copy()
    guide = guide[common].copy()

    # normalize + log
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)

    scores = module_scores(rna, MODULES)

    # parse guide calls
    Xg = guide.X
    if not sp.issparse(Xg): Xg = sp.csr_matrix(Xg)
    Xg = Xg.tocsr().astype(np.float32)
    total = np.asarray(Xg.sum(axis=1)).ravel()
    var = guide.var_names.astype(str)
    has = total >= 1
    top_idx = np.full(Xg.shape[0], -1, dtype=np.int64)
    if has.any():
        top_idx[has] = np.asarray(Xg[has].argmax(axis=1)).ravel()

    # source = guide_total >= 2 AND top_frac >= 0.7
    nz = has & (top_idx >= 0)
    top_guide = np.array([""] * Xg.shape[0], dtype=object)
    top_frac = np.zeros(Xg.shape[0], dtype=np.float32)
    if nz.any():
        rows = np.where(nz)[0]; cols = top_idx[rows]
        top_guide[rows] = var[cols]
        cell = Xg[rows, cols]
        max_vals = cell.A1 if hasattr(cell, "A1") else np.asarray(cell).ravel()
        top_frac[rows] = max_vals / np.maximum(total[rows], 1e-6)
    is_source = (total >= 2) & (top_frac >= 0.7)

    # per-guide rows
    rows = []
    for gi, gname in enumerate(var):
        is_ntc = bool(NTC_PATTERN.match(gname))
        if is_ntc:
            mask = (top_guide == gname) & is_source
            label = "NTC"
        else:
            mask = (top_guide == gname) & is_source
            label = gname
            if not gname.startswith("sg"): continue
        n_src = int(mask.sum())
        if n_src < 5: continue
        # NTC background (all NTC source bins)
        ntc_mask = np.array([bool(NTC_PATTERN.match(t)) if t else False for t in top_guide]) & is_source
        for module, sc_arr in scores.items():
            y_src = sc_arr[mask]
            y_ntc = sc_arr[ntc_mask]
            if len(y_ntc) < 5: continue
            try:
                u, p = mannwhitneyu(y_src, y_ntc, alternative="two-sided")
            except Exception:
                p = np.nan
            rows.append({
                "timepoint": timepoint, "rep": rep,
                "guide": label, "module": module,
                "n_source": n_src,
                "mean_source": float(y_src.mean()),
                "mean_ntc": float(y_ntc.mean()),
                "delta": float(y_src.mean() - y_ntc.mean()),
                "p": float(p),
            })
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for tp in ["Day4", "Day7", "Day10"]:
        for rep in ["rep1", "rep2"]:
            df = analyze_one(tp, rep)
            if df is not None:
                all_rows.append(df)
                print(f"  {tp}_{rep}: {len(df)} rows")
    df = pd.concat(all_rows, ignore_index=True)
    out = PROCESSED / "cohort3_timecourse.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote: {out}")
    print(f"total rows: {len(df)}")

    # focus: cd8_like + ifn_response per guide, average across reps
    focus = df[df["module"].isin(["cd8_like", "ifn_response"])].copy()
    summary = focus.groupby(["timepoint", "guide", "module"]).agg(
        mean_delta=("delta", "mean"),
        min_p=("p", "min"),
        n=("n_source", "sum"),
    ).reset_index()
    summary = summary.sort_values(["module", "timepoint", "mean_delta"])
    print("\n=== cd8_like + ifn_response summary (avg across reps) ===")
    print(summary.head(30).to_string(index=False))

    # figure: per guide, delta vs timepoint
    guides = sorted(focus["guide"].unique())
    n_guides = len(guides)
    TP_ORDER = ["Day4", "Day7", "Day10"]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, n_guides * 0.3)))
    for ax_i, module in enumerate(["cd8_like", "ifn_response"]):
        ax = axes[ax_i]
        sub = focus[focus["module"] == module]
        # mean across reps
        pivot = sub.groupby(["guide", "timepoint"])["delta"].mean().unstack()
        pivot = pivot.reindex(columns=TP_ORDER)
        # sort by max abs delta
        max_abs = pivot.abs().max(axis=1).sort_values(ascending=False)
        pivot = pivot.loc[max_abs.index]
        # plot
        im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(TP_ORDER)))
        ax.set_xticklabels(TP_ORDER)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title(f"{module}: Δ score (source - NTC) across time")
        ax.set_xlabel("Time point")
        plt.colorbar(im, ax=ax, fraction=0.04, label="Δ score")
    fig.suptitle("Cohort 3 (T cell perturbation): time-course response\n"
                 "Day4 → Day7 → Day10 evolution",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_fig = FIGS / "F16_cohort3_timecourse.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nwrote: {out_fig}")


if __name__ == "__main__":
    main()
