"""Utrn TCGA LUAD survival analysis.

Mirrors v1's v2_tcga_survival.py (BLNK) but for Utrn.
- Log-rank test: Utrn-high (Q4) vs Utrn-low (Q1) OS
- Kaplan-Meier curves
- Cox proportional hazards (univariate + multivariate age/sex if available)

Outputs:
  - processed/utrn_survival.json
  - figures/F8_utrn_survival.png
"""
from __future__ import annotations
import functools, json, time
from pathlib import Path

print = functools.partial(print, flush=True)

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")

import requests
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

CBIO = "https://www.cbioportal.org/api"
STUDY = "luad_tcga_gdc"
PROFILE = f"{STUDY}_rna_seq_mrna"
SAMPLE_LIST = f"{STUDY}_all"
UTRN_ENTREZ = 22290


def fetch_phgr1():
    url = f"{CBIO}/molecular-profiles/{PROFILE}/molecular-data"
    params = {"sampleListId": SAMPLE_LIST, "entrezGeneId": UTRN_ENTREZ}
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return {d["sampleId"]: d["value"] for d in r.json() if d.get("value") is not None}


def fetch_clinical():
    url = f"{CBIO}/studies/{STUDY}/clinical-data"
    params = {"clinicalDataType": "PATIENT", "projection": "DETAILED"}
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=60)
    df = pd.DataFrame(r.json())
    keep = df[df["clinicalAttributeId"].isin(
        {"OS_STATUS", "OS_MONTHS", "DFS_STATUS", "DFS_MONTHS",
         "AGE", "SEX"})]
    return keep.pivot_table(index="patientId", columns="clinicalAttributeId",
                             values="value", aggfunc="first")


def fetch_s2p():
    r = requests.get(f"{CBIO}/studies/{STUDY}/samples",
                     params={"projection": "SUMMARY"},
                     headers={"Accept": "application/json"}, timeout=60)
    return {s["sampleId"]: s["patientId"] for s in r.json()}


def main():
    print("========== Utrn TCGA LUAD survival ==========")
    print("fetching Utrn expression...")
    expr = fetch_phgr1()
    print(f"  {len(expr)} samples")
    print("fetching clinical...")
    clin = fetch_clinical()
    print(f"  {len(clin)} patients, cols: {list(clin.columns)}")
    s2p = fetch_s2p()
    phgr1_p = {s2p[s]: v for s, v in expr.items() if s in s2p}
    df = clin.join(pd.Series(phgr1_p, name="UTRN"), how="inner")
    df = df.dropna(subset=["UTRN"])
    print(f"merged: {len(df)} patients with Utrn + clinical")
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["UTRN"] = pd.to_numeric(df["UTRN"], errors="coerce")

    # event: 1 if OS_STATUS == "1:DECEASED"
    df["event"] = (df["OS_STATUS"] == "1:DECEASED").astype(int)
    valid = df.dropna(subset=["OS_MONTHS", "UTRN"]).copy()
    print(f"with OS data: {len(valid)} patients, {valid['event'].sum()} events")

    # quartile split
    q25, q75 = valid["UTRN"].quantile([0.25, 0.75])
    valid["group"] = "mid"
    valid.loc[valid["UTRN"] <= q25, "group"] = "low"
    valid.loc[valid["UTRN"] >= q75, "group"] = "high"
    low = valid[valid["group"] == "low"]
    high = valid[valid["group"] == "high"]
    print(f"low (Q1): {len(low)} patients, {low['event'].sum()} events")
    print(f"high (Q4): {len(high)} patients, {high['event'].sum()} events")
    print(f"median OS low: {low['OS_MONTHS'].median():.1f} mo")
    print(f"median OS high: {high['OS_MONTHS'].median():.1f} mo")

    # log-rank
    lr = logrank_test(low["OS_MONTHS"], high["OS_MONTHS"],
                      event_observed_A=low["event"], event_observed_B=high["event"])
    print(f"\nlog-rank p = {lr.p_value:.4g}")

    # Cox univariate
    cph = CoxPHFitter()
    cox_df = valid[["OS_MONTHS", "event", "UTRN"]].copy()
    cox_df["log_UTRN"] = np.log10(cox_df["UTRN"].clip(lower=1e-3))
    cph.fit(cox_df, duration_col="OS_MONTHS", event_col="event",
            formula="log_UTRN")
    hr = cph.hazard_ratios_.iloc[0]
    ci = cph.confidence_intervals_.iloc[0]
    p_cox = cph.summary.loc["log_UTRN", "p"]
    print(f"Cox PH (log_UTRN): HR={hr:.3f}, 95% CI [{ci.iloc[0]:.3f}, {ci.iloc[1]:.3f}], p={p_cox:.4g}")

    # multivariate if AGE/SEX available
    multi_result = None
    if "AGE" in valid.columns and "SEX" in valid.columns:
        valid["AGE_num"] = pd.to_numeric(valid["AGE"], errors="coerce")
        valid["SEX_male"] = (valid["SEX"] == "Male").astype(int)
        cox_m = valid[["OS_MONTHS", "event", "log_UTRN", "AGE_num", "SEX_male"]].copy() if "log_UTRN" in valid.columns else None
        if cox_m is None:
            valid["log_UTRN"] = np.log10(valid["UTRN"].clip(lower=1e-3))
            cox_m = valid[["OS_MONTHS", "event", "log_UTRN", "AGE_num", "SEX_male"]].copy()
        cox_m = cox_m.dropna()
        if len(cox_m) > 50:
            cph_m = CoxPHFitter()
            cph_m.fit(cox_m, duration_col="OS_MONTHS", event_col="event")
            multi_result = cph_m.summary
            print(f"\nMultivariate Cox (n={len(cox_m)}):")
            print(multi_result[["coef", "exp(coef)", "p"]].to_string())

    # save
    result = {
        "n_patients": int(len(valid)),
        "n_events": int(valid["event"].sum()),
        "n_low": int(len(low)),
        "n_high": int(len(high)),
        "median_os_low": float(low["OS_MONTHS"].median()),
        "median_os_high": float(high["OS_MONTHS"].median()),
        "logrank_p": float(lr.p_value),
        "cox_uni_HR": float(hr),
        "cox_uni_ci_low": float(ci.iloc[0]),
        "cox_uni_ci_high": float(ci.iloc[1]),
        "cox_uni_p": float(p_cox),
    }
    if multi_result is not None:
        result["cox_multi_summary"] = multi_result.reset_index().to_dict(orient="records")
    with open(PROCESSED / "utrn_survival.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nwrote: processed/utrn_survival.json")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    kmf = KaplanMeierFitter()
    for label, sub, color in [("Utrn-low (Q1)", low, "#9575cd"),
                              ("Utrn-high (Q4)", high, "#2e7d32")]:
        kmf.fit(sub["OS_MONTHS"], event_observed=sub["event"], label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
    ax.set_title(f"(a) Utrn Kaplan-Meier OS (TCGA LUAD)\n"
                 f"log-rank p={lr.p_value:.3g}")
    ax.set_xlabel("Months"); ax.set_ylabel("Survival probability")
    ax.grid(alpha=0.3)

    # (b) Cox forest plot
    ax = axes[1]
    if multi_result is not None:
        rows = []
        for var in multi_result.index:
            row = multi_result.loc[var]
            rows.append((var, row["exp(coef)"], row["exp(coef) lower 95%"],
                         row["exp(coef) upper 95%"], row["p"]))
        y = np.arange(len(rows))
        for i, (v, hr_v, lo, hi, pp) in enumerate(rows):
            color = "#d62728" if (hr_v > 1 and pp < 0.05) else "#1f77b4" if pp < 0.05 else "#7f7f7f"
            ax.errorbar(hr_v, i,
                        xerr=[[hr_v - lo], [hi - hr_v]],
                        fmt="o", color=color, capsize=5, markersize=10)
            label = f"{v}\nHR={hr_v:.2f}, p={pp:.3g}"
            ax.text(hi + 0.1, i, label, va="center", fontsize=9)
        ax.axvline(1, color="gray", linestyle="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel("Hazard Ratio (95% CI)")
        ax.set_title(f"(b) Multivariate Cox\n(n={len(valid)})")
        ax.grid(alpha=0.3, axis="x")
    else:
        ax.text(0.5, 0.5, "multivariate Cox not available\n(no AGE/SEX)",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Utrn survival analysis — TCGA LUAD (n=518)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F8_utrn_survival.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
