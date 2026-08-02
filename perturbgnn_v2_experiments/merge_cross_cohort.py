"""Merge cohort 1 (M001/M002/M003) + cohort 2 (subQ-1~5) scans.

8 slices total, ~1.5M bins, focus on cross-cohort replicated hits.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import combine_pvalues

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")

# focus genes (intersection of cohort 1 interesting + cohort 2 top)
FOCUS_GENES = ["Phgr1", "Utrn", "Cttn", "Ccn1", "Blnk", "Rab8a", "Bcam"]


def main():
    # gather all CSVs
    csvs = []
    # cohort 1
    for s in ["M001", "M002", "M003"]:
        p = PROCESSED / f"genome_scan_v2_{s}.csv" if s != "M002" else PROCESSED / "genome_scan_v2.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["cohort"] = "cohort1"
            df["slice"] = s
            csvs.append(df)
    # cohort 2
    for s in ["subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]:
        p = PROCESSED / f"cohort2_{s}_scan.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["cohort"] = "cohort2"
            csvs.append(df)

    merged = pd.concat(csvs, ignore_index=True)
    print(f"merged: {len(merged)} rows")
    print(f"slices: {sorted(merged['slice'].unique())}")

    # restrict to focus genes
    merged = merged[merged["gene"].isin(FOCUS_GENES)]
    print(f"focus genes: {len(merged)} rows")

    # cross-slice Stouffer per (gene, response)
    rows = []
    for (gene, resp), grp in merged.groupby(["gene", "response"]):
        ps = grp["durbin_p"].dropna().tolist()
        deltas = grp["durbin_delta"].dropna().tolist()
        slices = sorted(grp["slice"].unique())
        cohorts = sorted(grp["cohort"].unique())
        if not ps:
            continue
        if len(ps) >= 2:
            _, p_comb = combine_pvalues(ps, method="stouffer")
        else:
            p_comb = ps[0]
        signs = [np.sign(d) for d in deltas] if deltas else []
        dir_consist = len(set(signs)) == 1 if signs else False
        rows.append({
            "gene": gene, "response": resp,
            "n_slices": len(slices),
            "n_cohorts": len(cohorts),
            "combined_p": p_comb,
            "direction_consistent": dir_consist,
            "mean_delta": np.mean(deltas) if deltas else None,
            "min_p": min(ps),
            "slices": ",".join(slices),
            "cohorts": ",".join(cohorts),
        })
    combined = pd.DataFrame(rows)

    # BH FDR
    valid = combined["combined_p"].dropna()
    ranks = valid.rank(method="first")
    fdr = valid * len(valid) / ranks
    combined.loc[valid.index, "combined_q"] = fdr.clip(upper=1.0).values
    combined = combined.sort_values("combined_p")

    out = PROCESSED / "cross_cohort_combined.csv"
    combined.to_csv(out, index=False)
    print(f"\nwrote: {out}")

    # cross-cohort replicated (≥1 slice per cohort, direction consistent, q<0.05)
    cross_cohort = combined[
        (combined["n_cohorts"] >= 2) &
        (combined["direction_consistent"]) &
        (combined["combined_q"] < 0.05)
    ].sort_values("combined_q")
    out2 = PROCESSED / "cross_cohort_replicated.csv"
    cross_cohort.to_csv(out2, index=False)
    print(f"wrote: {out2}")

    print(f"\n=== {len(cross_cohort)} cross-cohort replicated (gene, response) pairs ===")
    if len(cross_cohort) > 0:
        print(cross_cohort[["gene", "response", "n_slices", "mean_delta",
                            "combined_q", "slices"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
