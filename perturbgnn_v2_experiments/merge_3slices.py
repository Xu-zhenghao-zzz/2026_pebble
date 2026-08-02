"""Merge 3 slices' scan results + cross-slice Stouffer + BH FDR.

Find genes that replicate across slices (the strongest causal claims).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import combine_pvalues

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")


def main():
    slices = ["M001", "M002", "M003"]
    dfs = []
    for s in slices:
        path = PROCESSED / f"genome_scan_v2_{s}.csv" if s != "M002" else PROCESSED / "genome_scan_v2.csv"
        df = pd.read_csv(path)
        df["slice"] = s
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    print(f"merged: {len(merged)} rows ({merged['slice'].value_counts().to_dict()})")
    print(f"unique genes: {merged['gene'].nunique()}")
    print(f"unique (gene,response): {merged.groupby(['gene','response']).ngroups}")

    # per (gene, response): collect p-values across slices where present
    rows = []
    for (gene, resp), grp in merged.groupby(["gene", "response"]):
        n_slices = grp["slice"].nunique()
        ps_durbin = grp["durbin_p"].dropna().tolist()
        ps_did = grp["did_p"].dropna().tolist() if "did_p" in grp.columns else []
        deltas = grp["durbin_delta"].dropna().tolist()
        # Stouffer combined p (Durbin)
        if len(ps_durbin) >= 1:
            if len(ps_durbin) >= 2:
                _, p_comb = combine_pvalues(ps_durbin, method="stouffer")
            else:
                p_comb = ps_durbin[0]
        else:
            p_comb = np.nan
        # direction consistency
        if deltas:
            signs = [np.sign(d) for d in deltas]
            dir_consist = (len(set(signs)) == 1)
        else:
            dir_consist = False
        rows.append({
            "gene": gene,
            "response": resp,
            "n_slices": n_slices,
            "combined_p_durbin": p_comb,
            "direction_consistent": dir_consist,
            "mean_durbin_delta": np.mean(deltas) if deltas else np.nan,
            "min_durbin_p": min(ps_durbin) if ps_durbin else np.nan,
            "slices_present": ",".join(sorted(grp["slice"].unique())),
        })
    combined = pd.DataFrame(rows)

    # BH FDR across all (gene, response)
    valid = combined["combined_p_durbin"].dropna()
    if len(valid) > 0:
        ranks = valid.rank(method="first")
        fdr = valid * len(valid) / ranks
        fdr = fdr.clip(upper=1.0)
        combined.loc[valid.index, "combined_q"] = fdr.values
    else:
        combined["combined_q"] = np.nan

    combined = combined.sort_values("combined_p_durbin")
    out = PROCESSED / "genome_scan_v2_combined_3slices.csv"
    combined.to_csv(out, index=False)
    print(f"\nwrote: {out}")

    # multi-slice replication: gene × response in >=2 slices, direction consistent
    multi = combined[(combined["n_slices"] >= 2) &
                     (combined["direction_consistent"]) &
                     (combined["combined_q"] < 0.05)]
    multi = multi.sort_values("combined_q")
    out2 = PROCESSED / "genome_scan_v2_replicated.csv"
    multi.to_csv(out2, index=False)
    print(f"wrote: {out2}")
    print(f"\n=== {len(multi)} replicated (≥2 slices, direction consistent, q<0.05) ===")
    if len(multi) > 0:
        print(multi.head(20).to_string(index=False))

    # cross-slice hits (in all 3 slices)
    all3 = combined[combined["n_slices"] == 3].sort_values("combined_q")
    print(f"\n=== {len(all3)} (gene, response) pairs present in all 3 slices ===")
    if len(all3) > 0:
        print(f"top 15 by combined_q:")
        print(all3.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
