"""V10: PHGR1 in melanoma ICB cohorts.

Melanoma had the STRONGEST pan-cancer PHGR1-CD8A correlation (rho=+0.72).
Three melanoma ICB cohorts on cBioPortal may have RNA-seq + response data:
  - mel_ucla_2016 (anti-PD-1, Cell 2016)
  - mel_dfci_2019 (anti-PD-1/CTLA-4, Nat Med 2019)
  - mel_iatlas_riaz_nivolumab_2017 (anti-PD-1, Cell 2017)

If PHGR1-high predicts response in melanoma (unlike bladder), that's
cancer-type-specific ICB prediction — the strongest possible clinical claim.
"""
from __future__ import annotations
import functools, json, time
from pathlib import Path
print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments"; FIG = ROOT / "figures"

import requests
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

CBIO = "https://www.cbioportal.org/api"
PHGR1 = 9398
COHORTS = [
    ("mel_ucla_2016", "Melanoma anti-PD-1 (UCLA, Cell 2016)"),
    ("mel_dfci_2019", "Melanoma anti-PD-1/CTLA-4 (DFCI, Nat Med 2019)"),
    ("mel_iatlas_riaz_nivolumab_2017", "Melanoma anti-PD-1 (Riaz, Cell 2017)"),
    ("mel_iatlas_hugo_ucla_2016", "Melanoma anti-PD-1 (Hugo, iAtlas)"),
    ("mel_iatlas_liu_2019", "Melanoma anti-PD-1/CTLA-4 (Liu, iAtlas)"),
    ("mel_iatlas_gide_2019", "Melanoma ICB (Gide, Cancer Cell 2019)"),
]


def fetch_expr(study):
    for suffix in ["_rna_seq_v2_mrna", "_rna_seq_mrna", "_mrna", "_rna_seq_v2_rsem"]:
        try:
            r = requests.get(f"{CBIO}/molecular-profiles/{study}{suffix}/molecular-data",
                             params={"sampleListId": f"{study}_all", "entrezGeneId": PHGR1},
                             headers={"Accept": "application/json"}, timeout=60)
            if r.status_code == 200 and len(r.json()) > 0:
                return {d["sampleId"]: d["value"] for d in r.json() if d.get("value") is not None}, suffix
        except:
            pass
    return {}, None


def fetch_clinical(study):
    r = requests.get(f"{CBIO}/studies/{study}/clinical-data",
                     params={"clinicalDataType": "PATIENT"}, headers={"Accept": "application/json"}, timeout=60)
    df = pd.DataFrame(r.json())
    return df.pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")


def fetch_s2p(study):
    r = requests.get(f"{CBIO}/studies/{study}/samples",
                     params={"projection": "SUMMARY"}, headers={"Accept": "application/json"}, timeout=60)
    return {s["sampleId"]: s["patientId"] for s in r.json()}


def main():
    t0 = time.time()
    print("========== V10: PHGR1 in melanoma ICB cohorts ==========")
    all_results = []

    for study, label in COHORTS:
        print(f"\n--- {label} ({study}) ---")
        expr, suffix = fetch_expr(study)
        print(f"  PHGR1 expression: {len(expr)} samples ({suffix})")
        if len(expr) < 20:
            print("  SKIP (no expression data)")
            continue

        clinical = fetch_clinical(study)
        s2p = fetch_s2p(study)
        print(f"  Clinical: {len(clinical)} patients, cols: {sorted(clinical.columns)[:15]}")

        # find response columns
        resp_cols = [c for c in clinical.columns
                     if any(k in c.lower() for k in ["response", "benefit", "recist", "binary", "responder"])]
        print(f"  Response cols: {resp_cols}")

        phgr1_p = {s2p.get(s): v for s, v in expr.items() if s in s2p}
        df = clinical.join(pd.Series(phgr1_p, name="PHGR1"), how="inner")
        df["PHGR1"] = pd.to_numeric(df["PHGR1"], errors="coerce")
        df = df.dropna(subset=["PHGR1"])
        print(f"  Paired: {len(df)} patients")

        # try each response column
        for resp_col in resp_cols:
            vals = df[resp_col].dropna().unique()
            print(f"    {resp_col}: values = {list(vals)[:10]}")

            # match common response patterns
            for r_pat, nr_pat, comp_label in [
                ("CR", "PD", "CR vs PD"),
                ("PR", "PD", "PR vs PD"),
                ("Y", "N", "Y vs N"),
                ("YES", "NO", "YES vs NO"),
                ("TRUE", "FALSE", "TRUE vs FALSE"),
                ("1", "0", "1 vs 0"),
                ("Responder", "Non-responder", "R vs NR"),
                ("R", "NR", "R vs NR"),
                ("responders", "non-responders", "resp vs non-resp"),
                ("complete", "progressive", "CR vs PD (text)"),
                ("partial", "progressive", "PR vs PD (text)"),
            ]:
                r_mask = df[resp_col].astype(str).str.contains(r_pat, case=False, na=False)
                nr_mask = df[resp_col].astype(str).str.contains(nr_pat, case=False, na=False)
                # exclude overlap
                both = r_mask & nr_mask
                r_mask = r_mask & ~both
                nr_mask = nr_mask & ~both

                if r_mask.sum() >= 5 and nr_mask.sum() >= 5:
                    r_phgr1 = df.loc[r_mask, "PHGR1"]
                    nr_phgr1 = df.loc[nr_mask, "PHGR1"]
                    u, p = mannwhitneyu(r_phgr1, nr_phgr1, alternative="greater")
                    sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"      {comp_label}: R n={r_mask.sum()} PHGR1={r_phgr1.mean():.1f}  "
                          f"NR n={nr_mask.sum()} PHGR1={nr_phgr1.mean():.1f}  "
                          f"p={p:.4f} {sig}")
                    all_results.append({
                        "cohort": study, "label": label,
                        "response_col": resp_col, "comparison": comp_label,
                        "n_responder": int(r_mask.sum()), "n_nonresponder": int(nr_mask.sum()),
                        "phgr1_R": float(r_phgr1.mean()), "phgr1_NR": float(nr_phgr1.mean()),
                        "p": float(p),
                    })

        # also survival
        if "OS_MONTHS" in df.columns and "OS_STATUS" in df.columns:
            df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
            df_os = df.dropna(subset=["OS_MONTHS"]).copy()
            df_os["event"] = df_os["OS_STATUS"].astype(str).str.contains("DEAD|DECEASED", case=False, na=False).astype(int)
            if len(df_os) > 30:
                q25, q75 = df_os["PHGR1"].quantile([0.25, 0.75])
                low = df_os[df_os["PHGR1"] <= q25]
                high = df_os[df_os["PHGR1"] >= q75]
                if len(low) > 5 and len(high) > 5:
                    u, p = mannwhitneyu(low["OS_MONTHS"], high["OS_MONTHS"], alternative="less")
                    print(f"    OS: low-median={low['OS_MONTHS'].median():.1f}mo  high-median={high['OS_MONTHS'].median():.1f}mo  p={p:.4f}")

    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(str(OUT / "phgr1_melanoma_icb.csv"), index=False)
        print(f"\n=== Summary ===")
        print(df_res[["cohort", "comparison", "n_responder", "n_nonresponder", "phgr1_R", "phgr1_NR", "p"]].to_string(index=False))

        # figure
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_res))
        w = 0.35
        ax.bar(x - w/2, df_res["phgr1_R"], w, color="#009E73", label="Responder")
        ax.bar(x + w/2, df_res["phgr1_NR"], w, color="#D55E00", label="Non-responder")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['label'][:25]}\n{r['comparison']}" for _, r in df_res.iterrows()], fontsize=7, ha="center")
        ax.set_ylabel("mean PHGR1 expression")
        ax.set_title("PHGR1 in melanoma ICB responder vs non-responder")
        for i, p_val in enumerate(df_res["p"]):
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.text(i, max(df_res["phgr1_R"].max(), df_res["phgr1_NR"].max()) * 1.02,
                    f"p={p_val:.3f} {sig}", ha="center", fontsize=7)
        ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(str(FIG / "F18_phgr1_icb.png"), dpi=130)
        plt.close(fig)
        print(f"[fig] F18_phgr1_icb.png")
    else:
        print("\n[no response data found in any melanoma ICB cohort]")

    json.dump({"results": all_results, "elapsed": time.time()-t0},
              open(str(OUT / "v10_summary.json"), "w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
