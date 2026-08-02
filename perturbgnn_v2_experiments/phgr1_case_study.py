"""Full causal diagnostics for Phgr1 across M001 + M002.

Outputs:
  - docs/PHGR1_CASE_STUDY.md
  - figures/F5_phgr1_case_study.png
"""
import sys, time, json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
DOCS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/docs")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")

from perturbgnn_v2.spatial import fit_durbin, DurbinConfig
from perturbgnn_v2.causal import CausalConfig, spatial_did, guide_iv, rosenbaum_sensitivity
from perturbgnn_v2.matching.match import _attach_module_scores


RESPONSES = ["score_ifn_response", "score_fibroblast", "score_hypoxia",
             "score_macrophage", "score_endothelial"]


def run_all_diagnostics(slice_name, gene="Phgr1"):
    print(f"\n=== {slice_name} / {gene} ===", flush=True)
    a = ad.read_h5ad(PROCESSED / f"{slice_name}_v2.h5ad")
    emb = np.load(PROCESSED / f"embed_{slice_name}_v3.npy")
    _attach_module_scores(a)
    out = {}
    for resp in RESPONSES:
        print(f"  {resp}...", flush=True, end="")
        t0 = time.time()
        r = {"response": resp}
        # Durbin
        try:
            d = fit_durbin(a, emb, gene, resp, DurbinConfig())
            if "error" not in d:
                b = d["best"]
                r["durbin_delta"] = b["delta"]
                r["durbin_p"] = b["delta_p"]
                r["durbin_lambda"] = b["lambda_um"]
                r["durbin_r2"] = b["r2"]
                r["n_clones"] = d["n_clones"]
                # all lambdas
                r["durbin_by_lambda"] = {
                    lam: {"delta": x["delta"], "p": x["delta_p"]}
                    for lam, x in d["all_lambdas"].items() if "error" not in x
                }
        except Exception as e:
            r["durbin_err"] = str(e)[:60]
        # DiD
        try:
            did = spatial_did(a, emb, gene, resp, CausalConfig())
            if "error" not in did:
                r["did_delta"] = did["delta_did"]
                r["did_p"] = did["p"]
                r["did_n"] = did["n"]
        except Exception as e:
            r["did_err"] = str(e)[:60]
        # IV
        try:
            iv = guide_iv(a, emb, gene, resp, CausalConfig())
            if "error" not in iv:
                r["iv_ratio"] = iv["iv_ratio"]
                r["iv_beta_ols"] = iv["beta_ols"]
                r["iv_beta_iv"] = iv["beta_iv"]
                r["iv_stage1_f"] = iv["stage1_f"]
        except Exception as e:
            r["iv_err"] = str(e)[:60]
        # Rosenbaum
        try:
            ros = rosenbaum_sensitivity(a, emb, gene, resp, CausalConfig())
            if "error" not in ros:
                r["ros_base_p"] = ros["base_p"]
                gs = ros["gamma_star"]
                r["ros_gamma_star"] = float(gs) if isinstance(gs, (int, float)) else 4.0
                r["ros_gamma_star_str"] = str(gs)
                r["ros_p_by_gamma"] = ros["by_Gamma"]
        except Exception as e:
            r["ros_err"] = str(e)[:60]
        out[resp] = r
        print(f" done in {time.time()-t0:.1f}s")
    return out


def main():
    results = {}
    for s in ["M001", "M002"]:
        results[s] = run_all_diagnostics(s, "Phgr1")

    # save raw
    with open(PROCESSED / "phgr1_full_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote: processed/phgr1_full_diagnostics.json")

    # Markdown summary
    md = ["# Phgr1 Case Study — v2 Full Causal Diagnostics",
          "",
          "**Target gene**: Phgr1 (Proline/Histidine/Glycine-Rich 1)",
          "**Slices**: M001 (395 sources) + M002 (640 sources)",
          "**Date**: 2026-08-01",
          "",
          "## Summary table",
          "",
          "| Slice | Response | Durbin δ | Durbin p | DiD Δ | DiD p | IV/OLS | Γ* |",
          "|---|---|---|---|---|---|---|---|"]
    for s in ["M001", "M002"]:
        for resp in RESPONSES:
            r = results[s].get(resp, {})
            dur_d = r.get("durbin_delta", "—")
            dur_p = r.get("durbin_p", "—")
            did_d = r.get("did_delta", "—")
            did_p = r.get("did_p", "—")
            iv = r.get("iv_ratio", "—")
            gs = r.get("ros_gamma_star_str", str(r.get("ros_gamma_star", "—")))
            def fmt(x, p=None):
                if isinstance(x, str): return x
                if x is None: return "—"
                if p is not None and isinstance(p, (int, float)) and p < 0.001:
                    return f"{x:+.3f}***"
                if p is not None and isinstance(p, (int, float)) and p < 0.01:
                    return f"{x:+.3f}**"
                return f"{x:+.3f}"
            md.append(f"| {s} | {resp.replace('score_','')} | {fmt(dur_d, dur_p)} | {dur_p:.3g} | {fmt(did_d, did_p)} | {did_p:.3g} | {iv:.2f} | {gs} |")
    md += ["",
           "## Direction consistency check (M001 vs M002)",
           "",
           "| Response | M001 δ | M002 δ | Consistent? |",
           "|---|---|---|---|"]
    for resp in RESPONSES:
        d1 = results["M001"].get(resp, {}).get("durbin_delta")
        d2 = results["M002"].get(resp, {}).get("durbin_delta")
        if d1 is not None and d2 is not None:
            consist = "✅" if np.sign(d1) == np.sign(d2) else "❌"
            md.append(f"| {resp.replace('score_','')} | {d1:+.3f} | {d2:+.3f} | {consist} |")
    (DOCS / "PHGR1_CASE_STUDY.md").write_text("\n".join(md))
    print(f"wrote: docs/PHGR1_CASE_STUDY.md")

    # Figure: 4 panels
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Durbin δ vs λ for each response, both slices
    ax = axes[0, 0]
    for resp in RESPONSES:
        for s, marker in [("M001", "o"), ("M002", "s")]:
            r = results[s].get(resp, {})
            if "durbin_by_lambda" in r:
                lams = sorted(r["durbin_by_lambda"].keys())
                deltas = [r["durbin_by_lambda"][l]["delta"] for l in lams]
                ax.plot(lams, deltas, marker=marker,
                        label=f"{s} {resp.replace('score_', '')}",
                        alpha=0.7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Kernel λ (µm)")
    ax.set_ylabel("Durbin δ")
    ax.set_title("(a) Phgr1 Durbin δ vs λ (M001=o, M002=s)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)

    # (b) DiD Δ per response, both slices
    ax = axes[0, 1]
    width = 0.35
    resps = [r.replace("score_", "") for r in RESPONSES]
    x = np.arange(len(resps))
    dids_m1 = [results["M001"].get(r, {}).get("did_delta", 0) for r in RESPONSES]
    dids_m2 = [results["M002"].get(r, {}).get("did_delta", 0) for r in RESPONSES]
    ax.bar(x - width/2, dids_m1, width, label="M001", color="steelblue", alpha=0.7)
    ax.bar(x + width/2, dids_m2, width, label="M002", color="coral", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(resps, rotation=30, ha="right")
    ax.set_ylabel("Spatial DiD Δ")
    ax.set_title("(b) Phgr1 spatial DiD per response")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (c) IV/OLS ratio
    ax = axes[1, 0]
    ivs_m1 = [results["M001"].get(r, {}).get("iv_ratio", 0) for r in RESPONSES]
    ivs_m2 = [results["M002"].get(r, {}).get("iv_ratio", 0) for r in RESPONSES]
    ax.bar(x - width/2, ivs_m1, width, label="M001", color="steelblue", alpha=0.7)
    ax.bar(x + width/2, ivs_m2, width, label="M002", color="coral", alpha=0.7)
    ax.axhline(1, color="green", lw=1, linestyle="--", label="ratio=1 (consistent)")
    ax.axhspan(0.5, 2, alpha=0.1, color="green")
    ax.set_xticks(x)
    ax.set_xticklabels(resps, rotation=30, ha="right")
    ax.set_ylabel("IV / OLS ratio")
    ax.set_title("(c) Confounding diagnostic (close to 1 = causal)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (d) Rosenbaum Γ* per response
    ax = axes[1, 1]
    gs_m1 = [results["M001"].get(r, {}).get("ros_gamma_star", 0) for r in RESPONSES]
    gs_m2 = [results["M002"].get(r, {}).get("ros_gamma_star", 0) for r in RESPONSES]
    ax.bar(x - width/2, gs_m1, width, label="M001", color="steelblue", alpha=0.7)
    ax.bar(x + width/2, gs_m2, width, label="M002", color="coral", alpha=0.7)
    ax.axhline(1.5, color="orange", lw=1, linestyle="--", label="Γ*=1.5 (moderate robustness)")
    ax.axhline(3.0, color="red", lw=1, linestyle="--", label="Γ*=3 (strong)")
    ax.set_xticks(x)
    ax.set_xticklabels(resps, rotation=30, ha="right")
    ax.set_ylabel("Rosenbaum Γ*")
    ax.set_title("(d) Sensitivity robustness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Phgr1 case study — v2 causal diagnostics (M001 + M002)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F5_phgr1_case_study.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
