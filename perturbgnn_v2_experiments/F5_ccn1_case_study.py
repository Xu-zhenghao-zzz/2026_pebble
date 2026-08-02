"""Phase 5 Figure 5: Ccn1 case study figure.

4-panel figure showing the causal diagnostics for Ccn1 in M002:
  (a) Durbin δ vs λ curve for 4 responses
  (b) Spatial DiD Δ with CI for 4 responses
  (c) IV vs OLS scatter (per response)
  (d) Rosenbaum p-value vs Γ curve

Run after test_causal.py has been run; reads results from logs.
"""
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/logs")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")
FIGS.mkdir(exist_ok=True)


def parse_causal_log(log_path):
    """Parse the test_causal output into per-response diagnostics."""
    text = open(log_path).read()
    responses = re.findall(r"=== Causal: Ccn1 × (\S+) ===", text)
    results = {}
    for resp in responses:
        # extract the block for this response
        m = re.search(rf"=== Causal: Ccn1 × {re.escape(resp)} ===\n"
                      r"(.*?)(?==== Causal:|$)", text, re.DOTALL)
        if not m:
            continue
        block = m.group(1)
        r = {"response": resp}
        # DiD
        did_m = re.search(r"Δ_DiD = ([+-]?[\d.]+) ±([\d.]+).*?p=([\d.e+-]+)", block)
        if did_m:
            r["did_delta"] = float(did_m.group(1))
            r["did_se"] = float(did_m.group(2))
            r["did_p"] = float(did_m.group(3))
        # IV
        iv_m = re.search(r"β_IV\s*=\s*([+-]?[\d.]+).*?IV/OLS ratio = ([+-]?[\d.]+)", block)
        if iv_m:
            r["iv_beta"] = float(iv_m.group(1))
            r["iv_ratio"] = float(iv_m.group(2))
        # Rosenbaum
        ros_m = re.search(r"Γ\*.*?=\s*([<>]?[\d.]+)", block)
        if ros_m:
            gs = ros_m.group(1)
            r["gamma_star"] = float(gs.lstrip(">")) if gs[0] != ">" else 4.0
            r["gamma_star_str"] = gs
        # p by Gamma
        pg_m = re.search(r"p by Γ: (\{[^\}]+\})", block)
        if pg_m:
            try:
                r["p_by_gamma"] = eval(pg_m.group(1))
            except Exception:
                pass
        results[resp] = r
    return results


def parse_durbin_log(log_path):
    """Parse Durbin output to get δ vs λ per response."""
    text = open(log_path).read()
    responses = re.findall(r"=== Durbin: Ccn1 × (\S+) ===", text)
    results = {}
    for resp in responses:
        m = re.search(rf"=== Durbin: Ccn1 × {re.escape(resp)} ===\n"
                      r"(.*?)(?==== Durbin:|$)", text, re.DOTALL)
        if not m:
            continue
        block = m.group(1)
        # parse all λ lines
        lam_delta = {}
        for lm in re.finditer(r"λ=\s*(\d+)µm: δ=([+-]?[\d.]+).*?p=([\d.e+-]+)",
                              block):
            lam_delta[int(lm.group(1))] = (float(lm.group(2)),
                                            float(lm.group(3)))
        if lam_delta:
            results[resp] = lam_delta
    return results


def main():
    # find latest logs
    causal_logs = sorted(LOGS.glob("p4_causal_smoke*.log"))
    durbin_logs = sorted(LOGS.glob("p3_durbin*.log"))
    if not causal_logs or not durbin_logs:
        print("missing logs")
        return
    causal = parse_causal_log(causal_logs[-1])
    durbin = parse_durbin_log(durbin_logs[-1])
    print(f"parsed: causal={list(causal.keys())}, durbin={list(durbin.keys())}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Durbin δ vs λ
    ax = axes[0, 0]
    for resp, lam_delta in durbin.items():
        lams = sorted(lam_delta.keys())
        deltas = [lam_delta[l][0] for l in lams]
        ax.plot(lams, deltas, marker="o", label=resp.replace("score_", ""))
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Kernel λ (µm)")
    ax.set_ylabel("Durbin δ (propagation slope)")
    ax.set_title("(a) Ccn1 Durbin δ vs kernel length scale")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # (b) DiD Δ with CI
    ax = axes[0, 1]
    resps = list(causal.keys())
    dids = [causal[r].get("did_delta", 0) for r in resps]
    ses = [causal[r].get("did_se", 0) for r in resps]
    y_pos = np.arange(len(resps))
    labels = [r.replace("score_", "") for r in resps]
    ax.barh(y_pos, dids, xerr=[1.96 * s for s in ses],
            color="steelblue", alpha=0.7, capsize=4)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spatial DiD Δ (95% CI)")
    ax.set_title("(b) Ccn1 spatial DiD per response")
    ax.grid(alpha=0.3, axis="x")

    # (c) IV vs OLS ratio
    ax = axes[1, 0]
    ratios = [causal[r].get("iv_ratio", 0) for r in resps]
    colors = ["#2ca02c" if 0.5 < abs(causal[r].get("iv_ratio", 0)) < 2
              else "#d62728" for r in resps]
    ax.barh(y_pos, ratios, color=colors, alpha=0.7)
    ax.axvline(1, color="gray", lw=1, linestyle="--", label="ratio=1 (consistent)")
    ax.axvspan(0.5, 2, alpha=0.1, color="green", label="consistent range")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("IV / OLS ratio")
    ax.set_title("(c) Confounding diagnostic\n(green = consistent, red = confounded)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, axis="x")

    # (d) Rosenbaum p vs Γ
    ax = axes[1, 1]
    for r in resps:
        if "p_by_gamma" in causal[r]:
            gammas = sorted(causal[r]["p_by_gamma"].keys())
            ps = [-np.log10(max(causal[r]["p_by_gamma"][g], 1e-30))
                  for g in gammas]
            ax.plot(gammas, ps, marker="o",
                    label=r.replace("score_", ""))
    ax.axhline(-np.log10(0.05), color="red", lw=1, linestyle="--",
               label="p=0.05")
    ax.set_xlabel("Rosenbaum Γ (hidden confounder strength)")
    ax.set_ylabel("-log10(p)")
    ax.set_title("(d) Rosenbaum sensitivity")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle("Ccn1 case study — v2 causal diagnostics (M002)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F5_ccn1_case_study.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
