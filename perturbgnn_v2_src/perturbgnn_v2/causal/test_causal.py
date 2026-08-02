"""Test causal diagnostics on M002 Ccn1."""
import sys, time
from pathlib import Path
import anndata as ad
import numpy as np

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")

from perturbgnn_v2.causal import (
    CausalConfig, spatial_did, guide_iv, rosenbaum_sensitivity,
)
from perturbgnn_v2.matching.match import _attach_module_scores


def main():
    print("[test] loading M002...", flush=True)
    a = ad.read_h5ad(PROCESSED / "M002_v2.h5ad")
    emb = np.load(PROCESSED / "embed_M002_v3.npy")
    _attach_module_scores(a)

    responses = ["score_ifn_response", "score_cd8_like",
                 "score_fibroblast", "score_macrophage"]

    for resp in responses:
        print(f"\n{'='*60}\n[test] === Causal: Ccn1 × {resp} ===\n{'='*60}")

        # (c1) Spatial DiD
        print(f"\n--- (c1) Spatial DiD ---")
        t0 = time.time()
        did = spatial_did(a, emb, "Ccn1", resp, CausalConfig())
        print(f"  ran in {time.time()-t0:.1f}s")
        if "error" in did:
            print(f"  FAILED: {did['error']}")
        else:
            print(f"  Δ_DiD = {did['delta_did']:+.4f} ±{did['se']:.4f} "
                  f"(t={did['t']:+.2f}, p={did['p']:.4g})")
            print(f"    y_pert near/far: {did['y_pert_near']:+.4f} / {did['y_pert_far']:+.4f}")
            print(f"    y_ctrl near/far: {did['y_ctrl_near']:+.4f} / {did['y_ctrl_far']:+.4f}")
            print(f"    n: {did['n']}")

        # (c2) Guide IV
        print(f"\n--- (c2) Guide IV (2SLS) ---")
        t0 = time.time()
        iv = guide_iv(a, emb, "Ccn1", resp, CausalConfig())
        print(f"  ran in {time.time()-t0:.1f}s")
        if "error" in iv:
            print(f"  FAILED: {iv['error']}")
        else:
            print(f"  β_OLS = {iv['beta_ols']:+.4f} ±{iv['se_ols']:.4f}")
            print(f"  β_IV  = {iv['beta_iv']:+.4f} ±{iv['se_iv']:.4f}")
            print(f"  IV/OLS ratio = {iv['iv_ratio']:.3f}  "
                  f"(close to 1 = consistent with causal)")
            print(f"  stage1 F = {iv['stage1_f']:.2f}  (>10 = strong instrument)")

        # (c3) Rosenbaum
        print(f"\n--- (c3) Rosenbaum sensitivity ---")
        t0 = time.time()
        ros = rosenbaum_sensitivity(a, emb, "Ccn1", resp, CausalConfig())
        print(f"  ran in {time.time()-t0:.1f}s")
        if "error" in ros:
            print(f"  FAILED: {ros['error']}")
        else:
            print(f"  base β = {ros['base_beta']:+.4f}, p = {ros['base_p']:.4g}")
            print(f"  Γ* (smallest Γ that flips p>0.05) = {ros['gamma_star']}")
            print(f"  p by Γ: {ros['by_Gamma']}")


if __name__ == "__main__":
    main()
