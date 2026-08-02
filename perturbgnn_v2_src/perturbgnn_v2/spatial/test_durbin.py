"""Test spatial Durbin on M002 Ccn1."""
import sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")

from perturbgnn_v2.spatial import fit_durbin, DurbinConfig
from perturbgnn_v2.matching.match import _attach_module_scores


def main():
    print("[test] loading M002...", flush=True)
    a = ad.read_h5ad(PROCESSED / "M002_v2.h5ad")
    emb = np.load(PROCESSED / "embed_M002_v3.npy")
    _attach_module_scores(a)

    responses = ["score_ifn_response", "score_cd8_like",
                 "score_fibroblast", "score_macrophage"]

    for resp in responses:
        print(f"\n{'='*60}\n[test] === Durbin: Ccn1 × {resp} ===\n{'='*60}")
        t0 = time.time()
        out = fit_durbin(a, emb, "Ccn1", resp, DurbinConfig())
        dt = time.time() - t0
        if "error" in out:
            print(f"  FAILED in {dt:.1f}s: {out['error']}")
            continue
        print(f"  ran in {dt:.1f}s, n_sources={out['n_sources']}, "
              f"n_clones={out['n_clones']}")
        print(f"\n  best λ = {out['best_lambda_um']} µm")
        b = out["best"]
        star = "***" if b["delta_p"] < 0.001 else "**" if b["delta_p"] < 0.01 else "*" if b["delta_p"] < 0.05 else ""
        print(f"    δ (propagation slope) = {b['delta']:+.4f} ±{b['delta_se']:.4f} "
              f"(t={b['delta_t']:+.2f}, p={b['delta_p']:.4g}) {star}")
        print(f"    β_self = {b['beta_self']:+.4f} (p={b['beta_self_p']:.4g})")
        print(f"    R² = {b['r2']:.4f}")
        print(f"\n  sensitivity across λ:")
        for lam, r in out["all_lambdas"].items():
            if "error" in r:
                print(f"    λ={lam:>3}µm: ERROR: {r['error']}")
            else:
                s = "*" if r["delta_p"] < 0.05 else ""
                print(f"    λ={lam:>3}µm: δ={r['delta']:+.4f} "
                      f"(p={r['delta_p']:.4g}) {s}")


if __name__ == "__main__":
    main()
