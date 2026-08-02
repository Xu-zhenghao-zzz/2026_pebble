"""Compare SLX (euclidean) vs Resistance-SLX on M002 Ccn1."""
import sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")

from perturbgnn_v2.spatial import fit_slx, SLXConfig
from perturbgnn_v2.spatial.resistance import fit_resistance_slx, ResistanceConfig
from perturbgnn_v2.matching.match import _attach_module_scores


def main():
    print("[test] loading M002...", flush=True)
    a = ad.read_h5ad(PROCESSED / "M002_v2.h5ad")
    emb = np.load(PROCESSED / "embed_M002_v3.npy")
    _attach_module_scores(a)

    responses = ["score_ifn_response", "score_cd8_like",
                 "score_fibroblast", "score_macrophage"]

    for resp in responses:
        print(f"\n{'='*60}\n[test] === {resp} ===\n{'='*60}")

        print(f"\n--- SLX (euclidean) ---")
        t0 = time.time()
        slx = fit_slx(a, emb, "Ccn1", resp, SLXConfig())
        print(f"  ran in {time.time()-t0:.1f}s")
        if "error" in slx:
            print(f"  FAILED: {slx['error']}")
        else:
            _print_thetas(slx, prefix="ring")

        print(f"\n--- Resistance-SLX ---")
        t0 = time.time()
        rslx = fit_resistance_slx(a, emb, "Ccn1", resp, ResistanceConfig())
        print(f"  ran in {time.time()-t0:.1f}s")
        if "error" in rslx:
            print(f"  FAILED: {rslx['error']}")
        else:
            _print_thetas(rslx, prefix="res_ring")


def _print_thetas(out, prefix="ring"):
    edges = (0, 20, 40, 80, 120, 200)
    print(f"  n_obs={out['n_obs']:,}  n_clones={out['n_clones']}  "
          f"global_p={out['global_p']:.4g}  R²={out['r2']:.4f}")
    print(f"  β_self = {out['beta_self']:+.4f} (p={out['beta_self_p']:.4g})")
    for k, (th, se, tt, pp) in enumerate(zip(out["theta_k"], out["theta_se"],
                                              out["theta_t"], out["theta_p"])):
        star = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
        print(f"    {edges[k]:>3}-{edges[k+1]:>3} µm: θ={th:+.4f} ±{se:.4f} "
              f"(p={pp:.4g}) {star}")


if __name__ == "__main__":
    main()
