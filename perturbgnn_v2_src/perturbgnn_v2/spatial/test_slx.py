"""Smoke test for SLX on M002 Ccn1."""
import sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
import sys
sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
from perturbgnn_v2.spatial import fit_slx, SLXConfig
from perturbgnn_v2.matching.match import _attach_module_scores


def main():
    print("[test] loading M002...", flush=True)
    a = ad.read_h5ad(PROCESSED / "M002_v2.h5ad")
    emb = np.load(PROCESSED / "embed_M002_v3.npy")
    _attach_module_scores(a)
    print(f"  shape: {a.shape}, embed: {emb.shape}", flush=True)
    print(f"  module score cols: {[c for c in a.obs.columns if c.startswith('score_')]}", flush=True)

    cfg = SLXConfig()
    print(f"\n[config] {cfg}")

    responses = ["score_ifn_response", "score_cd8_like",
                 "score_fibroblast", "score_macrophage"]
    for resp in responses:
        print(f"\n[test] === SLX Ccn1 × {resp} ===")
        t0 = time.time()
        out = fit_slx(a, emb, "Ccn1", resp, cfg)
        dt = time.time() - t0
        if "error" in out:
            print(f"  FAILED in {dt:.1f}s: {out['error']}")
            continue
        print(f"  ran in {dt:.1f}s, n_obs={out['n_obs']:,}, n_sources={out['n_sources']}, "
              f"n_clones={out['n_clones']}, R²={out['r2']:.4f}")
        print(f"  global F-test p = {out['global_p']:.4g}" if out['global_p'] else "  global F: NA")
        print(f"  β_self = {out['beta_self']:+.4f} (p={out['beta_self_p']:.4g})")
        print(f"\n  ring spillover θ_k:")
        edges = cfg.ring_edges_um
        for k, (th, se, tt, pp) in enumerate(zip(out["theta_k"], out["theta_se"],
                                                  out["theta_t"], out["theta_p"])):
            star = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
            print(f"    ring {edges[k]:>3}-{edges[k+1]:>3} µm: "
                  f"θ={th:+.4f} ±{se:.4f} (t={tt:+.2f}, p={pp:.4g}) {star}")


if __name__ == "__main__":
    main()
