"""Quick smoke test for matched control on M002."""
import sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from perturbgnn_v2.matching import (
    MatchConfig, matched_control_for_gene, ntc_sanity_check, 
)

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")

print("[test] loading M002 adata + embedding...", flush=True)
a = ad.read_h5ad(PROCESSED / "M002_v2.h5ad")
emb = np.load(PROCESSED / "embed_M002_v3.npy")
print(f"  adata shape: {a.shape}, embed shape: {emb.shape}", flush=True)

cfg = MatchConfig()
print(f"\n[config] {cfg}", flush=True)

# NTC sanity first
print("\n[test] === NTC sanity check ===", flush=True)
t0 = time.time()
ntc_df = ntc_sanity_check(a, emb, cfg)
print(f"  ran in {time.time()-t0:.1f}s, n_ntc_analysis_spots={len(ntc_df)}", flush=True)
if len(ntc_df) > 0:
    delta_cols = [c for c in ntc_df.columns if c.endswith("_delta")]
    print(f"\n  NTC matched Δ per response (should be ≈ 0):")
    for c in delta_cols:
        vals = ntc_df[c].dropna()
        if len(vals) > 0:
            print(f"    {c:40s}: mean={vals.mean():+.4f}, median={vals.median():+.4f}, n={len(vals)}")

# Then Ccn1 (top target gene in M002)
print("\n[test] === Ccn1 matched control ===", flush=True)
t0 = time.time()
df, bal = matched_control_for_gene(a, emb, "Ccn1", cfg)
print(f"  ran in {time.time()-t0:.1f}s, n_analysis_spots={len(df)}", flush=True)
if len(df) > 0:
    print(f"  n_sources: {df['n_sources'].iloc[0]}")
    delta_cols = [c for c in df.columns if c.endswith("_delta")]
    print(f"\n  Ccn1 matched Δ per response:")
    for c in delta_cols:
        vals = df[c].dropna()
        if len(vals) > 0:
            print(f"    {c:40s}: mean={vals.mean():+.4f}, median={vals.median():+.4f}, n={len(vals)}")

# Covariate balance (already computed inline)
print("\n[test] === Ccn1 covariate balance ===", flush=True)
if len(bal) > 0:
    print(f"  |SMD| max: {bal['smd'].abs().max():.4f}  (target < 0.1)")
    print(f"  |SMD| > 0.1 covariates: {(bal['smd'].abs() > 0.1).sum()} / {len(bal)}")
    print(f"  worst 5:")
    print(bal.reindex(bal.smd.abs().sort_values(ascending=False).index).head(5).to_string(index=False))
