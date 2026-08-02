"""Genome-wide causal scan (v2 Layer 4 orchestrator).

For each (gene G, response Y, slice s) with sufficient samples:
  - SLX (euclidean + resistance) — descriptive
  - Durbin — propagation slope δ
  - Spatial DiD — causal identification
  - Guide IV — confounding diagnosis
  - Rosenbaum Γ* — robustness

Output: genome_scan_v2.csv with all diagnostics per row.

Aggregation:
  - Stouffer combined p across slices (≥2 slices)
  - BH FDR across all (gene, response) pairs
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import json
import time
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues

from perturbgnn_v2.spatial import (
    fit_slx, SLXConfig,
    fit_resistance_slx, ResistanceConfig,
    fit_durbin, DurbinConfig,
)
from perturbgnn_v2.causal import (
    CausalConfig, spatial_did, guide_iv, rosenbaum_sensitivity,
)
from perturbgnn_v2.matching.match import _attach_module_scores


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
LOGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/logs")


# 8 responses we scan
DEFAULT_RESPONSES = (
    "score_malignant", "score_cd8_like", "score_macrophage",
    "score_fibroblast", "score_endothelial",
    "score_hypoxia", "score_ifn_response",
)


def _min_gene_source_count(adata, min_count: int = 30) -> list[str]:
    """Genes with enough source bins in this slice to be analyzable."""
    obs = adata.obs
    is_source = (
        (obs["guide_total"].to_numpy() >= 2)
        & (obs["top_fraction"].to_numpy() >= 0.70)
        & (~obs["is_ntc"].to_numpy())
    )
    counts = obs.loc[is_source, "target_gene"].value_counts()
    return counts[counts >= min_count].index.tolist()


def scan_one_slice(
    slice_name: str,
    responses: Iterable[str] = DEFAULT_RESPONSES,
    min_gene_sources: int = 30,
    durbin_only: bool = True,  # skip SLX/resistance for speed in full scan
    fast: bool = False,
    max_genes: int = 0,
) -> list[dict]:
    """Run all diagnostics for one slice. Returns list of row dicts."""
    print(f"\n[scan] === {slice_name} ===", flush=True)
    a = ad.read_h5ad(PROCESSED / f"{slice_name}_v2.h5ad")
    emb = np.load(PROCESSED / f"embed_{slice_name}_v3.npy")
    _attach_module_scores(a)

    genes = _min_gene_source_count(a, min_count=min_gene_sources)
    if max_genes > 0 and len(genes) > max_genes:
        genes = genes[:max_genes]
    print(f"[scan] {len(genes)} genes with ≥{min_gene_sources} sources", flush=True)

    rows = []
    for gi, gene in enumerate(genes):
        if gi % 10 == 0:
            print(f"[scan]   {slice_name} gene {gi+1}/{len(genes)}: {gene}", flush=True)
        for resp in responses:
            row = {"slice": slice_name, "gene": gene, "response": resp}
            # Durbin (always)
            try:
                d = fit_durbin(a, emb, gene, resp, DurbinConfig())
                if "error" not in d:
                    b = d["best"]
                    row["durbin_delta"] = b["delta"]
                    row["durbin_p"] = b["delta_p"]
                    row["durbin_lambda"] = b["lambda_um"]
                    row["durbin_beta_self"] = b["beta_self"]
                    row["durbin_r2"] = b["r2"]
                    row["n_clones"] = d["n_clones"]
            except Exception as e:
                row["durbin_error"] = str(e)[:80]

            # DiD
            try:
                did = spatial_did(a, emb, gene, resp, CausalConfig())
                if "error" not in did:
                    row["did_delta"] = did["delta_did"]
                    row["did_p"] = did["p"]
                    row["did_n_pert_near"] = did["n"]["pert_near"]
            except Exception as e:
                row["did_error"] = str(e)[:80]

            # IV (skip in fast mode)
            if not fast:
                try:
                    iv = guide_iv(a, emb, gene, resp, CausalConfig())
                    if "error" not in iv:
                        row["iv_beta_ols"] = iv["beta_ols"]
                        row["iv_beta_iv"] = iv["beta_iv"]
                        row["iv_ratio"] = iv["iv_ratio"]
                        row["iv_stage1_f"] = iv["stage1_f"]
                except Exception as e:
                    row["iv_error"] = str(e)[:80]

            # Rosenbaum (skip in fast mode)
            if not fast:
                try:
                    ros = rosenbaum_sensitivity(a, emb, gene, resp, CausalConfig())
                    if "error" not in ros:
                        row["ros_base_beta"] = ros["base_beta"]
                        row["ros_base_p"] = ros["base_p"]
                        gs = ros["gamma_star"]
                        row["ros_gamma_star"] = (
                            float(gs) if isinstance(gs, (int, float)) else 4.0
                        )
                except Exception as e:
                    row["ros_error"] = str(e)[:80]

            rows.append(row)
    return rows


def aggregate_fdr(rows: list[dict]) -> pd.DataFrame:
    """Build DataFrame, aggregate across slices, BH FDR."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # combined p per (gene, response) via Stouffer (slices with valid p)
    def combine(group):
        ps = group["durbin_p"].dropna().tolist()
        if len(ps) >= 2:
            _, comb = combine_pvalues(ps, method="stouffer")
            return comb
        elif len(ps) == 1:
            return ps[0]
        return np.nan

    combined = df.groupby(["gene", "response"]).apply(combine).reset_index()
    combined.columns = ["gene", "response", "combined_p"]

    # BH FDR
    valid = combined["combined_p"].dropna()
    if len(valid) > 0:
        ranks = valid.rank(method="first")
        fdr = valid * len(valid) / ranks
        fdr = fdr.clip(upper=1.0)
        combined.loc[valid.index, "combined_q"] = fdr.values
    else:
        combined["combined_q"] = np.nan

    return df, combined


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="M001,M002,M003")
    ap.add_argument("--out", default=str(PROCESSED / "genome_scan_v2.csv"))
    ap.add_argument("--min_sources", type=int, default=30)
    ap.add_argument("--fast", action="store_true",
                    help="skip IV and Rosenbaum (use when scanning many genes)")
    ap.add_argument("--max_genes", type=int, default=0,
                    help="cap number of genes (0 = no cap, for fast testing)")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    log_file = LOGS / f"p4_scan_{int(time.time())}.log"
    print(f"[scan] log: {log_file}", flush=True)
    def log(m):
        print(m, flush=True)
        with open(log_file, "a") as f: f.write(m + "\n")

    all_rows = []
    for s in args.slices.split(","):
        log(f"[scan] starting {s}...")
        t0 = time.time()
        rows = scan_one_slice(s, min_gene_sources=args.min_sources,
                                   fast=args.fast, max_genes=args.max_genes)
        log(f"[scan] {s} done in {time.time()-t0:.1f}s, {len(rows)} rows")
        all_rows.extend(rows)

    log(f"\n[scan] aggregating {len(all_rows)} rows...")
    df, combined = aggregate_fdr(all_rows)

    df.to_csv(args.out, index=False)
    comb_path = args.out.replace(".csv", "_combined.csv")
    combined.to_csv(comb_path, index=False)
    log(f"[scan] wrote: {args.out}")
    log(f"[scan] wrote: {comb_path}")

    # top hits summary
    if "combined_q" in combined.columns:
        sig = combined[combined["combined_q"] < 0.05].sort_values("combined_q")
        log(f"\n[scan] {len(sig)} significant (gene, response) pairs at q<0.05:")
        if len(sig) > 0:
            log(sig.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
