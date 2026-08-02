"""Multi-ring hierarchical Spatial Lag of X (SLX) model (v2 Layer 3).

For each target gene G and response Y in slice s:
  logit(E[Y_ij]) = α + β·T_ij + Σ_k θ_k·(W_k T)_ij
                   + γ·e_ij              # GNN embedding covariates (PC1-8)
                   + u_clone             # random effect

where T_ij = 1 if spot i is in a source clone of gene G,
      W_k = row-normalized spatial weights matrix for ring k
      (k ∈ {0-20, 20-40, 40-80, 80-120, 120-200} µm).

θ_k captures the spillover effect at distance ring k. The full
distance-response curve is the vector (θ_1, ..., θ_5).

This replaces the v1 concentric-ring Δ statistic with a proper
regression framework that:
  - controls for embedding (cell type + niche) via γ
  - has clone-level random effects (no pseudo-replication)
  - gives proper standard errors via the regression
  - allows global test H_0: θ_1 = ... = θ_5 = 0

Implementation note: for speed on 1.1M spots × multiple (gene, response)
combinations, we use OLS with cluster-robust SEs at the clone level
rather than full MixedLM. MixedLM is too slow for the genome scan.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import cKDTree


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
LOGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/logs")


@dataclass
class SLXConfig:
    ring_edges_um: tuple = (0, 20, 40, 80, 120, 200)
    embed_covariates: int = 8           # PC1..PC8 of embedding
    min_sources: int = 5
    min_analysis_spots: int = 30
    cluster_by: str = "clone_id"        # for cluster-robust SE


def _build_ring_weights(xy: np.ndarray, source_xy: np.ndarray,
                         ring_edges_um: tuple) -> np.ndarray:
    """For each spot, compute (K,) row-normalized ring exposure vector.

    For ring k, exposure = (# source points within ring_k of spot i) /
                          (# source points within max_ring of spot i).

    Returns (N, K) array of ring exposures in [0, 1].
    """
    src_tree = cKDTree(source_xy)
    N = xy.shape[0]
    K = len(ring_edges_um) - 1
    W = np.zeros((N, K), dtype=np.float32)
    # for each spot, count sources in each ring
    max_r = ring_edges_um[-1]
    # query all sources within max_r
    neigh_lists = src_tree.query_ball_point(xy, r=max_r)
    for i, srcs in enumerate(neigh_lists):
        if not srcs:
            continue
        dists = np.linalg.norm(source_xy[srcs] - xy[i], axis=1)
        for k in range(K):
            lo, hi = ring_edges_um[k], ring_edges_um[k + 1]
            W[i, k] = ((dists >= lo) & (dists < hi)).sum()
        # row-normalize
        s = W[i].sum()
        if s > 0:
            W[i] /= s
    return W


def _assign_clone_id(adata, target_gene: str) -> np.ndarray:
    """Each connected component of source spots → unique clone_id.
    Non-source spots inherit the clone_id of their nearest source.
    """
    obs = adata.obs
    is_source = (
        (obs["guide_total"].to_numpy() >= 2)
        & (obs["top_fraction"].to_numpy() >= 0.70)
        & (obs["target_gene"].to_numpy() == target_gene)
    )
    if is_source.sum() < 1:
        return np.zeros(len(obs), dtype=np.int64)
    xy = adata.obsm["spatial"].astype(np.float32)
    source_xy = xy[is_source]
    # connected components on source spots (DBSCAN with eps=24µm)
    from sklearn.cluster import DBSCAN
    src_labels = DBSCAN(eps=24, min_samples=1).fit_predict(source_xy)
    # each non-source spot inherits nearest source's clone
    src_tree = cKDTree(source_xy)
    _, nearest_src = src_tree.query(xy, k=1)
    clone_id = np.where(is_source, src_labels[nearest_src]
                        if len(src_labels) > 0 else 0, -1)
    # for non-source spots beyond max distance, clone_id = -1 (unexposed)
    max_d = 400  #µm
    dist_to_src, _ = src_tree.query(xy, k=1)
    clone_id[dist_to_src > max_d] = -1
    # but spots within 200µm of a source get that source's clone_id
    within_200 = (dist_to_src <= 200) & (~is_source)
    clone_id[within_200] = src_labels[nearest_src[within_200]]
    return clone_id


def fit_slx(
    adata: ad.AnnData,
    embed: np.ndarray,
    target_gene: str,
    response: str,
    cfg: SLXConfig = SLXConfig(),
) -> dict:
    """Fit multi-ring SLX for (gene, response) in one slice.

    Returns dict with:
      theta_k:    (K,) ring spillover coefficients
      theta_se:   (K,) standard errors
      theta_t:    (K,) t-statistics
      theta_p:    (K,) p-values
      beta_self:  cell-autonomous effect (source = 1)
      gamma:      embedding covariate coefficients
      n_obs:      sample size
      n_clones:   number of unique clones
      global_p:   LRT p-value for H_0: all θ_k = 0
    """
    obs = adata.obs
    xy = adata.obsm["spatial"].astype(np.float32)

    is_source = (
        (obs["guide_total"].to_numpy() >= 2)
        & (obs["top_fraction"].to_numpy() >= 0.70)
        & (obs["target_gene"].to_numpy() == target_gene)
    )
    n_sources = int(is_source.sum())
    if n_sources < cfg.min_sources:
        return {"error": f"too few sources: {n_sources}"}

    source_xy = xy[is_source]

    # restrict to analysis spots: within max_ring of any source
    src_tree = cKDTree(source_xy)
    dist_to_src, _ = src_tree.query(xy, k=1)
    in_analysis = dist_to_src <= cfg.ring_edges_um[-1]
    if in_analysis.sum() < cfg.min_analysis_spots:
        return {"error": f"too few analysis spots: {in_analysis.sum()}"}

    # response variable
    if response not in obs.columns:
        # try score_<response>
        if f"score_{response}" in obs.columns:
            response = f"score_{response}"
        else:
            return {"error": f"response {response} not in obs"}
    y_all = obs[response].to_numpy()
    valid = ~pd.isna(y_all)
    keep = in_analysis & valid
    if keep.sum() < cfg.min_analysis_spots:
        return {"error": f"too many NaN responses: keep={keep.sum()}"}

    # build design matrix on the kept subset
    y = y_all[keep].astype(np.float32)
    T = is_source[keep].astype(np.float32)
    X_embed = embed[keep][:, :cfg.embed_covariates]
    W_full = _build_ring_weights(xy, source_xy, cfg.ring_edges_um)
    W = W_full[keep]
    n_obs = int(keep.sum())

    # clone ID for cluster-robust SE
    clone_id = _assign_clone_id(adata, target_gene)[keep]
    n_clones = int((clone_id >= 0).sum() > 0 and len(set(clone_id[clone_id >= 0])))

    # design matrix
    K = W.shape[1]
    X = np.column_stack([T, W, X_embed])
    col_names = (["T_self"]
                 + [f"ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}"
                    for k in range(K)]
                 + [f"embed_pc{i+1}" for i in range(cfg.embed_covariates)])
    X_df = pd.DataFrame(X, columns=col_names)
    X_df = sm.add_constant(X_df)

    # OLS with cluster-robust SE by clone_id
    # use clusters only where clone_id >= 0; others get their own cluster
    clusters = np.where(clone_id >= 0, clone_id,
                       np.arange(n_obs) + n_clones + 1)
    try:
        model = sm.OLS(y, X_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )
    except Exception as e:
        return {"error": f"OLS fit failed: {e}"}

    # extract results
    theta_k = np.array([model.params[f"ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}"]
                         for k in range(K)])
    theta_se = np.array([model.bse[f"ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}"]
                          for k in range(K)])
    theta_t = theta_k / (theta_se + 1e-12)
    theta_p = np.array([model.pvalues[f"ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}"]
                         for k in range(K)])

    # global F-test (joint significance of all ring coefficients)
    ring_cols = [f"ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}" for k in range(K)]
    try:
        r_matrix = np.zeros((K, len(model.params)))
        for j, c in enumerate(ring_cols):
            r_matrix[j, list(model.params.index).index(c)] = 1.0
        f_test = model.f_test(r_matrix)
        global_p = float(f_test.pvalue)
        global_f = float(f_test.fvalue) if hasattr(f_test, "fvalue") else None
    except Exception:
        global_p = None
        global_f = None

    return {
        "theta_k": theta_k.tolist(),
        "theta_se": theta_se.tolist(),
        "theta_t": theta_t.tolist(),
        "theta_p": theta_p.tolist(),
        "beta_self": float(model.params["T_self"]),
        "beta_self_p": float(model.pvalues["T_self"]),
        "embed_gamma": [float(model.params[f"embed_pc{i+1}"])
                         for i in range(cfg.embed_covariates)],
        "n_obs": n_obs,
        "n_sources": n_sources,
        "n_clones": n_clones,
        "global_p": global_p,
        "global_f": global_f,
        "r2": float(model.rsquared),
    }
