"""Resistance-distance SLX (v2 Layer 3, key innovation), optimized.

Replaces Euclidean ring weights with effective resistance on the tissue
graph. Two spots at 50 µm Euclidean may have very different resistance
if separated by a necrotic region or cell-type boundary.

Approximation (for tractability on 1.1M spots):
    R(i,j) ≈ ||xy_i - xy_j|| · (1 + α_b · avg_barrier - α_v · avg_vessel)

where averages are sampled along the straight-line path (K points).

Optimization vs v1: only compute resistance for analysis spots (within
max_r of a source), not all 358K spots. This brings M002 Ccn1 from >3min
to <30s.
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


@dataclass
class ResistanceConfig:
    ring_edges_um: tuple = (0, 20, 40, 80, 120, 200)
    embed_covariates: int = 8
    min_sources: int = 5
    min_analysis_spots: int = 30
    barrier_alpha: float = 1.0
    vessel_alpha: float = 0.3
    path_sample_points: int = 5


def _compute_node_barrier_score(adata) -> np.ndarray:
    xy = adata.obsm["spatial"].astype(np.float32)
    ct = adata.obs["cell_type"].to_numpy()
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=13)
    nbr_ct = ct[idx[:, 1:]]
    n_distinct = np.array([len(set(r[r != "Missing"])) for r in nbr_ct],
                          dtype=np.float32)
    density = tree.query_ball_point(xy, r=40.0, return_length=True) - 1
    density = density.astype(np.float32)
    necrosis = np.clip(1.0 - density / 30.0, 0, 1)
    return ((n_distinct - 1) / 7.0 + necrosis).astype(np.float32)


def _compute_vessel_score(adata) -> np.ndarray:
    if "module_scores" not in adata.obsm:
        return np.zeros(adata.n_obs, dtype=np.float32)
    endo = adata.obsm["module_scores"][:, 4]
    return np.clip((endo - endo.min()) / (endo.max() - endo.min() + 1e-8),
                   0, 1).astype(np.float32)


def _resistance_ring_weights_subset(
    analysis_xy: np.ndarray,
    source_xy: np.ndarray,
    barrier_field: np.ndarray,
    vessel_field: np.ndarray,
    xy_all: np.ndarray,
    cfg: ResistanceConfig,
) -> np.ndarray:
    """For each analysis spot, compute (K,) resistance-ring exposures."""
    src_tree = cKDTree(source_xy)
    nA = analysis_xy.shape[0]
    K = len(cfg.ring_edges_um) - 1
    W = np.zeros((nA, K), dtype=np.float32)
    max_r = cfg.ring_edges_um[-1]
    neigh_lists = src_tree.query_ball_point(analysis_xy, r=max_r)
    all_tree = cKDTree(xy_all)
    ts = np.linspace(0, 1, cfg.path_sample_points)

    for i in range(nA):
        srcs = neigh_lists[i]
        if not srcs:
            continue
        srcs = np.array(srcs)
        n_srcs = len(srcs)
        paths = (analysis_xy[i][None, None, :]
                 + ts[None, :, None] *
                 (source_xy[srcs][:, None, :] - analysis_xy[i][None, None, :]))
        paths = paths.reshape(-1, 2)
        _, nearest = all_tree.query(paths, k=1)
        b_per = barrier_field[nearest].reshape(n_srcs, -1).mean(axis=1)
        v_per = vessel_field[nearest].reshape(n_srcs, -1).mean(axis=1)
        eucl = np.linalg.norm(source_xy[srcs] - analysis_xy[i], axis=1)
        factor = np.maximum(1.0 + cfg.barrier_alpha * b_per
                            - cfg.vessel_alpha * v_per, 0.1)
        d_res = eucl * factor
        for k in range(K):
            lo, hi = cfg.ring_edges_um[k], cfg.ring_edges_um[k + 1]
            W[i, k] = ((d_res >= lo) & (d_res < hi)).sum()
        s = W[i].sum()
        if s > 0:
            W[i] /= s
    return W


def fit_resistance_slx(
    adata: ad.AnnData,
    embed: np.ndarray,
    target_gene: str,
    response: str,
    cfg: ResistanceConfig = ResistanceConfig(),
) -> dict:
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
    src_tree = cKDTree(source_xy)
    dist_to_src, _ = src_tree.query(xy, k=1)
    in_analysis = dist_to_src <= cfg.ring_edges_um[-1]
    if in_analysis.sum() < cfg.min_analysis_spots:
        return {"error": f"too few analysis spots: {in_analysis.sum()}"}

    if response not in obs.columns:
        if f"score_{response}" in obs.columns:
            response = f"score_{response}"
        else:
            return {"error": f"response {response} not in obs"}
    y_all = obs[response].to_numpy()
    valid = ~pd.isna(y_all)
    keep = in_analysis & valid
    if keep.sum() < cfg.min_analysis_spots:
        return {"error": "too many NaN responses"}

    y = y_all[keep].astype(np.float32)
    T = is_source[keep].astype(np.float32)
    X_embed = embed[keep][:, :cfg.embed_covariates]

    print(f"[resistance] computing barrier + vessel fields...", flush=True)
    t0 = time.time()
    barrier_field = _compute_node_barrier_score(adata)
    vessel_field = _compute_vessel_score(adata)
    print(f"[resistance] fields done in {time.time()-t0:.1f}s", flush=True)

    print(f"[resistance] building resistance ring weights on "
          f"{int(keep.sum()):,} analysis spots...", flush=True)
    t0 = time.time()
    W = _resistance_ring_weights_subset(
        xy[keep], source_xy, barrier_field, vessel_field, xy, cfg,
    )
    print(f"[resistance] weights done in {time.time()-t0:.1f}s", flush=True)

    from .slx import _assign_clone_id
    clone_id = _assign_clone_id(adata, target_gene)[keep]
    n_clones = int((clone_id >= 0).sum() > 0
                    and len(set(clone_id[clone_id >= 0])))

    K = W.shape[1]
    X = np.column_stack([T, W, X_embed])
    col_names = (["T_self"]
                 + [f"res_ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}"
                    for k in range(K)]
                 + [f"embed_pc{i+1}" for i in range(cfg.embed_covariates)])
    X_df = pd.DataFrame(X, columns=col_names)
    X_df = sm.add_constant(X_df)

    clusters = np.where(clone_id >= 0, clone_id,
                       np.arange(int(keep.sum())) + n_clones + 1)
    try:
        model = sm.OLS(y, X_df).fit(
            cov_type="cluster", cov_kwds={"groups": clusters})
    except Exception as e:
        return {"error": f"OLS fit failed: {e}"}

    theta_k = np.array([model.params[f"res_ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}"]
                         for k in range(K)])
    theta_se = np.array([model.bse[f"res_ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}"]
                          for k in range(K)])
    theta_t = theta_k / (theta_se + 1e-12)
    theta_p = np.array([model.pvalues[f"res_ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}"]
                         for k in range(K)])

    ring_cols = [f"res_ring_{cfg.ring_edges_um[k]}_{cfg.ring_edges_um[k+1]}" for k in range(K)]
    try:
        r_matrix = np.zeros((K, len(model.params)))
        for j, c in enumerate(ring_cols):
            r_matrix[j, list(model.params.index).index(c)] = 1.0
        f_test = model.f_test(r_matrix)
        global_p = float(f_test.pvalue)
    except Exception:
        global_p = None

    return {
        "theta_k": theta_k.tolist(),
        "theta_se": theta_se.tolist(),
        "theta_t": theta_t.tolist(),
        "theta_p": theta_p.tolist(),
        "beta_self": float(model.params["T_self"]),
        "beta_self_p": float(model.pvalues["T_self"]),
        "embed_gamma": [float(model.params[f"embed_pc{i+1}"])
                         for i in range(cfg.embed_covariates)],
        "n_obs": int(keep.sum()),
        "n_sources": n_sources,
        "n_clones": n_clones,
        "global_p": global_p,
        "r2": float(model.rsquared),
    }
