"""Multi-source spatial Durbin model (v2 Layer 3, causal propagation target).

Unlike single-source SLX (which assigns each spot to nearest source),
Durbin models a continuous "exposure field" from all sources:

    exposure_field_i = Σ_c M_c · K(d(i, c), λ)

where M_c = source strength (clone size × guide confidence) and
       K = Gaussian kernel with learnable length scale λ.

This handles multi-source overlap correctly: a spot near 2 sources gets
combined exposure, and the regression coefficient δ captures the
per-unit-exposure effect.

Model:
    Y_i = α + β·T_i + δ·exposure_field_i + γ·X_i + ε_i

δ > 0 = positive propagation (perturbation increases Y beyond cell-autonomous)
δ < 0 = inhibitory propagation
δ ≈ 0 = no non-cell-autonomous effect

This is the v2 attempt at causal claim (c): does perturbing a gene
change the phenotype of *non-perturbed neighboring cells*?
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


@dataclass
class DurbinConfig:
    kernel_lambda_um: float = 60.0       # Gaussian length scale
    max_exposure_radius_um: float = 300.0 # cutoff for kernel sum
    embed_covariates: int = 8
    min_sources: int = 2
    min_analysis_spots: int = 30


def _compute_source_strength(adata, target_gene: str) -> np.ndarray:
    """Per-source strength M_c = clone_size_normalized × mean guide_confidence."""
    obs = adata.obs
    is_source = (
        (obs["guide_total"].to_numpy() >= 2)
        & (obs["top_fraction"].to_numpy() >= 0.70)
        & (obs["target_gene"].to_numpy() == target_gene)
    )
    if is_source.sum() < 1:
        return np.array([]), is_source

    xy = adata.obsm["spatial"].astype(np.float32)
    source_xy = xy[is_source]
    source_frac = obs["top_fraction"].to_numpy()[is_source]
    source_total = obs["guide_total"].to_numpy()[is_source]

    # cluster source spots into clones (DBSCAN eps=24µm)
    from sklearn.cluster import DBSCAN
    clone_labels = DBSCAN(eps=15, min_samples=1).fit_predict(source_xy)
    n_clones = len(set(clone_labels))
    M = np.zeros(n_clones, dtype=np.float32)
    for c in range(n_clones):
        mask = clone_labels == c
        size = mask.sum()
        conf = (source_frac[mask] * source_total[mask]).mean()
        M[c] = float(size) * float(conf)
    # normalize M to [0, 1] for numerical stability
    if M.max() > 0:
        M /= M.max()
    # also return clone centroids
    centroids = np.array([source_xy[clone_labels == c].mean(axis=0)
                          for c in range(n_clones)])
    return M, centroids, is_source


def _compute_exposure_field(xy: np.ndarray, centroids: np.ndarray,
                              M: np.ndarray, cfg: DurbinConfig) -> np.ndarray:
    """Vectorized: Σ_c M_c · exp(-d²/(2λ²)) using sparse matrix multiply."""
    import scipy.sparse as sp
    src_tree = cKDTree(centroids)
    # find all (spot, source) pairs within max_exposure_radius
    pairs = src_tree.query_ball_point(xy, r=cfg.max_exposure_radius_um)
    # build sparse COO: rows = spot idx, cols = source idx, vals = M_c · K(d)
    rows, cols, vals = [], [], []
    lam2 = 2 * cfg.kernel_lambda_um ** 2
    for i, srcs in enumerate(pairs):
        if not srcs:
            continue
        srcs = np.array(srcs)
        d2 = np.sum((centroids[srcs] - xy[i]) ** 2, axis=1)
        w = M[srcs] * np.exp(-d2 / lam2)
        rows.extend([i] * len(srcs))
        cols.extend(srcs.tolist())
        vals.extend(w.tolist())
    if not rows:
        return np.zeros(xy.shape[0], dtype=np.float32)
    W = sp.csr_matrix((vals, (rows, cols)),
                      shape=(xy.shape[0], len(centroids)))
    exposure = np.asarray(W @ M).ravel()
    return exposure.astype(np.float32)


def fit_durbin(
    adata: ad.AnnData,
    embed: np.ndarray,
    target_gene: str,
    response: str,
    cfg: DurbinConfig = DurbinConfig(),
    lambda_grid: tuple = (30, 60, 100, 150),
) -> dict:
    """Fit spatial Durbin with grid search over kernel λ.

    Returns best-λ result + all-λ grid for sensitivity.
    """
    obs = adata.obs
    xy = adata.obsm["spatial"].astype(np.float32)

    result = _compute_source_strength(adata, target_gene)
    if len(result) == 2:
        return {"error": "no sources"}
    M, centroids, is_source = result
    if len(centroids) < cfg.min_sources:
        return {"error": f"too few clones: {len(centroids)}"}

    src_tree = cKDTree(centroids)
    dist_to_src, _ = src_tree.query(xy, k=1)
    in_analysis = dist_to_src <= cfg.max_exposure_radius_um

    if response not in obs.columns:
        if f"score_{response}" in obs.columns:
            response = f"score_{response}"
        else:
            return {"error": f"response {response} not in obs"}
    y_all = obs[response].to_numpy()
    valid = ~pd.isna(y_all)
    keep = in_analysis & valid
    if keep.sum() < cfg.min_analysis_spots:
        return {"error": "too few analysis spots"}

    y = y_all[keep].astype(np.float32)
    T = is_source[keep].astype(np.float32)
    X_embed = embed[keep][:, :cfg.embed_covariates]

    from .slx import _assign_clone_id
    clone_id = _assign_clone_id(adata, target_gene)[keep]
    n_clones = int((clone_id >= 0).sum() > 0
                    and len(set(clone_id[clone_id >= 0])))

    results_per_lambda = {}
    best_p = float("inf")
    best_lambda = None
    best_result = None

    for lam in lambda_grid:
        cfg_lam = DurbinConfig(kernel_lambda_um=lam,
                                max_exposure_radius_um=cfg.max_exposure_radius_um,
                                embed_covariates=cfg.embed_covariates)
        t0 = time.time()
        exposure_full = _compute_exposure_field(xy, centroids, M, cfg_lam)
        exposure = exposure_full[keep]

        X = np.column_stack([T, exposure, X_embed])
        col_names = (["T_self", "exposure"]
                     + [f"embed_pc{i+1}" for i in range(cfg.embed_covariates)])
        X_df = pd.DataFrame(X, columns=col_names)
        X_df = sm.add_constant(X_df)

        clusters = np.where(clone_id >= 0, clone_id,
                            np.arange(int(keep.sum())) + n_clones + 1)
        try:
            model = sm.OLS(y, X_df).fit(
                cov_type="cluster", cov_kwds={"groups": clusters})
        except Exception as e:
            results_per_lambda[lam] = {"error": str(e)}
            continue

        delta = float(model.params["exposure"])
        delta_se = float(model.bse["exposure"])
        delta_t = delta / (delta_se + 1e-12)
        delta_p = float(model.pvalues["exposure"])

        result_lam = {
            "lambda_um": lam,
            "delta": delta,
            "delta_se": delta_se,
            "delta_t": delta_t,
            "delta_p": delta_p,
            "beta_self": float(model.params["T_self"]),
            "beta_self_p": float(model.pvalues["T_self"]),
            "r2": float(model.rsquared),
            "n_clones_in_model": len(centroids),
            "n_obs": int(keep.sum()),
            "compute_s": time.time() - t0,
        }
        results_per_lambda[lam] = result_lam
        if delta_p < best_p:
            best_p = delta_p
            best_lambda = lam
            best_result = result_lam

    return {
        "best_lambda_um": best_lambda,
        "best": best_result,
        "all_lambdas": results_per_lambda,
        "n_sources": int(is_source.sum()),
        "n_clones": len(centroids),
    }
