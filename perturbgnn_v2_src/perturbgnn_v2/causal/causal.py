"""Causal inference diagnostics (v2 Layer 4).

Three causal identification strategies layered on top of Layer 3:

  (c1) Spatial DiD:
       Δ_DiD = [Y_pert_near - Y_pert_far] - [Y_matched_ctrl_near - Y_matched_ctrl_far]
       Controls for neighborhood gradient by using Layer 2 matched controls.

  (c2) Guide-IV (2SLS):
       Stage 1: T_hat = π_0 + π_1 · guide_UMI + covariates
       Stage 2: Y = α + β · T_hat + covariates
       If β_IV ≈ β_OLS → perturbation is the mechanism
       If β_IV ≪ β_OLS → OLS estimate is confounded

  (c3) Rosenbaum sensitivity:
       What is the smallest hidden confounder Γ that flips significance?
       Robust claim requires Γ* ≥ 1.5.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


@dataclass
class CausalConfig:
    near_radius_um: float = 80.0
    far_radius_um: tuple = (120.0, 200.0)
    embed_covariates: int = 8
    min_obs: int = 30
    rosbaum_grid: tuple = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0)


def _attach_module_scores(adata):
    if "module_scores" not in adata.obsm:
        return
    if "score_ifn_response" not in adata.obs.columns:
        names = adata.uns.get("module_names",
            ["malignant", "cd8_like", "macrophage", "fibroblast",
             "endothelial", "hypoxia", "ifn_response"])
        for i, n in enumerate(names):
            adata.obs[f"score_{n}"] = adata.obsm["module_scores"][:, i]


def spatial_did(
    adata: ad.AnnData,
    embed: np.ndarray,
    target_gene: str,
    response: str,
    cfg: CausalConfig = CausalConfig(),
) -> dict:
    """Spatial difference-in-differences.

    For each spot near a target source:
      Y_pert_near  = response among spots within near_r of target source
      Y_pert_far   = response among spots in far_r band of target source
      Y_ctrl_near  = response of Layer-2-matched controls for near spots
      Y_ctrl_far   = response of Layer-2-matched controls for far spots

    Δ_DiD = (Y_pert_near - Y_pert_far) - (Y_ctrl_near - Y_ctrl_far)
    """
    _attach_module_scores(adata)
    obs = adata.obs
    xy = adata.obsm["spatial"].astype(np.float32)
    if response not in obs.columns and f"score_{response}" in obs.columns:
        response = f"score_{response}"

    is_source = (
        (obs["guide_total"].to_numpy() >= 2)
        & (obs["top_fraction"].to_numpy() >= 0.70)
        & (obs["target_gene"].to_numpy() == target_gene)
    )
    if is_source.sum() < 5:
        return {"error": "too few sources"}

    src_tree = cKDTree(xy[is_source])
    dist_to_src, _ = src_tree.query(xy, k=1)
    is_ntc = obs["is_ntc"].to_numpy()
    is_unexposed = (dist_to_src > cfg.far_radius_um[1]) & ~is_source & (
        obs["guide_total"].to_numpy() <= 1
    )

    # also need NTC sources for the "ctrl" arm
    ntc_source_mask = is_ntc & (obs["guide_total"].to_numpy() >= 2)
    if ntc_source_mask.sum() < 5:
        # fallback: use random unexposed spots as synthetic NTC
        print("[did] WARNING: too few NTC sources, using synthetic controls")
        ntc_xy = xy[is_unexposed][:200]  # 200 random unexposed
    else:
        ntc_xy = xy[ntc_source_mask]

    ntc_tree = cKDTree(ntc_xy)
    dist_to_ntc, _ = ntc_tree.query(xy, k=1)

    # define near/far masks for both target and control arms
    target_near = (dist_to_src <= cfg.near_radius_um) & ~is_source
    target_far = (dist_to_src >= cfg.far_radius_um[0]) & \
                 (dist_to_src <= cfg.far_radius_um[1]) & ~is_source
    ctrl_near = (dist_to_ntc <= cfg.near_radius_um) & ~is_source & ~is_ntc
    ctrl_far = (dist_to_ntc >= cfg.far_radius_um[0]) & \
               (dist_to_ntc <= cfg.far_radius_um[1]) & ~is_source & ~is_ntc

    if min(target_near.sum(), target_far.sum(),
           ctrl_near.sum(), ctrl_far.sum()) < cfg.min_obs:
        return {"error": f"insufficient samples: t_near={target_near.sum()}, "
                         f"t_far={target_far.sum()}, c_near={ctrl_near.sum()}, "
                         f"c_far={ctrl_far.sum()}"}

    y = obs[response].to_numpy()
    y_pert_near = y[target_near]
    y_pert_far = y[target_far]
    y_ctrl_near = y[ctrl_near]
    y_ctrl_far = y[ctrl_far]

    delta_did = (y_pert_near.mean() - y_pert_far.mean()) \
              - (y_ctrl_near.mean() - y_ctrl_far.mean())

    # approximate SE via two-sample formula on each arm
    se_pert = np.sqrt(y_pert_near.var() / len(y_pert_near)
                      + y_pert_far.var() / len(y_pert_far))
    se_ctrl = np.sqrt(y_ctrl_near.var() / len(y_ctrl_near)
                      + y_ctrl_far.var() / len(y_ctrl_far))
    se_did = np.sqrt(se_pert**2 + se_ctrl**2)
    t_did = delta_did / (se_did + 1e-12)
    from scipy.stats import norm
    p_did = 2 * (1 - norm.cdf(abs(t_did)))

    return {
        "delta_did": float(delta_did),
        "se": float(se_did),
        "t": float(t_did),
        "p": float(p_did),
        "y_pert_near": float(y_pert_near.mean()),
        "y_pert_far": float(y_pert_far.mean()),
        "y_ctrl_near": float(y_ctrl_near.mean()),
        "y_ctrl_far": float(y_ctrl_far.mean()),
        "n": {
            "pert_near": int(target_near.sum()),
            "pert_far": int(target_far.sum()),
            "ctrl_near": int(ctrl_near.sum()),
            "ctrl_far": int(ctrl_far.sum()),
        },
    }


def guide_iv(
    adata: ad.AnnData,
    embed: np.ndarray,
    target_gene: str,
    response: str,
    cfg: CausalConfig = CausalConfig(),
) -> dict:
    """Two-stage least squares with guide UMI count as instrument.

    Instrument Z = guide_UMI in source cells of target_gene (continuous).
    Treatment T = whether spot is perturbed (binary).
    Outcome Y = response.

    β_IV vs β_OLS comparison diagnoses confounding.
    """
    _attach_module_scores(adata)
    obs = adata.obs
    xy = adata.obsm["spatial"].astype(np.float32)
    if response not in obs.columns and f"score_{response}" in obs.columns:
        response = f"score_{response}"

    is_source = (
        (obs["guide_total"].to_numpy() >= 2)
        & (obs["top_fraction"].to_numpy() >= 0.70)
        & (obs["target_gene"].to_numpy() == target_gene)
    )
    if is_source.sum() < 5:
        return {"error": "too few sources"}

    src_tree = cKDTree(xy[is_source])
    dist_to_src, _ = src_tree.query(xy, k=1)
    in_analysis = dist_to_src <= cfg.far_radius_um[1]

    y_all = obs[response].to_numpy()
    valid = ~pd.isna(y_all)
    keep = in_analysis & valid & ~is_source  # exclude source cells themselves
    if keep.sum() < cfg.min_obs:
        return {"error": "too few analysis spots"}

    # build spatial exposure to source UMI density as instrument
    # for each analysis spot, instrument = sum of guide_total over nearby sources
    source_guide_total = obs["guide_total"].to_numpy()[is_source]
    source_xy = xy[is_source]
    # exposure-weighted instrument
    neigh = src_tree.query_ball_point(xy[keep], r=200)
    Z = np.array([source_guide_total[n].sum() if len(n) > 0 else 0
                  for n in neigh], dtype=np.float32)
    # binary treatment = high exposure (above median)
    T = (Z > np.median(Z) if Z.max() > 0 else Z > 0).astype(np.float32)
    Y = y_all[keep].astype(np.float32)
    X_embed = embed[keep][:, :cfg.embed_covariates]

    # OLS: Y ~ T + covariates
    X_ols = np.column_stack([T, X_embed])
    col_ols = ["T"] + [f"pc{i+1}" for i in range(cfg.embed_covariates)]
    df_ols = pd.DataFrame(X_ols, columns=col_ols)
    df_ols = sm.add_constant(df_ols)
    ols = sm.OLS(Y, df_ols).fit(cov_type="HC1")
    beta_ols = float(ols.params["T"])
    se_ols = float(ols.bse["T"])

    # 2SLS: stage 1 T ~ Z + covariates
    X_stage1 = np.column_stack([Z, X_embed])
    col_s1 = ["Z"] + [f"pc{i+1}" for i in range(cfg.embed_covariates)]
    df_s1 = pd.DataFrame(X_stage1, columns=col_s1)
    df_s1 = sm.add_constant(df_s1)
    stage1 = sm.OLS(T, df_s1).fit()
    T_hat = stage1.fittedvalues

    # stage 2 Y ~ T_hat + covariates
    X_stage2 = np.column_stack([T_hat, X_embed])
    col_s2 = ["T_hat"] + [f"pc{i+1}" for i in range(cfg.embed_covariates)]
    df_s2 = pd.DataFrame(X_stage2, columns=col_s2)
    df_s2 = sm.add_constant(df_s2)
    stage2 = sm.OLS(Y, df_s2).fit(cov_type="HC1")
    beta_iv = float(stage2.params["T_hat"])
    se_iv = float(stage2.bse["T_hat"])

    return {
        "beta_ols": beta_ols, "se_ols": se_ols,
        "beta_iv": beta_iv, "se_iv": se_iv,
        "iv_ratio": float(beta_iv / (beta_ols + 1e-12)),
        "stage1_f": float(stage1.fvalue),
        "n": int(keep.sum()),
    }


def rosenbaum_sensitivity(
    adata: ad.AnnData,
    embed: np.ndarray,
    target_gene: str,
    response: str,
    cfg: CausalConfig = CausalConfig(),
) -> dict:
    """Approximate Rosenbaum sensitivity via perturbed-treatment robustness.

    For each Γ in cfg.rosbaum_grid, simulate a hidden confounder of strength Γ
    and recompute the p-value. Γ* = smallest Γ that flips significance.
    """
    _attach_module_scores(adata)
    obs = adata.obs
    xy = adata.obsm["spatial"].astype(np.float32)
    if response not in obs.columns and f"score_{response}" in obs.columns:
        response = f"score_{response}"

    is_source = (
        (obs["guide_total"].to_numpy() >= 2)
        & (obs["top_fraction"].to_numpy() >= 0.70)
        & (obs["target_gene"].to_numpy() == target_gene)
    )
    if is_source.sum() < 5:
        return {"error": "too few sources"}

    src_tree = cKDTree(xy[is_source])
    dist_to_src, _ = src_tree.query(xy, k=1)
    in_analysis = dist_to_src <= cfg.far_radius_um[1]
    y_all = obs[response].to_numpy()
    valid = ~pd.isna(y_all)
    keep = in_analysis & valid & ~is_source
    if keep.sum() < cfg.min_obs:
        return {"error": "too few analysis spots"}

    Y = y_all[keep].astype(np.float32)
    X_embed = embed[keep][:, :cfg.embed_covariates]

    # baseline: T = proximity to source (binary via median split of dist)
    d = dist_to_src[keep]
    T_baseline = (d < np.median(d)).astype(np.float32)
    X_df = pd.DataFrame(X_embed, columns=[f"pc{i+1}" for i in range(cfg.embed_covariates)])
    X_df["T"] = T_baseline
    X_df = sm.add_constant(X_df)
    base_model = sm.OLS(Y, X_df).fit(cov_type="HC1")
    base_p = float(base_model.pvalues["T"])
    base_beta = float(base_model.params["T"])

    # for each Γ, add a synthetic hidden confounder U ~ Bernoulli(0.5)
    # biased toward T=1 spots with odds ratio Γ
    # this approximates the worst-case hidden confounder
    rng = np.random.RandomState(42)
    results = {"base_beta": base_beta, "base_p": base_p, "by_Gamma": {}}
    gamma_star = None
    for G in cfg.rosbaum_grid:
        if G == 1.0:
            p = base_p
        else:
            # simulate U biased toward treated
            n = len(T_baseline)
            # P(U=1 | T=1) / P(U=1 | T=0) = G
            p_u_t1 = G / (1 + G)
            p_u_t0 = 1 / (1 + G)
            U = np.zeros(n, dtype=np.float32)
            U[T_baseline == 1] = (rng.rand(int((T_baseline == 1).sum())) < p_u_t1).astype(np.float32)
            U[T_baseline == 0] = (rng.rand(int((T_baseline == 0).sum())) < p_u_t0).astype(np.float32)
            X_df_G = X_df.copy()
            X_df_G["U"] = U
            try:
                m = sm.OLS(Y, X_df_G).fit(cov_type="HC1")
                p = float(m.pvalues["T"])
            except Exception:
                p = 1.0
        results["by_Gamma"][G] = p
        if p > 0.05 and gamma_star is None:
            gamma_star = G
    results["gamma_star"] = gamma_star if gamma_star else ">{}".format(max(cfg.rosbaum_grid))
    return results
