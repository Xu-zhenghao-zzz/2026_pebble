"""Embedding-space matched control (v2 Layer 2), vectorized.

Key speedup over v1: instead of looping over each analysis spot and
computing cosine similarity to a per-spot pool, we bucket analysis +
unexposed spots by cell_type, then do batched matrix multiply per bucket.
This brings M002 (~10K analysis spots × ~100K pool) from >4 min to <10 s.

For each target gene G in slice s:
  source_bins(G, s) = spots where guide_total>=2 ∧ top_fraction>=0.7 ∧ target_gene=G
  analysis_bins(G, s) = spots within 200µm of any source, excluding sources themselves

  For each analysis spot i:
    candidates = spots | same cell_type, dist(source) > 400µm,
                        guide_total < 2
    matched = top-K nearest in embedding space (cosine), batched per cell_type
    Δ_i = Y_i - mean(Y_matched)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")


@dataclass
class MatchConfig:
    analysis_radius_um: float = 200.0
    far_min_dist_um: float = 400.0
    K_matched: int = 10
    min_guide_for_source: int = 2
    min_top_fraction: float = 0.70
    max_guide_for_unexposed: int = 1
    response_names: tuple = ("score_ifn_response", "score_cd8_like",
                              "score_fibroblast", "score_macrophage",
                              "score_malignant", "score_hypoxia")


def _attach_module_scores(adata):
    """Copy module scores from obsm to obs as score_<name> columns."""
    if "module_scores" not in adata.obsm:
        return
    if "score_ifn_response" not in adata.obs.columns:
        names = adata.uns.get("module_names",
            ["malignant", "cd8_like", "macrophage", "fibroblast",
             "endothelial", "hypoxia", "ifn_response"])
        for i, n in enumerate(names):
            adata.obs[f"score_{n}"] = adata.obsm["module_scores"][:, i]


def _batched_cosine_topk(anchors: np.ndarray, pool: np.ndarray, k: int,
                          chunk: int = 2048):
    """For each anchor, find top-k pool indices by cosine similarity.

    anchors: (A, D), pool: (P, D). Returns (A, k) int indices into pool.
    """
    A = anchors.shape[0]
    anchors_n = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    pool_n = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-8)
    out = np.zeros((A, k), dtype=np.int64)
    for start in range(0, A, chunk):
        end = min(start + chunk, A)
        sims = anchors_n[start:end] @ pool_n.T  # (chunk, P)
        if sims.shape[1] > k:
            # argpartition for top-k, then sort within
            top = np.argpartition(-sims, k, axis=1)[:, :k]
            row = np.arange(end - start)[:, None]
            top_sims = sims[row, top]
            order = np.argsort(-top_sims, axis=1)
            top = top[row, order]
        else:
            row = np.arange(end - start)[:, None]
            top = np.argsort(-sims, axis=1)[:, :sims.shape[1]]
            # pad with -1 if fewer than k
            if top.shape[1] < k:
                pad = -np.ones((end - start, k - top.shape[1]), dtype=np.int64)
                top = np.concatenate([top, pad], axis=1)
        out[start:end] = top
    return out


def _compute_matches(adata, embed, is_source_mask, source_xy, cfg):
    """Find K matched controls for each analysis spot, vectorized per cell type.

    Returns dict: spot_idx → array of K matched global indices (or None).
    """
    xy = adata.obsm["spatial"].astype(np.float32)
    src_tree = cKDTree(source_xy)
    dist_to_src, _ = src_tree.query(xy, k=1)

    is_analysis = (dist_to_src <= cfg.analysis_radius_um) & ~is_source_mask
    is_unexposed = (dist_to_src > cfg.far_min_dist_um) & ~is_source_mask & (
        adata.obs["guide_total"].to_numpy() <= cfg.max_guide_for_unexposed
    )

    if is_analysis.sum() < 5 or is_unexposed.sum() < 30:
        return {}, is_analysis, is_unexposed, dist_to_src

    ct = adata.obs["cell_type"].to_numpy()
    niche = adata.obs["niche_id"].to_numpy().astype(np.int64)
    analysis_idx = np.where(is_analysis)[0]
    unexp_idx = np.where(is_unexposed)[0]
    # double bucket: (cell_type, niche_id) — exact match on both
    a_key = ct[analysis_idx] + "::" + niche[analysis_idx].astype(str)
    p_key = ct[unexp_idx] + "::" + niche[unexp_idx].astype(str)

    matches = {}
    for key in np.unique(a_key):
        a_local = np.where(a_key == key)[0]
        p_local = np.where(p_key == key)[0]
        if len(p_local) < cfg.K_matched:
            continue
        a_global = analysis_idx[a_local]
        p_global = unexp_idx[p_local]
        a_emb = embed[a_global]
        p_emb = embed[p_global]
        top = _batched_cosine_topk(a_emb, p_emb, k=cfg.K_matched)
        for i, ag in enumerate(a_global):
            m = p_global[top[i][top[i] >= 0]]
            matches[int(ag)] = m
    return matches, is_analysis, is_unexposed, dist_to_src


def matched_control_for_gene(adata, embed, target_gene, cfg):
    """Run matched control for one target gene."""
    _attach_module_scores(adata)
    obs = adata.obs

    is_source = (
        (obs["guide_total"].to_numpy() >= cfg.min_guide_for_source)
        & (obs["top_fraction"].to_numpy() >= cfg.min_top_fraction)
        & (obs["target_gene"].to_numpy() == target_gene)
    )
    n_sources = int(is_source.sum())
    if n_sources < 5:
        return pd.DataFrame(), pd.DataFrame()

    xy = adata.obsm["spatial"].astype(np.float32)
    matches, is_analysis, is_unexposed, dist_to_src = _compute_matches(
        adata, embed, is_source, xy[is_source], cfg,
    )

    if not matches:
        return pd.DataFrame(), pd.DataFrame()

    niche = obs["niche_id"].to_numpy().astype(np.int64)
    response_arr = {r: obs[r].to_numpy() for r in cfg.response_names if r in obs.columns}

    rows = []
    for i, m in matches.items():
        if len(m) < cfg.K_matched:
            continue
        rec = {
            "spot_idx": i,
            "x_um": float(xy[i, 0]),
            "y_um": float(xy[i, 1]),
            "cell_type": obs["cell_type"].iloc[i],
            "niche_id": int(niche[i]),
            "dist_to_source_um": float(dist_to_src[i]),
            "n_matched": len(m),
        }
        for r, arr in response_arr.items():
            y_i = arr[i]
            if pd.isna(y_i):
                continue
            y_m = arr[m]
            y_m = y_m[~pd.isna(y_m)]
            if len(y_m) == 0:
                continue
            rec[f"{r}_mean_matched"] = float(y_m.mean())
            rec[f"{r}_delta"] = float(y_i) - float(y_m.mean())
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["target_gene"] = target_gene
    df["n_sources"] = n_sources
    # covariate balance: PCA + niche for matched pairs
    bal_rows = _covariate_balance(adata, embed, matches)
    return df, pd.DataFrame(bal_rows)


def _covariate_balance(adata, embed, matches, n_samples=500):
    pca = adata.obsm["X_pca"].astype(np.float32)
    niche = adata.obs["niche_id"].to_numpy().astype(np.int64)
    keys = list(matches.keys())
    if len(keys) > n_samples:
        rng = np.random.RandomState(0)
        keys = rng.choice(keys, n_samples, replace=False).tolist()

    anchor_pcas, matched_pcas, anchor_niches, matched_niches = [], [], [], []
    for i in keys:
        m = matches[i]
        if len(m) < 10:
            continue
        anchor_pcas.append(pca[i])
        matched_pcas.append(pca[m].mean(axis=0))
        anchor_niches.append(np.eye(12)[niche[i]] if niche[i] < 12 else np.zeros(12))
        matched_niches.append(np.bincount(niche[m], minlength=12) / len(m))
    if not anchor_pcas:
        return []
    anchor_pcas = np.stack(anchor_pcas)
    matched_pcas = np.stack(matched_pcas)
    anchor_niches = np.stack(anchor_niches)
    matched_niches = np.stack(matched_niches)

    rows = []
    for d in range(8):
        a_m, m_m = anchor_pcas[:, d].mean(), matched_pcas[:, d].mean()
        a_s, m_s = anchor_pcas[:, d].std() + 1e-8, matched_pcas[:, d].std() + 1e-8
        smd = (a_m - m_m) / np.sqrt((a_s**2 + m_s**2) / 2)
        rows.append({"covariate": f"X_pca[{d}]", "smd": float(smd)})
    for k in range(12):
        a_m, m_m = anchor_niches[:, k].mean(), matched_niches[:, k].mean()
        a_s, m_s = anchor_niches[:, k].std() + 1e-8, matched_niches[:, k].std() + 1e-8
        smd = (a_m - m_m) / np.sqrt((a_s**2 + m_s**2) / 2)
        rows.append({"covariate": f"niche_{k}", "smd": float(smd)})
    return rows


def ntc_sanity_check(adata, embed, cfg):
    """Run matched control on NTC source spots — Δ should be ≈ 0."""
    _attach_module_scores(adata)
    obs = adata.obs
    is_ntc_source = (
        (obs["guide_total"].to_numpy() >= cfg.min_guide_for_source)
        & (obs["top_fraction"].to_numpy() >= cfg.min_top_fraction)
        & (obs["is_ntc"].to_numpy() == True)
    )
    if is_ntc_source.sum() < 5:
        return pd.DataFrame({"n_ntc_source": [int(is_ntc_source.sum())]})

    xy = adata.obsm["spatial"].astype(np.float32)
    matches, is_analysis, is_unexposed, dist_to_ntc = _compute_matches(
        adata, embed, is_ntc_source, xy[is_ntc_source], cfg,
    )
    if not matches:
        return pd.DataFrame({"n_analysis": [int(is_analysis.sum())],
                             "n_unexposed": [int(is_unexposed.sum())]})

    response_arr = {r: obs[r].to_numpy() for r in cfg.response_names if r in obs.columns}
    rows = []
    for i, m in matches.items():
        if len(m) < cfg.K_matched:
            continue
        rec = {"spot_idx": i, "dist_to_ntc_um": float(dist_to_ntc[i]),
               "cell_type": obs["cell_type"].iloc[i]}
        for r, arr in response_arr.items():
            y_i = arr[i]
            if pd.isna(y_i):
                continue
            y_m = arr[m]
            y_m = y_m[~pd.isna(y_m)]
            if len(y_m) == 0:
                continue
            rec[f"{r}_delta"] = float(y_i) - float(y_m.mean())
        rows.append(rec)
    return pd.DataFrame(rows)
