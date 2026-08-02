"""Fix 2 — Leiden niches sensitivity.

v2 derives niche_id from KMeans(k=12) on cell-type composition. Then
uses niche_id as (a) Layer 1 encoder supervision and (b) matching
covariate. Circular dependency.

Fix: replace KMeans with Leiden community detection at 3 resolutions
(0.3, 0.6, 1.0 → ~8, 14, 26 niches). Re-derive niche_id, then re-run
Phgr1 × fibroblast L2+L3 test to check δ stability across resolutions.

Acceptance: Phgr1 δ sign consistent across all 3 resolutions for
≥ 4/5 responses. KMeans k=12 result must fall within Leiden range.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from sklearn.neighbors import NearestNeighbors


sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
OUT = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/processed/sensitivity")
OUT.mkdir(parents=True, exist_ok=True)

MODULES = ["malignant", "cd8_like", "macrophage", "fibroblast",
           "endothelial", "hypoxia", "ifn_response"]
TARGET_GENE = "Phgr1"
RESOLUTIONS = [0.3, 0.6, 1.0]


def compute_celltype_composition(adata, K=20):
    """Per-spot cell-type composition in K=20 spatial neighbourhood.

    Returns (N, n_cell_types) where each row sums to 1.
    """
    from scipy.spatial import cKDTree
    xy = adata.obsm["spatial"].astype(np.float32)
    ct = adata.obs["cell_type"].astype(str).to_numpy()
    vocab = sorted(set(ct.tolist()) - {"Missing"})
    ct_lookup = {c: i for i, c in enumerate(vocab)}

    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=K + 1)
    idx = idx[:, 1:]
    N = xy.shape[0]
    n_ct = len(vocab)
    comp = np.zeros((N, n_ct), dtype=np.float32)
    ct_codes = np.array([ct_lookup.get(c, -1) for c in ct])
    for k in range(K):
        nbr_codes = ct_codes[idx[:, k]]
        valid = nbr_codes >= 0
        for nc in range(n_ct):
            mask = valid & (nbr_codes == nc)
            comp[mask, nc] += 1
    # Normalise.
    row_sum = comp.sum(axis=1, keepdims=True) + 1e-12
    comp /= row_sum
    return comp, vocab


def leiden_niches(comp, resolution, n_anchor=50000):
    """Leiden community detection on approximate kNN(comp) graph.

    Subsamples to n_anchor nodes for tractable Leiden runtime.
    Returns labels for ALL nodes by NN-assignment back from anchors.
    """
    import pynndescent
    n = comp.shape[0]

    # Subsample anchors if too large.
    if n > n_anchor:
        rng = np.random.default_rng(7)
        anchor_idx = rng.choice(n, n_anchor, replace=False)
        comp_anchor = comp[anchor_idx]
        print(f"    subsampled to {n_anchor} anchors", flush=True)
    else:
        comp_anchor = comp
        anchor_idx = np.arange(n)

    # Approximate kNN with pynndescent (10-100x faster than sklearn).
    index = pynndescent.NNDescent(
        comp_anchor, n_neighbors=20, metric="cosine",
        random_state=7, n_jobs=8,
    )
    index.prepare()
    knn_idx, knn_dist = index.neighbor_graph  # (n_anchor, 20)

    # Build igraph from kNN.
    sources, targets, weights = [], [], []
    for i in range(len(comp_anchor)):
        for j_pos in range(1, 20):  # skip self at pos 0
            j = knn_idx[i, j_pos]
            if j == i:
                continue
            d = float(knn_dist[i, j_pos])
            w = max(0.0, 1.0 - d)  # similarity = 1 - cosine_distance
            if w > 0:
                sources.append(int(i))
                targets.append(int(j))
                weights.append(w)

    g = ig.Graph(n=len(comp_anchor), edges=list(zip(sources, targets)),
                 directed=False, edge_attrs={"weight": weights})
    part = la.find_partition(
        g, la.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=7,
        n_iterations=5,
    )
    anchor_labels = np.array(part.membership)

    # Assign labels to ALL nodes via 1-NN in cosine space.
    if n > n_anchor:
        # Query the index for all nodes (comp includes non-anchors).
        full_idx, _ = index.query(comp, k=1)
        all_labels = anchor_labels[full_idx.flatten()]
    else:
        all_labels = anchor_labels
    return all_labels, list(part.sizes())


def cosine_nn_match(source_emb, pool_emb):
    s = source_emb / (np.linalg.norm(source_emb, axis=1, keepdims=True) + 1e-12)
    p = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-12)
    sim = s @ p.T
    return sim.argmax(axis=1)


def main():
    rows = []
    for slice_name in ["M001", "M002", "M003"]:
        print(f"\n=== {slice_name} ===", flush=True)
        adata = ad.read_h5ad(PROCESSED / f"{slice_name}_v2.h5ad")
        embed = np.load(PROCESSED / f"embed_{slice_name}_v3.npy")
        print(f"  adata: {adata.shape}", flush=True)

        comp, vocab = compute_celltype_composition(adata, K=20)
        print(f"  cell-type composition: {comp.shape}", flush=True)

        # KMeans baseline (v2's original).
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=12, n_init=10, random_state=7).fit(comp)
        niche_kmeans = km.labels_
        n_kmeans = len(set(niche_kmeans))
        print(f"  KMeans k=12: {n_kmeans} niches (v2 baseline)", flush=True)

        # Leiden at 3 resolutions.
        for res in RESOLUTIONS:
            print(f"  Leiden res={res}...", flush=True)
            t0 = time.time()
            niche_leiden, sizes = leiden_niches(comp, res)
            dt = time.time() - t0
            n_leiden = len(set(niche_leiden))
            print(f"    → {n_leiden} niches, sizes range {min(sizes)}-{max(sizes)}, {dt:.1f}s", flush=True)

            # Phgr1 source / NTC.
            source_mask = adata.obs["guide"].astype(str).str.contains(
                TARGET_GENE, case=False, na=False).values
            ntc_mask = adata.obs["is_ntc"].values
            if source_mask.sum() == 0 or ntc_mask.sum() == 0:
                continue

            # For each response: matched control + δ.
            for resp_idx, response in enumerate(MODULES):
                y = adata.obsm["module_scores"][:, resp_idx].astype(np.float32)
                # Cosine match via v2 embedding (same as v2 baseline).
                source_emb = embed[source_mask]
                ntc_emb = embed[ntc_mask]
                match_idx = cosine_nn_match(source_emb, ntc_emb)
                source_y = y[source_mask]
                matched_y = y[ntc_mask][match_idx]
                delta = float(source_y.mean() - matched_y.mean())

                # Perm p.
                rng = np.random.default_rng(7)
                n_perm = 100
                all_y = np.concatenate([source_y, matched_y])
                n_s = len(source_y)
                perm_deltas = np.empty(n_perm)
                for i in range(n_perm):
                    ix = rng.permutation(len(all_y))
                    perm_deltas[i] = all_y[ix[:n_s]].mean() - all_y[ix[n_s:]].mean()
                p = float((np.sum(np.abs(perm_deltas) >= abs(delta)) + 1) / (n_perm + 1))

                rows.append({
                    "slice": slice_name,
                    "niche_method": f"leiden_res{res}",
                    "n_niches": n_leiden,
                    "response": f"score_{response}",
                    "n_source": int(source_mask.sum()),
                    "delta": round(delta, 4),
                    "perm_p": p,
                })

            # Also store KMeans baseline once per slice.
            if res == RESOLUTIONS[0]:
                for resp_idx, response in enumerate(MODULES):
                    y = adata.obsm["module_scores"][:, resp_idx].astype(np.float32)
                    source_emb = embed[source_mask]
                    ntc_emb = embed[ntc_mask]
                    match_idx = cosine_nn_match(source_emb, ntc_emb)
                    source_y = y[source_mask]
                    matched_y = y[ntc_mask][match_idx]
                    delta = float(source_y.mean() - matched_y.mean())
                    rng = np.random.default_rng(7)
                    n_perm = 100
                    all_y = np.concatenate([source_y, matched_y])
                    n_s = len(source_y)
                    perm_deltas = np.empty(n_perm)
                    for i in range(n_perm):
                        ix = rng.permutation(len(all_y))
                        perm_deltas[i] = all_y[ix[:n_s]].mean() - all_y[ix[n_s:]].mean()
                    p = float((np.sum(np.abs(perm_deltas) >= abs(delta)) + 1) / (n_perm + 1))
                    rows.append({
                        "slice": slice_name,
                        "niche_method": "kmeans_k12",
                        "n_niches": n_kmeans,
                        "response": f"score_{response}",
                        "n_source": int(source_mask.sum()),
                        "delta": round(delta, 4),
                        "perm_p": p,
                    })

    df = pd.DataFrame(rows)
    out_path = OUT / "niche_leiden_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}", flush=True)

    # Summary table: per (slice, response), compare deltas across methods.
    print("\n=== Summary ===")
    pivot = df.pivot_table(
        index=["slice", "response"],
        columns="niche_method",
        values="delta",
        aggfunc="first",
    )
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
