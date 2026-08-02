"""F2: GNN embedding quality diagnostics.

3 panels:
  (a) cell-type silhouette per slice (bar)
  (b) source vs NTC centroid cosine per slice (target ≥0.7)
  (c) t-SNE of M002 embedding colored by cell_type
"""
import json
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")


def main():
    val = json.load(open(PROCESSED / "embedding_validation.json"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    slices = ["M001", "M002", "M003"]
    sils = [val["slices"][s]["silhouette_celltype"] for s in slices]
    coses = [val["slices"][s]["src_ntc_centroid_cos"] for s in slices]

    # (a) silhouette
    ax = axes[0]
    ax.bar(slices, sils, color="steelblue", alpha=0.7)
    ax.axhline(0.3, color="red", linestyle="--", label="target=0.3")
    ax.set_ylabel("Cell-type silhouette")
    ax.set_title("(a) Embedding cell-type separability")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (b) src-NTC cosine
    ax = axes[1]
    ax.bar(slices, coses, color="#2ca02c", alpha=0.7)
    ax.axhline(0.7, color="red", linestyle="--", label="target=0.7")
    ax.set_ylim(0.99, 1.001)
    ax.set_ylabel("Source vs NTC centroid cosine")
    ax.set_title("(b) Perturbation non-leakage\n(higher = cleaner embedding)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (c) t-SNE of M002
    ax = axes[2]
    print("loading M002 for t-SNE...", flush=True)
    a = ad.read_h5ad(PROCESSED / "M002_v2.h5ad")
    emb = np.load(PROCESSED / "embed_M002_v3.npy")
    sub = np.random.RandomState(0).choice(len(a), 10000, replace=False)
    ct = a.obs["cell_type"].iloc[sub].to_numpy()
    emb_sub = emb[sub]
    print("running t-SNE...", flush=True)
    tsne = TSNE(n_components=2, perplexity=30, random_state=0,
                init="pca", learning_rate="auto").fit_transform(emb_sub)
    for ct_val in sorted(set(ct.tolist())):
        mask = ct == ct_val
        ax.scatter(tsne[mask, 0], tsne[mask, 1], s=3, alpha=0.5,
                   label=ct_val)
    ax.set_title("(c) M002 embedding t-SNE (10k subsample)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, loc="best", markerscale=3)
    ax.grid(alpha=0.3)

    fig.suptitle("GNN embedding quality (v2 Phase 1)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F2_embedding_quality.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
