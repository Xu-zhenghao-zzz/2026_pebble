"""Synthetic benchmark: inject known ground truth, measure v2 recovery.

For each (effect_size, clone_size, n_clones) combination:
  1. Take real M002 slice
  2. Pick N random "synthetic clone" locations
  3. Inject effect: Y_near_clone += effect_size (decaying with distance)
  4. Run v2 Durbin pipeline
  5. Measure: did we recover the effect? bias? power?

Output: power curve figure + calibration table.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import time
import json
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")

from perturbgnn_v2.spatial import fit_durbin, DurbinConfig
from perturbgnn_v2.matching.match import _attach_module_scores


def inject_synthetic_effect(adata, response_col, clone_centers_xy,
                              effect_size, lambda_um=60, noise_std=0.05,
                              max_radius=300):
    """Inject a known spatial effect into response_col.

    Effect = effect_size * exp(-d² / (2λ²)) for d < max_radius.
    Returns the modified adata (in-place on obs[response_col]).
    """
    xy = adata.obsm["spatial"].astype(np.float32)
    Y = adata.obs[response_col].to_numpy().astype(np.float32).copy()
    Y_original = Y.copy()

    if len(clone_centers_xy) == 0:
        return Y_original, np.zeros_like(Y)

    # total injected effect (sum over clones)
    lam2 = 2 * lambda_um ** 2
    tree = cKDTree(clone_centers_xy)
    # for each spot, distance to nearest clone center
    d, _ = tree.query(xy, k=1)
    injected = np.where(
        d < max_radius,
        effect_size * np.exp(-d ** 2 / lam2) + np.random.randn(len(Y)) * noise_std,
        0
    )
    Y_new = Y_original + injected
    return Y_new, injected


def benchmark_one_config(adata, emb, response_col, effect_size, n_clones,
                           clone_size=50, seed=0):
    """Run one benchmark configuration. Returns dict of metrics."""
    np.random.seed(seed)
    xy = adata.obsm["spatial"].astype(np.float32)

    # pick random clone centers (from existing source spots to be realistic)
    existing_source_pos = np.where(adata.obs["is_source"].to_numpy())[0]
    if len(existing_source_pos) == 0:
        return None
    center_indices = np.random.choice(existing_source_pos, min(n_clones, len(existing_source_pos)), replace=False)
    center_xy = xy[center_indices]

    # create synthetic "gene" mask: spots within clone_size of centers
    tree = cKDTree(center_xy)
    d, _ = tree.query(xy, k=1)
    is_synthetic_source = d < clone_size

    # inject effect into a COPY of response
    Y_original = adata.obs[response_col].to_numpy().astype(np.float32)
    Y_new, injected = inject_synthetic_effect(
        adata, response_col, center_xy,
        effect_size=effect_size, lambda_um=60, noise_std=0.05,
    )
    adata.obs[response_col + "_synthetic"] = Y_new

    # set a synthetic target_gene
    adata.obs["synthetic_target"] = ""
    adata.obs.loc[is_synthetic_source, "synthetic_target"] = "SYNTHETIC"
    # backup and override target_gene temporarily
    original_target = adata.obs["target_gene"].copy()
    adata.obs["target_gene"] = "SYNTHETIC"
    adata.obs.loc[~is_synthetic_source, "target_gene"] = ""
    adata.obs.loc[is_synthetic_source, "is_source"] = True

    try:
        result = fit_durbin(adata, emb, "SYNTHETIC", response_col + "_synthetic",
                            DurbinConfig())
    except Exception as e:
        result = {"error": str(e)[:80]}
    finally:
        # restore
        adata.obs["target_gene"] = original_target
        adata.obs["is_source"] = (adata.obs["guide_total"] >= 2) & \
                                  (adata.obs["top_fraction"] >= 0.7) & \
                                  (~adata.obs["is_ntc"])
        del adata.obs[response_col + "_synthetic"]

    if "error" in result:
        return {"effect_size": effect_size, "n_clones": n_clones,
                "error": result["error"]}

    best = result.get("best", {})
    return {
        "effect_size_truth": effect_size,
        "n_clones": n_clones,
        "injected_mean": float(np.mean(injected[injected != 0])),
        "durbin_delta": best.get("delta", np.nan),
        "durbin_p": best.get("delta_p", np.nan),
        "durbin_lambda": best.get("lambda_um", np.nan),
        "n_clones_detected": result.get("n_clones", 0),
    }


def main():
    print("[benchmark] loading M002...", flush=True)
    a = ad.read_h5ad(PROCESSED / "M002_v2.h5ad")
    emb = np.load(PROCESSED / "embed_M002_v3.npy")
    _attach_module_scores(a)

    # benchmark grid
    EFFECT_SIZES = [0.02, 0.05, 0.10, 0.20, 0.50]
    N_CLONES = [5, 20, 50]
    RESPONSE = "score_ifn_response"

    rows = []
    for es in EFFECT_SIZES:
        for nc in N_CLONES:
            print(f"\n[bench] effect={es}, n_clones={nc}...", flush=True)
            t0 = time.time()
            r = benchmark_one_config(a, emb, RESPONSE, es, nc, seed=42)
            r["dt_s"] = time.time() - t0
            print(f"  result: δ={r.get('durbin_delta', '?')}, p={r.get('durbin_p', '?')}")
            rows.append(r)

    df = pd.DataFrame(rows)
    out = PROCESSED / "synthetic_benchmark.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote: {out}")

    # power curve figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) recovery: injected vs detected δ
    ax = axes[0]
    for nc in N_CLONES:
        sub = df[df["n_clones"] == nc]
        if len(sub) == 0: continue
        valid = sub.dropna(subset=["durbin_delta"])
        ax.plot(valid["effect_size_truth"], valid["durbin_delta"],
                marker="o", label=f"{nc} clones", linewidth=2)
    ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.5, label="perfect recovery (y=x)")
    ax.set_xlabel("True effect size (injected)")
    ax.set_ylabel("Durbin δ (detected)")
    ax.set_title("(a) Effect size recovery (bias = deviation from y=x)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) power: P(detect) vs effect size, per n_clones
    ax = axes[1]
    for nc in N_CLONES:
        sub = df[df["n_clones"] == nc]
        if len(sub) == 0: continue
        valid = sub.dropna(subset=["durbin_p"])
        power = []
        for es in EFFECT_SIZES:
            r = valid[valid["effect_size_truth"] == es]
            if len(r) > 0:
                power.append(float((r["durbin_p"] < 0.05).mean()))
            else:
                power.append(0)
        ax.plot(EFFECT_SIZES, power, marker="o", label=f"{nc} clones", linewidth=2)
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="80% power")
    ax.set_xlabel("True effect size")
    ax.set_ylabel("Power (P(detect at p<0.05))")
    ax.set_title("(b) Power curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Synthetic benchmark: v2 pipeline recovery + power calibration",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_fig = FIGS / "F19_synthetic_benchmark.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"wrote: {out_fig}")


if __name__ == "__main__":
    main()
