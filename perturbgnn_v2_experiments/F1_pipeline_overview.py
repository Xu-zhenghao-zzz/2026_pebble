"""Phase 5 Figure 1: Pipeline overview.

Single-panel schematic of the 5-layer v2 pipeline:
  Layer 0  data + tissue graph
  Layer 1  GNN embedding (cross-modal)
  Layer 2  matched control in embedding space
  Layer 3  multi-ring SLX + resistance distance + Durbin
  Layer 4  DiD + IV + Rosenbaum causal diagnostics
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

FIGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/figures")
FIGS.mkdir(exist_ok=True)


def main():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # 5 layer boxes
    layers = [
        ("Layer 0: Data + Tissue Graph",
         "SPAC-seq H5 (3 slices, 1.1M bins)\n"
         "+ Delaunay/kNN graph\n"
         "+ 8-dim edge features (barrier, density, vessel)",
         "#9ecae1"),
        ("Layer 1: GNN Embedding (cross-modal)",
         "Dual encoder (PCA view + cell-type view)\n"
         "+ shared cell_type + niche classifier\n"
         "+ soft alignment (cosine)\n"
         "→ 64-dim spot embedding",
         "#a1d99b"),
        ("Layer 2: Matched Control",
         "For each analysis spot:\n"
         "  candidates = same (cell_type, niche_id)\n"
         "  + dist > 400µm, guide_total ≤ 1\n"
         "  + top-K in embedding cosine\n"
         "SMD < 0.13, NTC Δ ≈ 0",
         "#fdae6b"),
        ("Layer 3: Spatial Network Models",
         "Multi-ring SLX (euclidean)\n"
         "+ Resistance-weighted SLX (key innovation)\n"
         "+ Multi-source Durbin (propagation slope δ)",
         "#c6dbef"),
        ("Layer 4: Causal Diagnostics",
         "Spatial DiD (matched-control contrast)\n"
         "+ Guide-IV (2SLS dose-response)\n"
         "+ Rosenbaum Γ* (sensitivity)\n"
         "→ falsifiable causal claims",
         "#d9d9d9"),
    ]

    y_top = 7.0
    h = 1.2
    for i, (title, body, color) in enumerate(layers):
        y = y_top - i * (h + 0.15)
        box = patches.FancyBboxPatch(
            (0.3, y - h), 12.4, h,
            boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="black", facecolor=color, alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(0.5, y - 0.25, title, fontsize=11.5, fontweight="bold",
                va="top", ha="left")
        ax.text(0.5, y - 0.55, body, fontsize=8.5, va="top", ha="left",
                family="monospace")

        # arrow down
        if i < len(layers) - 1:
            ax.annotate("", xy=(6.5, y - h - 0.13),
                        xytext=(6.5, y - h + 0.02),
                        arrowprops=dict(arrowstyle="->", lw=1.4))

    # right side: input/output flow
    ax.text(12.7, 7.0 - 0 * (h + 0.15) - h / 2, "input",
            fontsize=8, ha="right", va="center", style="italic", color="#555")
    ax.text(12.7, 7.0 - 4 * (h + 0.15) - h / 2, "claim",
            fontsize=8, ha="right", va="center", style="italic", color="#555")

    # title
    ax.text(6.5, 7.35,
            "PerturbGNN v2 — Embedding-matched causal propagation framework",
            fontsize=14, fontweight="bold", ha="center", va="center")

    # bottom note
    ax.text(6.5, 0.1,
            "Replaces v1's vs-NTC FDR + concentric ring + exponential decay\n"
            "with causal identification + sensitivity analysis",
            fontsize=9, ha="center", va="bottom", style="italic", color="#444")

    fig.tight_layout()
    out = FIGS / "F1_pipeline_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
