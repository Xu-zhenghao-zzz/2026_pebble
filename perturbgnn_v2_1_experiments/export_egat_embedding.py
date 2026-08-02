"""Export EGAT embedding from a trained checkpoint.

Usage:
    python export_egat_embedding.py --checkpoint processed/egat_encoder_v3.pt \
        --slices M001,M002,M003 --suffix v1_bad
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perturbgnn_v2_1.egat import (
    EGATEncoder, EGATEncoderV2, EGATEncoderV3, EGATEncoderV4, assemble_node_features)


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
OUT_PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/processed")


def load_tissue_graph(slice_name: str):
    p = PROCESSED / f"tissue_graph_{slice_name}_v2.pt"
    g_orig = torch.load(p, weights_only=False)
    x_egat = assemble_node_features(g_orig)
    g = Data(
        x=x_egat,
        edge_index=g_orig.edge_index,
        edge_attr=g_orig.edge_attr.float(),
        pos=g_orig.pos,
        cell_type=g_orig.cell_type.long(),
        niche_id=g_orig.niche_id.long(),
        density_local=g_orig.density_local.float(),
        vessel_distance=g_orig.vessel_distance.float(),
        is_source=g_orig.is_source.long(),
        is_ntc=g_orig.is_ntc.long(),
    )
    g.n_cell_types = int(getattr(g_orig, "n_cell_types", 8))
    return g


def export_one_slice(model, g, slice_name, device, batch_size=8192,
                     n_neighbors=[15, 10]):
    """Run encoder over full slice via NeighborLoader in eval mode.

    Returns (N, embed_dim) embedding tensor on CPU."""
    model.eval()
    # We use NeighborLoader to get subgraphs; collect per-node embeddings.
    loader = NeighborLoader(
        g, num_neighbors=n_neighbors, batch_size=batch_size,
        shuffle=False, input_nodes=None,
    )
    N = g.x.shape[0]
    embed_dim = model.embed_dim
    out = torch.zeros(N, embed_dim, dtype=torch.float32)

    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index, batch.edge_attr)
            # batch.n_id maps subgraph-local index back to global.
            global_idx = batch.n_id
            out[global_idx] = z.cpu()
            n_seen += len(global_idx)
    print(f"  {slice_name}: exported {n_seen}/{N} nodes, embed_dim={embed_dim}", flush=True)
    return out


def compute_diagnostics(embed, g, slice_name):
    """Source-NTC centroid cosine, silhouette, etc."""
    diagnostics = {"slice": slice_name, "n_spots": int(embed.shape[0])}

    # Source / NTC masks.
    is_source = g.is_source.bool().numpy()
    is_ntc = g.is_ntc.bool().numpy()
    diagnostics["n_source"] = int(is_source.sum())
    diagnostics["n_ntc"] = int(is_ntc.sum())

    # Cosine of centroids.
    if is_source.sum() > 0 and is_ntc.sum() > 0:
        e_src = embed[is_source].mean(dim=0)
        e_ntc = embed[is_ntc].mean(dim=0)
        cos = F.cosine_similarity(e_src.unsqueeze(0), e_ntc.unsqueeze(0)).item()
        diagnostics["src_ntc_centroid_cos"] = round(cos, 4)
    else:
        diagnostics["src_ntc_centroid_cos"] = None

    # z collapse check.
    diagnostics["z_std_mean"] = round(float(embed.std(dim=0).mean()), 4)
    diagnostics["z_mean_norm"] = round(float(embed.norm(dim=1).mean()), 4)

    return diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--slices", default="M001,M002,M003")
    ap.add_argument("--suffix", default="egat")
    ap.add_argument("--encoder", default="v1", choices=["v1", "v2", "v3", "v4"],
                    help="v1=EGATEncoder, v2=EGATEncoderV2 (anti-collapse)")
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    ckpt_args = ckpt["args"]
    print(f"  trained at epoch {ckpt['epoch']}, val_metrics: {list(ckpt.get('val_metrics', {}).keys())}", flush=True)

    EncoderClass = {"v1": EGATEncoder, "v2": EGATEncoderV2, "v3": EGATEncoderV3, "v4": EGATEncoderV4}[args.encoder]
    common_kwargs = dict(
        in_dim=ckpt_args.get("in_dim", 43),
        hidden_dim=ckpt_args["hidden"],
        embed_dim=ckpt_args["embed_dim"],
        edge_dim=8,
        heads=ckpt_args["heads"],
        dropout=ckpt_args.get("dropout", 0.1),
        attn_dropout=ckpt_args.get("attn_dropout", 0.0),
    )
    if args.encoder in ("v3", "v4"):
        model = EncoderClass(
            n_niches=12,
            contrastive_weight=ckpt_args.get("contrastive_weight", 0.5),
            temperature=ckpt_args.get("temperature", 0.3),
            **common_kwargs,
        ).to(device)
    else:
        model = EncoderClass(
            n_cell_types=8,
            n_niches=12,
            **common_kwargs,
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_diagnostics = []
    for slice_name in args.slices.split(","):
        print(f"\n=== {slice_name} ===", flush=True)
        g = load_tissue_graph(slice_name)
        t0 = time.time()
        embed = export_one_slice(model, g, slice_name, device, args.batch)
        dt = time.time() - t0
        print(f"  export time: {dt:.1f}s", flush=True)

        # Save embedding.
        out_path = OUT_PROCESSED / f"embed_{slice_name}_{args.suffix}.npy"
        np.save(out_path, embed.numpy())
        print(f"  saved to {out_path}", flush=True)

        diag = compute_diagnostics(embed, g, slice_name)
        diag["export_time_s"] = round(dt, 1)
        all_diagnostics.append(diag)
        for k, v in diag.items():
            print(f"    {k}: {v}", flush=True)

    # Save diagnostics.
    diag_path = OUT_PROCESSED / f"embedding_validation_{args.suffix}.json"
    with open(diag_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "encoder_version": args.encoder,
            "slices": all_diagnostics,
        }, f, indent=2)
    print(f"\nDiagnostics saved to {diag_path}", flush=True)


if __name__ == "__main__":
    main()
