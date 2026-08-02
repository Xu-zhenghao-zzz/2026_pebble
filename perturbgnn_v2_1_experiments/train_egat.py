"""Train EGAT encoder on cohort 1 SPAC-seq tissue graphs.

Mirrors perturbgnn_v2/embedding/train_bt.py interface but uses EGAT
instead of set-pooling. Trains on M001+M002, validates on M003.
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
from perturbgnn_v2_1.egat import EGATEncoder, assemble_node_features


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
LOGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/logs")
OUT_PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/processed")


def load_tissue_graph(slice_name: str):
    """Load v2 tissue_graph_<slice>_v2.pt and rebuild a tensor-only Data.

    Returns (g, meta) where g has only tensor-typed fields in its store
    (so NeighborLoader can sample it) and meta carries the object arrays
    (target_gene, cell_type_vocab, section_names) separately.
    """
    p = PROCESSED / f"tissue_graph_{slice_name}_v2.pt"
    print(f"  loading {p.name}...")
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
        section_idx=g_orig.section_idx.long(),
    )
    if hasattr(g_orig, "module_scores"):
        g.module_scores = g_orig.module_scores.float()
    g.n_cell_types = int(getattr(g_orig, "n_cell_types", 8))

    meta = {
        "target_gene": getattr(g_orig, "target_gene", None),
        "cell_type_vocab": getattr(g_orig, "cell_type_vocab", None),
        "section_names": getattr(g_orig, "section_names", None),
    }
    print(f"    nodes={g.x.shape[0]}, edges={g.edge_index.shape[1]}, "
          f"node_feat={g.x.shape[1]}")
    return g, meta


def make_loader(g: Data, n_neighbors, batch_size: int, shuffle: bool):
    return NeighborLoader(
        g,
        num_neighbors=n_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        input_nodes=None,
    )


def evaluate(model, loader, device):
    model.eval()
    keys = ["loss", "ct_loss", "niche_loss", "ct_acc", "niche_acc", "std_mean"]
    acc = {k: [] for k in keys}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x, batch.edge_index, batch.edge_attr,
                cell_type=batch.cell_type,
                niche_id=batch.niche_id,
            )
            for k in keys:
                v = out[k]
                acc[k].append(v.item() if hasattr(v, "item") else float(v))
    return {k: float(np.mean(vs)) for k, vs in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="M001,M002")
    ap.add_argument("--val", default="M003")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n_neighbors", type=int, nargs="+", default=[15, 10])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn_dropout", type=float, default=0.0)
    ap.add_argument("--out", default=str(OUT_PROCESSED / "egat_encoder_v3.pt"))
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    OUT_PROCESSED.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"args: {vars(args)}")

    train_slices = args.train.split(",")
    val_slices = args.val.split(",")
    print(f"\nLoading train slices: {train_slices}")
    train_data = {s: load_tissue_graph(s) for s in train_slices}
    print(f"Loading val slices: {val_slices}")
    val_data = {s: load_tissue_graph(s) for s in val_slices}

    # Only the Data goes into loaders; meta is preserved for later use.
    train_graphs = {s: d[0] for s, d in train_data.items()}
    train_meta = {s: d[1] for s, d in train_data.items()}
    val_graphs = {s: d[0] for s, d in val_data.items()}

    g0 = next(iter(train_graphs.values()))
    n_cell_types = int(getattr(g0, "n_cell_types", 8))
    n_niches = int(g0.niche_id.max().item() + 1)
    in_dim = g0.x.shape[1]
    print(f"\nin_dim={in_dim}, n_cell_types={n_cell_types}, n_niches={n_niches}")

    model = EGATEncoder(
        in_dim=in_dim,
        hidden_dim=args.hidden,
        embed_dim=args.embed_dim,
        edge_dim=8,
        heads=args.heads,
        n_cell_types=n_cell_types,
        n_niches=n_niches,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    print(f"\nBuilding NeighborLoaders (n_neighbors={args.n_neighbors})...")
    train_loaders = []
    for s, g in train_graphs.items():
        loader = make_loader(g, args.n_neighbors, args.batch, shuffle=True)
        train_loaders.append((s, loader))
        print(f"  {s}: {len(loader)} batches")
    val_loaders = []
    for s, g in val_graphs.items():
        loader = make_loader(g, args.n_neighbors, args.batch, shuffle=False)
        val_loaders.append((s, loader))

    history = []
    best_val_loss = float("inf")
    best_epoch = -1

    print(f"\n=== Training {args.epochs} epochs ===")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ct_acc = 0.0
        epoch_niche_acc = 0.0
        n_batches = 0
        t0 = time.time()

        for slice_name, loader in train_loaders:
            for batch in loader:
                batch = batch.to(device)
                out = model(
                    batch.x, batch.edge_index, batch.edge_attr,
                    cell_type=batch.cell_type,
                    niche_id=batch.niche_id,
                )
                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                epoch_ct_acc += out["ct_acc"].item()
                epoch_niche_acc += out["niche_acc"].item()
                n_batches += 1

        scheduler.step()
        dt = time.time() - t0

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            val_metrics = {s: evaluate(model, l, device)
                           for s, l in val_loaders}
            val_loss_mean = np.mean([m["loss"] for m in val_metrics.values()])
            val_ct_acc = np.mean([m["ct_acc"] for m in val_metrics.values()])
            val_niche_acc = np.mean([m["niche_acc"] for m in val_metrics.values()])

            print(
                f"epoch {epoch+1:3d}/{args.epochs} | "
                f"train_loss={epoch_loss/n_batches:.4f} "
                f"ct_acc={epoch_ct_acc/n_batches:.3f} "
                f"niche_acc={epoch_niche_acc/n_batches:.3f} | "
                f"val_loss={val_loss_mean:.4f} "
                f"val_ct_acc={val_ct_acc:.3f} "
                f"val_niche_acc={val_niche_acc:.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"{dt:.1f}s/epoch"
            )
            history.append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss / n_batches,
                "train_ct_acc": epoch_ct_acc / n_batches,
                "train_niche_acc": epoch_niche_acc / n_batches,
                "val_loss": float(val_loss_mean),
                "val_ct_acc": float(val_ct_acc),
                "val_niche_acc": float(val_niche_acc),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_s": dt,
            })

            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                best_epoch = epoch + 1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                }, args.out)

    print(f"\n=== Done. Best epoch {best_epoch}, val_loss={best_val_loss:.4f} ===")
    print(f"Saved best model to {args.out}")

    hist_path = LOGS / "egat_train_history.json"
    with open(hist_path, "w") as f:
        json.dump({
            "args": vars(args),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "n_params": n_params,
        }, f, indent=2)
    print(f"Saved history to {hist_path}")


if __name__ == "__main__":
    main()
