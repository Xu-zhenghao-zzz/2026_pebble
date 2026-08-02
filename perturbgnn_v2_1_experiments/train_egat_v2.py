"""Train EGATEncoder v2 (anti-collapse fixes).

Differences from train_egat.py:
- Uses EGATEncoderV2 (no self_loops, attention entropy regulariser).
- lr=3e-4 (was 1e-3), weight_decay=1e-4 (was 1e-5).
- Early stopping patience=15.
- Saves best model + history with v2 suffix.
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
from perturbgnn_v2_1.egat import EGATEncoderV2, assemble_node_features


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
LOGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/logs")
OUT_PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/processed")


def load_tissue_graph(slice_name: str):
    """Same as v1: rebuild tensor-only Data + return (g, meta)."""
    p = PROCESSED / f"tissue_graph_{slice_name}_v2.pt"
    print(f"  loading {p.name}...", flush=True)
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
    print(f"    nodes={g.x.shape[0]}, edges={g.edge_index.shape[1]}", flush=True)
    return g, meta


def make_loader(g, n_neighbors, batch_size, shuffle):
    return NeighborLoader(g, num_neighbors=n_neighbors,
                          batch_size=batch_size, shuffle=shuffle,
                          input_nodes=None)


def evaluate(model, loader, device):
    model.eval()
    keys = ["loss", "ct_loss", "niche_loss", "ct_acc", "niche_acc",
            "ent1", "ent2", "std_mean"]
    acc = {k: [] for k in keys}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr,
                         cell_type=batch.cell_type, niche_id=batch.niche_id)
            for k in keys:
                v = out.get(k, 0.0)
                acc[k].append(v.item() if hasattr(v, "item") else float(v))
    return {k: float(np.mean(vs)) for k, vs in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="M001,M002")
    ap.add_argument("--val", default="M003")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n_neighbors", type=int, nargs="+", default=[15, 10])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--attn_dropout", type=float, default=0.1)
    ap.add_argument("--ct_weight", type=float, default=0.3)
    ap.add_argument("--niche_weight", type=float, default=1.0)
    ap.add_argument("--entropy_weight", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--out", default=str(OUT_PROCESSED / "egat_encoder_v3_fixed.pt"))
    ap.add_argument("--log_every", type=int, default=2)
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    OUT_PROCESSED.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    train_slices = args.train.split(",")
    val_slices = args.val.split(",")
    print(f"\nLoading train slices: {train_slices}", flush=True)
    train_data = {s: load_tissue_graph(s) for s in train_slices}
    print(f"Loading val slices: {val_slices}", flush=True)
    val_data = {s: load_tissue_graph(s) for s in val_slices}

    train_graphs = {s: d[0] for s, d in train_data.items()}
    val_graphs = {s: d[0] for s, d in val_data.items()}

    g0 = next(iter(train_graphs.values()))
    n_ct = int(getattr(g0, "n_cell_types", 8))
    n_niches = int(g0.niche_id.max().item() + 1)
    in_dim = g0.x.shape[1]
    print(f"\nin_dim={in_dim}, n_cell_types={n_ct}, n_niches={n_niches}", flush=True)

    model = EGATEncoderV2(
        in_dim=in_dim, hidden_dim=args.hidden, embed_dim=args.embed_dim,
        edge_dim=8, heads=args.heads,
        n_cell_types=n_ct, n_niches=n_niches,
        dropout=args.dropout, attn_dropout=args.attn_dropout,
        ct_loss_weight=args.ct_weight, niche_loss_weight=args.niche_weight,
        entropy_reg_weight=args.entropy_weight,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    print(f"\nBuilding NeighborLoaders...", flush=True)
    train_loaders = [(s, make_loader(g, args.n_neighbors, args.batch, True))
                     for s, g in train_graphs.items()]
    val_loaders = [(s, make_loader(g, args.n_neighbors, args.batch, False))
                   for s, g in val_graphs.items()]
    for s, l in train_loaders:
        print(f"  {s}: {len(l)} batches", flush=True)

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    print(f"\n=== Training {args.epochs} epochs (patience {args.patience}) ===", flush=True)
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_ct = 0.0; ep_niche = 0.0
        ep_ent1 = 0.0; ep_ent2 = 0.0
        n_b = 0
        t0 = time.time()
        for slice_name, loader in train_loaders:
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr,
                            cell_type=batch.cell_type, niche_id=batch.niche_id)
                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss += loss.item()
                ep_ct += out["ct_acc"].item()
                ep_niche += out["niche_acc"].item()
                ep_ent1 += float(out.get("ent1", 0))
                ep_ent2 += float(out.get("ent2", 0))
                n_b += 1
        scheduler.step()
        dt = time.time() - t0

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            val_metrics = {s: evaluate(model, l, device) for s, l in val_loaders}
            vlm = np.mean([m["loss"] for m in val_metrics.values()])
            v_ct = np.mean([m["ct_acc"] for m in val_metrics.values()])
            v_ni = np.mean([m["niche_acc"] for m in val_metrics.values()])
            v_e1 = np.mean([m.get("ent1", 0) for m in val_metrics.values()])
            v_e2 = np.mean([m.get("ent2", 0) for m in val_metrics.values()])
            print(
                f"epoch {epoch+1:3d}/{args.epochs} | "
                f"train_loss={ep_loss/n_b:.4f} ct_acc={ep_ct/n_b:.3f} "
                f"niche_acc={ep_niche/n_b:.3f} "
                f"ent1={ep_ent1/n_b:.3f} ent2={ep_ent2/n_b:.3f} | "
                f"val_loss={vlm:.4f} val_ct={v_ct:.3f} val_niche={v_ni:.3f} "
                f"val_ent1={v_e1:.3f} val_ent2={v_e2:.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} {dt:.1f}s",
                flush=True,
            )
            history.append({
                "epoch": epoch + 1,
                "train_loss": ep_loss / n_b,
                "train_ct_acc": ep_ct / n_b,
                "train_niche_acc": ep_niche / n_b,
                "val_loss": float(vlm),
                "val_ct_acc": float(v_ct),
                "val_niche_acc": float(v_ni),
                "val_ent1": float(v_e1),
                "val_ent2": float(v_e2),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_s": dt,
            })
            if vlm < best_val_loss:
                best_val_loss = vlm
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                }, args.out)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1} "
                          f"(patience {args.patience} exhausted)", flush=True)
                    break

    print(f"\n=== Done. Best epoch {best_epoch}, val_loss={best_val_loss:.4f} ===", flush=True)
    print(f"Saved best model to {args.out}", flush=True)
    with open(LOGS / "egat_train_history_v2.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss,
                   "n_params": n_params}, f, indent=2)


if __name__ == "__main__":
    main()
