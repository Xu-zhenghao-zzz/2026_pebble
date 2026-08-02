"""Train EGATEncoder v3 — drop cell_type, add SimCLR-style contrastive."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perturbgnn_v2_1.egat import EGATEncoderV3, assemble_node_features


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
LOGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/logs")
OUT_PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2_1/processed")


def load_tissue_graph(slice_name: str):
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
    )
    if hasattr(g_orig, "module_scores"):
        g.module_scores = g_orig.module_scores.float()
    g.n_cell_types = int(getattr(g_orig, "n_cell_types", 8))
    print(f"    nodes={g.x.shape[0]}, edges={g.edge_index.shape[1]}", flush=True)
    return g


def make_loader(g, n_neighbors, batch_size, shuffle):
    return NeighborLoader(g, num_neighbors=n_neighbors,
                          batch_size=batch_size, shuffle=shuffle,
                          input_nodes=None)


def evaluate(model, loader, device):
    model.eval()
    keys = ["loss", "niche_loss", "contr_loss", "niche_acc",
            "std_mean", "avg_cos_random"]
    acc = {k: [] for k in keys}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr,
                         niche_id=batch.niche_id)
            for k in keys:
                v = out.get(k, 0.0)
                acc[k].append(v.item() if hasattr(v, "item") else float(v))
    return {k: float(np.mean(vs)) for k, vs in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="M001,M002")
    ap.add_argument("--val", default="M003")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n_neighbors", type=int, nargs="+", default=[15, 10])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--attn_dropout", type=float, default=0.1)
    ap.add_argument("--contrastive_weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--out", default=str(OUT_PROCESSED / "egat_encoder_v3_contrastive.pt"))
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
    train_graphs = {s: load_tissue_graph(s) for s in train_slices}
    print(f"Loading val slices: {val_slices}", flush=True)
    val_graphs = {s: load_tissue_graph(s) for s in val_slices}

    g0 = next(iter(train_graphs.values()))
    n_niches = int(g0.niche_id.max().item() + 1)
    in_dim = g0.x.shape[1]
    print(f"\nin_dim={in_dim}, n_niches={n_niches}", flush=True)

    model = EGATEncoderV3(
        in_dim=in_dim, hidden_dim=args.hidden, embed_dim=args.embed_dim,
        edge_dim=8, heads=args.heads, n_niches=n_niches,
        dropout=args.dropout, attn_dropout=args.attn_dropout,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
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

    print(f"\n=== Training v3 {args.epochs} epochs (patience {args.patience}) ===", flush=True)
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_niche = 0.0; ep_contr = 0.0; ep_niche_acc = 0.0
        ep_cos = 0.0
        n_b = 0
        t0 = time.time()
        for slice_name, loader in train_loaders:
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr,
                            niche_id=batch.niche_id)
                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss += loss.item()
                ep_niche += float(out["niche_loss"])
                ep_contr += float(out["contr_loss"])
                ep_niche_acc += out["niche_acc"].item()
                ep_cos += float(out.get("avg_cos_random", 0))
                n_b += 1
        scheduler.step()
        dt = time.time() - t0

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            val_metrics = {s: evaluate(model, l, device) for s, l in val_loaders}
            vlm = np.mean([m["loss"] for m in val_metrics.values()])
            v_ni = np.mean([m["niche_acc"] for m in val_metrics.values()])
            v_cos = np.mean([m.get("avg_cos_random", 0) for m in val_metrics.values()])
            print(
                f"epoch {epoch+1:3d}/{args.epochs} | "
                f"loss={ep_loss/n_b:.4f} niche={ep_niche/n_b:.4f} "
                f"contr={ep_contr/n_b:.4f} niche_acc={ep_niche_acc/n_b:.3f} "
                f"avg_cos={ep_cos/n_b:.3f} | "
                f"val_loss={vlm:.4f} val_niche_acc={v_ni:.3f} val_cos={v_cos:.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} {dt:.1f}s",
                flush=True,
            )
            history.append({
                "epoch": epoch + 1,
                "train_loss": ep_loss / n_b,
                "train_niche_loss": ep_niche / n_b,
                "train_contr_loss": ep_contr / n_b,
                "train_niche_acc": ep_niche_acc / n_b,
                "train_avg_cos": ep_cos / n_b,
                "val_loss": float(vlm),
                "val_niche_acc": float(v_ni),
                "val_avg_cos": float(v_cos),
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
                    print(f"Early stopping at epoch {epoch+1}", flush=True)
                    break

    print(f"\n=== Done. Best epoch {best_epoch}, val_loss={best_val_loss:.4f} ===", flush=True)
    print(f"Saved best model to {args.out}", flush=True)
    with open(LOGS / "egat_train_history_v3.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss,
                   "n_params": n_params}, f, indent=2)


if __name__ == "__main__":
    main()
