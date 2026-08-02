"""Training loop for supervised cross-modal spot encoder (v2 Layer 1, design v3).

Pipeline:
  1. Load AnnData for one or more slices
  2. For each spot, gather K=15 nearest spatial neighbors' PCA + cell-type
  3. Train dual-view encoder (pca + cell-type) with:
       - cell_type classification (8-class)
       - niche_id classification (12-class)
       - soft cross-modal alignment (cosine sim)
  4. Periodically evaluate on held-out slice
  5. Save final encoder + per-spot embeddings

Usage:
  python -m perturbgnn_v2.embedding.train_bt \
      --train M001,M002 --val M003 \
      --epochs 200 --batch 4096 --K 15 \
      --out processed/bt_encoder_v3.pt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from .bt_encoder import SupervisedCrossModalEncoder


PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")
LOGS = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/logs")


def gather_neighborhoods(adata, K: int = 15):
    """For each spot, gather K nearest spatial neighbors' PCA + cell-type
    composition. Also return labels (cell_type_idx, niche_id) per spot.
    """
    xy = adata.obsm["spatial"].astype(np.float32)
    pca = adata.obsm["X_pca"].astype(np.float32)
    ct = adata.obs["cell_type"].to_numpy()
    niche = adata.obs["niche_id"].to_numpy().astype(np.int64)
    vocab = sorted(set(ct.tolist()) - {"Missing"})
    ct_lookup = {c: i for i, c in enumerate(vocab)}

    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=K + 1)
    idx = idx[:, 1:]  # drop self
    N = xy.shape[0]
    pca_dim = pca.shape[1]
    n_ct = len(vocab)

    nbr_pca = np.take(pca, idx, axis=0).astype(np.float32)
    nbr_ct = np.zeros((N, K, n_ct), dtype=np.float32)
    for k in range(K):
        ct_codes = np.array([ct_lookup.get(c, -1) for c in ct[idx[:, k]]])
        valid = ct_codes >= 0
        nbr_ct[valid, k, ct_codes[valid]] = 1.0
    miss_mask = nbr_ct.sum(axis=-1) == 0
    nbr_ct[miss_mask] = 1.0 / n_ct

    # anchor cell type idx (spot's own); "Missing" → 0 (majority class fallback)
    ct_self = np.array([ct_lookup.get(c, 0) for c in ct], dtype=np.int64)
    return nbr_pca, nbr_ct, ct_self, niche.astype(np.int64), vocab


class NeighborhoodDataset(Dataset):
    def __init__(self, nbr_pca, nbr_ct, ct_label, niche_label):
        self.pca = torch.from_numpy(nbr_pca)
        self.ct = torch.from_numpy(nbr_ct)
        self.ct_label = torch.from_numpy(ct_label).long()
        self.niche_label = torch.from_numpy(niche_label).long()

    def __len__(self):
        return self.pca.shape[0]

    def __getitem__(self, i):
        return (self.pca[i], self.ct[i],
                self.ct_label[i], self.niche_label[i])


def evaluate_embedding(model, loader, device):
    """Compute mean loss + diagnostics on eval set."""
    model.eval()
    import numpy as np
    keys = ["loss", "cls_loss", "align_loss", "cos_sim",
            "ct_acc1", "ct_acc2", "niche_acc1", "niche_acc2",
            "std1_mean", "std2_mean"]
    acc = {k: [] for k in keys}
    with torch.no_grad():
        for pca_b, ct_b, ct_l, niche_l in loader:
            pca_b = pca_b.to(device, non_blocking=True)
            ct_b = ct_b.to(device, non_blocking=True)
            ct_l = ct_l.to(device, non_blocking=True)
            niche_l = niche_l.to(device, non_blocking=True)
            out = model(pca_b, ct_b, ct_l, niche_l)
            for k in keys:
                v = out[k]
                acc[k].append(v.item() if hasattr(v, "item") else float(v))
    return {k: float(np.mean(vs)) for k, vs in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="M001,M002")
    ap.add_argument("--val", default="M003")
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--alpha_align", type=float, default=0.1)
    ap.add_argument("--n_niches", type=int, default=12)
    ap.add_argument("--out", default=str(PROCESSED / "bt_encoder_v3.pt"))
    ap.add_argument("--log_every", type=int, default=2)
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    log_file = LOGS / f"p1_bt_train_{int(time.time())}.log"
    print(f"[bt] log file: {log_file}", flush=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[bt] device: {device}")

    def load_multi(names):
        return [ad.read_h5ad(PROCESSED / f"{n}_v2.h5ad") for n in names]

    train_names = args.train.split(",")
    val_names = args.val.split(",") if args.val else []
    log(f"[bt] loading train: {train_names}, val: {val_names}")

    train_pca_parts, train_ct_parts = [], []
    train_ctl_parts, train_niche_parts = [], []
    for a in load_multi(train_names):
        log(f"[bt]   gathering {a.obs.section.iloc[0]} (n={a.n_obs:,})...")
        p, c, ctl, niche, vocab = gather_neighborhoods(a, K=args.K)
        train_pca_parts.append(p)
        train_ct_parts.append(c)
        train_ctl_parts.append(ctl)
        train_niche_parts.append(niche)
    train_pca = np.concatenate(train_pca_parts, axis=0)
    train_ct = np.concatenate(train_ct_parts, axis=0)
    train_ctl = np.concatenate(train_ctl_parts, axis=0)
    train_niche = np.concatenate(train_niche_parts, axis=0)
    log(f"[bt] train neighborhoods: {train_pca.shape}")

    val_pca = val_ct = val_ctl = val_niche = None
    if val_names:
        vp, vc, vl, vn = [], [], [], []
        for a in load_multi(val_names):
            log(f"[bt]   gathering val {a.obs.section.iloc[0]}...")
            p, c, ctl, niche, _ = gather_neighborhoods(a, K=args.K)
            vp.append(p); vc.append(c); vl.append(ctl); vn.append(niche)
        val_pca = np.concatenate(vp, axis=0)
        val_ct = np.concatenate(vc, axis=0)
        val_ctl = np.concatenate(vl, axis=0)
        val_niche = np.concatenate(vn, axis=0)
        log(f"[bt] val neighborhoods: {val_pca.shape}")

    pca_dim = train_pca.shape[-1]
    n_cell_types = train_ct.shape[-1]
    n_niches = int(max(args.n_niches, train_niche.max() + 1, val_niche.max() + 1 if val_niche is not None else 0))
    log(f"[bt] pca_dim={pca_dim}, n_cell_types={n_cell_types}, n_niches={n_niches}")

    model = SupervisedCrossModalEncoder(
        K=args.K,
        pca_dim=pca_dim,
        n_cell_types=n_cell_types,
        n_niches=n_niches,
        hidden=args.hidden,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        alpha_align=args.alpha_align,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[bt] model params: {n_params:,}")

    train_ds = NeighborhoodDataset(train_pca, train_ct, train_ctl, train_niche)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = None
    if val_pca is not None:
        val_ds = NeighborhoodDataset(val_pca, val_ct, val_ctl, val_niche)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=2, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss = 0.0; nb = 0
        ep_diag = {"cos": 0.0, "ct1": 0.0, "ct2": 0.0}
        for pca_b, ct_b, ct_l, niche_l in train_loader:
            pca_b = pca_b.to(device, non_blocking=True)
            ct_b = ct_b.to(device, non_blocking=True)
            ct_l = ct_l.to(device, non_blocking=True)
            niche_l = niche_l.to(device, non_blocking=True)
            optim.zero_grad()
            out = model(pca_b, ct_b, ct_l, niche_l)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += out["loss"].item(); nb += 1
            if nb == 1:
                ep_diag = {
                    "cos": out["cos_sim"].item(),
                    "ct1": out["ct_acc1"].item(),
                    "ct2": out["ct_acc2"].item(),
                    "niche1": out["niche_acc1"].item(),
                }
        sched.step()
        ep_loss /= max(nb, 1)

        msg = f"[bt] epoch {epoch:3d}/{args.epochs}  loss={ep_loss:.4f}  "
        msg += f"lr={sched.get_last_lr()[0]:.2e}  dt={time.time()-t0:.1f}s"
        msg += (f"  | train cos={ep_diag.get('cos', 0):.3f} "
                f"ct1={ep_diag.get('ct1', 0):.3f} "
                f"ct2={ep_diag.get('ct2', 0):.3f} "
                f"niche1={ep_diag.get('niche1', 0):.3f}")
        if val_loader is not None and (epoch % args.log_every == 0 or epoch == 1):
            stats = evaluate_embedding(model, val_loader, device)
            msg += (f"  | val: loss={stats['loss']:.4f} "
                    f"cos={stats['cos_sim']:.3f} "
                    f"ct1={stats['ct_acc1']:.3f} "
                    f"ct2={stats['ct_acc2']:.3f} "
                    f"niche1={stats['niche_acc1']:.3f}")
            if stats["loss"] < best_val:
                best_val = stats["loss"]
                torch.save({"model": model.state_dict(),
                            "args": vars(args),
                            "pca_dim": pca_dim,
                            "n_cell_types": n_cell_types,
                            "n_niches": n_niches,
                            "cell_type_vocab": vocab},
                           args.out)
                msg += "  *saved*"
        log(msg)

    torch.save({"model": model.state_dict(),
                "args": vars(args),
                "pca_dim": pca_dim,
                "n_cell_types": n_cell_types,
                "n_niches": n_niches,
                "cell_type_vocab": vocab},
               args.out)
    log(f"[bt] done. saved: {args.out}")


if __name__ == "__main__":
    main()
