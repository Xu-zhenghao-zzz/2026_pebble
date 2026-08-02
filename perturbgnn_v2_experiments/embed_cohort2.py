"""Compute embeddings for cohort 2 sections using existing bt_encoder_v3.pt.

Reuses the encoder trained on cohort 1 (M001+M002). Tests transfer
across cohorts.
"""
import sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import torch

sys.path.insert(0, "/mnt/data/xuzh/spac_seq/perturbgnn_v2/src")
PROCESSED = Path("/mnt/data/xuzh/spac_seq/perturbgnn_v2/processed")

from perturbgnn_v2.embedding import SupervisedCrossModalEncoder
from perturbgnn_v2.embedding.train_bt import gather_neighborhoods


def main():
    print("[c2-emb] loading bt_encoder_v3.pt...", flush=True)
    ckpt = torch.load(PROCESSED / "bt_encoder_v3.pt",
                       map_location="cuda", weights_only=False)
    model = SupervisedCrossModalEncoder(
        K=ckpt["args"]["K"],
        pca_dim=ckpt["pca_dim"],
        n_cell_types=ckpt["n_cell_types"],
        n_niches=ckpt["n_niches"],
        hidden=ckpt["args"]["hidden"],
        embed_dim=ckpt["args"]["embed_dim"],
        n_layers=ckpt["args"]["n_layers"],
        alpha_align=ckpt["args"]["alpha_align"],
    ).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[c2-emb] model loaded (pca_dim={ckpt['pca_dim']}, n_ct={ckpt['n_cell_types']})", flush=True)

    for sid in ["subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]:
        path = PROCESSED / f"{sid}_v2.h5ad"
        if not path.exists():
            print(f"[c2-emb] {sid}: not yet loaded, skip"); continue
        print(f"\n[c2-emb] === {sid} ===", flush=True)
        t0 = time.time()
        a = ad.read_h5ad(path)
        # build neighborhoods using cohort 2 cell_type vocab
        # gather_neighborhoods returns (nbr_pca, nbr_ct, ct_self, niche, vocab)
        # cohort 2 cell_type vocab differs from cohort 1 — re-encode using cohort 1 vocab
        cohort1_vocab = ckpt["cell_type_vocab"]
        nbr_pca, nbr_ct_full, ct_self_full, niche, c2_vocab = gather_neighborhoods(a, K=15)
        # re-encode nbr_ct to cohort 1 vocab (out-of-vocab -> uniform)
        n_c1 = len(cohort1_vocab)
        c1_lookup = {c: i for i, c in enumerate(cohort1_vocab)}
        N, K, _ = nbr_ct_full.shape
        nbr_ct_c1 = np.zeros((N, K, n_c1), dtype=np.float32)
        for k in range(K):
            for li, ct in enumerate(c2_vocab):
                if ct in c1_lookup:
                    mask = nbr_ct_full[:, k, li] > 0
                    nbr_ct_c1[mask, k, c1_lookup[ct]] = 1.0
        # out-of-vocab -> uniform
        miss = nbr_ct_c1.sum(axis=-1) == 0
        nbr_ct_c1[miss] = 1.0 / n_c1

        # batched inference
        batch = 4096
        embs = []
        with torch.no_grad():
            for i in range(0, N, batch):
                p = torch.from_numpy(nbr_pca[i:i+batch]).cuda()
                c = torch.from_numpy(nbr_ct_c1[i:i+batch]).cuda()
                z = model.encode_pca(p)
                embs.append(z.cpu().numpy())
        emb = np.concatenate(embs, axis=0)
        out = PROCESSED / f"embed_{sid}_v3.npy"
        np.save(out, emb)
        print(f"  shape: {emb.shape}, saved: {out}, dt={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
