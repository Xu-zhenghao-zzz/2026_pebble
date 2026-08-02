# GNN Selection for v2.1 — Technical analysis and recommendation

**Decision**: replace v2's set-pooling encoder (`bt_encoder.py`) with a
**2-layer Edge-aware Graph Attention Network (EGAT)** as the primary
v2.1 encoder. Implement **2-layer edge-augmented GraphSAGE** as a
fallback / ablation control.

This doc records the technical analysis behind that decision.

---

## 1. Hard constraints from v2's data

The encoder must operate within these fixed constraints — they are not
negotiable design choices, they are properties of the SPAC-seq tissue
graph (see `PIPELINE_LAYERS.md` L0):

| Constraint | Value | Implication |
|---|---|---|
| Node count per slice | 273K (M001) - 630K (subQ) | full-batch GNN must use sparse ops |
| Edge count per slice | 3-5M undirected | attention must be sparse |
| Average degree | ~12-15 | denser than citation graphs, sparser than image grids |
| Node feature dim | 42 | small, no need for deep feature extractor |
| **Edge feature dim** | **8** | **the GNN MUST consume edge features** |
| Edge features | `[dx, dy, dist, same_CT, src_endpoint, barrier, density_avg, vessel_dist]` | `barrier` and `vessel_dist` carry causal info; dropping them loses signal |
| Supervision targets | cell_type (8) + niche_id (12) | unchanged from v2 |
| Output dim | 64 (per spot) | must match v2's L2-L5 contract |
| Layers | 2 (max 3) | L>3 → over-smoothing → L2 match fails (SMD>0.13) |
| Hardware | 4× V100S-32GB (shared with 2608 PPO runs) | single-GPU ~24GB practical headroom |

**Critical**: the edge features encode **causal structure** — `barrier_score`
represents ECM discontinuity, `vessel_dist_avg` represents vascular access
for soluble signals. Any GNN that ignores edge features throws away the
physical prior that distinguishes NCA propagation from generic message passing.

---

## 2. Candidate GNN types — head-to-head

| Type | Expressive | Memory (358K nodes, 1 layer) | Train time | Edge features | Over-smoothing risk | PyG support |
|---|---|---|---|---|---|---|
| GCN (Kipf 2017) | weak | ~2 GB | 30 min | ❌ | high at L>2 | ✅ stable |
| ChebNet (Defferrard 2016) | medium | ~2 GB | 30 min | ❌ | medium | ✅ |
| GraphSAGE (Hamilton 2017) | medium | ~8 GB | 1-2 hr | ⚠ needs custom conv | low | ✅ |
| GAT (Veličković 2018) | strong | ~15 GB | 3-5 hr | ⚠ needs EGAT conv | medium | ✅ |
| GIN (Xu 2019) | strong | ~8 GB | 1-2 hr | ❌ | medium | ✅ |
| **EGAT** (edge-aware GAT) | very strong | ~22 GB | 5-8 hr | ✅ native | medium | ⚠ PyG 2.7 experimental |
| SIGN (Rossi 2020) | medium | ~2 GB | 20 min | ❌ | low | ✅ |
| Graph Transformer (full) | strongest | >50 GB | infeasible | ✅ | high | ⚠ |

### Why GCN / ChebNet / SAGE-native are eliminated

These three do not natively consume edge features. v2's edge features
encode ECM barriers and vascular access — dropping them means the GNN
sees only the topology, not the **physical** meaning of each edge.
This is unacceptable for a causal identification framework where the
edge semantics are load-bearing.

Workarounds (edge-feature concat into node features, then run vanilla
SAGE) lose the **per-edge** information — the model knows "node v has
high average barrier to neighbours" but not "edge (v,u) specifically
crosses a barrier".

### Why Graph Transformer is eliminated

Full self-attention on 358K nodes is O(N²) = ~10¹¹ attention entries,
~400 GB at fp32. Even with sparse attention windows (e.g. 2-hop), it
collapses to EGAT-equivalent compute with worse implementation quality.

Graph Transformers are SOTA on small molecule / protein graphs (N<10K).
On dense tissue graphs (N>10⁵), attention-based GNNs (EGAT) dominate.

### Final shortlist: EGAT (primary) + edge-SAGE (fallback)

---

## 3. Primary recommendation: 2-layer EGAT

### Architecture

```
Inputs per slice:
  x          (N, 42)  float32   node features
  edge_index (2, E)   int64     PyG edge list, undirected
  edge_attr  (E, 8)   float32   edge features

─────────────────────────────────────────────────────────────────────
EGAT Layer 1:
  EGenConv(in_dim=42, edge_dim=8, out_dim=128, heads=4, concat=True)
  → LayerNorm(128)
  → ReLU
  → Dropout(0.1)
  → Residual: x + linear(42→128)

EGAT Layer 2:
  EGenConv(in_dim=128, edge_dim=8, out_dim=64, heads=4, concat=False)
  → LayerNorm(64)
  → ReLU
  → Dropout(0.1)

Output:  z ∈ ℝ^{N × 64}        per-spot embedding

Heads (same as v2 for fair comparison):
  head_ct(z)    → 8-class  cell_type prediction
  head_niche(z) → 12-class niche_id prediction

Loss:
  L_cls = CE(head_ct(z), ct) + CE(head_niche(z), niche)
  L_align = 1 - cos(z, z_v2_set_pooling).mean()    # soft anchor to v2
  L = L_cls + 0.05 × L_align
```

### Why the soft alignment to v2's set-pooling embedding

v2's set-pooling embedding has `cos(source_centroid, ntc_centroid) = 0.9997`.
That is a *niche-balance* property we want to preserve: source and NTC
bins in the same niche should map to nearby embeddings so L2 matching
works. Adding a small (α=0.05) alignment loss to v2's embedding anchors
the EGAT output in the same niche-balanced subspace.

Without this, EGAT might learn a more discriminative embedding that
breaks L2 matching (SMD > 0.13). With it, EGAT explores **within the
niche-balanced subspace** rather than away from it.

### Training configuration

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data=pyg_data,
    num_neighbors=[15, 10],   # 1-hop sample 15, 2-hop sample 10
    batch_size=4096,           # anchor nodes per batch
    shuffle=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 100 epochs × 80 batches/epoch = 8000 steps
# ~3-5 hours on V100S-32GB
```

### Memory budget (single V100S-32GB)

```
x            (N=358K, 42)               =   60 MB
edge_index   (2, 5M)                    =   40 MB
edge_attr    (5M, 8)                    =  160 MB

EGAT L1 attention (heads=4, 5M edges × 4)  =   80 MB
EGAT L1 hidden   (358K, 128)              =  184 MB
EGAT L2 attention (heads=4, 5M × 4)       =   80 MB
EGAT L2 hidden   (358K, 64)               =   92 MB

Backward pass gradients                  ≈ 1.0 GB
Adam optimizer state (2× params)         ≈ 0.5 GB
PyTorch overhead                         ≈ 0.5 GB

Mini-batch sampling (NeighborLoader)     ≈ 4 GB
─────────────────────────────────────────────────
Total                                    ≈ 6 GB   ← fits comfortably in 32 GB
```

### Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| PyG EGATConv is experimental, may have bugs | medium | implement edge-augmented SAGE as parallel fallback |
| Attention collapses (all heads attend to same neighbours) | medium | log attention entropy, regularise with `-H(att)·Σ att·log(att)` |
| Over-smoothing at L=2 already kills SMD | low | if SMD > 0.13, drop to L=1 + wider hidden |
| Training time > 5 hours | medium | reduce epochs to 50 + early stopping on val L_cls |
| GPU contention with 2608 PPO runs | high | wait for PPO to finish or use different GPU |

---

## 4. Fallback: 2-layer edge-augmented GraphSAGE

If EGAT training is unstable or PyG's EGATConv has bugs, fall back to
a custom edge-aware SAGE:

### Architecture

```python
from torch_geometric.nn import MessagePassing

class EdgeSAGEConv(MessagePassing):
    """GraphSAGE with edge features concatenated into messages."""
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr='mean')
        self.message_lin = Linear(in_dim + edge_dim, out_dim)
        self.update_lin = Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops with zero edge_attr.
        self_loops = torch.arange(x.size(0), device=x.device).view(1, -1).repeat(2, 1)
        edge_index_full = torch.cat([edge_index, self_loops], dim=1)
        edge_attr_full = torch.cat([
            edge_attr,
            torch.zeros(x.size(0), edge_attr.size(1), device=x.device),
        ], dim=0)
        return self.propagate(edge_index_full, x=x, edge_attr=edge_attr_full)

    def message(self, x_j, edge_attr):
        return self.message_lin(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr, x):
        return self.update_lin(torch.cat([x, aggr], dim=-1))
```

Stack 2 layers (42→128→64), same heads, same loss as EGAT.

### Why edge-SAGE is the right fallback

- **Same edge-feature consumption** as EGAT
- **Mean aggregation** is more stable than attention
- **PyG MessagePassing** is rock-solid (used by thousands of papers)
- Train time ~1.5 hr (faster than EGAT)
- Memory ~8 GB (much lower than EGAT)

---

## 5. Why these and not GIN / ChebNet / SIGN

### GIN (Graph Isomorphism Network)

GIN is the most expressive message-passing GNN (as powerful as 1-WL test).
But:

- **No native edge features** in the standard GINConv
- Designed for **graph-level classification** (molecules, proteins),
  not **node-level embedding** in dense graphs
- Stronger expressiveness is wasted on v2's task (we don't need to
  distinguish non-isomorphic subgraphs; we need smooth spot embeddings)

### ChebNet

Chebyshev spectral GNN. Operates on graph Laplacian eigenvalues.

- **No edge features** in spectral framework (edge weights are baked
  into W, not into node updates)
- Spectral filters are global, not local — bad for spot-level tasks
- Fast (1st-order Chebyshev = GCN), but expressive power too weak

### SIGN (Scalable Inception Graph Network)

SIGN pre-computes A^k X for k=1..K and learns a linear mixer. Extremely
fast (no message passing during training).

- **No edge features** in A^k X (unless you hand-craft A with edge weights)
- Equivalent to a single-layer MLP on multi-hop aggregated features
- Expressiveness: same as a depth-1 GCN with K-hop features
- Could work but feels like a step backwards from v2's set-pooling

---

## 6. Comparison: v2 set-pooling vs EGAT vs edge-SAGE

| Property | v2 set-pooling | **EGAT (recommended)** | edge-SAGE (fallback) |
|---|---|---|---|
| Captures neighbour identities? | mean+max+anchor | attention-weighted | mean |
| Edge features used? | ❌ (only via v1 KNN topology) | ✅ native | ✅ via message concat |
| Directional info preserved? | ❌ set is unordered | ⚠ via learned attention | ❌ mean is symmetric |
| Train time | 1.5 hr | 3-5 hr | 1.5 hr |
| Memory | 4 GB | 6-22 GB | 8 GB |
| PyG stability | high (uses only nn.Linear) | medium (experimental conv) | high |
| L2 match SMD expected | 0.13 (current) | 0.10-0.15 (similar) | 0.10-0.15 |
| L3 Durbin δ expected | baseline | ±20% (Fix 5 decision rule) | ±20% |
| Methodological novelty for paper | low | **high** | medium |

### Expected v2.1 paper impact

If EGAT passes Fix 5 acceptance criteria (SMD not worse + δ within ±20%):

> "We replace the set-pooling encoder with a 2-layer edge-aware graph
> attention network. The Phgr1 cross-cohort effect (δ=0.43, q=3e-60)
> remains significant under the new encoder (δ=0.41, q=1e-58), demonstrating
> that the v2 result is not an artefact of the set-pooling choice."

If EGAT finds **stronger** effects:

> "EGAT discovers an additional 3 cross-cohort NCA hubs missed by the
> set-pooling encoder (Table N), demonstrating that edge-aware message
> passing captures biologically meaningful signal that the simpler
> encoder missed."

Either outcome is publishable.

---

## 7. Implementation plan

### Phase 1: Edge-SAGE first (lowest risk, 1 day)

1. Implement `EdgeSAGEConv` in `perturbgnn_v2_1_src/baseline/edge_sage.py`
2. Write training loop (mirror `train_bt.py` interface)
3. Train on M001+M002, validate on M003
4. Compute embedding diagnostics (cos, silhouette, ARI, NMI)
5. Run L2 matching + check SMD
6. If SMD > 0.13 → debug, otherwise commit

### Phase 2: EGAT (1-2 days)

1. Implement `EGATEncoder` using PyG `EGATConv` or custom conv
2. Same training loop, swap encoder
3. Add attention entropy regulariser (5e-3 weight)
4. Train + evaluate same as Phase 1
5. Compare δ vs set-pooling + edge-SAGE

### Phase 3: Full L2-L5 re-run (1-2 days)

1. Use EGAT embeddings as input to L2 matcher
2. Re-run L3 Durbin scan for Phgr1 across all 8 slices
3. Re-run L4 causal diagnostics
4. Re-run L5 cross-cohort meta-analysis
5. Compare `cross_cohort_replicated.csv` between v2 and v2-GNN

### Phase 4: Notebook integration (0.5 day)

Add §15.5 "GNN encoder sensitivity" to `SINGLE_MASTER_perturbgnn_v2.ipynb`
with the head-to-head comparison.

**Total: 4-5 working days** assuming server access is restored.

---

## 8. Decision summary

| Decision | Choice | Reason |
|---|---|---|
| Primary GNN type | **2-layer EGAT** | edge-aware, expressive, attention interpretable |
| Fallback GNN type | **2-layer edge-SAGE** | stable, lower memory, fast |
| Number of layers | **2** | over-smoothing risk + L2 SMD requirement |
| Hidden dim | **128 → 64** | match v2 output dim |
| Heads (EGAT) | **4** | standard, memory-fits |
| Training | mini-batch NeighborLoader [15,10] | 358K nodes can't go full-batch |
| Loss | supervised (cell_type + niche) + soft align to v2 | collapse prevention + L2 compatibility |
| Acceptance gate | SMD ≤ 0.13 + δ within ±20% of v2 + cos(z, z_v2) > 0.7 | three independent pass criteria |

**This is the path that converts v2's "set-pooling called GNN" into a
true message-passing GNN, with the minimum risk of breaking the L2-L5
causal pipeline that v2 depends on.**
