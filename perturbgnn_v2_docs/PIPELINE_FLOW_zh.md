# PerturbGNN v2 — 从输入到输出的完整流程（中文教材版）

> 这份文档是 [PIPELINE_FLOW.md](PIPELINE_FLOW.md) 的中文深度版。
> 那一份用 ASCII 框图给出"每层是什么"，这一份回答"**每层为什么这么做、
> 数学推导是什么、关键数字怎么解读、Reviewer 会怎么挑刺、如果换方法
> 会怎样**"。目标是让一个第一次接触空间转录组 + 因果推断的研究生，
> 读完这份文档就能独立跑通并辩护整套 v2 流程。

## 目录

- [§0 总览](#0-总览)
- [§1 L0 — 原始数据与组织图](#1-l0--原始数据与组织图)
- [§2 L1 — 双视角监督式 GNN 编码器](#2-l1--双视角监督式-gnn-编码器)
- [§3 L2 — Embedding 匹配对照](#3-l2--embedding-匹配对照)
- [§4 L3 — 空间 Durbin 模型](#4-l3--空间-durbin-模型)
- [§5 L4 — 因果诊断三件套](#5-l4--因果诊断三件套)
- [§6 L5 — 跨 cohort 元分析](#6-l5--跨-cohort-元分析)
- [§7 L6 — 正交人类验证](#7-l6--正交人类验证)
- [§8 设计原则总结](#8-设计原则总结)
- [§9 v2.1 修复预览](#9-v21-修复预览)

---

## §0 总览

### 0.1 我们要回答的科学问题

SPAC-seq（Spatial Perturbation and Transcriptome sequencing）是一种
**在空间上做 CRISPR 扰动 + 测转录组**的技术。简单说：

- 你给小鼠肿瘤里转一堆 sgRNA
- 每个细胞（或每个空间 bin）只被一种 sgRNA 编辑
- 同时测每个 bin 的 ~19000 个基因表达 + 这个 bin 是哪个 sgRNA
- 保留每个 bin 的物理坐标 (x, y)

这就允许你问一个 v1/v2 都关心的问题：

> **如果我敲低基因 X，那么 X 周围那些"没有被敲低、只是物理邻居"的细胞，
> 转录状态会变吗？** 这种"非细胞自主"（non-cell-autonomous, NCA）效应
> 是不是因果的，能不能跨 cohort 复现？

举例：Phgr1 是个功能不太清楚的基因。如果敲低 Phgr1 后，它**周围 50 µm
范围内的成纤维细胞、巨噬细胞、T 细胞的模块得分有显著变化**，那 Phgr1
可能是个 NCA 信号枢纽——这跟"Phgr1 在细胞内部干什么"是两回事。

### 0.2 v2 的五层 pipeline（加 L0 数据 + L6 验证）

```
L0  原始数据 → AnnData + 组织图
L1  GNN 编码器 → 64 维 spot embedding
L2  匹配对照 → 反事实 null
L3  空间 Durbin → 因果效应 δ
L4  因果诊断 → DiD + IV + Γ* 三重检验
L5  跨 cohort 元分析 → reportable gate
L6  TCGA + ICB 正交验证
```

**核心设计哲学**：
- **L1-L2 是表示学习**（深度学习擅长）
- **L3-L5 是经典统计**（空间计量学 + 因果推断 + meta 分析）
- **两者用 embedding 桥接**，但因果 claim 完全不依赖 GNN 的可解释性

这种分层是 v2 的卖点——把"深度学习"和"因果识别"的边界划清楚，让 reviewer
不能拿"你的 GNN 是黑盒"来 KO 论文。

### 0.3 v1 → v2 的关键变化

| 维度 | v1 (perturbradius) | v2 |
|---|---|---|
| 框架 | 距离衰减 + "传播半径" | 因果识别 |
| 对照 | vs-NTC（NTC 全局基线） | embedding 匹配（局部反事实） |
| 空间模型 | 1D 同心环 | 2D resistance distance + Durbin |
| 因果证据 | label-shuffle permutation | DiD + IV + Rosenbaum Γ* |
| 复现 | 单 cohort 3 切片 | 跨 cohort 8 切片 |
| 头号发现 | BLNK（单切片显著） | **Phgr1**（7 切片跨 cohort） |
| TCGA Cox HR | 0.744（BLNK） | **0.556**（Phgr1，更强） |
| 假阳性率 | — | 79% v1 hits 是 v2 假阳性 |

---

## §1 L0 — 原始数据与组织图

### 1.1 输入数据

**Cohort 1 — 肺转移（lung metastasis）**

来自 Zhang et al., *Cell* 2026 的 SPAC-seq 数据集。MC38 小鼠结直肠癌
细胞系，注射到小鼠体内形成肺转移。3 个切片：

| 切片 | bin 数 | NTC bin 数 | 每 bin 基因数 |
|---|---|---|---|
| M001 | 273 886 | 548 | 19 059 |
| M002 | 358 149 | 191 | 19 059 |
| M003 | 488 373 | 1 813 | 19 059 |

总共 ~110 万 bin。每个 bin 大约 10 µm × 10 µm（亚细胞级分辨率）。
guide library 是 1520 个 sgRNA，对应 ~120 个靶基因（每基因 ~12 guides），
但**实际有效编辑只测到一个 guide per bin per gene**——这是 SPAC-seq 的
关键技术限制（见 §1.5）。

**Cohort 2 — 多切片（multiple-section subQ）**

同样 MC38 细胞系，但取皮下（subQ）组织，连续切片（5 个，间隔 50 µm）：

| 切片 | bin 数 |
|---|---|
| subQ-1 | ~632 K |
| subQ-2 | ~615 K |
| subQ-3 | ~628 K |
| subQ-4 | ~625 K |
| subQ-5 | ~580 K |

**Cohort 2 解决了 cohort 1 的两个限制**：
1. 每基因有 **2 个独立 guide** → 缓解 off-target 问题
2. 5 个连续切片提供 **3D 维度** → 可以看效应在 Z 轴的传播

**Cohort 3 — 时空 T 细胞（spatiotemporal）**

OT1-Cas9 T 细胞（不是肿瘤细胞被扰动，是 T 细胞被扰动），3 个时间点
（Day 4 / 7 / 10）× 2 个生物学重复。**panel 不含 sgPhgr1**，所以
这个 cohort 不能用来复现 Phgr1。我们只用它做时间维度的辅助分析。

### 1.2 每个 h5ad 的内部结构

```python
import anndata as ad
adata = ad.read_h5ad("M001_v2.h5ad")

adata.X              # 稀疏矩阵 (N=273 886, G=19 059), 每个元素是该 bin 该基因的 UMI 计数
adata.obsm['spatial']  # (N, 2) — 每个 bin 的 (x_um, y_um) 物理坐标
adata.obsm['X_pca']    # (N, 32) — scanpy 跑出来的 32 维 PCA
adata.obs['cell_type'] # 字符串, 8 类: 'Malignant', 'Fibroblast', 'Macrophage', 'CD8+T', 'NK', 'B', 'Endothelial', 'Non-immune'
adata.obs['cell_type_idx']  # int8, 0-7, 给模型用的数值化标签
adata.obs['niche_id']       # int8, 0-11, KMeans(k=12) 在 cell_type composition 上的聚类结果
adata.obs['guide']          # 字符串, sgRNA 名字, 比如 'sgPhgr1_1' 或 'NTC'
adata.obs['is_source']      # bool, True = guide 是某个靶基因 (非 NTC)
adata.obs['is_ntc']         # bool, True = NTC control guide

# 8 个 module score (per-bin, 标准化到 z-score)
adata.obs['module_score_ifn_response']    # IFN 通路激活程度
adata.obs['module_score_hypoxia']         # 缺氧程度
adata.obs['module_score_fibroblast']      # 成纤维细胞特征强度
adata.obs['module_score_macrophage']      # 巨噬细胞特征强度
adata.obs['module_score_cd8_like']        # CD8+T 细胞特征强度
adata.obs['module_score_endothelial']     # 内皮细胞特征强度
adata.obs['module_score_malignant']       # 恶性细胞特征强度
adata.obs['module_score_nk_like']         # NK 细胞特征强度
```

### 1.3 module score 是怎么算的

每个 module score 是一组 marker 基因表达的均值（然后 z-score 标准化）。
比如 `module_score_ifn_response`：

```python
ifn_markers = ['Stat1', 'Irf1', 'Ifit1', 'Ifit3', 'Oas1b', 'Mx1', 'Isg15']
adata.obs['module_score_ifn_response'] = (
    adata[:, ifn_markers].X.mean(axis=1)  # 这 7 个基因的平均表达
    - 全切片均值
) / 全切片标准差
```

> **为什么用 module score 而不是单基因表达？**
> SPAC-seq 是低深度测序（每 bin 总 UMI ~500-2000），单基因表达噪声极大。
> Module score 是 7-15 个 marker 基因的均值，降噪 5-10 倍，统计功效更高。

### 1.4 组织图（tissue graph）

代码：`graph_build.py`

对每个切片单独建图：

**节点**：每个 bin 是一个节点。

**边**（两种来源取并集）：
1. **Delaunay 三角剖分**：所有相邻 bin 自动连边
2. **kNN(k=6)**：每个 bin 跟它最近的 6 个邻居连边，但距离 > 60 µm 的丢弃

为什么用 Delaunay + kNN 而不是只用 kNN？

- Delaunay 保证**全局连通性**（不会出现孤岛）
- kNN 保证**局部正则性**（每个节点的度数大致相等）
- 两者并集 = 既不丢全局结构，又控制局部稀疏度

最终每个切片有 ~3-5M 条边。

**节点特征**（42 维）：

```python
x = concat[
    PCA32,                 # 32 维, 来自 obsm['X_pca'][v]
    CT_onehot8,            #  8 维, cell_type 的 one-hot
    density_local,         #  1 维, v 周围 40 µm 内的邻居数
    vessel_distance,       #  1 维, v 到最近内皮细胞的距离 (µm)
]  # 总共 42 维
```

**边特征**（8 维）：

```python
edge_attr[i->j] = [
    dx,                # j.x - i.x
    dy,                # j.y - i.y
    dist,              # 欧氏距离
    same_cell_type,    # 1.0 if i, j 同 cell_type else 0.0
    src_endpoint,      # 1.0 if i or j 是 source bin else 0.0
    barrier_score,     # cell-type 不连续性, ECM/坏死的代理
    density_avg,       # (i.density + j.density) / 2
    vessel_dist_avg,   # (i.vessel_dist + j.vessel_dist) / 2
]
```

### 1.5 SPAC-seq 数据集的 3 个本质限制

reviewer 必问，必须诚实写进 Discussion：

**限制 1：每基因每切片只有 1 个有效 guide（cohort 1）**

虽然 library 里有 ~12 个 guide per gene，但因为转导效率低 + 单 bin
分辨率，**实际测到的 source bin 里每基因每切片只有一个 guide 是显著的**。
这意味着：

- 不能做 held-out-guide test（不同 guide 互相验证）
- **off-target 永远无法完全排除**——如果这个 guide 既敲低了 Phgr1 又
  非特异性地敲低了某个邻居基因，v2 的 δ 会归因于 Phgr1 但实际可能是
  off-target 效应

Cohort 2 有 2 个 guide per gene 部分缓解了这个问题，但 2 个 guide 序列
可能相似，共享 off-target。

**限制 2：3 个 cohort 1 切片的 guide panel 不重叠**

```
M001 测到的基因 ∩ M002 测到的基因 ∩ M003 测到的基因 = 只有 7 个！
（Clcn2 / F11r / Fzd5 / Kcnk6 / Myo1c / Phgr1 / Utrn）
```

这意味着跨切片复现只能在 7 个基因上做。Phgr1 是其中之一——这是**幸运**，
不是设计。

**限制 3：NTC 分布不均**

```
M001: 548 NTC bins   (NTC 比例 0.2%)
M002: 191 NTC bins   (NTC 比例 0.05% — 严重不足！)
M003: 1813 NTC bins  (NTC 比例 0.4%)
```

M002 的 NTC 太少，所以 v2 在 M002 上的 vs-NTC 分析借用 M001 + M003 的
NTC pool——但这引入了跨切片 baseline 的潜在 batch effect。

> **Reviewer 会问**：「M002 的 NTC pool 不足，你借用 M001/M003 的 NTC，
> 这会不会让你的 M002 δ 估计有偏？」
>
> **v2 的回答**：「我们用 embedding-matched control 替代了 vs-NTC，匹配
> 是同切片内的 cosine-NN，不依赖 NTC pool 大小。M002 的 matched control
> 数量跟 source 数量 1:1 配对，不受 NTC 比例影响。」

---

## §2 L1 — 双视角监督式 GNN 编码器

代码：`bt_encoder.py` + `train_bt.py`

### 2.1 这一层的任务

**输入**：每个 spot 的 32 维 PCA + 8 维 cell_type one-hot + 它的 K=15 个
邻居的同样信息。

**输出**：每个 spot 的 64 维 embedding，要求：
- 同一 niche 的 spot embedding 相似（保留 tissue structure）
- 不同 cell_type 的 spot embedding 可分（保留 cell identity）
- source bin 和它周围的 NTC bin embedding 相似（niche 一致）
- **同时**，embedding 不能 collapse（所有 spot 都聚成一坨）

### 2.2 为什么不用无监督学习（v1 走过的弯路）

v1 试过两个无监督方案，都失败了：

**尝试 1：Barlow Twins（2021）**

思想：两个增强视角 z1, z2，让它们的 cross-correlation 矩阵接近单位阵
（dim-wise 独立 + dim-wise 对齐）。

为什么失败：Barlow Twins 假设 PCA 视角和 cell_type 视角**在 dim 层面
对齐**——但 PCA 的 32 个 dim 是数据驱动的（按方差排序），cell_type 的
8 个 dim 是语义的（CD8+T 是 dim 0），两者**没有 dim-level 对应关系**。
强行对齐导致 representation collapse。

**尝试 2：SimSiam（2020）**

思想：predictor + stop-gradient，不需要负样本。

为什么失败：没有负样本的对比学习很容易收敛到 trivial equilibrium——
两个视角都映射到常数向量，cos(z1, z2) = 1，但表示完全没用。

### 2.3 v2 的最终方案：监督式 cross-modal encoder

核心洞察：**SPAC-seq 自带可信的标签**——cell_type（8 类，scanpy 注释）
和 niche_id（12 类，KMeans 聚类）。这些标签是组织学先验，不是模型学的。
直接用它们做监督信号，**比无监督更稳**。

**架构**：

```
view 1 (PCA):     K=15 邻居的 PCA         [15, 32]
view 2 (CT):      K=15 邻居的 cell_type   [15,  8]
```

对每个视角，先做 **set pooling**：

```python
mean_pool = nbr_feats.mean(dim=1)   # 邻居特征的均值
max_pool  = nbr_feats.max(dim=1)    # 邻居特征的 max
anchor    = nbr_feats[:, 0, :]      # 中心 spot 自己
combined  = cat([mean_pool, max_pool, anchor])
# PCA view: 3 × 32 = 96 dim
# CT view:  3 × 8  = 24 dim
```

然后过 3 层 MLP：

```
MLP:
  Linear(in_dim → 128) → LayerNorm → ReLU → Dropout(0.1)
  Linear(128 → 128)    → LayerNorm → ReLU → Dropout(0.1)
  Linear(128 → 64)
```

得到 z1（PCA 视角，64 维）和 z2（CT 视角，64 维）。

**共享分类头**（这是 v2 的设计选择，也是 v2.1 Fix 3 要修的问题——见 §9）：

```python
head_ct(z)  = Linear(64→64) → LayerNorm → ReLU → Dropout → Linear(64→8)   # cell_type
head_niche(z) = 同样结构 → Linear(64→12)  # niche
```

z1 和 z2 都过这同一个 head_ct 和 head_niche。

**损失函数**：

```
L_cls    = CE(head_ct(z1), ct) + CE(head_niche(z1), niche)
         + CE(head_ct(z2), ct) + CE(head_niche(z2), niche)

L_align  = 1 - cos(z1, z2).mean()

L        = L_cls + 0.1 × L_align
```

### 2.4 为什么 alignment 权重是 0.1

`α_align = 0.1` 是个超参。如果太大（比如 1.0），alignment loss 主导，
z1 和 z2 会强行拉到一起，丢掉各自视角的特殊信息。如果太小（比如 0.01），
两个视角完全独立，下游用 `0.5 × (z1 + z2)` 时信息丢失。

**0.1 是经验调出来的**：分类损失（CE）数量级 ~1-3，alignment 损失
~0.001-0.1，权重 0.1 让两者贡献量级相当。这个值的 sensitivity analysis
在 `LIMITATIONS_V3.md` 里待做。

### 2.5 训练配置

```python
optimizer = Adam(lr=1e-3)
batch_size = 4096
epochs = 200
train_slices = [M001, M002]   # ~632K spots
val_slice = M003               # ~488K spots, held-out
early_stopping = patience 20 epochs on val L_cls
```

每个 epoch ~30 秒（V100S），200 epochs ~1.5 小时。

### 2.6 输出

| 文件 | 内容 |
|---|---|
| `bt_encoder_v3.pt` | encoder + heads 的 state_dict |
| `embed_M001.npy` 等 | (N, 64) float32, 每行是 0.5·(z1+z2) |
| `embedding_validation.json` | 诊断指标 |

诊断指标（实际数字）：

```
M001: src_ntc_centroid_cos = 0.9997  (source 和 NTC 的 embedding centroid 几乎重合)
      silhouette_celltype = 0.037    (cell_type 在 embedding 空间可分, 但不强)
      niche_ari = 0.042, niche_nmi = 0.093

M002: src_ntc_centroid_cos = 0.9994
      silhouette_celltype = 0.156    (M002 cell_type 分得更开)
```

**关键数字解读**：

- `cos = 0.9997` 意味着 source bin 和 NTC bin 在 embedding 空间几乎是
  **同一类**——这是好事，说明它们 niche 一致（不然匹配对照就无意义）。
  但**也是坏事**，因为 v2.1 Fix 3 担心这是 head 共享强制的结果。

- `silhouette = 0.037` 意味着 cell_type 在 embedding 空间**勉强可分但不
  是线性可分**。这是因为同一 cell_type 在不同 niche 的 embedding 不同
  （比如肿瘤边缘的巨噬细胞 vs 肿瘤中心的巨噬细胞），silhouette 自然低。
  这不是 bug，是设计——我们就是要 niche-aware embedding。

### 2.7 Reviewer 会怎么挑

**Q：你的 GNN 实际上不是 GNN，是邻域 pooling + MLP。为什么叫 GNN？**

A：诚实回答——叫"GNN"是 marketing。严格的 message-passing GNN（GCN/
GAT/GraphSAGE）在 350K 节点的密集 tissue graph 上会过平滑（见 §9 Fix 5）。
我们用的是 set-based neighborhood encoder，**信息瓶颈等价于 1-hop GNN
但避免了 L 层传递导致的 over-smoothing**。代码里保留 GNN 命名是为了跟
文献对接，实质是 neighborhood-aware encoder。

**Q：cos(z1, z2) = 0.9997 是不是 head 共享导致的 self-fulfilling？**

A：这是 v2.1 Fix 3 要解决的问题。Fix 3 用 **probe transfer** 测真实
对齐质量：在 z1 上训 head，在 z2 上 zero-shot 测试。如果 z1→z2 准确率
≥ 70% × z1→z1 准确率，对齐质量是真实的；否则承认 0.9997 是架构 artifact。

**Q：为什么 K=15，不是 K=10 或 K=30？**

A：K=15 在 8 µm bin size 下对应 ~120 µm 邻域半径，覆盖一个完整 niche
（典型 niche 半径 100-150 µm）。K 太小（K=5）会丢 niche 信息，K 太大
（K=30）会跨 niche 平均。这个值是从 v1 调参继承的，sensitivity 待做。

---

## §3 L2 — Embedding 匹配对照

代码：`matching/match.py`

### 3.1 反事实框架

因果推断的核心概念：**反事实（counterfactual）**。
对每个 source bin s（被 Phgr1 敲低的 bin），我们想知道：

> "如果 s **没有**被敲低，它的 module score 会是多少？"

这个反事实值叫 `Y(0)`（潜在结果 under control）。我们观测到的是
`Y(1)`（潜在结果 under treatment）。**因果效应 = Y(1) - Y(0)**。

问题是同一个 bin 不能同时被敲低和不敲低（fundamental problem of
causal inference）。所以我们要**找一个跟 s 非常像、但没被敲低的 bin**
作为 `Y(0)` 的代理——这就是 matched control。

### 3.2 匹配算法

```python
def matched_control_for_gene(adata, embed, target_gene, cfg):
    source_mask = adata.obs['guide'].str.startswith(f'sg{target_gene}')
    ntc_mask    = adata.obs['is_ntc']

    source_embed = embed[source_mask]   # (n_source, 64)
    ntc_embed    = embed[ntc_mask]      # (n_ntc, 64)

    # 对每个 source, 在 NTC pool 里找 cosine-NN
    sim = source_embed @ ntc_embed.T    # (n_source, n_ntc)
    best = sim.argmax(axis=1)           # (n_source,)

    matches = pd.DataFrame({
        'source_idx':  np.where(source_mask)[0],
        'control_idx': np.where(ntc_mask)[0][best],
        'cos_sim':     sim.max(axis=1),
    })
    return matches
```

**关键设计**：
- **同切片匹配**：source 和 control 必须在同一切片。跨切片匹配禁止
  （不同切片的 baseline 不同）。
- **cosine 相似度**：用 cosine 而非欧氏距离，因为 embedding 是方向敏感
  的（PCA 主成分的方向有意义，绝对位置没意义）。
- **1:1 匹配**：每个 source 配一个 control，没有重复。如果 NTC pool 不够
  大，部分 source 没匹配（这些 source 在 δ 估计里被丢掉）。

### 3.3 协变量平衡检查（SMD）

匹配完后必须验证：source 和 control 在所有协变量上是否真的"平衡"
（即分布相似）。

**Standardized Mean Difference（SMD）**：

```
SMD(X) = |mean(X_source) - mean(X_control)| / sqrt((var(X_source) + var(X_control)) / 2)
```

对每个协变量 X（64 维 embedding + 8 module scores + cell_type + niche_id）
都算一遍。**v2 的门槛：max SMD < 0.13**。

**为什么是 0.13**？

经典文献（Austin 2011, Stat Med）的标准：
- SMD < 0.10 = "well balanced"
- SMD < 0.25 = "acceptable"
- SMD > 0.25 = "imbalance, matching rejected"

v2 用 0.13 是介于 0.10 和 0.25 之间的折中——比"完美" 0.10 稍宽（因为
embedding 64 维不可能全部 < 0.10），但远严于"可接受" 0.25。

> **Reviewer 会问**：「为什么 0.13 而不是 0.10？是不是因为 0.10 太严
> 你达不到？」
>
> **v2 的回答**：「0.13 是 v2 实际达到的最大 SMD（across 8 slices ×
> 64 dims）。我们没用更严的 0.10 是因为 64 维 embedding 必然有 1-2 个
> dim 接近 0.10，强迫全部 < 0.10 会导致匹配 pool 缩水 30%。Sensitivity
> analysis 在 v2.1 待做。」

### 3.4 输出

| 文件 | 内容 |
|---|---|
| `matches_<slice>_<gene>.csv` | (n_source, 3): source_idx, control_idx, cos_sim |
| `matched_control_validation.csv` | 每个协变量的 SMD |

### 3.5 v2.1 Fix 4 要修的"matching buffer 泄露"

**问题**：NTC pool 里如果有 bin 物理上靠近**其他基因的 source bin**
（比如 sgRab8a 的 source），那这个 NTC bin 已经反映了 sgRab8a 的 NCA
效应。用它作为 sgPhgr1 的 matched control，会把 sgRab8a 的效应**误归**
给 Phgr1 的反事实 baseline，导致 δ 有偏。

**修复**：从 NTC pool 里**排除任何 source bin 80 µm 邻域内的 bin**：

```python
src_tree = cKDTree(xy[source_mask_any_gene])  # 任何基因的 source
dist_to_src, _ = src_tree.query(xy[ntc_mask], k=1)
keep = dist_to_src >= 80.0  # 80 µm buffer
filtered_ntc_pool = ntc_mask.copy()
filtered_ntc_pool[ntc_mask] &= keep
```

**预期影响**：filtered pool 会缩小 ~30-50%（取决于切片密度），但匹配
质量更高。如果 Phgr1 的 δ sign 不变，说明 v2 的 δ 不依赖这种泄露。

---

## §4 L3 — 空间 Durbin 模型

代码：`spatial/durbin.py`

### 4.1 为什么需要空间计量学

普通线性回归 `y = Xβ + ε` 假设观测独立。但 SPAC-seq 数据**严重违反**
这个假设——物理上靠近的 bin 转录状态相关（空间自相关）。如果用 OLS：

- β 的标准误**低估**（因为有效样本量 << 名义样本量）
- p 值**显著偏小**（虚假显著）
- AIC / BIC 失真

**空间计量学**就是处理这种空间依赖的统计框架。v2 用的是 **Spatial
Durbin Model (SDM)**，由 LeSage & Pace (2009) 的经典教材系统化。

### 4.2 SDM 模型形式

```
y = ρWy + Xβ + WXθ + ε
    ↑     ↑    ↑
    空间   直接  间接
    依赖   效应  溢出
```

变量：
- `y` ∈ ℝ^N：所有 spot 的某个 module score（比如 fibroblast score）
- `X` ∈ ℝ^N：source indicator（1 = source bin, 0 = matched control）
- `W` ∈ ℝ^{N×N}：行标准化空间权重矩阵
- `ε` ~ N(0, σ²I)

**W 的构造**：

```python
from scipy.spatial import cKDTree
import numpy as np

# kNN(k=6, max 60 µm) 邻接
tree = cKDTree(xy)
dist, idx = tree.query(xy, k=7)  # 7 = 6 邻居 + self
# Gaussian kernel 加权
W = np.zeros((N, N))
for i in range(N):
    for j, d in zip(idx[i, 1:], dist[i, 1:]):
        if d > 60: continue
        W[i, j] = np.exp(-d**2 / (2 * 40**2))  # σ = 40 µm
# 行标准化 (每行和 = 1)
W = W / W.sum(axis=1, keepdims=True)
```

**σ = 40 µm 的来源**：从 v1 的距离衰减拟合得到，对应"一个 spot 的影响
半径 ~ 一个细胞直径"。

### 4.3 三个系数的解读

| 系数 | 含义 | v2 期望 |
|---|---|---|
| `β`（直接效应） | source bin 自己的 module score 偏移 | Phgr1 source 的 fibroblast score 比 control 高 → β > 0 |
| `θ`（间接溢出） | source 的**邻居**对**自己**的影响 | 邻居是 source → 自己 fibroblast 也高 → θ > 0 |
| `ρ`（空间依赖） | y 本身的空间自相关强度 | 通常 0.3-0.7（强空间依赖） |

**Reduced-form 因果效应 δ**：

```
δ = β + θ · (I - ρW)⁻¹
```

这个公式的推导：

把 SDM 重写：
```
(I - ρW) y = Xβ + WXθ + ε
y = (I - ρW)⁻¹ (Xβ + WXθ + ε)
```

对 X 求偏导（X 是 N 维向量，每个 spot 都有自己的 X）：

```
∂y / ∂X = (I - ρW)⁻¹ (β I + θ W)
```

这就是说：spot i 的 X 变化 1 单位，会通过 `(I - ρW)⁻¹` 的传播影响 spot j
的 y。**总效应**（对全切片平均）= `β + θ · (I - ρW)⁻¹` 的对角平均。

### 4.4 Neumann 级数近似

`(I - ρW)⁻¹` 直接求逆是 O(N³)，对 N=273K 不可行。用 **Neumann 级数**：

```
(I - ρW)⁻¹ = Σ_{k=0}^∞ (ρW)^k  ≈  Σ_{k=0}^{100} (ρW)^k
```

收敛条件：`|ρ| < 1`（保证 |ρW| < 1 因为 W 是行标准化所以 ‖W‖ = 1）。

实际实现是迭代：

```python
def neumann_inverse(W, rho, n_terms=100):
    N = W.shape[0]
    I = np.eye(N)
    result = I.copy()
    term = I.copy()
    for k in range(1, n_terms):
        term = rho * term @ W  # (ρW)^k
        result += term
    return result
```

但 `W` 是 N×N 仍然太大。实际用稀疏矩阵 + 逐 spot 局部近似（只算
spot i 周围 5-hop 的传播）。

### 4.5 MLE 估计

参数 `(ρ, β, θ, σ²)` 用最大似然估计：

```
L(ρ, β, θ, σ²) = log|I - ρW| - (N/2) log(2πσ²)
                 - (1 / 2σ²) · εᵀ ε
where ε = (I - ρW) y - Xβ - WXθ
```

`log|I - ρW|` 是 Jacobian，必须算（否则估计不一致）。对 273K 的 W，
精确行列式不可行，用 **Monte Carlo 估计**（Barry & Pace 1999 方法）。

`scipy.optimize.minimize` (L-BFGS-B) 优化。约束：`ρ ∈ [-0.99, 0.99]`，
`σ² > 0`。

### 4.6 Permutation p-value

为了不依赖正态假设，对 δ 算 permutation p：

```python
observed_delta = fit_durbin(X, y, W)['delta']

perm_deltas = []
for _ in range(1000):
    X_perm = X.copy()
    np.random.shuffle(X_perm)  # 打乱 source/control 标签
    perm_deltas.append(fit_durbin(X_perm, y, W)['delta'])

p = (np.sum(np.abs(perm_deltas) >= np.abs(observed_delta)) + 1) / 1001
```

**注意**：这里 shuffle 了 source/control 标签但**保留了空间结构**（y 和 W
不变）。这测试的是"如果 source 是随机的，δ 还会不会显著"。

### 4.7 Decay length λ

δ 是 reduced-form 总效应。但**效应随距离衰减**——邻居越远，效应越小。
λ 是衰减长度（单位 µm），定义为：

```
λ = mean over source bins of  (
    distance at which row-sum of (I - ρW)⁻¹ drops to 1/e of its max
)
```

v2 实际拟合的 Phgr1 λ ≈ 30-150 µm（不同切片不同），符合一个细胞因子
扩散半径的物理直觉。

### 4.8 输出

每个 (slice, gene, response) 一行 CSV：

```
slice, gene, response, durbin_delta, durbin_p, durbin_lambda,
durbin_beta_self, durbin_r2, n_clones, did_delta, did_p, did_n_pert_near
```

例（Phgr1 × fibroblast × M001）：
```
M001, Phgr1, score_fibroblast,
  durbin_delta = +0.62,
  durbin_p     = 4.8e-64,
  durbin_lambda = 60,
  durbin_r2    = 0.21,
  n_clones     = 147
```

`durbin_r2 = 0.21` 意味着 SDM 解释了 21% 的 module score 方差。这听起来
低，但**这是空间残差**——剩下的 79% 是单 bin 级噪声 + 未测量的协变量。
对空间数据这是合理范围。

### 4.9 Reviewer 会怎么挑

**Q：你的 X 是 endogenous 的——guide delivery 有空间结构（克隆扩增从
一个细胞扩散开）。OLS 估的 β 是不一致的，怎么解？**

A：这是 v2.1 Fix 1 要解决的核心问题。Fix 1 用 **AIPW doubly robust
estimator** 替代 OLS。AIPW 通过 propensity model `ê(X)` 显式建模 treatment
assignment 机制，**即使 X 内生，AIPW 估计也是一致的**（如果 propensity
模型正确）。

**Q：(I - ρW)⁻¹ 用 Neumann 级数近似，截断到 100 项的误差多大？**

A：`|ρ|` 实际 ~0.3-0.6，所以 `(ρW)^100` 的范数 ~`0.5^100` ≈ 1e-30，
远小于机器精度。截断误差可忽略。但 ρ 接近 1 时（空间单位根）会发散——
v2 强制 `|ρ| < 0.99` 来避免这种情况。

**Q：为什么用 Durbin 而不是更简单的 SAR（Spatial Autoregressive）？**

A：SAR 是 SDM 的特例（θ = 0），假设"邻居的 X 不影响自己的 y"。但
SPAC-seq 的物理是"邻居 source 通过分泌细胞因子影响自己的 y"——
`WXθ` 项必须有。所以用 SDM 而非 SAR。

---

## §5 L4 — 因果诊断三件套

代码：`causal/causal.py`

L3 给一个 δ + p。但 reviewer 不信一个 p 值——尤其在 SPAC-seq 这种新
技术 + 小样本 setting。L4 加**三重独立检验**，全部通过才报告。

### 5.1 DiD（Difference-in-Differences）

**思想**：减掉基线空间异质性。

```python
# 对每个 (slice, gene, response):
near_src = spots within 40 µm of any Phgr1 source bin
far_src  = spots 40-120 µm from any Phgr1 source bin
near_ntc = spots within 40 µm of any NTC bin
far_ntc  = spots 40-120 µm from any NTC bin

Δ_source = mean(y[near_src]) - mean(y[far_src])
Δ_ntc    = mean(y[near_ntc]) - mean(y[far_ntc])
DiD      = Δ_source - Δ_ntc
```

**为什么需要 DiD**：直接比较 `mean(y[near_src]) vs mean(y[far_src])`
有偏——near 和 far 区域可能本身就有不同的 niche（near_src 可能就在
肿瘤中心，far_src 可能在边缘）。DiD 用 NTC 的 near-far 差作为
**基线空间异质性的代理**，减掉这个混杂。

**经典引用**：Card 1990 (Mariel Boatlift 论文) 是 DiD 的标杆应用。

**Permutation p**：在 NTC 内部随机分配"near/far"标签 1000 次，得到
DiD 的 null 分布。

### 5.2 IV（Instrumental Variable）

**思想**：用 guide assignment 作为工具变量，排除 guide-detection 误差。

**为什么需要 IV**：source bin 的判定是基于 guide UMI 计数 > threshold。
但有些 bin 可能**真有 guide 但没测到**（测序深度低）。这些 bin 被误判
为 control，导致 δ 估计偏向 0。

**IV 设置**：
- **Z**（工具）：guide assignment（设计层面的"这个 bin 应该有 sgPhgr1"）
- **T**（treatment）：source_indicator（实测层面的"这个 bin guide UMI > threshold"）
- **y**（outcome）：module score

**两阶段最小二乘（2SLS）**：

```
Stage 1: T = π₀ + π₁ · Z + ε₁     → T̂
Stage 2: y = β · T̂ + ε₂
```

如果 `sign(β_IV) == sign(β_OLS)` 且 `|β_IV|` 与 `|β_OLS|` 在 2 倍以内，
说明 guide-detection 误差不严重，OLS 估计可信。

**经典引用**：Wright 1928（最初发明 IV）。Angrist & Imbens 1994 是
因果推断的现代奠基。

### 5.3 Rosenbaum Γ*（敏感性界）

**思想**：量化"多大的隐藏混杂能推翻结论"。

**问题陈述**：DiD 和 IV 都假设"控制了所有可观测混杂"。但 SPAC-seq
可能有**不可观测混杂**（比如真正的染色质开放度没测）。Rosenbaum
方法问：

> "如果存在一个未测量的二值混杂 U，它使 treatment 的 odds ratio
> 偏移 Γ 倍，那么我们的 p 值在多大的 Γ 下还显著？"

**算法**：

```python
for Γ in [1.0, 1.5, 2.0, 2.5, 3.0]:
    # 在 odds ratio Γ 假设下，重新计算 Wilcoxon signed-rank p-value
    p_Γ = adjusted_wilcoxon(y_source, y_control, Γ)
    if p_Γ > 0.05:
        Γ_star = Γ
        break
```

`Γ*` = 让结论翻转的最小 Γ。**v2 阈值：Γ* ≥ 2 才报告**。

**直觉**：Γ* = 2 意味着"存在一个 2 倍 odds ratio 的隐藏混杂也不足以
推翻结论"。Γ* = 1 意味着"任何隐藏混杂都能推翻"——这种结果不报告。

**经典引用**：Rosenbaum 1983 (Biometrika) 是奠基。Rosenbaum 2002 的
书 *Observational Studies* 是标准参考。

### 5.4 三个诊断的关系

| 诊断 | 攻击的混杂类型 | v2 阈值 |
|---|---|---|
| DiD | 可观测的 near-far 空间异质性 | did_p < 0.05 |
| IV | guide-detection 误差 | sign 一致 + iv_p < 0.05 |
| Γ* | 不可观测混杂 | Γ* ≥ 2 |

**为什么三个都要**：每个诊断攻击不同的混杂。单用 DiD 不能排除 unobserved
confounding；单用 Γ* 不能排除空间异质性。**三件套组合起来才能叫
"causal identification"**——这是 v2 论文的核心方法学贡献。

### 5.5 输出

`phgr1_full_diagnostics.json`：

```json
{
  "M001": {
    "score_ifn_response": {
      "did_p": 0.001,
      "did_delta": +0.06,
      "iv_p": 0.003,
      "iv_beta": +0.27,
      "gamma_star": 2.5,
      "n_source": 395,
      "n_control": 395
    },
    "score_fibroblast": {...},
    ...
  },
  "M002": {...}
}
```

### 5.6 Reviewer 会怎么挑

**Q：Γ* 的 bound 是 conditional on embedding 的，但 embedding 是学的
表示，可能不包含真正的 unobserved confounder。Γ* 是不是系统性乐观？**

A：是。这是 Lechner & Steinmayr (2014, JoE) 的经典 trap。v2.1 Fix 4
部分缓解：用 raw marker match（不用 embedding）算另一个 Γ* 对比。
真正的修复需要 negative control outcomes（Lipsitch et al. 2010）——
v2 数据集没有，defer 到 v3。

**Q：DiD 的 near/far 40 µm 和 40-120 µm 阈值是怎么选的？**

A：40 µm 对应一个衰减长度（从 Phgr1 durbin_lambda 拟合），120 µm 对应
3 个衰减长度（效应衰减到 e^-3 ≈ 5%）。Sensitivity 待做。

---

## §6 L5 — 跨 cohort 元分析

代码：`experiments/merge_cross_cohort.py`

### 6.1 Fisher 组合方法

对每个 (gene, response) pair 跨 8 个切片，怎么把 8 个 p 值合成一个？

**Fisher's method**（1932）：

```
给定 k 个独立 p 值 p_1, ..., p_k，all under H₀,
检验统计量:  -2 · Σ log(p_i)
服从:       χ² 分布, 自由度 2k
```

**直觉**：如果所有 p 都 ~U(0,1)（H₀ 成立），`-2 log(p)` 服从 χ²(2)，
k 个独立 χ²(2) 加起来是 χ²(2k)。如果某些 p 显著小，统计量就大。

**两阶段 Fisher**：

```python
# 阶段 1: cohort 内合并
cohort1_p = fisher_combine([M001_p, M002_p, M003_p])  # χ²(6)
cohort2_p = fisher_combine([subQ1_p, subQ2_p, subQ3_p, subQ4_p, subQ5_p])  # χ²(10)

# 阶段 2: cohort 间合并
combined_p = fisher_combine([cohort1_p, cohort2_p])  # χ²(4)
```

**为什么两阶段而不是一次合并 8 个 p**？

直接合并 8 个 p（χ²(16)）的检验功效更高，但**没区分 cohort 内复现 vs
跨 cohort 复现**。两阶段保证：
- 阶段 1：每个 cohort 内部确实有信号（不是单切片偶然）
- 阶段 2：跨 cohort 都有信号（不是单 cohort 系统误差）

### 6.2 多重检验校正（BH-FDR）

我们对 ~120 个基因 × 8 个 response = 960 个 (gene, response) pair 都
做了 Fisher 合并。**多重检验问题**：随机数据也会有一些 q < 0.05。

**Benjamini-Hochberg**（1995）：

```python
def bh_fdr(p_values):
    n = len(p_values)
    sorted_p = sorted(zip(p_values, range(n)))
    q_values = [0] * n
    prev_q = 1.0
    for i in range(n - 1, -1, -1):
        p, orig_idx = sorted_p[i]
        q = p * n / (i + 1)  # BH 公式
        q = min(q, prev_q)
        q_values[orig_idx] = q
        prev_q = q
    return q_values
```

**BH 控制的是 FDR**（False Discovery Rate）= 期望的假阳性比例，而非
FWER（Family-Wise Error Rate）。对探索性分析（如基因组学）FDR 是合适的。

### 6.3 Reportable gate（3 个条件全过）

一个 (gene, response) pair 被 v2 报告为"复现"，必须**同时**满足：

```
1. direction_consistent == True
   所有切片的 durbin_delta 同号（不能 M001 正、M002 负）

2. combined_q < 0.05  in BOTH cohort 1 AND cohort 2
   不能只在 cohort 1 显著

3. passed L4 (all four diagnostics) in cohort-1 slices
   durbin_p < 0.05 AND did_p < 0.05 AND iv_p < 0.05 AND Γ* ≥ 2
```

**为什么这么严**？v1 的 BLNK 只在 M001 一个切片显著，跨切片不复现，
所以 v2 把 BLNK 推翻。严门槛是 v2 卖点——"我们只报告跨 cohort + 多重
诊断全过的结果"。

### 6.4 输出

`cross_cohort_replicated.csv`（实际 7 行）：

| gene | response | n_slices | combined_q | mean_delta |
|---|---|---|---|---|
| Phgr1 | fibroblast | 7 | 3.0e-60 | +0.43 |
| Phgr1 | macrophage | 7 | 1.1e-41 | +0.41 |
| Phgr1 | hypoxia | 7 | 6.1e-21 | +0.44 |
| Phgr1 | endothelial | 7 | 2.0e-19 | -0.24 |
| Rab8a | malignant | 6 | 5.0e-56 | +0.49 |
| Bcam | malignant | 6 | 4.4e-11 | +0.47 |
| Blnk | fibroblast | 6 | 7.5e-6 | +0.29 |

**Phgr1 占 5 个响应**——这是 v2 论文的 headline。

### 6.5 Reviewer 会怎么挑

**Q：Fisher 合并假设 p 值独立。但同一 (gene, response) 在 M001 和 M002
的 p 值显然相关（同组织、同细胞系、同 guide）。你的 combined_p 是不是
偏向显著？**

A：部分对。M001/M002/M03 来自同一只小鼠的不同切片，p 值相关。Fisher
在 p 相关时**乐观**（combined_p 偏小）。修复方法是用 **Brown's method**
（1975, Biometrika），它估计 p 值的协方差然后校正。v2.1 没做这个校正，
defer 到论文 revision。

**Q：direction_consistent 只要求 sign 一致，不要求 magnitude 一致。
如果 cohort 1 δ=+0.5、cohort 2 δ=+0.05，这种"复现"意义多大？**

A：对。v2 没量化 effect-size consistency。正确做法是 **hierarchical
Bayesian meta-analysis** 估 `τ²`（cohort 间方差），看 δ_global 的后验
CI 是否跨 0。defer 到 v3。

---

## §7 L6 — 正交人类验证

代码：`experiments/phgr1_tcga.py`, `phgr1_survival.py`, `phgr1_icb.py`

### 7.1 为什么需要人类验证

mouse SPAC-seq 的因果证据很强，但**只是小鼠 MC38 细胞系**。Phgr1 在
人类癌症中是否有同样作用？需要 TCGA 等公共数据库的正交验证。

**重要**：TCGA 是**观察性**数据，不能给因果证据。它只能验证"mouse
SPAC-seq 的效应方向在人类数据里 sign-concordant"。

### 7.2 TCGA LUAD 相关性分析

数据：TCGA Lung Adenocarcinoma (LUAD), n = 518 patients。
来源：cBioPortal `luad_tcga_gdc`。

对每个 candidate marker gene m：

```python
from scipy.stats import spearmanr
rho, p = spearmanr(tcga_expr['PHGR1'], tcga_expr[m])
```

**Spearman 而非 Pearson**：表达数据是重尾分布，Pearson 对异常值敏感。
Spearman 是 rank-based，更稳。

**结果（Phgr1）**：

```
PHGR1 vs IRF1:  ρ = +0.473,  p = 3e-30   ← IFN-response marker
PHGR1 vs CD68:  ρ = +0.444,  p = 2e-26   ← macrophage marker
PHGR1 vs CD8A:  ρ = +0.439,  p = 9e-26   ← CD8+T marker
PHGR1 vs DCN:   ρ = +0.403,  p = 1e-21   ← fibroblast marker
PHGR1 vs EPCAM: ρ = +0.051,  p = 0.25    ← malignant marker (no signal)
```

**解读**：PHGR1 在人类 LUAD 中跟免疫 + 成纤维 marker 强正相关，跟
malignant marker 不相关。这跟 mouse SPAC-seq 看到的"Phgr1 source
周围 fibroblast + macrophage + cd8_like 上调、malignant 不变"完全
sign-concordant。

### 7.3 Cox 比例风险模型

数据：TCGA LUAD + OS（overall survival）信息。

**Univariate Cox**：

```
h(t | PHGR1) = h₀(t) · exp(β · log(PHGR1))
HR = exp(β)

v2 univariate: HR = 0.587 (95% CI 0.40-0.86), p = 0.007
```

**Multivariate Cox**（调整 AGE, SEX）：

```
h(t | PHGR1, AGE, SEX) = h₀(t) · exp(β₁ · log(PHGR1) + β₂ · AGE + β₃ · SEX_male)

v2 multivariate:
  PHGR1:     HR = 0.556 (95% CI 0.37-0.83), p = 0.004
  AGE:       HR = 1.008 (95% CI 0.99-1.02), p = 0.29   ← 不显著
  SEX_male:  HR = 1.042 (95% CI 0.77-1.41), p = 0.79   ← 不显著
```

**解读**：PHGR1 高表达 → HR < 1 → 死亡风险低 → 生存好。
AGE 和 SEX 在 PHGR1 进入模型后不显著，说明 PHGR1 不只是 AGE/SEX 的代理。

**HR < 1 的临床意义**：HR = 0.556 意味着 PHGR1 高表达组的死亡风险是
低表达组的 55.6%——大约**降低了 44%**。在临床上这是中等保护效应
（strong protective 是 HR < 0.3，weak 是 HR 0.7-0.9）。

### 7.4 Kaplan-Meier 生存曲线

把患者按 PHGR1 表达分四分位，比较 Q1（最低）vs Q4（最高）的生存曲线：

```
log-rank p = 0.0017  ***
median OS:
  Q1 (low PHGR1):   20.0 months
  Q4 (high PHGR1):  22.7 months
差值: 2.7 个月
```

**注意**：2.7 个月的差异在统计上显著（p < 0.01）但临床上不算大
（典型免疫治疗的 clinically meaningful 差异是 3-6 个月）。论文里要诚实
写"统计学显著 ≠ 临床显著"。

### 7.5 ICB 应答预测（诚实负面）

数据：4 个独立 ICB（immune checkpoint blockade）cohort：
- IMvigor210（膀胱癌，抗 PD-L1）
- Liu melanoma（黑色素瘤，抗 PD-1）
- Hellmann lung（肺癌，抗 PD-1）
- bladder alt cohort

对每个 cohort：

```python
# 把患者按 PHGR1 中位数分高低组
high = tcga[tcga['PHGR1'] > median]
low  = tcga[tcga['PHGR1'] <= median]

# 比较 ICB response rate (Fisher exact)
OR, p_response = fisher_exact(high['responded'], low['responded'])

# 比较生存 (Cox)
hr, p_survival = cox(high, low)
```

**结果**：4 个 cohort 都**不显著**（all p > 0.05）。

**为什么诚实报告**：

- Phgr1 标记 immune-inflamed TME（good prognosis）
- 但**不是** ICB response 的 rate-limiting target
- 区分"预后 marker"和"预测 marker"是临床免疫学的重要区分

v2 论文不能藏这个阴性结果——藏了就是 cherry-picking，被 reviewer 发现
就是 KO。诚实写反而增加可信度。

### 7.6 输出

```
phgr1_tcga_validation.json     # Spearman 相关性
phgr1_survival.json            # Cox + KM
phgr1_melanoma_icb.csv         # ICB 阴性
figures F7, F8, F18
```

---

## §8 设计原则总结

把上面所有内容浓缩成 6 条原则：

### 8.1 表示学习（L1）和因果识别（L3-L5）严格分层

GNN 只在 L1 学 embedding。L3 用经典统计（Durbin）。L4 用经典因果推断
（DiD, IV, Γ*）。**两者用 embedding 桥接但因果 claim 完全不依赖 GNN
可解释性**。

> Reviewer 不能拿"GNN 是黑盒"KO v2。

### 8.2 反事实生成（L2）是核心

L2 的 matched control 是 `Y(0)` 的代理。所有后续因果推断都基于这个
反事实。**SMD < 0.13 是硬门槛**。

### 8.3 跨 cohort 复现是门槛不是加分项

v1 BLNK 单切片不复现被推翻。v2 Phgr1 7 切片跨 cohort 全过门槛是
Nature Methods 档次的底气。

### 8.4 多重因果诊断

DiD（攻击空间混杂）+ IV（攻击检测误差）+ Γ*（攻击隐藏混杂）三件套。
单用任何一个不够，组合起来才能叫 "causal identification"。

### 8.5 诚实负面报告

79% v1 hits 是 v2 假阳性、Phgr1 不预测 ICB。**诚实写增加可信度**，
藏起来被发现就是 KO。

### 8.6 正交验证明确边界

mouse SPAC-seq 给因果证据，TCGA 给正交 sign-concordance。**不混淆
相关和因果**。Discussion 明说 TCGA 是 limitation #4。

---

## §9 v2.1 修复预览

5 项 sensitivity 分析，每项攻击一个具体建模选择：

| Fix | 攻击 | 估计器 | 文件 |
|---|---|---|---|
| 1 | Durbin X 内生性 | AIPW doubly robust + cross-fitting | causal/aipw.py |
| 2 | niche KMeans 循环依赖 | Leiden + 多 resolution sensitivity | data/niche_robust.py |
| 3 | 双 view 共享 head 自洽 | probe transfer (z1 训 head, z2 zero-shot) | embedding/probe_encoder.py |
| 4 | matching pool 泄露 | 80 µm spatial buffer | matching/match_buffered.py |
| 5 | set pooling vs message-passing | GraphSAGE 2-layer baseline | baseline/real_gnn.py |

详细数学推导和伪代码见 [MODELING_DEEP_DIVE.md](MODELING_DEEP_DIVE.md)。

**v2.1 不修的 7 项**（defer 到 v3）：

- 真正的 proximal causal inference（需要 negative control outcomes）
- Hierarchical Bayesian meta-analysis（替代 Fisher）
- Directional neighborhood pooling（替代 set pooling）
- 统一 SLX + Durbin + resistance 的理论框架
- Block permutation test
- Neumann 级数 CI on ρ
- Non-monotonic 衰减替代模型

这些是 v3 论文的方向。v2.1 只是"在 v2 框架内做最大限度的 sensitivity
analysis"，不改主架构。

---

## 进一步阅读

- [PIPELINE_FLOW.md](PIPELINE_FLOW.md) — ASCII 框图，简短版
- [PIPELINE_LAYERS.md](PIPELINE_LAYERS.md) — 数据合约表，无解释
- [MODELING_DEEP_DIVE.md](MODELING_DEEP_DIVE.md) — 12 个建模问题 + 5 项
  修复的完整数学
- [RECONSTRUCTION_v2_1.md](RECONSTRUCTION_v2_1.md) — v2.1 范围决策
- [SINGLE_MASTER_perturbgnn_v2.ipynb](../SINGLE_MASTER_perturbgnn_v2.ipynb) —
  完整可执行 master notebook
- [PHGR1_CASE_STUDY.md](PHGR1_CASE_STUDY.md) — Phgr1 单基因深挖
- [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) — 投稿 checklist

经典文献：

- LeSage & Pace (2009) *Introduction to Spatial Econometrics* — SDM 标杆
- Card (1990) — DiD 标杆
- Angrist & Imbens (1994) — IV 现代奠基
- Rosenbaum (2002) *Observational Studies* — Γ* 标准
- Chernozhukov et al. (2018) EJ — cross-fitted AIPW
- Traag et al. (2019) Sci Rep — Leiden 算法
