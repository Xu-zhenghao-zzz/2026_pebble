# Biology Backlog — computer-implementation-orthogonal work

This document captures tasks that **cannot be done by Claude/computer** and
require wet-lab experiments, literature deep-dives by domain experts, or
collaborator input.

---

## 1. Phgr1 mechanism review

**v2 finding**: Phgr1 KO in MC38 cells drives a coordinated multi-response
NCA effect (5 responses replicated across cohort 1 + cohort 2):
- IFN-response ↑ (δ=+0.73, q=6e-66)
- Hypoxia ↑ (δ=+0.53, q=4e-28)
- Macrophage ↑ (δ=+0.42, q=1e-54)
- Fibroblast ↑ (δ=+0.55, q=1e-71)
- Endothelial ↓ (δ=-0.21, q=2e-19)

### Known Phgr1 biology
- **Oltedal et al. 2018** (PMC5809026): PHGR1 is a NSCLC lymph-node
  metastasis marker, glycine metabolism.
- **No prior NCA / SPAC-seq work on Phgr1** — v2 is first.

### Open questions
1. Is Phgr1's effect on IFN mediated by cell-autonomous IFN signaling
   upregulation, or via paracrine macrophage recruitment?
2. Why does Phgr1 KO reduce endothelial module score?
3. Does Phgr1's glycine metabolism connect to hypoxia (glycolytic switch)?

**Action**: 1-2 day lit review, 500-word Discussion paragraph.

---

## 2. Phgr1 KO rescue experiment design

**Goal**: prove Phgr1 is causal.

### Design
- MC38 + sgRNA-resistant Phgr1 overexpression
- Control: empty vector + sgPhgr1
- Readout: SPAC-seq on rescued vs KO-only

### Predictions (from v2)
If causal:
- Rescue abolishes the IFN/macrophage/fibroblast NCA effect
- Durbin δ should drop from ~+0.7 to ~0
- TCGA-style Cox HR shifts from 0.556 toward 1.0

If not causal (off-target):
- Rescue does NOT change the NCA effect
- v2 finding downgraded to "sgPhgr1-associated"

### JAK-inhibitor co-treatment (mediation test)
- Phgr1 KO + JAK inhibitor → if NCA effect disappears, IFN-signaling
  is the mediator.
- Most informative single experiment for mechanism.

**Action**: 2-page protocol, ~6 weeks total (cell work + SPAC-seq).

---

## 3. Bcam biology

**v2 finding**: Bcam→malignant δ=+0.50 (q=1e-11) across 5 slices, but
TCGA Cox HR=1.32 (worse prognosis, opposite of Phgr1's HR=0.556).

### Open questions
1. Why does Bcam correlate with stromal markers (DCN, COL1A1) in TCGA but
   worsen survival? Marker of therapy-resistant CAF state?
2. Bcam = Basal Cell Adhesion Molecule — KO effect on adhesion/migration?

**Action**: focused mini-review, 200 words.

---

## 4. Cross-species single-cell validation

Search GEO for "MC38 Phgr1" / "Phgr1 knockout scRNA-seq".
- Phgr1 KO malignant cells should have higher IFN-response module score
- Neighboring wild-type cells should show partial upregulation (NCA)

---

## 5. Collaborator outreach

- **Oltedal lab** (PHGR1 discoverers): validate cohort 2/TCGA concordance
- **SPAC-seq authors** (Zhang et al. Cell 2026): share cohort 3 CellType JSON
- **MC38 KO expertise**: labs with Phgr1 KO mouse models

---

## Summary

| Task | Time | Owner |
|---|---|---|
| Phgr1 lit review | 1-2 days | Biologist/PI |
| KO rescue protocol | 1 week | Wet-lab |
| KO rescue execution | 4-8 weeks | Wet-lab |
| Bcam mini-review | 1 day | Biologist |
| scRNA-seq search | 1 day | Bioinformatician |
| Collaborator outreach | ongoing | PI |

**Computer work that complements this**:
- Power calibration (F19 synthetic benchmark)
- TCGA pan-cancer Phgr1-CD8A correlation (~1 hour)
- IMvigor210 + melanoma ICB prediction (done, Phgr1 NOT predictive)
