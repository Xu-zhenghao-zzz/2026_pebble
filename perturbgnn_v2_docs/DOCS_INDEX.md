# Documentation Index — perturbgnn_v2_docs/

20 documents, 6350 lines total. This index organises them by purpose
so you can find the right document quickly.

---

## Quick navigation by question

| If you want to know... | Read this |
|---|---|
| What does the v2 pipeline do end-to-end? | `PIPELINE_FLOW.md` (ASCII) or `PIPELINE_FLOW_zh.md` (中文教材级) |
| What are the precise data contracts per layer? | `PIPELINE_LAYERS.md` |
| Why was each modelling choice made? | `MODELING_DEEP_DIVE.md` |
| What did v2.1 fix, and what was deferred? | `RECONSTRUCTION_v2_1.md` |
| Why didn't EGAT work? | `GNN_FAILURE_REPORT.md` (canonical) |
| What secondary findings are hidden in the data? | `SECONDARY_FINDINGS.md` |
| What is the Phgr1 case study? | `PHGR1_CASE_STUDY.md` |
| What is the cross-cohort replication evidence? | `CROSS_COHORT_REPLICATION.md` |
| What are the limitations? | `COHORT3_LIMITATIONS.md`, `FIX4_BUFFER_INFEASIBLE.md` |
| What wet-lab work is queued? | `BIOLOGY_BACKLOG.md` |
| How do I submit the paper? | `SUBMISSION_CHECKLIST.md` |
| What is the current paper draft? | `PAPER_DRAFT_FINAL.md` |

---

## Document inventory

### Pipeline understanding (4 docs)

| Document | Lines | Purpose |
|---|---|---|
| `PIPELINE_FLOW.md` | 351 | ASCII input-to-output flow, English, quick reference |
| `PIPELINE_FLOW_zh.md` | 1223 | Chinese textbook-level walkthrough, with full maths derivation + reviewer Q&A per layer |
| `PIPELINE_LAYERS.md` | 417 | Precise data contracts (file names, tensor shapes, schema) per layer |
| `DESIGN.md` | — | Original 5-layer design doc (historical, less polished than above) |

### v2.1 sensitivity analyses (5 docs)

| Document | Lines | Purpose |
|---|---|---|
| `RECONSTRUCTION_v2_1.md` | 250 | Scope decision: what v2.1 fixes vs defers to v3 |
| `MODELING_DEEP_DIVE.md` | 707 | 12 modelling problems + 5 fix specifications with maths |
| `GNN_SELECTION.md` | 342 | EGAT vs other GNN types, technical trade-off analysis |
| `GNN_FAILURE_REPORT.md` | 616 | **Canonical** EGAT failure report (4 variants + theory) |
| `EGAT_NEGATIVE_RESULT.md` | 277 | Earlier EGAT failure report (v1-v3 only, superseded by above) |
| `FIX4_BUFFER_INFEASIBLE.md` | 132 | Why 80µm matching buffer doesn't work on SPAC-seq |

### Biological case studies (3 docs)

| Document | Lines | Purpose |
|---|---|---|
| `PHGR1_CASE_STUDY.md` | 29 | Phgr1 single-gene deep dive (very short, placeholder) |
| `CCN1_CASE_STUDY.md` | — | Ccn1 (v1's headline) comparison with v2 |
| `CROSS_COHORT_REPLICATION.md` | — | Detailed cross-cohort evidence |

### Paper writing (4 docs)

| Document | Lines | Purpose |
|---|---|---|
| `PAPER_DRAFT_FINAL.md` | 233 | Current paper draft (to be updated with v2.1 sensitivity) |
| `PAPER_DRAFT.md` | 215 | Earlier draft (superseded) |
| `PAPER_OUTLINE.md` | 263 | Outline / structure of the paper |
| `SUBMISSION_CHECKLIST.md` | 103 | Cover letter, reviewer Q&A, venue decision |

### Hypotheses and findings (3 docs)

| Document | Lines | Purpose |
|---|---|---|
| `SECONDARY_FINDINGS.md` | 390 | 6 exploratory findings beyond the Phgr1 headline |
| `BIOLOGY_BACKLOG.md` | — | Wet-lab work for collaborators (JAK inhibitor rescue etc.) |
| `COHORT3_LIMITATIONS.md` | — | Why cohort 3 cannot directly replicate Phgr1 |

---

## Reading order suggestions

### For a new reader (1 hour)

1. `PIPELINE_FLOW.md` (15 min) — get the big picture
2. `PHGR1_CASE_STUDY.md` + `CROSS_COHORT_REPLICATION.md` (10 min) — the headline
3. `SECONDARY_FINDINGS.md` (30 min) — what's hidden
4. `SUBMISSION_CHECKLIST.md` (5 min) — venue strategy

### For a reviewer defending the paper (2 hours)

1. `PIPELINE_LAYERS.md` (30 min) — exact data contracts
2. `MODELING_DEEP_DIVE.md` (1 hr) — 12 known modelling problems + fixes
3. `GNN_FAILURE_REPORT.md` (30 min) — why set-pooling not real GNN
4. `SECONDARY_FINDINGS.md` (15 min) — pre-empted biology

### For a v3 follow-up author (3 hours)

1. `MODELING_DEEP_DIVE.md` Part IV — what v2.1 does NOT fix
2. `GNN_FAILURE_REPORT.md` §8 — what would need to change for message-passing
3. `BIOLOGY_BACKLOG.md` — wet-lab rescue priorities
4. `SECONDARY_FINDINGS.md` N.6.4 + N.6.5 — Bcam paradox + Rab8a under-explored

---

## Document health

| Status | Documents |
|---|---|
| ✅ Authoritative (current) | `PIPELINE_FLOW.md`, `PIPELINE_FLOW_zh.md`, `PIPELINE_LAYERS.md`, `MODELING_DEEP_DIVE.md`, `RECONSTRUCTION_v2_1.md`, `GNN_SELECTION.md`, `GNN_FAILURE_REPORT.md`, `FIX4_BUFFER_INFEASIBLE.md`, `SECONDARY_FINDINGS.md`, `SUBMISSION_CHECKLIST.md`, `PAPER_DRAFT_FINAL.md` |
| ⚠️ Superseded but kept for history | `EGAT_NEGATIVE_RESULT.md` (superseded by `GNN_FAILURE_REPORT.md`), `PAPER_DRAFT.md` (superseded by `PAPER_DRAFT_FINAL.md`), `DESIGN.md` (superseded by `PIPELINE_LAYERS.md`) |
| 📝 Stub / placeholder | `PHGR1_CASE_STUDY.md` (29 lines), `BIOLOGY_BACKLOG.md`, `COHORT3_LIMITATIONS.md`, `CCN1_CASE_STUDY.md`, `CROSS_COHORT_REPLICATION.md`, `PAPER_OUTLINE.md` |

Stubs should be expanded before paper submission (1-2 days of writing).

---

## Stats

- Total documents: 20
- Total lines: 6 350
- Largest: `PIPELINE_FLOW_zh.md` (1 223 lines, Chinese textbook)
- Smallest (non-stub): `SUBMISSION_CHECKLIST.md` (103 lines)
