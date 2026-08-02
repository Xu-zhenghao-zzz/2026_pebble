# Phgr1 Case Study — v2 Full Causal Diagnostics

**Target gene**: Phgr1 (Proline/Histidine/Glycine-Rich 1)
**Slices**: M001 (395 sources) + M002 (640 sources)
**Date**: 2026-08-01

## Summary table

| Slice | Response | Durbin δ | Durbin p | DiD Δ | DiD p | IV/OLS | Γ* |
|---|---|---|---|---|---|---|---|
| M001 | ifn_response | +0.266*** | 6.39e-08 | +0.059** | 0.00119 | 0.89 | >3.0 |
| M001 | fibroblast | +0.619*** | 4.77e-64 | -0.069*** | 0.00053 | -9.26 | 1.0 |
| M001 | hypoxia | +1.341*** | 2.19e-19 | +0.049 | 0.102 | -1.54 | >3.0 |
| M001 | macrophage | +0.346*** | 9e-11 | -0.031 | 0.108 | 0.93 | >3.0 |
| M001 | endothelial | -0.099** | 0.00294 | +0.037** | 0.00816 | 1.12 | 1.0 |
| M002 | ifn_response | +1.844*** | 1.01e-116 | +0.377*** | 0 | 2.39 | >3.0 |
| M002 | fibroblast | +0.091*** | 0.000457 | +0.180*** | 0 | 0.57 | >3.0 |
| M002 | hypoxia | +0.174*** | 7.79e-06 | -0.010 | 0.575 | -19.46 | 1.0 |
| M002 | macrophage | +0.179*** | 8.12e-08 | +0.132*** | 2.12e-11 | 2.01 | >3.0 |
| M002 | endothelial | -0.299*** | 7.83e-20 | -0.124*** | 1.25e-11 | 1.55 | >3.0 |

## Direction consistency check (M001 vs M002)

| Response | M001 δ | M002 δ | Consistent? |
|---|---|---|---|
| ifn_response | +0.266 | +1.844 | ✅ |
| fibroblast | +0.619 | +0.091 | ✅ |
| hypoxia | +1.341 | +0.174 | ✅ |
| macrophage | +0.346 | +0.179 | ✅ |
| endothelial | -0.099 | -0.299 | ✅ |