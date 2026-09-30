# P5 CQL Generalization Evidence Summary

## Status

P5 primary consequence analysis and stored secondary analyses have been audited from the frozen analysis artifacts. No new RL training or consequence collection was performed during this evidence-package step.

## Experiment

- Task: `mujoco/hopper/medium-v0`
- Environment: Hopper-v5
- Algorithm: CQL
- Policy seeds: 0, 1, 2, 3, 4
- Decision states: 100 per policy seed
- Total frozen records: 14,000
- Primary horizon: H=10
- Primary condition: sigma > 0
- Nonzero sigma levels: 0.01, 0.025, 0.05, 0.10, 0.20, 0.30

## Primary question

Determine whether the action-disagreement to downstream-consequence relationship established for IQL is reproduced for CQL.

## Primary outcome

`C10 = absolute_consequence` at H=10.

## Primary predictor

`action_disagreement`.

## Independent unit

Policy seed is the independent replication unit. Individual states, sigma levels, and consequence records are not treated as independent policy replicates.

## Primary estimand

For each decision state, regress C10 on action disagreement across the six nonzero sigma levels and average the resulting state-level slopes within each policy seed.

## Primary seed-level results

| Seed | Records | States | Finite slopes | Mean state slope | State-slope SD | Positive fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 600 | 100 | 100 | 0.23064449 | 0.21378831 | 1.00 |
| 1 | 600 | 100 | 100 | 1.85321008 | 3.62980495 | 0.98 |
| 2 | 600 | 100 | 100 | 0.17355650 | 0.21162938 | 1.00 |
| 3 | 600 | 100 | 100 | 2.01691202 | 3.36773337 | 0.94 |
| 4 | 600 | 100 | 100 | 0.74664450 | 0.53813318 | 1.00 |

## Cross-seed primary inference

- Mean seed-level slope: `1.00419352`
- Sample SD: `0.88049636`
- 95% t-based CI: `[-0.08908711, 2.09747414]`
- Positive seed-level slopes: `5/5`
- Exact one-sided sign-flip p: `0.03125`
- Exact two-sided sign-flip p: `0.0625`
- Exact sign assignments: `32`

The five stored seed-level slopes are:

0.23064449, 1.85321008, 0.17355650, 2.01691202, 0.74664450

## Secondary dose-response analysis

The stored secondary analysis reports dose-response summaries over the six nonzero sigma levels for each policy seed. All five seeds have Spearman rho = 1.0 and Kendall tau approximately 1.0 for the dose-response between action disagreement and downstream consequence.

| Seed | Spearman rho | Kendall tau |
|---:|---:|---:|
| 0 | 1.000000 | 1.000000 |
| 1 | 1.000000 | 1.000000 |
| 2 | 1.000000 | 1.000000 |
| 3 | 1.000000 | 1.000000 |
| 4 | 1.000000 | 1.000000 |

## Secondary support-conditioned analyses

After averaging each state over the six nonzero sigma levels, the stored analysis reports descriptive regressions against support distance.

| Seed | Support→action slope | R² | Support→C10 slope | R² |
|---:|---:|---:|---:|---:|
| 0 | 0.00213998 | 0.055067 | -0.00096357 | 0.137734 |
| 1 | -0.01873882 | 0.169899 | -0.07929886 | 0.187589 |
| 2 | 0.03426421 | 0.234449 | 0.00515979 | 0.015715 |
| 3 | 0.03757456 | 0.126249 | 0.33316567 | 0.209829 |
| 4 | 0.00571443 | 0.008938 | -0.02013963 | 0.127256 |

These support-conditioned regressions are secondary descriptive analyses and are not used as the primary generalization test.

## Dose-response values

The stored per-seed dose means are reproduced from `P5_CQL_SECONDARY_ANALYSIS.json` for auditability.

| Seed | Sigma | Mean action disagreement | Mean absolute consequence |
|---:|---:|---:|---:|
| 0 | 0.010 | 0.00187408 | 0.00030594 |
| 0 | 0.025 | 0.00470402 | 0.00077748 |
| 0 | 0.050 | 0.00955193 | 0.00158383 |
| 0 | 0.100 | 0.01892235 | 0.00312081 |
| 0 | 0.200 | 0.03708633 | 0.00636705 |
| 0 | 0.300 | 0.05553260 | 0.01033961 |
| 1 | 0.010 | 0.00459690 | 0.00420646 |
| 1 | 0.025 | 0.01145260 | 0.02044233 |
| 1 | 0.050 | 0.02296964 | 0.04744205 |
| 1 | 0.100 | 0.04675154 | 0.09561440 |
| 1 | 0.200 | 0.10300555 | 0.20067229 |
| 1 | 0.300 | 0.16975185 | 0.32256055 |
| 2 | 0.010 | 0.01149947 | 0.00140308 |
| 2 | 0.025 | 0.02852561 | 0.00348083 |
| 2 | 0.050 | 0.05598417 | 0.00704281 |
| 2 | 0.100 | 0.10602660 | 0.01407698 |
| 2 | 0.200 | 0.18632106 | 0.04200678 |
| 2 | 0.300 | 0.24827775 | 0.06086415 |
| 3 | 0.010 | 0.01046933 | 0.03734402 |
| 3 | 0.025 | 0.02582369 | 0.05943172 |
| 3 | 0.050 | 0.05157233 | 0.18704043 |
| 3 | 0.100 | 0.09939986 | 0.31935532 |
| 3 | 0.200 | 0.18361123 | 0.47422617 |
| 3 | 0.300 | 0.24644495 | 0.55649594 |
| 4 | 0.010 | 0.00388294 | 0.00240757 |
| 4 | 0.025 | 0.00977566 | 0.00599701 |
| 4 | 0.050 | 0.01961779 | 0.01190162 |
| 4 | 0.100 | 0.03893089 | 0.02432638 |
| 4 | 0.200 | 0.07407493 | 0.04713740 |
| 4 | 0.300 | 0.10829038 | 0.07781740 |

## Data integrity checks

- Primary seed rows: `5`
- State-slope rows: `500`
- Primary records per seed: `[600]`
- Decision states per seed: `[100]`
- Finite state slopes per seed: `[100]`
- Positive state-slope fractions: `1.00, 0.98, 1.00, 0.94, 1.00`

## Interpretation boundary

The primary result supports reproduction of the evaluated action-disagreement/downstream-consequence relationship for the CQL Hopper setting.

It does not establish causality, universal OOD detection, or generalization to all offline-RL algorithms, datasets, environments, or CQL configurations.

CQL and IQL policy seeds are not pooled for primary inferential analysis.

The 95% t-based confidence interval and exact sign-flip p-values are reported as distinct inferential procedures.

## Frozen artifacts

- `results/analysis/P5_CQL/P5_CQL_PRIMARY_ANALYSIS.json`
- `results/analysis/P5_CQL/P5_CQL_SECONDARY_ANALYSIS.json`
- `results/analysis/P5_CQL/P5_CQL_SEED_SUMMARY.csv`
- `results/analysis/P5_CQL/P5_CQL_STATE_SLOPES.csv`

## Source hashes

| Artifact | SHA-256 |
|---|---|
| `results/analysis/P5_CQL/P5_CQL_PRIMARY_ANALYSIS.json` | `05dad16ea1c7c9b297539e8f2d5a9118050fe612d1c7733d639e9746553be49a` |
| `results/analysis/P5_CQL/P5_CQL_SECONDARY_ANALYSIS.json` | `b09042ef218cc50fa09c066d6f21b1ebdd2d1d4db1f5882024c2655cfe69435b` |
| `results/analysis/P5_CQL/P5_CQL_SEED_SUMMARY.csv` | `d8393d6fa7faa2b269b961a3f56e05a5ffcac2245436bc8bedabb49d72c05c2a` |
| `results/analysis/P5_CQL/P5_CQL_STATE_SLOPES.csv` | `564ab84f50cc8d167aaf457cc744d0ab8606363009a8b4ca5634afe103c3e764` |
| `experiments/reliability/P5_CQL_ANALYSIS_PROTOCOL.md` | `6ca0bf7cd2d2ffad1f3be2551f1d12e2a1dcb4a2ee7acac739ee07fee784914d` |
| `experiments/reliability/P5_CQL_GENERALIZATION_PROTOCOL.md` | `48fddb9d9973aca4ac460f73f515e7558cad9d3f62990be10b2b2253734b42d2` |

## Provenance

- Evidence-builder repository commit: `fd1720258cdac4e39c3d53e3516dc5b88244f675`
- Repository status before evidence generation: `?? results/analysis/P10/
?? src/evaluation/analyze_p10_standalone_ood.py
?? src/evaluation/build_p5_evidence_package.py`
- Python: `3.10.20 (main, Jun 11 2026, 15:17:37) [GCC 14.3.0]`
- Platform: `Linux-7.2.7-arch1-1-x86_64-with-glibc2.44`
- NumPy: `2.2.6`
- Pandas: `2.3.3`

## Original analysis provenance

- P5 primary-analysis commit recorded in JSON: `2a811a6571e36cf1349c9c97981482cdc6232900`

## Status

**P5 CQL Hopper consequence analysis: complete.**
