# Close the `557_analcatdata_apnea1` Gap (Experiment 13)

**Date:** 2026-04-25
**Context:** Experiment 12 got SplitBoost to 6/7 within 10% of XGBoost at matched capacity, stable across 3 seeds. The sole remaining holdout is `557_analcatdata_apnea1` (n=475, k=3), stuck at mean ratio 1.15. Lowering `min_samples_leaf_eml` only moved it from 1.15 → 1.15 — the loss isn't about EML-leaf activation, it's about expressiveness. This experiment sweeps `max_depth` and `n_eml_candidates` to test whether more internal-split exploration or deeper trees can close the gap.

## Goal

Sweep `max_depth ∈ {6, 8}` × `n_eml_candidates ∈ {10, 30, 100}` (5 useful combinations) on all 7 PMLB datasets × 3 seeds with all other Experiment-12-best defaults fixed. Determine whether some config closes apnea1 to within 10% of XGBoost without regressing the 6 datasets currently at 10%-or-better.

## Non-goals

- **No library changes.** Both hyperparameters already exist. Pure measurement + runner + report.
- No sweep of `k_leaf_eml` (always 1), `leaf_eml_cap_k` (always 2.0), `min_samples_leaf_eml` (always 30). These are Experiment 11/12 defaults that should stay.
- No `max_depth=10` — at n_train=380 for apnea1 with `min_samples_leaf=20`, depth=8 is already the practical max; depth=10 would overfit without adding real capacity.
- No `n_eml_candidates=300` — 100 already samples 1.6% of the 6400-tree search space at k=3; going higher has diminishing returns per the standard random-forest intuition.
- No full PMLB suite. That's Experiment 15's scope.

## Design

### Configurations (5 SplitBoost variants)

| id | `max_depth` | `n_eml_candidates` | intent |
|---|---|---|---|
| D6_C10 | 6 | 10 | baseline — Experiment 12's config |
| D6_C30 | 6 | 30 | more candidates at baseline depth |
| D6_C100 | 6 | 100 | saturation check: 1.6% of 6400-tree space |
| D8_C10 | 8 | 10 | deeper trees, baseline candidates |
| D8_C30 | 8 | 30 | both axes bumped |

`D8_C100` is **excluded** because it compounds the deeper-tree overfit risk with the most aggressive candidate count; if apnea1's problem is expressiveness, either D8_C* or D6_C100 should close it. If neither does, D8_C100 wouldn't either.

### Fixed config across all runs

```
max_rounds          = 200
learning_rate       = 0.1
patience            = 15
val_fraction        = 0.15
k_eml               = 3
k_leaf_eml          = 1
min_samples_leaf    = 20
min_samples_leaf_eml = 30     # Experiment 12 default
leaf_eml_gain_threshold = 0.05
leaf_eml_ridge      = 0.0
leaf_eml_cap_k      = 2.0     # Experiment 11 default
use_stacked_blend   = False
n_bins              = 256
test_size           = 0.20
seeds               = [0, 1, 2]
```

### Baselines — matched depth

XGBoost and LightGBM run at **matched depth** per config, not at a single fixed depth. Since we sweep `max_depth ∈ {6, 8}`, that's 2 XGBoost runs and 2 LightGBM runs per (dataset, seed) — not 5 each. When computing "ratio vs XGBoost" for a SplitBoost config, the denominator is the XGBoost run at the **same depth** (so D6_C30's ratio uses XGB at depth 6; D8_C30's uses XGB at depth 8).

This matches the project's "matched capacity" ethos: we're not testing SplitBoost against under-configured baselines.

### Total fits

7 datasets × 3 seeds × (5 SplitBoost + 2 XGBoost + 2 LightGBM) = **189 fits**. Estimated runtime ~30-40 min on RTX 3090 (dominated by `564_fried` at depth 8).

### Outputs

- `experiments/experiment13/summary.csv` — one row per (dataset, seed, config).
- `experiments/experiment13/summary.json` — config + aggregates, indexed by dataset × config.
- `experiments/experiment13/pmlb_rmse.png` — grouped bar chart with error bars, log-scale ratio panel.
- `experiments/experiment13/report.md` — narrative.
- `experiments/experiment13/run.log` — full console output.

## Experiment 13 plan

### Files changed

**Created:**
- `experiments/run_experiment13_apnea1_capacity.py` — runner. Structural fork of `run_experiment12_min_leaf_sweep.py`, with the sweep axis changed to (max_depth, n_eml_candidates) and the matched-depth baseline wiring.
- `experiments/experiment13/` directory populated by the runner.

**Unchanged:**
- `eml_boost/tree_split/tree.py`, `ensemble.py`, `nodes.py` — no code changes. The parameters already exist.

## Success criteria

- **S-A (primary): some config gets `557_analcatdata_apnea1` mean ratio below 1.10** (inside the 10% band for the first time).
- **S-B: the apnea1-winning config does not regress any current winner by > 0.03 on mean ratio.** The winners are `192_vineyard` (0.82), `523_analcatdata_neavote` (0.55), `210_cloud` (1.10), `529_pollen` (0.98), `562_cpu_small` (0.87), `564_fried` (1.00).
- **S-C: no stability regression.** No RMSE on any dataset × seed exceeds 10× the matched-depth XGBoost RMSE.

Partial success is fine: if a config closes apnea1 at the cost of a single dataset regressing by 0.04-0.05, we can document it as a tuning option rather than flipping the default.

## Action on verdict

- **Clean win (S-A + S-B + S-C):** flip defaults to the winning config (e.g., if D6_C30 wins, change `n_eml_candidates=10 → 30`).
- **Partial success (S-A met, S-B marginally violated):** keep current defaults, document the apnea1-specific tuning in the report.
- **Negative outcome (no config closes apnea1 within the 10% band):** document the structural cap, keep defaults. The dataset's 1.13-1.15 plateau may be a genuine expressiveness wall that no hyperparameter sweep can close.

## Risks

- **Deeper trees on small datasets (vineyard, cloud, neavote):** `max_depth=8` may cause overfit. The val-based early stopping (patience=15) should catch this, but worth watching train-vs-test RMSE per seed. If early stopping fires aggressively on small-n at depth 8, that's informative.
- **`n_eml_candidates=100` runtime:** candidate evaluation is batched GPU per node. 10× candidates is not 10× slower because the kernel launch overhead dominates for small batches. Expect maybe 1.5-2× slowdown on the SplitBoost fits with C100.
- **Matched-depth baselines add runtime:** 4 extra baseline fits per dataset × seed (2 for XGB, 2 for LGB) vs a single-depth baseline. LGB/XGB are fast (0.2-0.5 s each), so ~35 extra baseline-seconds per dataset × 7 datasets = ~4 min. Acceptable.
- **Seed variance on apnea1:** Experiment 11 showed apnea1 has high seed-to-seed variance (σ ≈ 350 on RMSE out of ~1200 mean). A "win" at one config and seed might not replicate. 3-seed mean is the verdict, not any single seed.
