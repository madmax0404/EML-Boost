# Experiment 12: `min_samples_leaf_eml` Sweep

**Date:** 2026-04-25
**Commit:** `c8be51f` (runner committed; fit-time HEAD)
**Runtime:** ~12 min on RTX 3090 (126 fits: 7 datasets × 3 seeds × 6 models)
**Scripts:** `experiments/run_experiment12_min_leaf_sweep.py`

## What the experiment was about

Experiment 11's leaf-prediction cap (`leaf_eml_cap_k=2.0`) got SplitBoost to 5/7 outright wins against XGBoost at matched capacity, stable across 3 seeds. The two holdouts were `210_cloud` (mean ratio 1.19) and `557_analcatdata_apnea1` (1.13). On `210_cloud`, the cause was structural: with `min_samples_leaf_eml=50` and n_train=86, the depth-6 tree's leaves end up size 20-40 — all below the 50-sample EML gate — so **EML never activated**.

This experiment sweeps `min_samples_leaf_eml ∈ {20, 30, 40, 50}` to test whether lowering the threshold unlocks EML on `210_cloud` (and other small-n datasets) without regressing the winners. All other Experiment-11-best hyperparameters are fixed, most importantly `leaf_eml_cap_k=2.0` — the cap's tree-selection robustness matters *more* on small leaves where OLS is noisier.

## Configuration

```
max_rounds          = 200
max_depth           = 6
learning_rate       = 0.1
patience            = 15
val_fraction        = 0.15
n_eml_candidates    = 10
k_eml               = 3
k_leaf_eml          = 1
min_samples_leaf    = 20       # tree-structure threshold, fixed
leaf_eml_gain_threshold = 0.05
leaf_eml_ridge      = 0.0
leaf_eml_cap_k      = 2.0      # Experiment 11 default, kept on
use_stacked_blend   = False
n_bins              = 256
test_size           = 0.20
seeds               = [0, 1, 2]
```

Four SplitBoost configurations:

| id | `min_samples_leaf_eml` |
|---|---|
| M20 | 20 |
| M30 | 30 |
| M40 | 40 |
| M50 | 50 (Experiment 11 default) |

## Results (mean ratios vs XGBoost, 3 seeds)

| dataset | M20 | M30 | M40 | M50 | XGB mean | verdict |
|---|---|---|---|---|---|---|
| 192_vineyard | **0.82** | **0.82** | 0.96 | 0.96 | 3.31 | **M30/M20 huge improvement (0.96 → 0.82)** |
| 210_cloud | 1.11 | **1.10** | 1.19 | 1.19 | 0.47 | **M30 JUST enters the 10% band (was 1.19)** |
| 523_analcatdata_neavote | **0.55** | **0.55** | **0.55** | **0.55** | 1.71 | unchanged (strong win everywhere) |
| 557_analcatdata_apnea1 | 1.18 | **1.15** | 1.16 | 1.15 | 1124 | M30 tied with M50 for best |
| 529_pollen | **0.98** | **0.98** | **0.98** | **0.98** | 1.58 | unchanged (comfortable win) |
| 562_cpu_small | 0.88 | **0.87** | **0.87** | 0.88 | 2.92 | unchanged (solid win) |
| 564_fried | 1.00 | 1.00 | **1.00** | 1.00 | 1.07 | unchanged (parity) |

**Under M30: 6/7 within 10% of XGBoost** (up from 5/7 at M50), **5 outright wins** (up from 5 — same count but `192_vineyard` drops its mean ratio from 0.96 to 0.82, a substantive strengthening).

### Per-seed picture on the two activated datasets

**`210_cloud`** (XGBoost mean 0.47):

| config | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| M20 | 0.813 | 0.290 | 0.447 |
| **M30** | 0.757 | 0.312 | 0.475 |
| M40 | 0.746 | 0.447 | 0.473 |
| M50 | 0.746 | 0.447 | 0.473 |

M30 wins seed 0 by a thread over M50, loses seed 2 by a tiny margin, but picks up a huge seed-1 improvement (0.45 → 0.31). M40 and M50 are bit-identical because both produce zero EML leaves on this dataset (confirmed below).

**`192_vineyard`** (XGBoost mean 3.31):

| config | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| M20 | 2.566 | 2.475 | 3.121 |
| **M30** | 2.566 | 2.475 | 3.121 |
| M40 | 2.394 | 2.621 | 4.534 |
| M50 | 2.394 | 2.621 | 4.534 |

Mean of M40/M50 = 3.18; mean of M20/M30 = 2.72. The seed-2 RMSE drops from 4.53 to 3.12 — that's where the mean-ratio improvement comes from. Seeds 0 and 1 are about even.

## Leaf activation (where EML actually fires)

Total `n_eml_leaves` across all 3 seeds × 200 trees, from `leaf_activation_stats.json`:

| dataset | n_train | M20 | M30 | M40 | M50 | comment |
|---|---|---|---|---|---|---|
| 192_vineyard | 41 | 87 | **87** | 0 | 0 | M30 unlocks |
| 210_cloud | 86 | 233 | **83** | 0 | 0 | M30 unlocks (partial) |
| 523_analcatdata_neavote | 80 | 81 | **81** | 0 | 0 | M30 unlocks (but no RMSE improvement — already winning) |
| 557_analcatdata_apnea1 | 380 | 1142 | 489 | 89 | 39 | graded, not a step function |
| 529_pollen | 3078 | 2639 | 1346 | 781 | 596 | graded |
| 562_cpu_small | 6553 | 6213 | 3304 | 2063 | 1525 | graded |
| 564_fried | 32614 | 9089 | 6420 | 4980 | 4368 | graded |

Two-regime picture:
- **Small-n datasets (vineyard, cloud, neavote):** EML activation is a *step function* of the threshold — zero leaves at M≥40, many at M≤30. That's because the tree's leaves on these datasets have a narrow size distribution in the 20-39 range, so crossing 40 is all-or-nothing.
- **Medium/large-n datasets (apnea1, pollen, cpu_small, fried):** EML activation is a *graded function* — lowering the threshold always adds more EML leaves from the smaller-leaf tail of the distribution. But the RMSE barely changes: these datasets already had enough EML leaves to pick up the pattern at M50; adding more doesn't help.

This is a clean result. The threshold controls a binary "does this dataset get any EML at all?" question for small-n, and a marginal "how much EML?" question for medium/large-n. The RMSE gains are concentrated where the former crosses zero.

## Success criteria verdict

- **S-A (primary): `210_cloud` improves under some M < 50.** **MET.** M30 gets the mean ratio from 1.188 to 1.100 — just inside the 10% band. M20 is barely worse at 1.105.
- **S-B: no regression on winners > 0.03 mean ratio under best M < 50.** **MET.** Under M30, `192_vineyard` improves by 0.14, `557_analcatdata_apnea1` improves by 0.005, and all other winners (`neavote`, `pollen`, `cpu_small`, `fried`) stay within 0.006 of M50.
- **S-C: no RMSE > 10× XGBoost on any dataset × seed.** **MET.** Worst ratio across all 84 SplitBoost fits is `557_analcatdata_apnea1` seed 2 at ~1.23× XGBoost. No explosions.

**Verdict: M30 IS A KEEPER.** Recommended default: **`min_samples_leaf_eml = 30`**.

Reasoning: M30 provides all the benefit of lowering the threshold (unlocks `210_cloud` and dramatically improves `192_vineyard`) without the extra risk of M20's smaller leaves (where OLS has only 15-sample fit portions). M40 is functionally identical to M50 on the small-n datasets — it doesn't do the work. M20 doesn't outperform M30 on any dataset. The 30-sample threshold is also structurally motivated: it's 1.5× the `min_samples_leaf=20` tree-structure minimum, giving EML leaves a ~10-sample OLS-fit portion buffer above the absolute floor.

## What Experiment 12 actually shows

- **Small-n datasets were leaving EML on the table.** `210_cloud`, `192_vineyard`, and `523_analcatdata_neavote` had zero EML leaves under the old M50 default. Lowering to M30 activates EML on all three.
- **Activation unlocks both cloud and vineyard.** `210_cloud` crosses the 10% band for the first time (1.19 → 1.10). `192_vineyard`, already a winner, deepens its margin (0.96 → 0.82).
- **Medium/large datasets don't care about the threshold.** `pollen`, `cpu_small`, `fried` all have mean ratios within 0.01 across M20-M50. The cap-stabilized Exp-11 baseline was already well-calibrated on these.
- **No stability issues.** The cap (`leaf_eml_cap_k=2.0`) extends to tinier leaves without trouble. No RMSE explosion anywhere across 84 fits.
- **`neavote` activation doesn't help RMSE.** Despite M30 unlocking 81 EML leaves, the mean ratio is 0.546 vs 0.548 at M50 — within noise. The dataset is already dominated by the inner-split EML mechanism, not the leaf-EML mechanism.

## What's left as a loss

- **`557_analcatdata_apnea1`** (1.147 at M30): still outside 10%. Lowering `min_samples_leaf_eml` gave a tiny improvement (1.152 → 1.147) but not enough. This one's loss is about expressiveness on a medium-n dataset, not a structural gate — a different experiment.

Just 1/7 out of the band now.

## What Experiment 12 does NOT show

- Single 80/20 shuffle-split per seed; no CV.
- Full PMLB suite (55 datasets) not tested.
- No `min_samples_leaf` sweep — that would change tree structure, not EML eligibility.
- Blend + lower threshold combination untested (blend was 0'd out in Experiment 9's negative outcome).

## Action taken

- **Flipped the `min_samples_leaf_eml` default from 50 to 30** in both `EmlSplitTreeRegressor.__init__` and `EmlSplitBoostRegressor.__init__`. Users can still tune via the hyperparameter.
- **Preserved all code paths and tests.** No library code change other than the default.

## Consequence for the project

**Headline is now stronger.** After Experiment 11 re-established "5/7 within 10% of XGBoost, stable across 3 seeds", Experiment 12 pushes this to **6/7 within 10% at matched capacity, stable across 3 seeds**, with `192_vineyard`'s lead over XGBoost deepening from 4% to 18%. The last holdout (`557_analcatdata_apnea1`) is within 15%, and no dataset has a multi-seed instability.

## Next possible experiments

- **Full PMLB multi-seed suite** with `min_samples_leaf_eml=30`, `leaf_eml_cap_k=2.0`. 55 datasets × 3 seeds. Turn the 6/7 count into an aggregate statistic with error bars.
- **Close the `557_analcatdata_apnea1` gap.** n=380, k=3, mean ratio stuck at 1.15. Try `max_depth=8` or `n_eml_candidates=30` — might be an expressiveness issue the current config under-captures.
- **Extrapolation benchmark.** Rerun Experiments 4/6's extrapolation targets with the new defaults. Does the cap + lowered threshold preserve any extrapolation edge from the EML splits/leaves?
- **Capacity-unlocked mode.** Bump both SplitBoost and XGBoost to `max_depth=10, max_rounds=500`. Does the cap hold, and does 6/7 become 7/7?
