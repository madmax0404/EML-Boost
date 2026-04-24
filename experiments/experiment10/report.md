# Experiment 10: Ridge-Regularized EML Leaves

**Date:** 2026-04-25
**Commit:** `bb0d050` (Task 3 runner at fit time)
**Runtime:** ~14 min on RTX 3090 (168 fits: 7 datasets × 3 seeds × 8 models)
**Scripts:** `experiments/run_experiment10_leaf_ridge.py`

## What the experiment was about

Experiment 9 surfaced catastrophic numerical explosions on `562_cpu_small` and `564_fried` under both blend-off and blend-on — traced by reading XGBoost v3.2.0 source to the fact that our EML-leaf OLS is unregularized. XGBoost's `w* = −G/(H+λ)` with default `reg_lambda=1.0` provably bounds `|w*| ≤ max|residual|`. Experiment 10 ports that idea: add a `leaf_eml_ridge` hyperparameter that shrinks the OLS slope `η` via `η = Sxy/(Sxx+λ)`. A 4-value ridge grid under the gate and a 2-value grid under the blend test whether soft shrinkage alone can stabilize the pathological datasets.

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
min_samples_leaf_eml = 50
leaf_eml_gain_threshold = 0.05
n_bins              = 256
test_size           = 0.20
seeds               = [0, 1, 2]
```

The six SplitBoost configurations:

| id | selection | `leaf_eml_ridge` | expected η shrinkage |
|---|---|---|---|
| G0 | gated | 0.0 | none (Exp 9 blend-off baseline) |
| G_weak | gated | 0.1 | ~10% |
| G_strong | gated | 1.0 | ~50% |
| G_vstrong | gated | 10.0 | ~91% |
| B0 | blend | 0.0 | none (Exp 9 blend-on baseline) |
| B_strong | blend | 1.0 | ~50% under blend |

## Results (mean ratios vs XGBoost, 3 seeds)

| dataset | G0 | G_weak (0.1) | G_strong (1.0) | G_vstrong (10.0) | B0 | B_strong (1.0) | verdict |
|---|---|---|---|---|---|---|---|
| 192_vineyard | **0.96** | **0.96** | **0.96** | **0.96** | **0.96** | **0.96** | ridge no-op (small-n, EML never activates) |
| 210_cloud | 1.19 | 1.19 | 1.19 | 1.19 | 1.19 | 1.19 | ridge no-op (small-n) |
| 523_analcatdata_neavote | **0.51** | **0.51** | **0.51** | **0.51** | **0.51** | **0.51** | ridge no-op (small-n) |
| 557_analcatdata_apnea1 | 1.13 | 1.14 | 1.13 | 1.18 | 1.14 | 1.13 | neutral (±0.01 across configs) |
| 529_pollen | **0.98** | **0.98** | 0.98 | 1.41 | 0.99 | 1.18 | G0 best; strong ridge regresses |
| **562_cpu_small** | 64,223 | **12.8** | 45,012 | 36,139 | 97,069 | 75,405 | G_weak helpful but still explodes on some seeds |
| **564_fried** | 4,167 | 5,921 | 3,542 | 3,379 | **2.12** | 3,614 | ridge makes fried worse; B0 stable |

**Mean ratios above are dominated by single-seed explosions. See per-seed table below for the real story.**

### Per-seed picture — RMSE on the two problematic datasets

| dataset / config | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `562_cpu_small` (XGBoost ≈ 2.9) | | | |
| &nbsp;&nbsp;G0 (ridge=0) | 2.56 | 2.56 | **563,005** |
| &nbsp;&nbsp;G_weak (ridge=0.1) | **98.6** | **10.3** | 2.98 |
| &nbsp;&nbsp;G_strong (ridge=1.0) | 2.54 | 2.62 | **394,595** |
| &nbsp;&nbsp;G_vstrong (ridge=10.0) | 2.52 | **3,773** | **313,036** |
| &nbsp;&nbsp;B0 (blend, ridge=0) | **2,470** | **182,510** | **665,978** |
| &nbsp;&nbsp;B_strong (blend, ridge=1) | **4,856** | **341,294** | **314,890** |
| `564_fried` (XGBoost ≈ 1.07) | | | |
| &nbsp;&nbsp;G0 | 1.07 | **13,386** | 1.07 |
| &nbsp;&nbsp;G_weak | **19,019** | 1.06 | 1.07 |
| &nbsp;&nbsp;G_strong | 1.07 | **11,376** | 1.07 |
| &nbsp;&nbsp;G_vstrong | 1.07 | **10,852** | 1.07 |
| &nbsp;&nbsp;B0 | 1.07 | 3.49 | 2.27 |
| &nbsp;&nbsp;B_strong | 1.07 | **11,608** | 1.07 |

Bolded cells are RMSE > 10× XGBoost — what I defined as "explosion" in the success criteria. Every SplitBoost config has at least one explosion across the 7 datasets × 3 seeds, except B0 on fried (three stable seeds: 1.07, 3.49, 2.27) — but B0 exploded spectacularly on cpu_small.

## |η| magnitudes

`max|η|` across 3 seeds per config per dataset (from `eta_stats.json`, aggregated across all leaves in all 200 trees):

| dataset | G0 | G_weak | G_strong | G_vstrong | B0 | B_strong |
|---|---|---|---|---|---|---|
| `562_cpu_small` | ~10⁴ | ~10² | ~10⁴ | ~10⁴ | ~10⁵ | ~10⁴ |
| `564_fried` | ~10⁴ | ~10⁵ | ~10⁴ | ~10⁴ | ~1 | ~10⁴ |
| `529_pollen` | ~1 | ~1 | ~1 | ~10 | ~1 | ~2 |
| `557_analcatdata_apnea1` | ~1 | ~1 | ~1 | ~1 | ~1 | ~1 |

Ridge does shrink η meaningfully in the monotonicity unit test (clean `y = exp(x_0)` signal showed `max|η|` going 14.9 → 0.5 → 0.27 → 0.09), but on the real heavy-tailed datasets the OLS fit is so unstable to begin with that even 50%+ shrinkage leaves η in the thousands. Ridge cannot turn η=10,000 into η<1 without strengths that also kill well-behaved datasets (as seen in `G_vstrong` regressing pollen from 0.98 to 1.41).

## Success criteria verdict

- **S-A (stability — no RMSE > 10× XGBoost on any dataset × seed under some ridge > 0):** **NOT MET.** Every positive-ridge config has at least one explosion on cpu_small or fried. The closest is G_weak, which still has cpu_small seed 0 at 98.6 (32× XGBoost).
- **S-B (cpu_small mean ratio < 2.0 under some ridge > 0):** **NOT MET.** The best ridge config on cpu_small is G_weak at 12.8 — 6× too high to satisfy the criterion.
- **S-C (no regression on prior wins):** **NOT MET.** G_vstrong regresses pollen from 0.98 → 1.41; every ridge > 0 config makes fried worse than the blend baseline.

**Negative-outcome check:** the spec's fallback says "if no positive ridge value brings any explosion-dataset within 10× of XGBoost, document the finding, keep the parameter in place, scope hard-capping as next experiment." **This fallback is triggered.**

## Why ridge alone is insufficient

The unit-test results show ridge works on well-behaved synthetic data: `max|η|` went 14.9 → 0.09 across a 0–10 ridge sweep, a ~160× shrinkage. That's not the mechanism failing.

The real issue is that on `562_cpu_small` and `564_fried`, the unregularized OLS fit produces |η| values of order 10⁴ on occasional seeds. A 50% shrinkage (ridge=1) takes 10⁴ → 5·10³ — still catastrophic. Even 91% shrinkage (ridge=10) takes 10⁴ → 10³. To kill the explosion via shrinkage alone we'd need ridge > 1,000, which destroys well-calibrated datasets where η's informative signal is of order 1.

This mirrors XGBoost's design reasoning: ridge (`reg_lambda`) is a *default* regularizer for benign shrinkage, but the hard-cap variant (`max_delta_step`) is the tool for bounding extreme leaf outputs. The analogy for our code is a hard cap `|η| ≤ C · max|residual|`, applied post-fit to every candidate tree before selection.

## Action taken

- **Kept `leaf_eml_ridge` default at 0.0** — no default change. The parameter stays in the public API for future experiments but isn't recommended without the companion hard cap.
- **Preserved the code path and all tests.** `_fit_leaf`'s ridge logic, the two unit tests, and the Task-1 backward-compat smoke test remain.

## What Experiment 10 actually shows

- **Ridge works in principle but not at useful strengths on this data.** The math and the unit tests confirm the implementation is correct. The benchmark shows the disease is resistant to this medicine.
- **The B0 (blend, ridge=0) result on `564_fried` is surprising** — three stable seeds (1.07, 3.49, 2.27) vs Experiment 9's blend-on (19.23, 13300, 1.07). I can't reproduce Experiment 9's fried blowups under the current code; possible CUDA non-determinism or a subtle interaction with Task 1's constructor-level changes. Noted as an open loose end, not a blocker for the Experiment 10 verdict.
- **The failure is consistent across ridge strengths and paths (gate and blend).** Every ridge > 0 config has at least one seed > 10× XGBoost on cpu_small or fried. The problem is deeper than just OLS shrinkage.
- **Small-n datasets (vineyard, cloud, neavote) are immune because EML leaves never activate.** With `min_samples_leaf_eml=50` and train sets of ~80 samples, no leaf hits the EML eligibility threshold. Ridge is a no-op on these.

## What's left as a loss

- **`562_cpu_small` and `564_fried`** remain fundamentally unstable. No single-config best: different ridge strengths produce explosions on different seeds. The pathology is in the OLS fit's extreme η magnitudes, which ridge can dampen but not cap.
- **`210_cloud` and `557_analcatdata_apnea1`** remain outside the 10% band — neither ridge nor blend helps. These are small-medium datasets where SplitBoost is competitive but not winning; Experiment 9 already flagged them.

## What Experiment 10 does NOT show

- Single 80/20 shuffle-split per seed; no CV.
- No hard-cap (max|η|) alternative tested — that's Experiment 11's scope.
- No L1 alternative (`reg_alpha` analog) tested.
- The η-magnitude summary for the heavy-tail datasets is approximate ("~10⁴") — see `eta_stats.json` for exact per-seed values.
- The Experiment 9 vs Experiment 10 B0 discrepancy on fried isn't explained. A focused reproduction attempt is worth an issue, not an experiment.

## Consequence for the project

**Unchanged from Experiment 9's status.** The headline claim from Experiment 8 (5/7 within 10% of XGBoost on single seed) is not rescued by ridge alone. The EML-leaf stability problem on heavy-tailed features persists. The next reasonable experiment is a hard cap on `|η|`, modeled on XGBoost's `max_delta_step`.

## Next possible experiments

- **Experiment 11: hard cap on |η|.** Port XGBoost's `max_delta_step` concept. After OLS fits (η, β), clip η to `±C · max|residual|/max|eml|` before tree selection. Test the same 6-ish config grid (gate × {no cap, cap=1, cap=10}) to find the sweet spot.
- **Root-cause the Exp 9 → Exp 10 B0 drift on fried.** The current B0 config on fried is suspiciously stable compared to Experiment 9's identical-spec fit. Worth a single-dataset reproduction with instrumented RNG traces.
- **Feature-screening heuristic.** On datasets where `max(|x|)/std(x) > threshold` per feature, skip EML leaves on that feature entirely. Simple and might dodge the heavy-tail issue without new hyperparameters.
- **Full PMLB multi-seed suite.** With the unstable-baseline story confirmed, the whole 55-dataset suite needs multi-seed evaluation before any win percentages from Experiment 8 can be considered settled.
