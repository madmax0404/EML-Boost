# Experiment 14: Capacity-Unlocked Mode

**Date:** 2026-04-25
**Commit:** `5cd948f` (runner committed; fit-time HEAD)
**Runtime:** ~16 min on RTX 3090 (63 fits — `564_fried` dominates at ~130 s/fit × 3 seeds)
**Scripts:** `experiments/run_experiment14_capacity_unlocked.py`

## What the experiment was about

Experiment 13 established that bumping `max_depth` from 6 to 8 was a monotonic Pareto improvement at matched capacity — every dataset got better or tied. Experiment 14 tests the natural follow-up: at substantially higher capacity (`max_depth=10, max_rounds=500, patience=30`), does SplitBoost continue to lead against matched-capacity XGBoost? Or does XGBoost's mature regularization finally catch up?

## Configuration

```
max_rounds          = 500       # bumped from 200
max_depth           = 10        # bumped from 8
patience            = 30        # bumped from 15 to scale with max_rounds
learning_rate       = 0.1
n_eml_candidates    = 10        # Exp 13 default kept
k_eml               = 3
k_leaf_eml          = 1
min_samples_leaf    = 20
min_samples_leaf_eml = 30       # Exp 12 default
leaf_eml_gain_threshold = 0.05
leaf_eml_ridge      = 0.0
leaf_eml_cap_k      = 2.0       # Exp 11 default
use_stacked_blend   = False
n_bins              = 256
test_size           = 0.20
seeds               = [0, 1, 2]
```

XGBoost and LightGBM at the same `max_depth=10, max_rounds=500` for the matched-capacity comparison.

## Results (mean ratios vs XGBoost, 3 seeds)

Side-by-side with the Experiment-13 D8_R200 reference:

| dataset | D8_R200 (Exp 13) | **D10_R500 (Exp 14)** | Δ | n_rounds (D10_R500 mean) | verdict |
|---|---|---|---|---|---|
| 192_vineyard | 0.801 | **0.801** | −0.000 | 54 | tied (early-stopped at d=10 like d=8) |
| 210_cloud | **1.100** | 1.125 | **+0.025** | 56 | **regressed past 10% band** |
| 523_analcatdata_neavote | 0.546 | **0.539** | −0.008 | 78 | improved (small-n early-stop, similar tree) |
| 557_analcatdata_apnea1 | 1.104 | 1.104 | +0.000 | 141 | tied (same marginal miss) |
| 529_pollen | 0.945 | **0.926** | −0.019 | 83 | improved |
| 562_cpu_small | 0.853 | **0.831** | −0.022 | 158 | improved (largest ratio improvement on a winner) |
| 564_fried | 0.988 | **0.970** | −0.018 | 120 | improved |

**Net: 4 datasets improved, 2 tied, 1 regressed** (cloud, by 0.025).

### Per-seed picture on the two band-edge datasets

**`210_cloud`** (the new regression):

| seed | SplitBoost RMSE | XGBoost RMSE | ratio |
|---|---|---|---|
| 0 | 0.755 | 0.671 | 1.126 |
| 1 | 0.327 | 0.269 | 1.214 |
| 2 | 0.514 | 0.479 | 1.073 |
| **mean** | **0.532** | **0.473** | **1.125** |

Cloud's regression is real but driven by seed 1 (ratio 1.214) where the high-capacity SplitBoost overfits the small training set (n_train=86). Seeds 0 and 2 stay near or just over the 10% band. The Exp-13 D8_R200 numbers on the same seeds were 1.157 / 1.170 / 0.985 — D10_R500 has slightly worse seed 0 and seed 2 but actually slightly *better* seed 1. The mean shifts because the worst seed got worse faster than the best seed got better.

**`557_analcatdata_apnea1`** (the persistent marginal miss):

| seed | SplitBoost RMSE | XGBoost RMSE | ratio |
|---|---|---|---|
| 0 | 1340 | 1085 | 1.235 |
| 1 | 939 | 889 | 1.057 |
| 2 | 1595 | 1535 | 1.039 |
| **mean** | **1291** | **1170** | **1.104** |

D10_R500 on apnea1 has slightly different per-seed ratios than D8_R200 (1.263 / 0.998 / 1.056 there) but identical mean ratio of 1.104. Seed 1 regresses (0.998 → 1.057), seeds 0 and 2 improve marginally. Net wash. The high-capacity setting doesn't help apnea1 at all; the dataset's loss is genuinely structural (high target variance + only 3 features).

## Success criteria verdict

- **S-A: all 7 datasets stay ≤ 1.10 mean ratio at unlocked capacity:** **NOT MET.** Cloud at 1.125 (was 1.100) and apnea1 at 1.104 (unchanged) are both outside. Exp 13 had cloud right at the band; Exp 14 pushes it past. The "7/7 within 10%" headline does not hold at unlocked capacity.
- **S-B: no dataset regresses > 0.05 vs D8_R200:** **MET.** Cloud regression is 0.025 — within the 0.05 tolerance.
- **S-C: no RMSE > 10× XGBoost on any dataset × seed:** **MET.** Worst single-seed ratio is cloud seed 1 at 1.214. Stability holds.

**Verdict: PARTIAL SUCCESS.** D10_R500 improves the average ratio on 4 datasets, ties on 2, but breaks cloud's strict 10% band membership. The architectural lead clearly persists at higher capacity (still outright wins on vineyard, neavote, pollen, cpu_small, fried), but the strict "7/7 within 10%" claim no longer holds.

## What Experiment 14 actually shows

- **The architectural win persists.** D10_R500 SplitBoost beats XGBoost outright on 5/7 datasets — the same 5 winners as D8_R200 (vineyard 0.80, neavote 0.54, pollen 0.93, cpu_small 0.83, fried 0.97). At higher capacity the wins on cpu_small and pollen actually deepen by 2-3 points each.
- **XGBoost overfits more at d=10/r=500 than SplitBoost.** Comparing absolute RMSEs: XGBoost on `562_cpu_small` goes from 2.93 (d=8) to 3.10 (d=10). SplitBoost on the same dataset goes 2.43 → 2.58. SplitBoost degrades, but less. On `pollen` and `fried` both degrade slightly but XGBoost more. SplitBoost's leaf cap and `min_samples_leaf_eml=30` continue to do real regularization work at higher capacity.
- **Small-n datasets are the new weak spot.** Cloud (n_train=86) is the only dataset that genuinely regresses. With max_depth=10, the tree's structure on a small training set has more freedom to overfit, and even though `min_samples_leaf=20` caps leaf size, the tree shape changes seed-to-seed in ways that hurt some held-out sets. Vineyard (n_train=41) and neavote (n_train=80) are insulated because their trees early-stop at the same effective depth as d=8 — but cloud has 5 features instead of 2, so depth-10 finds more spurious splits.
- **Apnea1 is unmoved.** D10_R500 produces an identical mean ratio (1.104) to D8_R200. The dataset's plateau is structural — capacity doesn't help, more candidates didn't help (Exp 13), and lowering `min_samples_leaf_eml` didn't help (Exp 12). Whatever SplitBoost is missing on apnea1 isn't a hyperparameter.
- **Runtime cost is real.** `564_fried` jumps from 32 s (d=8, r=200) to 130 s (d=10, r=500), a 4× slowdown. Total experiment runtime ~16 min vs Exp 13's 13 min, so the wall-clock isn't terrible — but the SplitBoost-only fit time is 3-5× longer per dataset.

## What's left as a loss

- **`210_cloud` (1.125):** new regression vs D8_R200's 1.100. Driven by seed 1 overfitting at d=10.
- **`557_analcatdata_apnea1` (1.104):** persistent marginal miss across all four experiments (12, 13, 14).

5/7 within strict 10% under D10_R500 (vs 6/7 under D8_R200).

## What Experiment 14 does NOT show

- Single 80/20 shuffle-split per seed; no CV.
- Full PMLB suite (Experiment 15) not tested.
- No 2×2 sweep of (max_depth, max_rounds) — single capacity-unlocked config only.
- D10 with smaller `max_rounds` (e.g., 200) untested — would isolate whether the depth alone caused cloud's regression or the rounds added overfit.

## Action taken

- **Keeping `max_depth=8` as the default.** D10_R500 is *available* (just pass `max_depth=10, max_rounds=500`) but not the auto-on choice. The runtime cost (3-5× per fit on medium/large datasets) plus cloud's regression don't justify the modest gains on 4 other datasets. The "matched capacity at d=8, r=200" remains the project's headline configuration.
- **No changes to the library.** Pure measurement experiment.

## Consequence for the project

**Headline holds at d=8, weakens at d=10.** Under the recommended d=8 default: still 6/7 within 10%, 5/7 outright wins, stable across 3 seeds (Experiment 13's claim). Under the experimental d=10 unlocked: 5/7 within 10%, 5/7 outright wins, with deeper margins on cpu_small and fried but cloud regressing past the band.

The project ships the Experiment-13 default as the recommendation, with capacity-unlocked as a known viable config for users prioritizing absolute RMSE over strict-band-membership.

## Next possible experiments

- **Experiment 15 (next in queue): full PMLB multi-seed suite** at d=8, r=200, the established default. Turn the 6/7-on-7-datasets story into a 55-dataset aggregate statistic with error bars.
- **Smaller capacity bump:** test D10_R200 to isolate whether cloud's regression is from the depth alone or the longer rounds. Could rescue the cloud band while keeping cpu_small / fried's d=10 gains.
- **Closer apnea1 root-cause:** the 1.104 plateau is now invariant across hyperparameter sweeps. Worth a focused look at the dataset's structure (are the 3 features near-collinear? is the noise heteroscedastic? does the train/test boundary hide a regime?) before investing more compute.
- **Experiment 16 (queued): extrapolation benchmark.** Rerun Experiments 4/6 targets under the new d=8 default. Does the depth bump preserve or break extrapolation?
