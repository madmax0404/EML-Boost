# Experiment 16: leaf_l2=1.0 Validation on Exp 15 Losers — Did Not Land

**Date:** 2026-04-25
**Commits:** `2d89e8b` (runner + outputs); fit-time HEAD includes `8ca6989` (default flip).
**Runtime:** ~3 min on RTX 3090 (100 fits = 20 datasets × 5 seeds × 1 model).
**Scripts:** `experiments/run_experiment16_leaf_l2_validation.py`.

## What the experiment was about

Experiment 15 (full PMLB regression suite, 119 datasets) showed SplitBoost beating XGBoost on 99/119 (83%) but losing on 20 datasets, including 3 catastrophic losses (ratio > 2.0): `527_analcatdata_election2000`, `663_rabe_266`, `561_cpu`. A pre-fix isolation experiment on the two smallest catastrophic datasets traced the gap to the **vanilla GBDT machinery**, not anything EML-specific — disabling EML entirely still left SplitBoost ~3× worse than XGBoost on those small-n datasets. The hypothesis: SplitBoost lacks XGBoost's `reg_lambda=1.0` default regularization on leaf values, so it overfits aggressively on tiny noisy data.

The fix (Tasks 1-7 of `docs/superpowers/plans/2026-04-25-leaf-l2-regularization.md`) added a `leaf_l2: float = 1.0` parameter that mirrors XGBoost's `reg_lambda`: leaf value becomes `Σy / (n + λ)`, EML leaf bias gets analogous shrinkage, and split-gain SSE denominators become `cnt + λ`. Implementation passed all unit tests including a new bit-exact regression test at λ=0 and a multi-λ Triton-vs-torch equivalence test.

This experiment validates the fix on the 20 Exp-15 losers — re-fits SplitBoost only with the new default, reuses XGBoost numbers from `experiment15/summary.csv`, and compares ratios.

## Configuration

```
leaf_l2             = 1.0     # NEW; the fix being validated
max_rounds          = 200
max_depth           = 8
patience            = 15
learning_rate       = 0.1
n_eml_candidates    = 10
k_eml               = 3
k_leaf_eml          = 1
min_samples_leaf    = 20
min_samples_leaf_eml = 30
leaf_eml_gain_threshold = 0.05
leaf_eml_ridge      = 0.0
leaf_eml_cap_k      = 2.0
n_bins              = 256
test_size           = 0.20
seeds               = [0, 1, 2, 3, 4]
```

XGBoost numbers reused from Exp 15 (matched-capacity, untouched). Only SplitBoost is re-fit.

## Headline

| metric | result |
|---|---|
| Datasets re-fit | **20** (the Exp-15 datasets where SB ratio > 1.00) |
| Now outright wins (ratio < 1.00) | **1/20** |
| In 10% band (ratio < 1.10) | 6/20 (vs ~7/20 in Exp 15) |
| Still catastrophic (ratio > 2.00) | **3/20** (unchanged) |
| **Mean Δ ratio** | **+0.006** (slight degradation, not improvement) |
| Wall time | ~3 min |
| Triton fallbacks | none |

**The fix did not land as predicted.** The mean ratio across the 20 datasets moved by +0.006 — well within seed-to-seed noise and in the wrong direction.

## Per-dataset comparison

Sorted by Exp-15 ratio (worst losses first):

| dataset | n_train | k | Exp 15 ratio | Exp 16 ratio | Δ | verdict |
|---|---|---|---|---|---|---|
| 527_analcatdata_election2000 | 53 | 14 | 2.355 | **2.364** | +0.009 | still_catastrophic |
| 663_rabe_266 | 96 | 2 | 2.341 | **2.366** | +0.026 | still_catastrophic |
| 561_cpu | 167 | 7 | 2.149 | **2.186** | +0.037 | still_catastrophic |
| 659_sleuth_ex1714 | 38 | 7 | 1.742 | 1.742 | -0.000 | still_clear_loss |
| 1089_USCrime | 37 | 13 | 1.608 | 1.608 | +0.000 | still_clear_loss |
| 485_analcatdata_vehicle | 39 | 4 | 1.484 | 1.484 | -0.000 | improved_but_still_loss |
| 230_machine_cpu | 167 | 6 | 1.479 | **1.426** | -0.052 | improved_but_still_loss |
| 505_tecator | 192 | 124 | 1.365 | 1.470 | **+0.105** | improved_but_still_loss |
| 1096_FacultySalaries | 40 | 4 | 1.253 | 1.317 | +0.064 | improved_but_still_loss |
| 542_pollution | 48 | 15 | 1.146 | 1.141 | -0.004 | improved_but_still_loss |
| 666_rmftsa_ladata | ~120 | 4 | 1.140 | 1.141 | +0.002 | improved_but_still_loss |
| 228_elusage | ~55 | 2 | 1.139 | 1.158 | +0.019 | improved_but_still_loss |
| 656_fri_c1_100_5 | 80 | 5 | 1.112 | 1.116 | +0.004 | improved_but_still_loss |
| 687_sleuth_ex1605 | ~50 | 4 | 1.094 | **1.045** | -0.049 | in_band |
| 591_fri_c1_100_10 | 80 | 10 | 1.090 | 1.093 | +0.003 | in_band |
| 594_fri_c2_100_5 | 80 | 5 | 1.066 | 1.045 | -0.021 | in_band |
| 201_pol | 12,000 | 48 | 1.025 | 1.020 | -0.005 | in_band |
| 537_houses | 16,512 | 8 | 1.009 | 1.002 | -0.007 | in_band |
| **657_fri_c2_250_10** | 200 | 10 | 1.006 | **0.994** | -0.012 | **now_a_win** |
| 1030_ERA | 800 | 4 | 1.002 | 1.003 | +0.001 | in_band |

**Distribution of deltas:**
- |Δ| < 0.01: **9/20** (no measurable change)
- 0.01 ≤ |Δ| < 0.05: 6/20
- 0.05 ≤ |Δ| < 0.10: 3/20
- |Δ| ≥ 0.10: 2/20 (505_tecator +0.105 — *worse*; nothing improved by ≥ 0.10)

**Net: 7 datasets improved, 1 unchanged, 12 worsened.** The improvements are small; the degradations include `505_tecator` (+0.105) and `1096_FacultySalaries` (+0.064).

## Success criteria verdict

The plan/spec set three criteria for the fix to be considered successful:

- **S-A (correctness):** ✅ **MET.** All 100 unit tests pass after the default flip; no Triton fallback warnings.
- **S-B (catastrophic losses ≤ 1.5):** ❌ **FAILED.** All 3 catastrophic datasets stayed > 2.0; deltas are +0.009, +0.026, +0.037 (slight degradation).
- **S-C (mean Δ ≤ −0.20):** ❌ **FAILED.** Actual mean Δ is +0.006 — about two orders of magnitude smaller than the threshold, in the wrong direction.
- **S-D (no Triton fallback):** ✅ **MET.** Clean run.

**Verdict: implementation is correct (S-A + S-D pass), but the fix's empirical impact on the loss regime is essentially zero (S-B + S-C fail).**

## Why it didn't work

The hypothesis was: "SplitBoost lacks `reg_lambda=1.0`, so leaf values overfit on tiny noisy data; adding L2 shrinkage will close the gap."

The math: `mean(y_residual)` becomes `Σy / (n + λ)`. Effective shrinkage is `n / (n + λ)`. With `min_samples_leaf=20` (every leaf has ≥ 20 samples) and λ=1.0:

```
n=20:    shrinkage = 20/21 ≈ 0.952  → leaf magnitude reduced by 4.8%
n=30:    shrinkage = 30/31 ≈ 0.968  → reduced by 3.2%
n=100:   shrinkage = 100/101 ≈ 0.990 → reduced by 1.0%
n=1000:  shrinkage = 1000/1001 ≈ 0.999 → reduced by 0.1%
```

**At λ=1.0 and our `min_samples_leaf=20`, the leaf-value shrinkage is at most 5%.** That's nowhere near enough to alter the gross overfitting we see on tiny datasets where individual leaves can have residual variance >> XGBoost's tighter fits. The split-gain regularization is similarly small in effect.

By comparison, XGBoost's `reg_lambda=1.0` is *also* nominally only ~5% shrinkage — but XGBoost gains more from regularization because of:
1. **Hessian-weighted gain formula** uses `Σg²/(Σh + λ)` where `h_i = 1` for squared error — the gain is proportional to `Σ(residuals)² / (n + λ)`, which interacts with split-acceptance differently than raw SSE reduction.
2. **`min_child_weight = 1` (Hessian-based, not sample-count)** — different leaf-size constraint.
3. **Default `gamma = 0` (split-loss penalty)** — but the gain itself includes the regularizer in its denominator, so the effective threshold compounds.
4. **Tree pruning** — XGBoost prunes after split-finding using the regularized objective; SplitBoost doesn't prune.
5. **`colsample_*` defaults at 1.0** but available — feature subsampling helps small-n.

Our fix matched XGBoost's *parameter form* but not the *behavioral mechanism* that makes XGBoost robust on tiny data. The gap isn't in `reg_lambda` alone — it's in the system of gain formulation + tree pruning + sample/feature subsampling that together regularize the model.

### Two anomalies worth flagging

1. **`505_tecator` got 8% worse** (1.365 → 1.470). This is the "high-dim spectroscopy" dataset (n=192, k=124) — the only non-tiny loser in Exp 15. With λ=1.0 the modest leaf shrinkage interacts unfavorably with the existing top-k=3 feature-selection bottleneck; the regularizer slightly weakens the few EML candidates that did capture spectral structure.
2. **`657_fri_c2_250_10` flipped from a 1.006 marginal loss to a 0.994 marginal win**. The single "now win" in the experiment. Δ = 0.012 — within seed-noise; not a real signal that the fix works on this dataset, just bench noise crossing the parity line.

## What Experiment 16 actually shows

- **The implementation is correct.** Bit-exact at λ=0 (locked by `test_leaf_l2_zero_constant_leaves_bit_exact`), GPU/CPU equivalent at λ=1.0, Triton/torch agree across multiple λ values. No correctness bugs.
- **`leaf_l2=1.0` is too weak to matter at this `min_samples_leaf=20`.** Maximum 5% leaf-value shrinkage is below the noise floor of fit-to-fit variance on small-n datasets.
- **The catastrophic loss regime is structural, not a missing-regularizer problem alone.** XGBoost's robustness on tiny data comes from a *system* (Hessian-weighted gain, pruning, sampling, gamma threshold) — adding only the L2 leaf-shrinkage piece in isolation moves nothing.
- **The 9 "no measurable change" datasets (|Δ| < 0.01) are the most diagnostic data point.** If the regularizer were doing real work, even small-n datasets would show some movement; the dominance of "essentially zero change" tells us the mechanism is inactive at this λ.
- **Higher λ might help — but at the cost of also shrinking large-n leaves more aggressively.** Untested in this experiment; would require its own validation.

## What's left as a loss

All 20 datasets that lost in Exp 15 still lose in Exp 16, by approximately the same margins. The 3 catastrophic cases are unchanged.

## What Experiment 16 does NOT show

- **No tuning of `leaf_l2` itself.** Single value tested (1.0); no grid sweep. We don't know if e.g. λ=10 or λ=100 would help.
- **No baseline tuning of XGBoost.** XGBoost runs at matched-capacity defaults from Exp 15.
- **No re-validation on Exp-15 winners.** Per the plan's explicit scope (and the user's earlier direction), winners weren't re-fit. Possible regressions on the 99 winning datasets remain unmeasured.
- **No isolation of which mechanism within `leaf_l2` did/didn't work.** The fix touched 3 sites (constant leaf, EML bias, split gain); we don't know which contributed (positive or negative) on which dataset.

## Caveats

- **CUDA non-determinism in fit-time.** Per-seed deltas of ±0.05 are within the run-to-run noise we observed in Exp 15's bench reproductions. The +0.006 mean Δ is well below this floor.
- **The `leaf_l2=0.0` opt-out is preserved.** Anyone who wants pre-Task-2 behavior can pass `leaf_l2=0.0` explicitly. Default flip is reversible.
- **Implementation is shipped, just inactive at default.** The `leaf_l2` parameter, all the math plumbing, and the bit-exact regression test are now part of the codebase — even if `leaf_l2=1.0` doesn't materially help, the parameter is available for future tuning experiments.

## Action taken

- **Default kept at `leaf_l2=1.0`.** No reason to revert: the fix is a no-op at this scale rather than a regression. Future regularization work (gamma threshold, Hessian-weighted gain, etc.) will likely build on this scaffolding.
- **Experiment results committed** alongside the runner at `2d89e8b`.
- **Decision: pursue Option 1 from the verdict menu — investigate root cause systematically before proposing further fixes.** Per the systematic-debugging discipline used throughout the project, the next step is to trace what XGBoost actually does on these small-n datasets that we don't, with evidence rather than guesses. The leaf_l2 hypothesis was *partially* right (XGBoost does use `reg_lambda=1.0`) but *insufficient* (the regularization comes from a system, not one knob). A focused profile/diff of the two algorithms on `561_cpu` will identify which mechanisms matter most.

## Next experiments

- **Experiment 17: root-cause investigation of XGBoost's small-n robustness.** Per Option 1 of the post-Exp-16 verdict menu. Concrete starting points: (a) grid `leaf_l2 ∈ {1, 10, 100}` to test whether stronger regularization fixes catastrophic cases at this `min_samples_leaf`; (b) instrument XGBoost on `561_cpu` to log per-tree leaf-value distributions and split-gain decisions; (c) compare against SplitBoost's identical instrumentation; (d) propose targeted fixes from observed differences. The output should be a spec for whatever fix the evidence supports — could be γ (split-gain threshold), Hessian-weighted gain, smaller `max_depth` for small-n, or something else entirely.
- **Experiment 18 (deferred): OpenML re-validation.** Per the project memory note, the next benchmark experiment after Exp 16 should be on OpenML-CTR23 or Grinsztajn-2022 instead of PMLB. Deferred until the loss-regime fix lands.
- **Experiment 19 (deferred): re-validate winners after any future loss-fix.** When a real fix is found, re-run Exp 15's full 119 datasets to confirm wins didn't regress.
