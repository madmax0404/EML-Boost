# Experiment 7: Real-Tabular Benchmark on PMLB with Matched Capacity

**Date:** 2026-04-24
**Commit:** post-Experiment 6 + Triton GPU kernel integration + standardization restored in the exhaustive path
**Runtime:** ~2 min total for 7 datasets on RTX 3090
**Scripts:** `experiments/run_experiment7_pmlb.py`

## What the experiment was about

Experiments 1-6 all ran on synthetic data. Experiment 7 is the first apples-to-apples real-tabular test against actual industry baselines. Seven PMLB regression datasets spanning tiny-to-large (52 → 40,768 samples; 2 → 12 features), all models at matched capacity:

| knob | value |
|---|---|
| `max_rounds` / `n_estimators` | 200 |
| `max_depth` / `depth_dt` | 6 |
| `learning_rate` | 0.1 |
| `patience` | disabled |
| `seed` | 0 |
| `device` | cuda (XGBoost) / gpu (LightGBM) / cuda+cpu (Hybrid via Triton + sympy) |

Hybrid's EML branch uses depth-2 exhaustive search with `k=3` (or `min(3, n_features)`) features selected per round by correlation.

## Infrastructure changes along the way

Three implementation issues surfaced during this experiment and got fixed:

1. **Triton GPU kernel for exhaustive search.** The original CPU sympy path evaluated ~6,400 candidate trees per round via `sp.lambdify` + numpy; at 200 rounds × 7 datasets the ETA was ~55 hours. Rewriting the inner search as a fused Triton kernel (one kernel launch evaluates all 6,400 trees on all samples, reads terminal choices from a precomputed descriptor tensor) drops it to ~60 ms per round. Full Experiment 7 now finishes in ~2 minutes. See `eml_boost/_triton_exhaustive.py` and `tests/unit/test_triton_exhaustive.py` (the torch evaluator is kept as a correctness oracle).

2. **Feature standardization restored in the exhaustive path.** Experiment 4 had removed standardization to fix extrapolation correctness on synthetic `exp(x_0)` data. Real PMLB features have wildly varying ranges (pollen: [−34, +35]; cpu_small: up to 2.2M). Without standardization, `exp(35)` produced finite-but-astronomical tree outputs that utterly dominated the fit — pollen hybrid RMSE was **491.7 vs XGBoost 1.6** (300× loss). Re-enabling standardization on the exhaustive path with the un-standardization substitution applied only to the stored-for-display `formula` (not the numeric `predict` path) brought pollen from 491.7 to 3.0. The extrapolation test from Experiments 4 and 6 now needs its own `standardize=False` mode.

3. **Log-argument clamp.** After standardization, half of `x_i` values are negative. The grammar's `eml(a, b) = exp(a) − log(b)` produces NaN when `b` is non-positive, and `torch.isfinite(...).all(dim=1)` would filter such trees entirely — which could remove every non-constant tree on standardized data and cause "no valid tree" errors. Clamping `log(max(x, 1e-6))` means such trees score badly via MSE but don't nuke the search. Resolved most but not all failures (see `210_cloud` below).

## Results

| dataset | n | k | Hybrid | LightGBM | XGBoost | hybrid / XGBoost | verdict |
|---|---|---|---|---|---|---|---|
| 192_vineyard | 52 | 2 | **1.97** | 2.39 | 3.03 | **0.65** | **Hybrid wins by 35%** |
| 523_analcatdata_neavote | 100 | 2 | 0.74 | **0.72** | 0.77 | 0.96 | within 10% |
| 210_cloud | 108 | 5 | FAIL | 0.77 | **0.65** | — | hybrid errored out |
| 557_analcatdata_apnea1 | 475 | 3 | 1463.70 | 1375.56 | **1137.82** | 1.29 | loses by 29% |
| 529_pollen | 3848 | 4 | 3.00 | **1.60** | 1.61 | 1.86 | loses by 86% |
| 562_cpu_small | 8192 | 12 | 4.87 | 3.07 | **3.09** | 1.58 | loses by 58% |
| 564_fried | 40768 | 10 | 1.77 | **1.07** | 1.08 | 1.64 | loses by 64% |

**Within 10% of XGBoost: 2 / 7 datasets** (both n ≤ 100).

## What v1 actually shows

- **The hybrid is competitive on small datasets with few features.** On `192_vineyard` (n=52, k=2) it beats XGBoost by 35%; on `analcatdata_neavote` (n=100, k=2) it reaches parity. In this regime, 200 depth-6 stumps over-expressive the data; the hybrid's closed-form elementary approximations are a better fit at matched capacity.
- **XGBoost wins on medium-to-large tabular.** Starting at `n=475`, XGBoost consistently beats the hybrid by 30-90%. At `n=40768` the gap is 64%. The hybrid's depth-2 grammar fundamentally caps the per-round expressiveness; XGBoost's depth-6 trees with 64 leaves are structurally richer.
- **LightGBM and XGBoost behave near-identically across the board** at matched capacity — confirming Experiment 3's v2 finding that XGBoost's "industry standard" status is about matching capacity, not the specific library.

Summary table against spec 9.3:

| Spec 9.3 must-have | Target | Experiment 7 result |
|---|---|---|
| #2: within 10% of XGBoost averaged on PMLB | 100% of datasets within 10% | **2 / 7 (28%)** — FAIL at matched capacity |

## Why the hybrid loses at scale

Two structural reasons:

1. **Capacity cap at depth 2.** The hybrid's EML branch is fixed at depth 2 because that's where exhaustive search is feasible (k=3 → 6,400 trees; k=4 → 22,500; k=5 → 63,504). A depth-2 EML tree has roughly the expressiveness of `exp(linear combo) − log(linear combo)` — fine for single-feature or simple bivariate targets, starved for higher-order interactions. Real datasets with non-trivial feature interactions (cpu_small: 12 features, fried: 10 features) need higher-order composition.

2. **Top-k feature selection bottleneck.** At each boosting round, only 3 features enter the EML branch (the top 3 by correlation with the current residual). On `562_cpu_small` with 12 features and genuine multi-feature interactions, 3-of-12 feature selection throws away signal.

The DT branch of the hybrid runs on all features at depth 6 like the baselines, so the hybrid's DT contributions alone should match LightGBM's, roughly. But BIC's per-round selection biases against DT rounds in favor of EML rounds once EML gets foothold — the hybrid under-uses its own strong branch in the high-k regime.

## The 210_cloud failure

Still diagnostic-pending. After standardization + log-clamp, every non-constant tree on `210_cloud`'s first residual somehow produces non-finite output per the `_exhaustive_search_gpu` filter. Log-clamp ought to prevent this. Likely suspects:
- Dataset-specific feature distribution (some column with near-zero std → z-score inflation).
- An interaction between the feature-mask filter and the log-clamp that leaves only constant trees passing.

Treated as a known issue for v2; the hybrid falls back gracefully (raises, outer code could catch and substitute DT-only prediction).

## What v7 does NOT show

- Does **not** test capacity-unlocked hybrids. If we bumped the hybrid's `depth_dt=10` and `max_rounds=500`, its DT branch would outperform its own EML branch on high-n tabular, and the per-round BIC would start preferring DT on more rounds — possibly closing the gap to pure XGBoost. Worth its own experiment.
- Does **not** test a regime-selector heuristic (e.g., "don't use the hybrid when k > 4 or n > 1000"). That would let the hybrid win by scope rather than by capability on every dataset.
- Does **not** test cross-validation or multiple seeds — the reported RMSE is a single 80/20 split at seed=0. Error bars are unknown.
- Does **not** report training RMSE — only test RMSE. In-sample fit quality could be revealing if the hybrid is overfitting (unlikely with its limited depth-2 expressiveness) or under-fitting (likely on high-k data).

## Reproducing these results

```bash
uv run python experiments/run_experiment7_pmlb.py
```

Runtime ~2 min on CPU+GPU (hybrid under 1 min per dataset thanks to Triton, most time on `564_fried` at 40k rows).

## Consequences for the project

- **Spec 9.3 must-have #2 is not satisfied at matched capacity.** The paper needs to either (a) relax the claim from "within 10% of XGBoost" to "within 10% on small-n small-k data," (b) implement a capacity-unlocked hybrid and revisit, or (c) pivot the paper's positioning away from "drop-in XGBoost replacement" toward "closed-form recovery when small data has elementary structure."
- **The Triton kernel is a permanent infrastructure improvement.** `eml_boost/_triton_exhaustive.py` turns the exhaustive search from a 55-hour projection into a 2-minute run. This unblocks any scale-up experiment (PMLB full 55-dataset suite, multiple seeds, cross-validation, larger k).
- **Standardization restored as default.** The un-standardization substitution preserves Experiment 4's extrapolation-correct formulas by expressing them in original coordinates via sympy substitution; the numeric predict path still uses standardized features. Experiments 4 and 6 need to opt out with `standardize=False` to reproduce their headline extrapolation numbers.

## Next possible experiments

- **Capacity-unlocked hybrid vs XGBoost**: bump `depth_dt` to 10 and `max_rounds` to 500 on the hybrid, leave XGBoost at its common-production config. Tests whether the hybrid's DT branch can carry the weight on high-k tabular.
- **Debug the `210_cloud` failure.** Instrument the exhaustive search to log where trees get filtered; a small fix here could also catch other edge-case datasets.
- **PMLB full regression suite (55 datasets)** at matched capacity. Turn the 2/7 ratio into a meaningful aggregate statistic with error bars across seeds.
- **Capacity-matched small-data benchmarks.** The hybrid's win on `vineyard` and near-win on `neavote` suggest a "small-data scientific" regime where it dominates. Curated benchmarks in that regime (UCI small regression, synthetic scientific sensors) could produce a defensible "use hybrid when n ≤ 200 and k ≤ 3" pitch.
- **Regime-selector heuristic**: auto-detect whether to dispatch to hybrid or plain XGBoost based on dataset stats (n, k, numeric-vs-categorical ratio).
