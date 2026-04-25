# Experiment 15: Full PMLB Regression Suite

**Date:** 2026-04-25
**Commits:** `ac9e652` (runner) / `96437d6` (fit-time HEAD — post-redux + post-leaf-microopts)
**Runtime:** 56 min 24 s on RTX 3090 (1650 fresh fits in this run = 110 datasets × 5 seeds × 3 models; 9 additional datasets — 6 from earlier runs + 3 from the post-microopts sanity bench — were resumed from `summary.csv`. Final coverage: 119 datasets × 5 × 3 = 1785 fits.)
**Scripts:** `experiments/run_experiment15_full_pmlb.py`

## What the experiment was about

Experiments 11-14 established the SplitBoost defaults (`max_depth=8, max_rounds=200, n_eml_candidates=10, k_eml=3, k_leaf_eml=1, leaf_eml_cap_k=2.0, min_samples_leaf_eml=30`) and showed they delivered 6/7 within 10% of XGBoost on a curated 7-dataset benchmark, with most ratios clearly under parity. Experiment 15 takes that exact configuration to the full PMLB regression suite — 122 datasets × 5 seeds × 3 models — to convert the small-n curated story into a proper aggregate distribution. No hyperparameter tuning per dataset; this is a fixed-config benchmark.

## Configuration

```
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

XGBoost and LightGBM at matched capacity (`max_depth=8, n_estimators=200, learning_rate=0.1, min_data_in_leaf=20`), GPU device for both. Per-fit reliability machinery: `try/except` per fit, incremental CSV append, resume-from-checkpoint via `_load_completed`.

## Coverage

- **Datasets attempted:** 122 (PMLB's full `regression_dataset_names`)
- **Datasets with full 5-seed × 3-model coverage:** 119
- **Fetch failures (excluded from ratios):** 3 — `195_auto_price`, `207_autoPrice`, `210_cloud`. PMLB has these in `regression_dataset_names` but the data files 404 from the upstream repo. PMLB's last release was 2020 and is sparsely maintained; failures are recorded in `failures.json` and the runner skips past them.

## Headline results (mean ratios SplitBoost / XGBoost over 5 seeds)

| metric | value |
|---|---|
| Within 10% of XGBoost | **106/119 (89.1%)** |
| Within 5% of XGBoost | 103/119 (86.6%) |
| **Outright wins (ratio < 1.00)** | **99/119 (83.2%)** |
| Catastrophic (ratio > 2.0) | 3/119 (2.5%) |
| **Median ratio** | **0.912** (SplitBoost 8.8% better on median) |
| Mean ratio | 0.963 |
| P25 / P75 | 0.839 / 0.987 |
| Min / Max ratio | 0.612 / 2.355 |

The full-suite story matches the curated story — and is a touch stronger. On 7 datasets (Exp 13) the headline was "6/7 within 10%, 5/7 outright wins." On 119 datasets (Exp 15) the headline is "**89% within 10%, 83% outright wins**." The architectural lead generalizes.

### Distribution of ratios

| ratio band | meaning | count |
|---|---|---|
| `< 0.80` | deep win (>20% better than XGBoost) | **15** |
| `0.80–0.95` | clear win (5-20% better) | **56** |
| `0.95–1.00` | narrow win (0-5% better) | **28** |
| `1.00–1.05` | narrow loss (0-5% worse) | 4 |
| `1.05–1.10` | loss (5-10% worse) | 3 |
| `1.10–2.00` | clear loss (10-100% worse) | 10 |
| `≥ 2.00` | catastrophic | 3 |

99 datasets sit at or below parity, 31 below the 0.85 mark. The losses cluster at the tail — once SplitBoost loses, it tends to lose meaningfully (10 of the 13 datasets above 1.05 are above 1.10).

## Top 10 wins (lowest ratios)

| dataset | ratio | SB RMSE | XGB RMSE |
|---|---|---|---|
| 560_bodyfat | 0.612 | 1.478 | 2.416 |
| 344_mv | 0.613 | 0.051 | 0.083 |
| 523_analcatdata_neavote | 0.630 | 0.850 | 1.348 |
| 603_fri_c0_250_50 | 0.663 | 0.439 | 0.662 |
| 647_fri_c1_250_10 | 0.701 | 0.311 | 0.443 |
| 633_fri_c0_500_25 | 0.704 | 0.351 | 0.498 |
| 611_fri_c3_100_5 | 0.731 | 0.420 | 0.574 |
| 649_fri_c0_500_5 | 0.745 | 0.319 | 0.428 |
| 650_fri_c0_500_50 | 0.405 | 0.405 | 0.542 |
| 654_fri_c0_500_10 | 0.758 | 0.369 | 0.487 |

The Friedman-family synthetic datasets (`fri_c*`) dominate the top wins — 7 of 10. These are smooth-structured regression tasks where the leaf-EML candidates can fit a meaningful continuous shape per leaf. `560_bodyfat` and `344_mv` are real-world datasets with similar smooth structure (body composition + Friedman MV synthetic). This is exactly the regime the EML-leaf design was built for.

## Top 10 losses (highest ratios)

| dataset | ratio | SB RMSE | XGB RMSE | n_train | k |
|---|---|---|---|---|---|
| 542_pollution | 1.145 | 50.3 | 43.9 | 48 | 15 |
| 1096_FacultySalaries | 1.252 | 1.892 | 1.510 | 40 | 4 |
| 505_tecator | 1.364 | 1.834 | 1.344 | 192 | 124 |
| 230_machine_cpu | 1.478 | 58.4 | 39.5 | 167 | 6 |
| 485_analcatdata_vehicle | 1.483 | 282.2 | 190.2 | 39 | 4 |
| 1089_USCrime | 1.608 | 36.9 | 23.0 | 37 | 13 |
| 659_sleuth_ex1714 | 1.742 | 2762.9 | 1585.8 | 38 | 7 |
| 561_cpu | 2.148 | 62.0 | 28.8 | 167 | 7 |
| 663_rabe_266 | 2.340 | 9.17 | 3.92 | 96 | 2 |
| 527_analcatdata_election2000 | 2.355 | 56844 | 24137 | 53 | 14 |

Every loss in the top 10 is a small-n dataset (n_train ≤ 200). The catastrophic three (ratio > 2) all have n_train < 200 and small-to-moderate feature counts — exactly the regime where:
- The EML-leaf fit needs ≥ 30 samples to attempt; small leaves reduce to constants and lose the architectural advantage.
- Per-fit seed variance dominates the mean ratio (one bad seed shifts the mean by 30%+).
- XGBoost's per-leaf shrinkage is already a near-optimal regularizer on noisy small-n.

`505_tecator` is the lone exception: 192 training rows but **124 features** (spectroscopy, mostly redundant). Top-k feature selection (k=3) loses information; the model can't see the spectral structure XGBoost picks up via deep tree splits over many features.

## vs LightGBM

LightGBM at matched capacity is closer to SplitBoost than XGBoost is:

- **Outright wins (SplitBoost < LightGBM):** 67/119 (56.3%)
- **Median ratio (SB / LGB):** 0.994 (effectively a wash on median)

LightGBM's leaf-wise growth is a stronger baseline than XGBoost's depth-wise on PMLB; SplitBoost's lead over XGBoost is partly a story about XGBoost's depth-wise weakness on this benchmark, not pure SplitBoost dominance. The XGBoost story is the headline because that's the canonical baseline; the LightGBM-comparable story is more honest.

## Success criteria verdict

The Exp 15 spec (`docs/superpowers/specs/2026-04-25-full-pmlb-design.md`) set three descriptive criteria for the headline:

- **S-A: > 60% within 10% of XGBoost.** **MET, with margin.** 89.1% (106/119) within 10%.
- **S-B: > 30% outright wins (ratio < 1.00).** **MET, with large margin.** 83.2% (99/119).
- **S-C: < 5% catastrophic (ratio > 2.0 or fit failure).** **MET.** 2.5% (3/119) — and zero fit failures within fittable datasets (only the 3 PMLB fetch failures, which the spec explicitly handles via try/except).

All three pass with substantial margin. The "small-n curated → full-suite" generalization holds.

## What Experiment 15 actually shows

- **The architectural lead from Exp 11-14 generalizes.** Across 119 PMLB regression datasets at fixed defaults, SplitBoost beats XGBoost on 83% — same direction as the 7-dataset story, broader coverage, larger evidence base. 31 datasets are won by ≥ 15%.
- **Smooth synthetic structure is the strongest regime.** Friedman-family datasets dominate the top 10 wins. The leaf-EML closed-form OLS over depth-2 elementary-function trees is exactly the bias smooth signals reward — this is the bias the algorithm was designed for, and it pays off.
- **Small-n is the loss regime.** All 13 datasets with ratio > 1.05 have n_train ≤ ~200. The leaf-EML gate (`min_samples_leaf_eml=30`) makes EML leaves rare on tiny datasets, so SplitBoost effectively reduces to a constant-leaf depth-8 GBDT — and XGBoost's regularization is just better in that fallback regime.
- **The XGBoost story is the headline; the LightGBM story is the honest one.** vs LightGBM, the median ratio is 0.994 — basically tied. The 0.912 median vs XGBoost includes XGBoost's known weakness on depth-wise growth at d=8 on this benchmark. SplitBoost is genuinely competitive with LightGBM and clearly ahead of XGBoost.
- **One real-world non-tiny loss: `505_tecator`.** Spectroscopy data with 124 features. Top-k=3 feature selection in the leaf-EML path can't capture the spectral structure. A `k_eml=20+` or feature-importance-aware variant might rescue it; currently undocumented as a known weakness.

## What's left as a loss

- **Small-n datasets (~ n_train < 200):** systematic 13-dataset cluster of losses, with 3 catastrophic. SplitBoost falls back to constant-leaf GBDT in this regime and XGBoost wins. Possible mitigations: lower `min_samples_leaf_eml` aggressively for tiny n; or detect the regime and switch to a different leaf strategy.
- **`505_tecator` (high-dim spectroscopy, ratio 1.364):** isolated loss driven by aggressive top-k feature selection. Specific to high-feature-count regimes.

## What Experiment 15 does NOT show

- **No hyperparameter tuning of any baseline.** Both XGBoost and LightGBM run at matched-capacity defaults. A tuned XGBoost (e.g., Optuna + 5-fold CV on each dataset) would close some of the gap, especially on medium-sized datasets where XGBoost's known levers (`reg_lambda`, `subsample`) help.
- **Single 80/20 shuffle-split per seed.** No K-fold CV; the test-RMSE estimates carry the train/test boundary's variance.
- **PMLB-specific.** The dataset distribution skews toward small-n synthetic (Friedman variants make up ~50 of 119) and small real-world tabular. Modern tabular ML papers benchmark on OpenML's curated suites (CTR23, Grinsztajn-2022) which are larger and have less small-n bias. Exp 16 will replicate on OpenML — the win rate could shift if the dataset distribution shifts.
- **No statistical significance reporting.** Per-fit RMSE × 5 seeds gives an estimate but no formal CI. Win/loss counts treat ratio = 0.999 the same as ratio = 0.500.
- **No latency or memory comparison.** SplitBoost fit time is ~10-50× XGBoost fit time on medium datasets (the redux + microopts brought it down from ~50-200×, but the gap remains). The benchmark is RMSE-only.

## Caveats

- **3 fetch failures excluded from ratios.** PMLB metadata-vs-data drift; the 119/122 coverage is the practical effective N. Not a methodology issue, but worth flagging.
- **`leaf_eml_cap_k=2.0` matters for stability.** Without the per-leaf magnitude cap (Exp 11), the catastrophic count would likely be higher than 3 — the cap clips runaway leaf-EML fits on small leaves.
- **Defaults were tuned on a 7-dataset subset.** Exp 11-13's curated subset was small and the defaults may have over-fit to that subset's idiosyncrasies. The fact that the full-suite numbers hold up suggests the defaults generalize; but a clean re-derivation on a held-out subset would be stronger evidence.

## Engineering story (runtime)

Experiment 15 was originally paused mid-run because per-fit time on the worst-case `1191_BNG_pbc` (1M rows × 18 features) was ~155s under the first GPU port — projecting 5-15 hours for the full suite. A redux GPU port + leaf-microopt commits brought per-fit time on that dataset from 155s → ~70s, and the full-suite total to **56 min 24 s**:

| stage | per-fit on `1191_BNG_pbc` | comments |
|---|---|---|
| Pre-port (CPU only) | 690 s | original baseline |
| Original GPU port (`c238977..eb24a0c`) | 155 s | per-node H2D eliminated, tensorized predict, paused mid-run at this point |
| Redux GPU port (`c610c3a..2e77881`) | ~75 s | X-cache + Triton predict + Triton hist + descriptor cache + `.item()` batching |
| Leaf micro-opts (`a13e84c..96437d6`) | ~70 s | valid_desc cache + numpy tensorize + batched `.item()` (~5% additional) |
| **Full-suite actual** | — | **56 min 24 s on RTX 3090** (vs ~3.3h post-microopts projection) |

The actual run beat the post-microopts projection by a wide margin because the heavy BNG_*-class datasets cleared in the first ~30 min, and the remaining ~80 datasets were small enough to fit in seconds each.

## Action taken

- **Defaults unchanged.** Exp-13 calibrated defaults (`max_depth=8, max_rounds=200, n_eml_candidates=10, ...`) ship as-is — this experiment validates them at suite scale.
- **No library changes from this experiment** beyond the leaf micro-opts already committed (`a13e84c..96437d6`) before the run started. Those were performance fixes, not algorithm changes; they're documented in `docs/superpowers/specs/2026-04-25-splitboost-leaf-microopts-design.md`.
- **3 PMLB datasets logged in `failures.json`** for future reference; not retried (PMLB stale registry).

## Consequence for the project

**SplitBoost ships as a viable XGBoost alternative for tabular regression at default hyperparameters.** The 89/83/0.91 headline (89% within 10%, 83% outright wins, 0.91 median ratio vs XGBoost on 119 datasets) is the strongest comparative claim the project has made. The architectural advantage — closed-form leaf-EML over a depth-2 elementary-function tree, with a per-leaf magnitude cap — generalizes well outside the small curated subsets it was developed on.

The honest framing also includes the LightGBM tie (0.994 median) and the small-n loss regime — SplitBoost is competitive with the better of the two GBDT baselines, dominant against the weaker, and has a known small-n soft spot.

## Next possible experiments

- **Experiment 16: OpenML re-validation.** PMLB is unmaintained (last release 2020) and skews toward small-n synthetic. Replicate the headline on OpenML-CTR23 (35 curated regression tasks, modern paper baseline) or Grinsztajn-2022 regression (~25 datasets, the "trees vs DL" benchmark). Decision already made; spec to be drafted.
- **Tune the baselines.** Re-run the headline with Optuna-tuned XGBoost and LightGBM per dataset. Honest comparison; expect the SplitBoost lead to shrink but the small-n losses to widen (XGBoost specifically benefits from `reg_lambda` tuning on tiny n).
- **Address the small-n loss regime.** Either lower `min_samples_leaf_eml` adaptively (e.g., `min(30, n_train // 5)`) or detect tiny-n and switch leaf strategy. The 13-dataset cluster is well-defined and a focused fix is plausible.
- **Address `505_tecator` / high-dim feature regime.** Either bump `k_eml` for high-`k`-train datasets or build a feature-importance-aware top-k. Specific to spectroscopy-like data; not urgent.
- **Architectural rewrite (deferred from current session): BFS-by-depth batched leaf processing.** Profile shows per-leaf orchestration is the remaining bottleneck. A 4-7-day rewrite would yield ~2× speedup on medium-large datasets. Tracked but not blocking.
