# Experiment 17: Full PMLB re-validation under truly-matched hyperparameters

**Date:** 2026-04-25
**Context:** Experiment 15 (full PMLB regression, 119 datasets) reported "89% within 10%, 83% outright wins, median ratio 0.912 vs XGBoost." Experiment 16 (validating the `leaf_l2=1.0` fix on the 20 Exp-15 losers) showed the fix was structurally inactive at SplitBoost's `min_samples_leaf=20` default — leaf-shrinkage by `n/(n+1) ≈ 0.95` is at most 5% magnitude reduction, well below the noise floor. A Phase-1 root-cause investigation (see `profile_loss_regime/investigate_xgb_vs_sb_smalln.py`) identified that `min_samples_leaf=20` was the load-bearing constraint: setting SplitBoost to `min_samples_leaf=1` (matching XGBoost's `min_child_weight=1` default) closes the catastrophic-loss gap on the 3 worst Exp-15 datasets to within ~3% of XGBoost on each. The corollary is that Exp 15 was NOT a fully-matched comparison — XGB ran at `min_child_weight=1` (default), `reg_lambda=1.0` (default), and no early stopping; SB ran at `min_samples_leaf=20`, `leaf_l2=0` (Exp 15's pre-fix state), and `patience=15` early stopping. Three of those five regularization axes were unmatched, biasing the comparison in different directions on different dataset regimes. This experiment re-runs the full 119-dataset suite under truly-matched hyperparameters across all three regularization axes that ARE matchable, documenting the LGB leaf-wise-vs-depth-wise growth-policy difference as an unavoidable algorithmic confound.

## Goal

Produce a methodologically defensible head-to-head comparison of SplitBoost vs XGBoost vs LightGBM on the full PMLB regression suite, where the three regularization axes that *can* be matched (leaf-floor, L2 on leaf weights, early stopping) ARE matched. The result is the project's new headline-quality baseline; the Exp 15 results stand as a "off-the-shelf defaults" comparison and will need a methodological-caveats note linking to Exp 17.

After this work, the project should be able to truthfully report: "Under matched leaf-floor, matched L2 leaf-weight regularization, and matched early-stopping policy, SplitBoost achieves [X]% wins vs matched XGBoost (median ratio [Y]) on PMLB regression." Whatever the actual numbers turn out to be, that statement will be defensible.

## Non-goals

- **No baseline tuning beyond the 3 matched axes.** No Optuna sweep, no per-dataset hyperparameter search. The goal is matched-hyperparameter rigor, not optimal-hyperparameter rigor.
- **No matching of LGB's growth policy.** LightGBM is fundamentally a leaf-wise (best-first) algorithm; there is no clean way to force depth-wise behavior. Documented in the report as a known confound.
- **No matching of EML.** The EML internal-split + leaf-EML mechanism is SplitBoost's intentional architectural differentiator. XGB and LGB don't have analogs; we don't add fake substitutes.
- **No new SplitBoost code beyond what already shipped in Tasks 1-7 of the leaf_l2 plan.** The default flip from `min_samples_leaf=20` to `1` IS a behavior change that lives in this experiment's runner (a SplitBoost call with `min_samples_leaf=1` explicit), not in the library defaults. **The `EmlSplitBoostRegressor()` library default remains `min_samples_leaf=20`** for backwards compatibility; this experiment validates whether changing that default would be wise.
- **No re-running Experiment 15.** Exp 15's 119 results stay on disk as the off-the-shelf baseline. Exp 17 produces a parallel set under matched hyperparameters.

## Design overview

### Matched axes (the three we CAN match)

| axis | SB (Exp 17) | XGB (Exp 17) | LGB (Exp 17) | Exp-15 difference |
|---|---|---|---|---|
| **1.** leaf-floor (sample-based) | `min_samples_leaf=1` | `min_child_weight=1` (existing default) | `min_data_in_leaf=1` (was 20) | SB & LGB had 20; XGB had 1 |
| **2.** L2 on leaf weights | `leaf_l2=1.0` | `reg_lambda=1.0` (existing default) | `reg_lambda=1.0` (was 0) | SB had 0; LGB had 0; XGB had 1 |
| **3.** early stopping | `patience=15, val_fraction=0.15` (existing) | `early_stopping_rounds=15` + 15% inner-val (NEW) | `early_stopping(stopping_rounds=15)` + 15% inner-val (NEW) | SB had it; XGB & LGB ran all 200 rounds |

### Unmatchable axes (documented as algorithmic differences)

| axis | SB | XGB | LGB | nature |
|---|---|---|---|---|
| **4.** growth policy | depth-wise (recursive) | depth-wise | leaf-wise (best-first) | algorithmic; not a hyperparameter |
| **5.** EML | enabled (10 internal-split candidates per node, 1-feature leaf-EML with cap_k=2.0) | n/a | n/a | SB's intentional differentiator |

The `report.md` for Exp 17 will note that LGB's leaf-wise growth and SB's EML are confounding factors that can't be neutralized in an apples-to-apples sense. Comparisons should be read with this in mind.

### Inner train/val split for early stopping

SplitBoost's `val_fraction=0.15` does an internal train/val split inside `EmlSplitBoostRegressor.fit()`, seeded by `random_state` (the outer seed). To match this for XGB and LGB, the runner does an explicit `train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed)` BEFORE passing to those `.fit()` calls — using the same `seed` as the outer 80/20 train/test split.

**Same-seed rationale:** with the same seed, the inner-val rows ARE THE SAME 15% of the original X_tr across all three algorithms. That eliminates a noise source — different algorithms aren't seeing different validation distributions for early stopping. The inner-val split happens AFTER the outer train/test split, so the test set is untouched and identical across algos.

The SB internal split MAY use a different RNG draw than `train_test_split` (different libraries' RNG implementations), so the inner val sets won't be byte-identical across SB vs XGB/LGB — but they'll be the same shape (15% of train), drawn from the same seed. Close enough for the matched-comparison goal; calling out as a minor caveat in the report.

## Implementation

### Runner: `experiments/run_experiment17_matched_revalidation.py`

Mirror `experiments/run_experiment15_full_pmlb.py` exactly:
- Same dataset list (`pmlb.regression_dataset_names`).
- Same per-fit reliability machinery (try/except per fit, `failures.json`, incremental CSV append, `_load_completed`).
- Same SEEDS = [0, 1, 2, 3, 4].
- Same TEST_SIZE = 0.20.
- Same outer `train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)`.

The three `_fit_*` functions change to apply the matched-hyperparameter conditions:

```python
INNER_VAL_FRACTION = 0.15
EARLY_STOPPING_PATIENCE = 15


def _fit_split_boost(X_tr, y_tr, seed):
    """SplitBoost matched: min_samples_leaf=1, leaf_l2=1.0,
    early stopping via internal val_fraction=0.15."""
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=1,                       # CHANGED from 20 (matched)
        leaf_l2=1.0,                              # explicit (Task-7 default; matched)
        n_eml_candidates=N_EML_CANDIDATES,
        k_eml=K_EML,
        n_bins=N_BINS,
        histogram_min_n=500,
        use_gpu=True,
        k_leaf_eml=K_LEAF_EML,
        min_samples_leaf_eml=MIN_SAMPLES_LEAF_EML,
        leaf_eml_gain_threshold=LEAF_EML_GAIN_THRESHOLD,
        leaf_eml_ridge=LEAF_EML_RIDGE,
        leaf_eml_cap_k=LEAF_EML_CAP_K,
        use_stacked_blend=False,
        patience=EARLY_STOPPING_PATIENCE,
        val_fraction=INNER_VAL_FRACTION,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _fit_xgb(X_tr, y_tr, seed):
    """XGBoost matched: min_child_weight=1 (existing default, made explicit),
    reg_lambda=1.0 (existing default, made explicit), early_stopping_rounds=15
    using the same-seed 15% inner val split."""
    start = time.time()
    X_inner_tr, X_inner_val, y_inner_tr, y_inner_val = train_test_split(
        X_tr, y_tr, test_size=INNER_VAL_FRACTION, random_state=seed,
    )
    m = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=MAX_DEPTH,
        n_estimators=MAX_ROUNDS,
        learning_rate=LEARNING_RATE,
        device="cuda",
        verbosity=0,
        min_child_weight=1,                       # explicit (existing default; matched)
        reg_lambda=1.0,                           # explicit (existing default; matched)
        early_stopping_rounds=EARLY_STOPPING_PATIENCE,  # NEW
        random_state=seed,
    )
    m.fit(
        X_inner_tr, y_inner_tr,
        eval_set=[(X_inner_val, y_inner_val)],
        verbose=False,
    )
    return m, time.time() - start


def _fit_lgb(X_tr, y_tr, seed):
    """LightGBM matched: min_data_in_leaf=1 (was 20), reg_lambda=1.0 (was 0),
    early_stopping_rounds=15 using the same-seed 15% inner val split."""
    start = time.time()
    X_inner_tr, X_inner_val, y_inner_tr, y_inner_val = train_test_split(
        X_tr, y_tr, test_size=INNER_VAL_FRACTION, random_state=seed,
    )
    train_set = lgb.Dataset(X_inner_tr, label=y_inner_tr)
    val_set = lgb.Dataset(X_inner_val, label=y_inner_val, reference=train_set)
    m = lgb.train(
        dict(
            objective="regression_l2",
            max_depth=MAX_DEPTH,
            num_leaves=2**MAX_DEPTH,
            min_data_in_leaf=1,                   # CHANGED from 20 (matched)
            reg_lambda=1.0,                       # NEW (was 0; matched)
            learning_rate=LEARNING_RATE,
            device="gpu",
            seed=seed,
            verbose=-1,
        ),
        train_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_PATIENCE, verbose=False),
        ],
    )
    return m, time.time() - start
```

Other module-level constants (`MAX_ROUNDS=200, MAX_DEPTH=8, LEARNING_RATE=0.1, N_EML_CANDIDATES=10, K_EML=3, K_LEAF_EML=1, MIN_SAMPLES_LEAF_EML=30, LEAF_EML_GAIN_THRESHOLD=0.05, LEAF_EML_RIDGE=0.0, LEAF_EML_CAP_K=2.0, N_BINS=256, TEST_SIZE=0.20, SEEDS=[0,1,2,3,4]`) carry over from `run_experiment15_full_pmlb.py` verbatim. Note: `MIN_SAMPLES_LEAF` constant from Exp 15 is no longer used directly — replaced inline with `min_samples_leaf=1` in `_fit_split_boost`.

### Outputs

`experiments/experiment17/`:
- `summary.csv` — per-fit rows (`dataset, seed, config, rmse, fit_time, n_rounds`); same schema as Exp 15.
- `summary.json` — per-dataset aggregates + headline_stats + ratios; mirrors Exp 15's structure.
- `pmlb_rmse.png` — sorted-bars + histogram (same code path as Exp 15's runner).
- `report.md` — narrative writeup including:
  - Headline numbers (within-10%, outright-wins, median ratio, catastrophic count) for SB-vs-XGB AND SB-vs-LGB.
  - **Per-dataset side-by-side with Exp 15** (Exp-15-ratio vs Exp-17-ratio per dataset, sorted by Δ).
  - Discussion of which datasets shifted most and why (likely small-n closing the gap on losses; medium/large-n shrinking the win margin).
  - Methodological-caveats section: LGB leaf-wise growth, SB EML — neither matchable.
- `failures.json` — per-fit failures (PMLB fetch failures expected to repeat from Exp 15).
- `comparison_to_exp15.md` — auto-generated table comparing this run's ratios to Exp 15's per dataset, similar to Exp 16's `comparison.md` but for all 119 datasets.

### Estimated runtime

- SB at `min_samples_leaf=1` builds richer trees → ~1.5-2× per fit on medium data.
- XGB and LGB with early stopping → likely 0.5-0.8× their Exp-15 fits (fewer rounds when overfit detected).
- Net: comparable to or slightly slower than Exp 15's 56 minutes.
- **Estimated total: 1.5-2.0 hours on RTX 3090.** Overnight-friendly. Per the user's "checkpoint before long runs" rule, the runner should be kicked off only after explicit user confirmation.

## Success criteria

This is a measurement experiment — not pass/fail. The headline numbers ARE the deliverable. But for the report writeup to be conclusive, three things must hold:

- **S-A (correctness):** all 100 unit tests still pass; no Triton fallback warnings during the run; `failures.json` only contains the same 3 PMLB stale-registry failures from Exp 15 (`195_auto_price`, `207_autoPrice`, `210_cloud`), no new fit failures.
- **S-B (catastrophic regime closes):** the 3 Exp-15 catastrophic datasets (`527_analcatdata_election2000`, `663_rabe_266`, `561_cpu`) all drop to ratio ≤ 1.5 vs matched-XGB. The Phase-1 isolation experiment showed C2 (SB msl=1, λ=1) ≈ C6 (XGB mcw=1, λ=1) on these three; if Exp 17 doesn't reproduce that, something in the cross-dataset re-run differs from the isolation result and needs investigation.
- **S-C (the LGB story is unchanged or tighter):** vs LGB-matched, SB's median ratio stays in [0.95, 1.05]. (Exp 15 was 0.994 vs unmatched LGB. Adding `reg_lambda=1.0` to LGB makes LGB more regularized, which should narrow LGB's variance — so SB's ratio vs LGB might shift but should stay near parity.)

If all three hold, the headline numbers are reportable as-is. If S-B fails, escalate before drawing conclusions — it would suggest the catastrophic loss isn't fully explained by msl-floor mismatch and there's another mechanism we missed.

## Risks

- **SB wins regress significantly.** This is the central risk and a possible result. If SB's "83% wins, 0.912 median" was largely the unmatched-defaults artifact (msl=20 acting as XGB-incompatible regularizer + early-stopping that XGB lacked), Exp 17 will reveal it. The result would be: "SB is competitive with matched-XGB (~50% wins, ~1.0 median) and ties matched-LGB; the 0.91-vs-XGB headline doesn't survive matched-hyperparameter scrutiny." That's the most honest possible story to ship and is fine if it's what the data shows.
- **Inner-val split RNG inconsistency between SB and XGB/LGB.** SplitBoost uses its internal `random_state`-seeded split; XGB/LGB use `train_test_split` with the same seed but a different RNG implementation. The inner-val ROWS may differ. Mitigated by being the same FRACTION (15%) of the same outer train set, but not byte-identical. Document in the report; treat as minor noise.
- **LGB at `min_data_in_leaf=1` may overfit catastrophically on small-n.** LGB's leaf-wise growth + tiny leaves + reg_lambda=1 may not be a stable combo on n < 100. If a few datasets blow up, they'll show up in `failures.json` or as catastrophic ratios — accept and note in report.
- **LGB at `reg_lambda=1.0` shifts wins/losses on medium-large data**. LGB's default `reg_lambda=0.0` means no leaf-weight regularization; adding it changes LGB's behavior on every dataset. SB-vs-LGB comparison shifts in unpredictable ways. Acceptable — that's part of the matched-hyperparameter point.
- **Runtime estimate may be off by ~50%.** SB at msl=1 could be slower than projected; early-stopping savings on XGB/LGB could be smaller than expected. If wall time exceeds 2.5h, the runner is still resume-safe — just kill and resume.

## Action on verdict

- **All criteria met (S-A + S-B + S-C):** ship the report.md as the new project headline. Update Exp 15's report.md with a methodological-caveats note pointing to Exp 17. Decide separately whether to flip the LIBRARY default `min_samples_leaf` from 20 to 1 (a separate spec; this experiment validates the runner-level change, not the library default).
- **S-A fails (test regressions or new fit failures):** investigate before drawing conclusions. Most likely culprit is a mismatch in the new `_fit_xgb` / `_fit_lgb` early-stopping wiring (e.g., XGB's `eval_set` requires both X and y in a tuple; LGB's `early_stopping` callback signature differs across versions). Fix and re-run.
- **S-B fails (catastrophic regime doesn't close):** the Phase-1 hypothesis is incomplete. Re-run the isolation experiment cross-checking against Exp 17 to identify what's different at scale. Possible additional mechanisms: subsample interactions, gradient-clipping, base_score initialization. Defer headline writeup until root cause confirmed.
- **S-C fails (SB vs LGB shifts unexpectedly):** likely an LGB-specific overfit at `min_data_in_leaf=1` on small-n, or `reg_lambda=1.0` overshooting on some dataset distribution. Acceptable — note in report; LGB-matching at axis 2 is itself a methodology change that may behave non-trivially.

## Files changed

**Created:**
- `experiments/run_experiment17_matched_revalidation.py` — runner.
- `experiments/experiment17/{summary.csv, summary.json, pmlb_rmse.png, report.md, failures.json, comparison_to_exp15.md}` — outputs.

**Modified:**
- `experiments/experiment15/report.md` — adds a methodological-caveats note pointing to Exp 17 (post-experiment edit, after Exp 17's results land).

**Unchanged:**
- The library code (`eml_boost/`).
- The unit-test suite.
- All prior experiments' artifacts.
- `experiments/run_experiment15_full_pmlb.py` (the off-the-shelf comparison stays as-is for reproducibility).
- `experiments/run_experiment16_leaf_l2_validation.py` and its outputs (the loser-only validation history stays).

## Naming consistency

Subsequent benchmark experiments after Exp 17 will be on OpenML-CTR23 / Grinsztajn-2022 (per the project memory note about deprecating PMLB after Exp 17). Exp 17 itself stays on PMLB because the comparison-to-Exp-15 requires the same dataset universe.
