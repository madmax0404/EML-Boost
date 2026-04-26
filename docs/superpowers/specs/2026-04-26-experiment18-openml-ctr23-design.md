# Experiment 18: SplitBoost benchmark on OpenML-CTR23

**Date:** 2026-04-26
**Context:** Experiment 17 produced a methodologically defensible matched-hyperparameter
comparison of SplitBoost vs XGBoost vs LightGBM on the full PMLB regression suite (119
datasets), landing at 78% wins / 0% catastrophic / 0.97 median ratio. A post-hoc 20-seed
re-validation revealed the actual win rate is ~90%+ — the 5-seed protocol was
underpowered for tiny-n datasets, hiding wins as noisy "losses." Only `banana` is a
real algorithmic loss, and a targeted fix doesn't generalize across other low-k datasets.
Per the project memory note, the next benchmark experiment moves off PMLB (last released
2020, sparsely maintained, registry has unfetchable datasets, skews small-n synthetic)
to OpenML — specifically OpenML-CTR23 (Curated Tabular Regression 2023), the modern
35-dataset standard reference for tabular regression benchmarks. This experiment also
flips the SplitBoost library default `min_samples_leaf` from 20 to 1 to match the Exp-17
runner-level setting that produced the matched-comparison results.

## Goal

Produce SplitBoost's first cross-suite benchmark result on a modern, actively-maintained
benchmark (OpenML-CTR23). The headline measures whether the Exp-17 matched-hyperparameter
story (78% wins / 0% catastrophic / 0.97 median vs XGB) holds on a less small-n-skewed
dataset distribution, AND whether the now-flipped library default (`min_samples_leaf=1`)
produces a competitive off-the-shelf experience without any per-experiment runner overrides.

After this work, the project should be able to truthfully report:
- "On OpenML-CTR23 (35 modern regression tasks), SB-default (msl=1, leaf_l2=1) achieves
  [X]% wins / [Y]% within 10% / [Z] median ratio vs matched-XGB and -LGB."
- The library defaults are now `min_samples_leaf=1`, `leaf_l2=1.0` — Exp 17's runner-level
  overrides are now the off-the-shelf experience.

## Non-goals

- **No baseline tuning beyond the matched axes.** Same as Exp 17: leaf-floor=1 + L2=1.0 +
  early-stopping=patience-15-with-15%-inner-val. No Optuna sweeps, no per-dataset HP search.
- **No comparison to PMLB Exp-17 results dataset-by-dataset.** OpenML-CTR23 and PMLB
  datasets do not overlap in name; cross-dataset comparison is meaningless. Aggregate
  shape comparison (median ratio, win count) IS the comparison.
- **No PMLB re-runs.** PMLB is documented as deprecated for benchmark purposes per the
  project memory; Exp 17 stands as the final PMLB headline.
- **No 20-seed protocol** in this experiment. Per user direction, 5 seeds is sufficient
  for OpenML-CTR23 (datasets skew larger; 5-seed noise floor is much lower than on PMLB
  small-n). The 20-seed methodology recommendation from Exp 17 applies to small-n claims
  specifically; it doesn't auto-apply to medium/large datasets.
- **No `banana` follow-up.** Documented as a known weakness in Exp 17; banana isn't part
  of OpenML-CTR23.

## Design overview

### Pre-step: flip library default `min_samples_leaf` from 20 to 1

In `eml_boost/tree_split/tree.py` and `eml_boost/tree_split/ensemble.py`:

```python
# was:
min_samples_leaf: int = 20,
# now:
min_samples_leaf: int = 1,                # was 20; flipped post-Exp-17 to match the matched-comparison setting
```

This is a user-visible behavior change, but per direction "the repo is ~2 days old, in
development, no external users" — no migration burden. The flip is justified by Exp 17's
empirical evidence: at `msl=1, leaf_l2=1` (matched XGB), SB closed all 3 catastrophic
losses to outright wins and the win rate stayed at 78% (vs 83% off-the-shelf, but with
zero structural failures). The post-hoc 20-seed analysis suggests the actual win rate
under proper measurement is 90%+.

Existing unit tests that hardcoded `min_samples_leaf=20` may need their expected values
adjusted. Implementation plan triages each.

### Experiment configuration

| component | value |
|---|---|
| Dataset universe | **OpenML-CTR23** — 35 curated regression tasks via `openml.study.get_suite('OpenML-CTR23')` |
| Seeds | 5 (range(5)) |
| Outer split | 80/20 train/test, `random_state=seed` |
| SB hyperparams | All library defaults (post-flip): `max_depth=8, max_rounds=200, learning_rate=0.1, min_samples_leaf=1, leaf_l2=1.0, n_eml_candidates=10, k_eml=3, k_leaf_eml=1, min_samples_leaf_eml=30, leaf_eml_gain_threshold=0.05, leaf_eml_ridge=0.0, leaf_eml_cap_k=2.0, n_bins=256, histogram_min_n=500, use_gpu=True, use_stacked_blend=False, patience=15, val_fraction=0.15` |
| XGB matched | `objective="reg:squarederror", max_depth=8, n_estimators=200, learning_rate=0.1, device="cuda", min_child_weight=1` (default), `reg_lambda=1.0` (default), `early_stopping_rounds=15` with 15% inner-val (`train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed)`) |
| LGB matched | `objective="regression_l2", max_depth=8, num_leaves=2**8, min_data_in_leaf=1` (was 20 in PMLB), `reg_lambda=1.0` (was 0), `learning_rate=0.1`, `device="gpu"`, `early_stopping(stopping_rounds=15)` callback with same 15% inner-val |

### Runner

`experiments/run_experiment18_openml_ctr23.py`. Mirrors `run_experiment17_matched_revalidation.py`
verbatim except:

1. Dataset loader switches from `pmlb.fetch_data` + `regression_dataset_names` to
   `openml.study.get_suite('OpenML-CTR23')` + `openml.datasets.get_dataset(task.dataset_id).get_data(target=...)`.
2. SB call passes NO explicit `min_samples_leaf` or `leaf_l2` (uses post-flip library defaults).
   This is the key user-facing demonstration: the off-the-shelf SB call is now competitive.
3. `RESULTS_DIR` → `experiment18/`.
4. No `comparison_to_exp17.md` generation (different dataset universe; meaningless to compare).
5. Plot renamed `openml_rmse.png` (descriptive of the suite).

Per-fit reliability machinery (try/except, `_load_completed`, `_append_rows`,
`failures.json`, resume-from-checkpoint) is preserved verbatim from Exp 17.

### Outputs

`experiments/experiment18/`:
- `summary.csv` — per-fit rows (`dataset, seed, config, rmse, fit_time, n_rounds`).
- `summary.json` — per-dataset aggregates + headline_stats + ratios; mirrors Exp-17 structure.
- `openml_rmse.png` — sorted-bars + histogram (same code path as Exp-17's runner).
- `failures.json` — per-fit failures (expected to be empty given OpenML's better data
  hygiene vs PMLB).
- `report.md` — narrative writeup (Task 4 of the plan).

### Estimated runtime

OpenML-CTR23 has 35 datasets, sized from ~500 to ~50k rows (a few larger). SB at msl=1
is somewhat slower than at msl=20 (more leaves per tree). XGB and LGB with early stopping
are typically fast.

Rough estimate: **1-2.5 hours on RTX 3090**. Overnight-friendly, but should checkpoint
with the user before kicking off (per the user's "checkpoint before long runs" rule).

## Success criteria

This is a measurement experiment — the headline numbers ARE the deliverable. For the
result to be reportable as the new project benchmark:

- **S-A (correctness):** all unit tests pass after the library default flip; the runner
  completes without unexpected failures (a small number of OpenML fetch failures is OK
  if any datasets are temporarily unavailable, mirrored as Exp-15-style fetch-failure
  handling).
- **S-B (matched-comparison story holds):** SB-vs-matched-XGB median ratio is in
  `[0.85, 1.05]` (Exp-17 was 0.97; OpenML may shift it). If it's WORSE than 1.05,
  investigate before publishing — could indicate OpenML-CTR23 is harder for SB or
  exposes a new loss regime.
- **S-C (catastrophic regime):** ≤ 3% catastrophic ratios > 2.0 (Exp-17 was 0%; the
  banana case wouldn't repeat since banana isn't in CTR23, but other PMLB-absent
  datasets could surface new catastrophic cases).
- **S-D (no Triton fallback):** clean `run.log` with no `RuntimeWarning` or unexpected
  fallback messages.

If S-A + S-D pass and S-B + S-C are within their bands, the headline numbers are
reportable as-is. If S-B or S-C surface unexpected behavior, document and investigate
in the report.md without blocking the experiment as failed.

## Risks

- **OpenML-CTR23 access requires network and possibly an API key.** The `openml`
  package fetches dataset metadata over HTTP. If the OpenML server is down or rate-limits
  us, the run could partially fail. Mitigation: existing try/except per dataset; the
  runner can resume after re-fetching.
- **Some CTR23 datasets may have categorical features that need encoding.** PMLB pre-
  encodes everything as float; OpenML provides raw mixed-type data. Mitigation: the
  runner converts to float64 via `np.asarray` and skips datasets that can't be cleanly
  numeric. Document the count in the report.
- **CTR23 dataset distribution differs from PMLB** in ways that may shift SB's win
  profile. CTR23 has fewer tiny-n synthetics, more medium/large real-world tasks. SB's
  EML mechanism showed strongest wins on Friedman synthetics in PMLB; if CTR23 has fewer
  smooth-signal datasets, SB's median win margin may shrink. Acceptable — that's the
  point of testing on a different distribution.
- **Library default flip may break existing tests** that hardcode `min_samples_leaf=20`
  expectations. Mitigation: implementation plan includes a test triage step that
  re-runs the unit suite after the flip and updates any breaking assertions (without
  weakening test intent).
- **Runtime overrun.** Estimate 1-2.5h is best-effort; could go to 4h on the larger
  CTR23 datasets. Resume-from-checkpoint mitigates: if killed, re-run picks up from
  `summary.csv`.

## Action on verdict

- **All criteria met:** ship the headline as the new project standard. Update relevant
  `current_state.md` (or equivalent) to reflect that the OpenML-CTR23 benchmark is the
  current reference. Library defaults remain at `msl=1, leaf_l2=1` (the post-Exp-18 flip).
- **S-A fails (test regressions):** triage individually. Most likely cause is a hardcoded
  `min_samples_leaf=20` expectation in a smoke test. Widen or update the assertion;
  don't revert the library flip.
- **S-B fails (matched comparison shifts unexpectedly):** investigate before drawing
  conclusions. Possible causes: CTR23 has a category of datasets SB hasn't seen
  (e.g., very high-dim, or specific time-series-like structures); or the Exp-17 78%
  win rate was overstated and CTR23 reveals it. Document in the report; possibly queue
  an Exp-19 deeper dive.
- **S-C fails (new catastrophic cases surface):** identify which dataset(s) and apply
  the same systematic-debugging discipline used for Exp-15's catastrophic regime
  investigation. Document; queue follow-up.
- **S-D fails (Triton fallback fires):** investigate the kernel; XGB/LGB are unaffected
  but SB's GPU path needs to stay clean. Probably an OpenML dataset shape edge case
  (e.g., k > 500 features triggers a kernel-size limit).

## Files changed

**Created:**
- `experiments/run_experiment18_openml_ctr23.py` — runner.
- `experiments/experiment18/{summary.csv, summary.json, openml_rmse.png, failures.json, report.md}` — outputs.

**Modified:**
- `eml_boost/tree_split/tree.py` — `min_samples_leaf: int = 20` → `1` in `EmlSplitTreeRegressor.__init__`.
- `eml_boost/tree_split/ensemble.py` — same change in `EmlSplitBoostRegressor.__init__`.
- `tests/unit/test_eml_split_*.py` — possibly: widen any assertions that depended on
  msl=20 specifically. Triage in implementation plan; only change if necessary.

**Unchanged:**
- The Triton kernels.
- The CPU pipeline.
- The Exp-15, Exp-16, Exp-17 runners and their outputs (frozen as historical record).
- The `EmlBoostRegressor` (older API) which is out of scope.

## Naming consistency

The library default flip is included AS PART OF Exp 18 (Task 1 of the plan) rather
than as a separate spec, per the user's direction ("no users; flip in the same commit
sequence"). If a future user asks "when did `min_samples_leaf` change from 20 to 1?",
the answer is "as part of Exp 18, commit `<sha>`, justified by Exp-17's matched-comparison
results." A separate `min_samples_leaf-default-flip` spec would be over-ceremony for a
2-line change.

Subsequent experiments after Exp 18 (Exp 19+) should default to OpenML-based suites
unless there's a specific reason to use PMLB or another benchmark.
