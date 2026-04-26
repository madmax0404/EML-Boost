# Experiment 18 — OpenML-CTR23 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip the SplitBoost library default `min_samples_leaf=20 → 1` (justified by Exp-17 results), then run Exp-18: SB at post-flip library defaults vs matched-XGB and matched-LGB on the OpenML-CTR23 regression suite (35 datasets, 5 seeds, with early stopping for all three models).

**Architecture:** Task 1 is a 2-line library code change + unit-test triage. Task 2 builds the Exp-18 runner by copying the Exp-17 runner and swapping the dataset loader from PMLB to OpenML-CTR23, then smoke-tests on 2-3 datasets. Task 3 pauses for user checkpoint then runs the full ~1-2.5h experiment in background. Task 4 writes the report mirroring Exp-17's structure.

**Tech Stack:** Python 3.12, numpy, torch, Triton, xgboost, lightgbm, openml, scikit-learn. Run via `uv run`.

**Spec:** `docs/superpowers/specs/2026-04-26-experiment18-openml-ctr23-design.md`.

---

## File Structure

| file | role | tasks that touch it |
|---|---|---|
| `eml_boost/tree_split/tree.py` | flip `min_samples_leaf` default 20 → 1 | Task 1 |
| `eml_boost/tree_split/ensemble.py` | flip `min_samples_leaf` default 20 → 1 | Task 1 |
| `tests/unit/test_eml_split_*.py` | possibly: triage assertions that depend on msl=20 | Task 1 |
| `experiments/run_experiment18_openml_ctr23.py` | NEW — runner | Task 2 |
| `experiments/experiment18/{summary.csv, summary.json, openml_rmse.png, failures.json}` | NEW — auto-generated outputs | Task 3 |
| `experiments/experiment18/report.md` | NEW — manual writeup | Task 4 |

The Triton kernels, CPU pipeline, and prior experiments' artifacts are NOT touched.

---

## Task 1: Flip library default `min_samples_leaf` 20 → 1

**Goal:** Change the SplitBoost library default for `min_samples_leaf` from 20 to 1 in both regressor classes. Justified by Exp-17 results (at msl=1 + leaf_l2=1, SB closes all 3 catastrophic losses to outright wins). Triage any unit tests that depended on the prior default.

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (one default value)
- Modify: `eml_boost/tree_split/ensemble.py` (one default value)
- Modify (possibly): `tests/unit/test_eml_split_tree.py` and `tests/unit/test_eml_split_boost.py`

### Step 1: Flip the default in `EmlSplitTreeRegressor`

- [ ] In `eml_boost/tree_split/tree.py`, find the `__init__` method of `EmlSplitTreeRegressor`. Locate the line:

```python
        min_samples_leaf: int = 20,
```

- [ ] Change to:

```python
        min_samples_leaf: int = 1,                # was 20; flipped post-Exp-17 to match the matched-comparison setting
```

### Step 2: Flip the default in `EmlSplitBoostRegressor`

- [ ] In `eml_boost/tree_split/ensemble.py`, find the `__init__` method of `EmlSplitBoostRegressor`. Locate the line:

```python
        min_samples_leaf: int = 20,
```

- [ ] Change to:

```python
        min_samples_leaf: int = 1,                # was 20; mirrors EmlSplitTreeRegressor (Exp-18 default)
```

### Step 3: Run the full unit suite

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected outcomes:
  - `test_leaf_l2_zero_constant_leaves_bit_exact` — PASS (it explicitly passes `min_samples_leaf=20`).
  - `test_leaf_l2_gpu_cpu_equivalence_at_one` — PASS (likely uses `min_samples_leaf=20` via Exp-17 defaults; verify).
  - `test_predict_triton_matches_torch` — likely PASS (smoke check on predict path).
  - `test_xcache_boost_loop_runs_cleanly` — likely PASS (smoke check on boost loop).
  - Other tests that don't explicitly pass `min_samples_leaf` will run with the new default of 1; behavior may shift on small fixtures.
  - `test_fit_recovers_simple_formula` — still failing (pre-existing, leave alone).

### Step 4: Triage any failures

- [ ] For each newly-failing test:
  - **Smoke check (assert finite predictions, assert MSE < some-loose-threshold)**: usually passes; if it doesn't, the threshold may have been very tight. Widen by ~50% (e.g., `< 0.5` → `< 0.75`) and add a comment: `# threshold widened post-msl-default-flip; see commit <SHA>`.
  - **Hardcoded prediction values or RMSE values**: this should NOT exist in the current suite. If found, regenerate or convert to smoke check.
  - **Floating-point allclose with tight rtol**: `min_samples_leaf=1` allows finer tree splits, changing predictions meaningfully; relax to `rtol=1e-2` (or whatever the test's intent supports) and add a comment.
- [ ] DO NOT modify the model behavior to make tests pass. Only relax test assertions.
- [ ] If a test has a clear behavior assertion that breaks at msl=1 (e.g., "this exact split is chosen" or "this leaf has exactly N samples"), either:
  - Convert it to a more semantic assertion (e.g., "the model fits the signal within X RMSE"), or
  - Add `min_samples_leaf=20` explicitly to the fixture if the test's INTENT was to test msl=20-specific behavior, with a comment noting the pin.

### Step 5: Re-run the suite

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: all previously-passing tests pass. Test count should match what it was before (1 pre-existing failure unchanged).

### Step 6: Commit

- [ ] Run:

```bash
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py tests/unit/test_eml_split_tree.py tests/unit/test_eml_split_boost.py
git commit -m "$(cat <<'EOF'
feat: flip min_samples_leaf default 20 -> 1 (Exp-18 prereq)

Changes the SplitBoost library default for min_samples_leaf from 20
(the historical Exp-13 calibrated value) to 1, matching what Exp 17
used at the runner level. Justified by Exp-17 evidence: at msl=1
combined with leaf_l2=1.0 (already the post-Task-7 default), SB
closes all 3 PMLB catastrophic losses to outright wins, and the
post-hoc 20-seed analysis suggests the actual win rate is 90%+.

Now-default behavior: anyone calling EmlSplitBoostRegressor() with
no args gets msl=1 (XGBoost-style fine-grained leaves with leaf_l2
regularization). To recover the prior msl=20 behavior, pass it
explicitly. The repo is in active development with no external users
per the project notes; flip is intentional and not a migration burden.

Tests with widened thresholds: <list any tests touched and the
specific old → new threshold; if none, write "none — all tests
passed without modification">.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(If no test thresholds needed widening, drop the "Tests with widened thresholds" block from the message.)

---

## Task 2: Build the Exp-18 runner + smoke-test on 2-3 datasets

**Goal:** Produce a working `experiments/run_experiment18_openml_ctr23.py` by copying Exp-17's runner and swapping the dataset loader from PMLB to OpenML-CTR23. Smoke-test on 2-3 datasets before kicking off Task 3's full run, to catch wiring bugs (especially around OpenML data loading and categorical-feature handling).

**Files:**
- Create: `experiments/run_experiment18_openml_ctr23.py`
- Read for reference (do not modify): `experiments/run_experiment17_matched_revalidation.py`

### Step 1: Copy the Exp-17 runner as a starting template

- [ ] Run:

```bash
cp experiments/run_experiment17_matched_revalidation.py experiments/run_experiment18_openml_ctr23.py
```

### Step 2: Update the module docstring

- [ ] Edit `experiments/run_experiment18_openml_ctr23.py` — replace the existing module docstring at the top with:

```python
"""Experiment 18: SplitBoost benchmark on OpenML-CTR23 (matched hyperparameters).

Runs SB at library defaults (post-Exp-18 Task 1: min_samples_leaf=1,
leaf_l2=1.0, EML enabled with default settings) against XGBoost and
LightGBM at matched-Exp-17 settings on the OpenML-CTR23 (Curated
Tabular Regression 2023) suite — 35 modern regression tasks. Same
matched-comparison framework as Exp 17:
- leaf-floor (sample-based): all → 1
- L2 on leaf weights: all → 1.0
- early stopping: all → patience=15, 15% inner-val (same outer seed)

LGB's leaf-wise growth policy and SB's EML mechanism are documented
algorithmic confounds, same as Exp 17.

Spec: docs/superpowers/specs/2026-04-26-experiment18-openml-ctr23-design.md

Estimated runtime: 1-2.5h on RTX 3090. Run with:
  PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment18_openml_ctr23.py 2>&1 | tee experiments/experiment18/run.log
"""
```

### Step 3: Update RESULTS_DIR

- [ ] In the same file, find the line:

```python
RESULTS_DIR = Path(__file__).resolve().parent / "experiment17"
```

- [ ] Change to:

```python
RESULTS_DIR = Path(__file__).resolve().parent / "experiment18"
```

### Step 4: Replace the dataset loader (PMLB → OpenML-CTR23)

- [ ] In the same file, find the imports near the top. Locate:

```python
from pmlb import fetch_data, regression_dataset_names
```

- [ ] Replace with:

```python
import openml
import pandas as pd
```

- [ ] Find the `DATASETS = list(regression_dataset_names)` line near the top and replace with a function that fetches the CTR23 task IDs:

```python
def _load_ctr23_dataset_names() -> list[str]:
    """Fetch the OpenML-CTR23 suite task IDs and resolve dataset names.

    Returns a list of dataset name strings (one per task) suitable for
    passing to the per-dataset loop below. Each name is the OpenML
    dataset name (e.g., 'pol', 'boston', 'kin8nm').
    """
    suite = openml.study.get_suite("OpenML-CTR23")
    names = []
    for task_id in suite.tasks:
        try:
            task = openml.tasks.get_task(task_id, download_data=False, download_qualities=False)
            ds = openml.datasets.get_dataset(
                task.dataset_id, download_data=False, download_qualities=False,
            )
            names.append(ds.name)
        except Exception as e:
            print(f"  failed to resolve task_id={task_id}: {type(e).__name__}: {e}", flush=True)
    return names


DATASETS = _load_ctr23_dataset_names()
```

- [ ] Find the `try: X, y = fetch_data(name, return_X_y=True)` block inside `main()` and replace with an OpenML-equivalent loader. Search for `fetch_data` to find the right line. The current Exp-17 block looks like:

```python
        try:
            X, y = fetch_data(name, return_X_y=True)
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[mask], y[mask]
            if len(X) < 20 or X.shape[1] < 1:
                raise ValueError(
                    f"dataset too small after sanitization: n={len(X)}, k={X.shape[1] if len(X) else 0}"
                )
            print(f"  n={len(X):>6}  k={X.shape[1]:>3}", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FETCH FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append(_summarize_failure(name, "ALL_SEEDS", "data_fetch", "fetch", e))
            _save_failures(failures_path, failures)
            continue
```

- [ ] Replace with an OpenML loader that handles categorical features via one-hot encoding:

```python
        try:
            ds = openml.datasets.get_dataset(name)
            X_df, y_ser, _, _ = ds.get_data(target=ds.default_target_attribute)
            # One-hot encode categorical features (CTR23 datasets may have them);
            # drop_first to avoid perfect multicollinearity.
            X_df = pd.get_dummies(X_df, drop_first=True)
            X = X_df.to_numpy(dtype=np.float64)
            y = np.asarray(y_ser, dtype=np.float64)
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[mask], y[mask]
            if len(X) < 20 or X.shape[1] < 1:
                raise ValueError(
                    f"dataset too small after sanitization: n={len(X)}, k={X.shape[1] if len(X) else 0}"
                )
            print(f"  n={len(X):>6}  k={X.shape[1]:>3}", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FETCH FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append(_summarize_failure(name, "ALL_SEEDS", "data_fetch", "fetch", e))
            _save_failures(failures_path, failures)
            continue
```

### Step 5: Remove the `comparison_to_exp15.md` generation block

- [ ] In the same file, find the block that starts with `# ---- comparison_to_exp15.md ----` (around line 497 in Exp 17's file). Delete the entire block — Exp 18 has no PMLB-baseline comparison since the dataset universe is different.
- [ ] Verify the deletion: search for `comparison_to_exp15` in the file and confirm zero matches remain.

### Step 6: Adjust the `_fit_split_boost` function to use library defaults

- [ ] In the same file, find `_fit_split_boost`. Currently it has explicit `min_samples_leaf=1, leaf_l2=1.0` — these are now library defaults post-Task-1. Change to:

```python
def _fit_split_boost(X_tr, y_tr, seed):
    """SplitBoost at post-Task-1 library defaults (msl=1, leaf_l2=1.0).
    No experiment-level overrides — the off-the-shelf SB call IS the
    matched-comparison configuration."""
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        # min_samples_leaf=1 (library default, post-Task-1)
        # leaf_l2=1.0 (library default, post-Task-7 of leaf_l2 plan)
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
```

The `_fit_xgb` and `_fit_lgb` functions stay unchanged from Exp 17 — they continue to explicitly set `min_child_weight=1` / `min_data_in_leaf=1` and `reg_lambda=1.0` and `early_stopping_rounds=15` (XGB defaults match what we want; LGB needs explicit setting).

### Step 7: Update the plot path

- [ ] In the same file, find the line setting `plot_path` (search for `pmlb_rmse.png`):

```python
plot_path = RESULTS_DIR / "pmlb_rmse.png"
```

- [ ] Change to:

```python
plot_path = RESULTS_DIR / "openml_rmse.png"
```

- [ ] If the plot title references PMLB, update that too. Search for `Experiment 17` in the file (titles, comments) and replace with `Experiment 18 (OpenML-CTR23)` where appropriate.

### Step 8: Smoke-test the runner on 2-3 datasets

- [ ] First, do a one-off filter to limit `DATASETS` for the smoke test. Edit the `DATASETS = _load_ctr23_dataset_names()` line. Temporarily replace with:

```python
DATASETS = ["boston", "kin8nm", "pol"]  # SMOKE TEST ONLY — revert before committing
```

(`boston` is small and standard, `kin8nm` is medium 8k-row, `pol` is medium 15k-row — gives variety.)

- [ ] Run the smoke test:

```bash
mkdir -p experiments/experiment18
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment18_openml_ctr23.py 2>&1 | tee experiments/experiment18/smoke.log
```

- [ ] Expected output: 3 datasets × 5 seeds × 3 models = 45 fits. Should complete in 2-5 minutes. The output should:
  - Print `=== dataset: <name> ===` for each.
  - Print per-fit lines `[seed=N] split_boost (Xs, Y rounds) RMSE=Z` etc.
  - **n_rounds for XGB and LGB should be < 200 on at least some seeds** (early stopping kicked in).
  - Print `=== Headline statistics ===` at the end with the 3 datasets covered.
  - Generate `experiments/experiment18/{summary.csv, summary.json, openml_rmse.png}`.

- [ ] Inspect `experiments/experiment18/summary.csv` — confirm it has 45 rows + header.

- [ ] If anything fails: STOP and debug. Most likely culprits:
  - OpenML API needs an API key for some calls (set via `openml.config.apikey = "..."` or the env var `OPENML_API_KEY`). If a key is required, document it as a prerequisite.
  - Some CTR23 datasets have all-categorical features that produce empty X after `pd.get_dummies` if there's a parsing error. Mitigated by the `if len(X) < 20 or X.shape[1] < 1` check, which will log a fetch failure and continue.
  - `default_target_attribute` may not be set on some datasets; need to fall back to `task.target` or similar. If this happens, add a `try/except` to fetch the target via the task instead.

### Step 9: Restore full DATASETS + clean smoke artifacts before commit

- [ ] **Restore the full DATASETS list** by reverting Step 8's edit:

```python
DATASETS = _load_ctr23_dataset_names()
```

- [ ] Delete the smoke-test outputs:

```bash
rm -f experiments/experiment18/smoke.log experiments/experiment18/summary.csv experiments/experiment18/summary.json experiments/experiment18/openml_rmse.png experiments/experiment18/failures.json
```

(The `experiments/experiment18/` directory itself stays — Task 3's full run will populate it.)

### Step 10: Run the full unit suite as a final sanity check

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: same gate as after Task 1 (all previously-passing tests still pass; pre-existing `test_fit_recovers_simple_formula` still fails).

### Step 11: Commit the runner

- [ ] Run:

```bash
git add experiments/run_experiment18_openml_ctr23.py
git commit -m "$(cat <<'EOF'
add: Experiment 18 runner for OpenML-CTR23 matched comparison

Mirrors run_experiment17_matched_revalidation.py's structure with
three changes:
1. Dataset loader: PMLB → OpenML-CTR23 via openml.study.get_suite.
   pd.get_dummies handles categorical features (drop_first=True).
2. SB call uses library defaults (post-Task-1 flip): no explicit
   min_samples_leaf or leaf_l2 — the off-the-shelf SB call is now
   the matched-comparison configuration.
3. RESULTS_DIR points at experiments/experiment18/; plot renamed
   openml_rmse.png; comparison_to_exp15.md generation removed
   (different dataset universe, no meaningful comparison).

XGB and LGB matched-config preserved verbatim from Exp 17:
min_child_weight=1, min_data_in_leaf=1, reg_lambda=1.0 each, with
early_stopping_rounds=15 + 15% inner-val using the same outer seed.

Smoke-tested on boston + kin8nm + pol (3 datasets x 5 seeds x 3
models = 45 fits) before commit; runner produces expected outputs
including early-stopping behavior on XGB/LGB.

Spec: docs/superpowers/specs/2026-04-26-experiment18-openml-ctr23-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Kick off the full Exp-18 run (~1-2.5h)

**Goal:** Run the validated Exp-18 runner over the full OpenML-CTR23 suite (35 datasets) and produce `summary.{csv,json}`, `openml_rmse.png`, `failures.json`. Multi-hour run: MUST checkpoint with the user before kicking off.

**Files:** None modified. Outputs land in `experiments/experiment18/`. Logs gitignored (`*.log`).

### Step 1: Checkpoint with the user before kicking off

- [ ] STOP. Report to the user that Task 1 + Task 2 are complete (library default flipped, runner committed, smoke test green) and that Task 3 is the multi-hour run. Ask for explicit confirmation before launching.
- [ ] If the user says "go": proceed to Step 2.
- [ ] If the user says "wait" or wants something else first: pause; do not launch.

### Step 2: Launch the full run in the background

- [ ] Run (in background, with output tee'd to a log):

```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment18_openml_ctr23.py 2>&1 | tee experiments/experiment18/run.log
```

Use Bash with `run_in_background=true` and a timeout high enough to cover the run (the actual process keeps running past the Bash-tool timeout because the background pattern detaches it). Get the background process ID for reference.

### Step 3: Set up an error-only monitor for the run

- [ ] Use the Monitor tool to watch for fit failures, Triton fallbacks, and the run's completion banner:

```
tail -f experiments/experiment18/run.log | grep -E --line-buffered "FETCH FAILED|SPLIT FAILED|FAILED:|Traceback|RuntimeWarning|finalizing|Headline statistics|wrote .*png"
```

This avoids spamming the controller with the 35 dataset-boundary lines and ~525 successful fit lines, while catching anything that requires intervention.

### Step 4: Wait for completion (notification-based — don't poll)

- [ ] When the background process completes, the controller is notified. While waiting:
  - Do NOT poll the log file repeatedly.
  - Do NOT spawn other long-running work in parallel.
  - DO answer user questions or do unrelated short tasks.

### Step 5: Verify the run completed cleanly

- [ ] When notified, confirm the run finished (exit code 0) and check:

```bash
tail -25 experiments/experiment18/run.log
```

- [ ] Expected: should see the "=== Headline statistics ===" block with numbers populated, plus `wrote .../openml_rmse.png`.

- [ ] Run:

```bash
grep -iE "warning|fallback|FAILED|Traceback|RuntimeWarning" experiments/experiment18/run.log | head -20
```

- [ ] Expected: ideally zero matches; CTR23 has better data hygiene than PMLB. A small number of fetch failures is acceptable if some OpenML datasets are temporarily unavailable.

- [ ] If unexpected failures appear: STOP. Investigate before proceeding to Task 4.

### Step 6: Verify the success criteria from the spec

The spec specifies four success criteria. Check each:

- [ ] **S-A (correctness):** unit suite passes (run `uv run pytest tests/unit/ -q`); failures.json is empty or only contains expected fetch-related failures.

- [ ] **S-B (matched-comparison story holds):** SB-vs-matched-XGB median ratio is in `[0.85, 1.05]`. Read from `experiments/experiment18/summary.json`:

```bash
uv run python -c "
import json
with open('experiments/experiment18/summary.json') as f:
    s = json.load(f)
print(f'median ratio vs XGB: {s[\"headline_stats\"][\"median_ratio\"]:.3f}')
print(f'  → {\"PASS\" if 0.85 <= s[\"headline_stats\"][\"median_ratio\"] <= 1.05 else \"FAIL\"}')
"
```

- [ ] **S-C (catastrophic regime):** ≤ 3% catastrophic ratios > 2.0:

```bash
uv run python -c "
import json
with open('experiments/experiment18/summary.json') as f:
    s = json.load(f)
n_total = s['headline_stats']['n_total_datasets']
n_cat = s['headline_stats']['n_catastrophic']
pct = 100 * n_cat / max(n_total, 1)
print(f'catastrophic: {n_cat}/{n_total} ({pct:.1f}%)')
print(f'  → {\"PASS\" if pct <= 3.0 else \"FAIL\"}')
"
```

- [ ] **S-D (no Triton fallback):** see Step 5 grep output; should be clean.

- [ ] If S-A + S-B + S-C + S-D all pass: proceed to Task 4 (writeup).
- [ ] If any fails: STOP, report to user, do not proceed to writeup until investigation.

### Step 7: Commit the experiment artifacts

- [ ] Stage the generated files (logs are gitignored):

```bash
git add experiments/experiment18/
```

- [ ] Inspect `git status` to confirm what's staged. Expected: `summary.csv`, `summary.json`, `openml_rmse.png`, `failures.json`. NOT `run.log` (gitignored). NOT `report.md` yet (Task 4).

- [ ] Commit:

```bash
git commit -m "$(cat <<'EOF'
exp 18 done: SB benchmark on OpenML-CTR23 (matched hyperparameters)

Headline (mean ratios SB at library defaults / matched-XGB across 5
seeds, on the OpenML-CTR23 suite):
- <fill in: within 10%, outright wins, median ratio>
- vs matched-LGB: <fill in: median ratio, win count>

Catastrophic (ratio > 2.0): <fill in: count / 35 datasets>.

Per-dataset summary in experiments/experiment18/summary.json. Headline
plot in openml_rmse.png. Methodological caveats (LGB leaf-wise growth,
SB EML mechanism — both unmatchable as hyperparameters) carry over
from Exp 17 and are documented in the report.md.

Wall time: <fill in> on RTX 3090. Logs gitignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `<fill in>` placeholders with actual values from the run BEFORE committing.)

---

## Task 4: Write Exp-18 report.md

**Goal:** Produce a narrative writeup of Exp-18's results following the same shape as `experiments/experiment17/report.md`. Include cross-references to the Exp-17 PMLB result for context (the experiment changed both the dataset universe AND made the matched-config the library default — both shifts should be acknowledged in the report).

**Files:**
- Create: `experiments/experiment18/report.md`

### Step 1: Read the Exp-18 outputs for the writeup

- [ ] Read in parallel for context:
  - `experiments/experiment18/summary.json` (full per-dataset aggregates + headline_stats + ratios)
  - `experiments/experiment18/openml_rmse.png` (the headline plot)
  - `experiments/experiment17/report.md` (for style match and the existing narrative structure)

- [ ] Optionally inspect:
  - Top wins / top losses in Exp 18 to mention in the report
  - Whether any Exp-17 catastrophic-regime patterns reappear or are absent on CTR23

### Step 2: Draft `experiments/experiment18/report.md`

Mirror Exp 17's report.md structure. The required sections:

1. **Title + metadata** — `# Experiment 18: SplitBoost benchmark on OpenML-CTR23 (matched hyperparameters)`. Date 2026-04-26, commits, runtime, scripts.
2. **What the experiment was about** — paragraph explaining the PMLB → OpenML pivot and library default flip, and what this experiment tests.
3. **Configuration** — table or code block showing the matched hyperparameters per algorithm and noting that SB uses library defaults (no runner-level overrides).
4. **Coverage** — datasets attempted (35 in CTR23), fittable, fetch failures.
5. **Headline results (mean ratios SplitBoost / matched-XGBoost)** — table with: within-10%, within-5%, outright-wins, catastrophic, mean ratio, median ratio, P25, P75. Compare to Exp 17's same metrics in parens or a side-by-side row (different dataset universe, but the SHAPE is comparable).
6. **Distribution of ratios** — banding (deep wins, clear wins, narrow wins, narrow losses, losses, clear losses, catastrophic). Counts.
7. **Top 5-10 wins / Top 5-10 losses** — tables.
8. **vs LightGBM (matched)** — separate paragraph.
9. **Comparison to Exp 17 PMLB** — short paragraph noting how the headline shifted across benchmarks. CTR23 has a different dataset distribution (fewer tiny-n synthetics, more medium real-world); this should reshape the win profile.
10. **What Exp 18 actually shows** — interpretive paragraph(s).
11. **What's left as a loss** — datasets where SB still loses; pattern analysis.
12. **What Exp 18 does NOT show** — caveats list (no baseline tuning, single split, LGB leaf-wise unmatched, SB EML unmatched, no statistical significance reporting at 5 seeds — refer back to Exp-17's 20-seed methodology lesson).
13. **Methodological caveats** — same inner-val-split note as Exp 17; OpenML data hygiene differences from PMLB; categorical-feature one-hot encoding.
14. **Action taken** — record any unit-test threshold widenings from Task 1; library defaults now `msl=1, leaf_l2=1`; project headline updates to OpenML-CTR23 numbers.
15. **Next experiments** — Optuna-tuned baselines for stronger comparison; possibly Grinsztajn-2022 as a second benchmark for cross-validation; investigation of any surprising loss patterns.

The writeup should be ~150-200 lines, similar in length to Exp 17's report.

### Step 3: Commit the report

- [ ] Run:

```bash
git add experiments/experiment18/report.md
git commit -m "$(cat <<'EOF'
exp 18 report: SplitBoost on OpenML-CTR23

Writes experiments/experiment18/report.md following the same shape
as Exp 17's report. Documents the headline (matched-comparison story
on the OpenML-CTR23 35-dataset suite), comparison to Exp 17's PMLB
result, vs-LGB story, methodological caveats, and queued follow-ups.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 4: Report final summary to the user

After committing, report to the user:
- Final headline (Exp 18's matched-hyperparameter ratios on OpenML-CTR23 vs Exp 17's PMLB ratios)
- Whether the Exp-17 narrative held on a different dataset distribution
- Any new loss patterns surfaced
- Recommendations for next steps (typically: Optuna-tuned baselines as Exp 19, OR Grinsztajn-2022 for cross-validation as Exp 19b, OR move to a different research direction)

---

## Implementation order recap

1. **Task 1** — Library default flip (msl=20 → 1) + unit test triage + commit.
2. **Task 2** — Build Exp-18 runner (OpenML loader + library-default SB) + smoke-test on 3 datasets + commit.
3. **Task 3** — Pause for user confirmation; launch full ~1-2.5h run; verify success criteria; commit experiment artifacts.
4. **Task 4** — Read outputs, write report.md, commit.

After Task 4, regardless of outcome, report to the user. Do NOT autonomously kick off Exp 19 or any further work — those are separate decisions for the user.
