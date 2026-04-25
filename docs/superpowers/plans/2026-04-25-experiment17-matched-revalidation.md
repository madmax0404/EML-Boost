# Experiment 17 Matched-Hyperparameter Re-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the Exp-17 PMLB-suite re-validation under truly-matched hyperparameters (leaf-floor=1, L2 leaf-weight=1.0, early-stopping=15 patience with 15% inner-val), then write the report and link a methodological-caveats note from Exp 15.

**Architecture:** New runner `experiments/run_experiment17_matched_revalidation.py` mirrors `run_experiment15_full_pmlb.py` with three modified `_fit_*` functions. Inner train/val split for XGB/LGB uses `train_test_split` with the same outer seed. Auto-generated `comparison_to_exp15.md` table. Run is ~1.5-2h on RTX 3090, overnight-friendly. Per the user's "checkpoint before long runs" rule, the controller MUST pause for user confirmation between Task 1 (smoke test passes) and Task 2 (the full ~1.5-2h run).

**Tech Stack:** Python 3.12 / numpy / torch / Triton (for SB GPU) / xgboost / lightgbm / pmlb / scikit-learn. Run via `uv run`.

**Spec:** `docs/superpowers/specs/2026-04-25-experiment17-matched-revalidation-design.md`.

---

## File Structure

| file | role | tasks |
|---|---|---|
| `experiments/run_experiment17_matched_revalidation.py` | Runner: per-fit reliability machinery + 3 modified `_fit_*` functions + summary aggregation + comparison_to_exp15 generation | Task 1 |
| `experiments/experiment17/{summary.csv, summary.json, pmlb_rmse.png, failures.json, comparison_to_exp15.md}` | Auto-generated outputs | Task 2 |
| `experiments/experiment17/report.md` | Manual writeup after results land | Task 3 |
| `experiments/experiment15/report.md` | Add methodological-caveats note pointing to Exp 17 | Task 3 |

The library code (`eml_boost/`) is NOT modified. The library default `min_samples_leaf=20` stays unchanged; this experiment overrides it at the runner level only.

---

## Task 1: Build the Exp 17 runner + smoke-test on 2-3 datasets

**Goal:** Produce a working `run_experiment17_matched_revalidation.py` that's identical in structure to `run_experiment15_full_pmlb.py` except for the three `_fit_*` functions (matched-hyperparameter conditions) and a new `comparison_to_exp15.md` generation step. Smoke-test on 2-3 datasets before kicking off the full run to catch wiring bugs.

**Files:**
- Create: `experiments/run_experiment17_matched_revalidation.py`
- Read for reference (do not modify): `experiments/run_experiment15_full_pmlb.py`, `experiments/experiment15/summary.json`

### Step 1: Copy the Exp-15 runner as a starting template

- [ ] Run:

```bash
cp experiments/run_experiment15_full_pmlb.py experiments/run_experiment17_matched_revalidation.py
```

The new file will be modified in subsequent steps.

### Step 2: Update the module docstring

- [ ] Edit `experiments/run_experiment17_matched_revalidation.py` — replace the existing module docstring at the top with:

```python
"""Experiment 17: full PMLB regression suite under TRULY-MATCHED hyperparameters.

Re-runs Exp-15's 119-dataset suite with three regularization axes matched
across SplitBoost, XGBoost, and LightGBM:
- leaf-floor (sample-based): all → 1
- L2 on leaf weights: all → 1.0
- early stopping: all → patience=15, 15% inner-val (same outer seed)

LGB's leaf-wise growth policy and SB's EML mechanism are documented as
algorithmic confounds (not matchable). See spec at:
  docs/superpowers/specs/2026-04-25-experiment17-matched-revalidation-design.md

Runtime estimate: 1.5-2h on RTX 3090. Run with:
  PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment17_matched_revalidation.py 2>&1 | tee experiments/experiment17/run.log
"""
```

### Step 3: Add the inner-val constants

- [ ] In the same file, find the module-level constants block (around lines 36-51 — the `MAX_ROUNDS = 200` block). Add two new constants right after `SEEDS = [0, 1, 2, 3, 4]`:

```python
# NEW: matched-early-stopping config for XGB/LGB inner-val split.
# SB already uses val_fraction=0.15 + patience=15 internally.
INNER_VAL_FRACTION = 0.15
EARLY_STOPPING_PATIENCE = 15
```

### Step 4: Update RESULTS_DIR to point at the new experiment folder

- [ ] In the same file, find the line near the top:

```python
RESULTS_DIR = Path(__file__).resolve().parent / "experiment15"
```

- [ ] Change to:

```python
RESULTS_DIR = Path(__file__).resolve().parent / "experiment17"
```

### Step 5: Replace `_fit_split_boost` with the matched version

- [ ] In the same file, locate `def _fit_split_boost(X_tr, y_tr, seed):` (currently around line 76). Replace its body with the matched-hyperparameter version. The change is `min_samples_leaf=1` (was `MIN_SAMPLES_LEAF`=20) and `leaf_l2=1.0` explicit:

```python
def _fit_split_boost(X_tr, y_tr, seed):
    """SplitBoost matched: min_samples_leaf=1, leaf_l2=1.0,
    early stopping via internal val_fraction=0.15."""
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=1,                        # CHANGED from MIN_SAMPLES_LEAF (20) — matched
        leaf_l2=1.0,                               # explicit (matched to XGB reg_lambda default)
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
        patience=EARLY_STOPPING_PATIENCE,           # 15
        val_fraction=INNER_VAL_FRACTION,            # 0.15
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start
```

The `MIN_SAMPLES_LEAF` constant is no longer referenced from `_fit_split_boost`, but other code may still reference it (it's used implicitly by LGB's old config) — keep the constant defined at module level for now; it'll go away in Step 7.

### Step 6: Replace `_fit_xgb` with the matched version (early stopping + inner val split)

- [ ] In the same file, locate `def _fit_xgb(X_tr, y_tr, seed):` (currently around line 121). Replace its body with the matched-hyperparameter version:

```python
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
        min_child_weight=1,                        # explicit (existing default; matched)
        reg_lambda=1.0,                            # explicit (existing default; matched)
        early_stopping_rounds=EARLY_STOPPING_PATIENCE,  # NEW (was no early stopping)
        random_state=seed,
    )
    m.fit(
        X_inner_tr, y_inner_tr,
        eval_set=[(X_inner_val, y_inner_val)],
        verbose=False,
    )
    return m, time.time() - start
```

### Step 7: Replace `_fit_lgb` with the matched version

- [ ] In the same file, locate `def _fit_lgb(X_tr, y_tr, seed):` (currently around line 102). Replace its body with the matched-hyperparameter version:

```python
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
            min_data_in_leaf=1,                    # CHANGED from MIN_SAMPLES_LEAF (20) — matched
            reg_lambda=1.0,                        # NEW (was 0; matched)
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

After Steps 5-7, `MIN_SAMPLES_LEAF` and `MIN_SAMPLES_LEAF_EML` are no longer referenced from any of the three fit functions. Keep `MIN_SAMPLES_LEAF_EML` (still used by SB's EML config). Delete `MIN_SAMPLES_LEAF` from the module-level constants block — it's now dead code.

- [ ] In the constants block, delete the line:

```python
MIN_SAMPLES_LEAF = 20
```

### Step 8: Add `comparison_to_exp15.md` generation at the end of `main()`

- [ ] In the same file, locate the end of `main()` — after the headline-statistics print block but before `return 0`. Add a new section that generates `comparison_to_exp15.md`:

```python
    # ---- comparison_to_exp15.md ----
    md_path = RESULTS_DIR / "comparison_to_exp15.md"
    exp15_summary_path = RESULTS_DIR.parent / "experiment15" / "summary.json"
    if exp15_summary_path.exists():
        with exp15_summary_path.open() as fp:
            exp15 = json.load(fp)
        rows_for_md = []
        for name in sorted(ratios.keys()):
            new_r = ratios[name]["ratio"]
            old_r = exp15["ratios"].get(name, {}).get("ratio")
            if old_r is None:
                rows_for_md.append((name, None, new_r, None))
            else:
                rows_for_md.append((name, old_r, new_r, new_r - old_r))
        rows_for_md.sort(key=lambda r: r[3] if r[3] is not None else 0.0)

        with md_path.open("w") as fp:
            fp.write("# Experiment 17 vs Experiment 15 — per-dataset ratio comparison\n\n")
            fp.write("Sorted by Δ (most-improved first; positive Δ = SB got worse vs matched-XGB).\n\n")
            fp.write(f"**Datasets in both runs:** {sum(1 for r in rows_for_md if r[1] is not None)}\n")
            fp.write(f"**Mean Δ:** {mean(r[3] for r in rows_for_md if r[3] is not None):+.3f}\n")
            n_improved = sum(1 for r in rows_for_md if r[3] is not None and r[3] < 0)
            n_regressed = sum(1 for r in rows_for_md if r[3] is not None and r[3] > 0)
            fp.write(f"**Improved (Δ < 0):** {n_improved}\n")
            fp.write(f"**Regressed (Δ > 0):** {n_regressed}\n\n")
            fp.write("## Per-dataset table\n\n")
            fp.write("| dataset | Exp 15 ratio (off-the-shelf) | Exp 17 ratio (matched) | Δ |\n")
            fp.write("|---|---|---|---|\n")
            for name, old_r, new_r, delta in rows_for_md:
                if old_r is None:
                    fp.write(f"| {name} | — | {new_r:.3f} | — |\n")
                else:
                    fp.write(f"| {name} | {old_r:.3f} | {new_r:.3f} | {delta:+.3f} |\n")
        print(f"wrote {md_path}", flush=True)
```

This block uses `ratios`, `mean`, and `json` which are already imported / defined earlier in `main()` (verify that `from statistics import mean` is in the imports near the top — it should be, copied from Exp 15's runner).

### Step 9: Smoke-test the runner on 2-3 datasets

- [ ] First, do a one-off filter to limit `DATASETS` for the smoke test. Edit the `DATASETS` line near the top of `experiments/run_experiment17_matched_revalidation.py`. Find:

```python
DATASETS = list(regression_dataset_names)  # all 122 regression datasets
```

Temporarily replace with (note: this is reverted at the end of Step 9 before commit):

```python
DATASETS = ["1027_ESL", "663_rabe_266", "344_mv"]  # SMOKE TEST ONLY — revert before committing
```

- [ ] Run the smoke test:

```bash
mkdir -p experiments/experiment17
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment17_matched_revalidation.py 2>&1 | tee experiments/experiment17/smoke.log
```

- [ ] Expected output: 3 datasets × 5 seeds × 3 models = 45 fits. Should complete in 1-3 minutes. The output should:
  - Print `=== dataset: <name> ===` for each.
  - Print per-fit lines `[seed=N] split_boost (Xs, Y rounds) RMSE=Z` etc.
  - **n_rounds for XGB and LGB should be < 200 on at least some seeds** (early stopping kicked in).
  - Print `=== Headline statistics ===` at the end with the 3 datasets covered.
  - Generate `experiments/experiment17/{summary.csv, summary.json, pmlb_rmse.png, comparison_to_exp15.md}`.

- [ ] Inspect `experiments/experiment17/summary.csv` — confirm it has 45 rows + header (1 row per fit).

- [ ] Inspect `experiments/experiment17/comparison_to_exp15.md` — confirm it lists the 3 smoke datasets with both Exp-15 and Exp-17 ratios + delta.

- [ ] If any of the above fail: STOP and debug. Most likely culprits:
  - XGB's `early_stopping_rounds` API needs the model parameter or a direct `early_stopping_rounds=` kwarg in `.fit()` — check XGBoost version against `pyproject.toml` (xgboost>=3.2.0).
  - LGB's `early_stopping` callback needs to be a list (`callbacks=[...]`).
  - PMLB fetch or `train_test_split` import errors.

- [ ] **Restore the full DATASETS list** before committing. Find:

```python
DATASETS = ["1027_ESL", "663_rabe_266", "344_mv"]  # SMOKE TEST ONLY — revert before committing
```

Replace with:

```python
DATASETS = list(regression_dataset_names)  # all 122 regression datasets
```

- [ ] Delete the smoke-test outputs (we want a clean experiment17/ directory before the full run):

```bash
rm experiments/experiment17/smoke.log experiments/experiment17/summary.csv experiments/experiment17/summary.json experiments/experiment17/pmlb_rmse.png experiments/experiment17/comparison_to_exp15.md
rm -f experiments/experiment17/failures.json  # may not exist if smoke had no failures
```

(The `experiments/experiment17/` dir itself stays — Task 2's full run will populate it.)

### Step 10: Run the full unit suite as a sanity check

- [ ] Run: `uv run pytest tests/unit/ -q`
- [ ] Expected: `100 passed, 1 failed` (the same pre-existing failure). The runner is a new file under `experiments/`, not under `tests/` or `eml_boost/`, so it can't affect the unit tests — but a quick re-run confirms nothing broke.

### Step 11: Commit the runner

- [ ] Run:

```bash
git add experiments/run_experiment17_matched_revalidation.py
git commit -m "$(cat <<'EOF'
add: Experiment 17 runner for matched-hyperparameter re-validation

Mirrors run_experiment15_full_pmlb.py's structure, with three changes:
1. SB: min_samples_leaf=1 (was 20), leaf_l2=1.0 (matches XGB reg_lambda)
2. XGB: min_child_weight=1 + reg_lambda=1.0 made explicit; new
   early_stopping_rounds=15 with 15% inner-val split (same outer seed)
3. LGB: min_data_in_leaf=1 (was 20), reg_lambda=1.0 (was 0); new
   early_stopping callback with same 15% inner-val split

Adds comparison_to_exp15.md auto-generation (per-dataset ratio shift
sorted by Δ). RESULTS_DIR points at experiments/experiment17/.

Smoke-tested on 1027_ESL + 663_rabe_266 + 344_mv (3 datasets × 5 seeds
× 3 models = 45 fits) before commit; runner produces expected outputs.
The full 119-dataset run is Task 2 of the plan.

Spec: docs/superpowers/specs/2026-04-25-experiment17-matched-revalidation-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Kick off the full Exp-17 run (~1.5-2h)

**Goal:** Run the validated Exp-17 runner over all 119 PMLB regression datasets and produce `summary.{csv,json}`, `pmlb_rmse.png`, `failures.json`, `comparison_to_exp15.md`. The runtime is multi-hour, so MUST checkpoint with the user before kicking off (per the project's "checkpoint before long runs" rule).

**Files:** None modified. Outputs land in `experiments/experiment17/`. Logs are gitignored (`*.log`).

### Step 1: Checkpoint with the user before kicking off

- [ ] STOP. Report to the user that Task 1 is complete (smoke test green, runner committed) and that Task 2 is the multi-hour run. Ask for explicit confirmation before launching.
- [ ] If the user says "go": proceed to Step 2.
- [ ] If the user says "wait" or wants something else first: pause; do not launch.

### Step 2: Launch the full run in the background

- [ ] Run (in background, with output tee'd to a log):

```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment17_matched_revalidation.py 2>&1 | tee experiments/experiment17/run.log
```

Recommended: use Bash with `run_in_background=true` and a 10-minute Bash-tool timeout (the actual process will keep running past the 10-min Bash timeout because the background pattern detaches it). Get the background process ID for reference.

### Step 3: Set up an error-only monitor for the run

- [ ] Use Monitor to watch for fit failures, Triton fallbacks, and the run's completion banner:

```
tail -f experiments/experiment17/run.log | grep -E --line-buffered "FETCH FAILED|SPLIT FAILED|FAILED:|Traceback|RuntimeWarning|finalizing|Headline statistics|wrote .*png"
```

This avoids spamming the controller with the 119 dataset-boundary lines and ~1800 successful fit lines, while catching anything that requires intervention.

### Step 4: Wait for completion (notification-based — don't poll)

- [ ] When the background process completes, the controller is notified. While waiting:
  - Do NOT poll the log file repeatedly.
  - Do NOT spawn other long-running work in parallel.
  - DO answer user questions or do unrelated short tasks.

### Step 5: Verify the run completed cleanly

- [ ] When notified, confirm the run finished (exit code 0) and check:

```bash
tail -25 experiments/experiment17/run.log
```

- [ ] Expected: should see the "=== Headline statistics ===" block with numbers populated, plus `wrote .../comparison_to_exp15.md`.

- [ ] Run:

```bash
grep -iE "warning|fallback|FAILED|Traceback|RuntimeWarning" experiments/experiment17/run.log | head -20
```

- [ ] Expected: only the 3 known PMLB-stale-registry FETCH FAILED lines (`195_auto_price`, `207_autoPrice`, `210_cloud`); no Triton fallback warnings; no fit failures other than those.

- [ ] If unexpected failures appear: STOP. Investigate before proceeding to Task 3.

### Step 6: Verify the success criteria from the spec

The spec specifies three success criteria. Check each:

- [ ] **S-A (correctness):** the unit suite passes (run `uv run pytest tests/unit/ -q` — confirm `100 passed, 1 failed` with the pre-existing failure unchanged); `failures.json` contains only the 3 PMLB stale-registry failures.

- [ ] **S-B (catastrophic regime closes):** read `experiments/experiment17/summary.json` and confirm the 3 catastrophic Exp-15 datasets each have new ratio ≤ 1.5:

```bash
uv run python -c "
import json
with open('experiments/experiment17/summary.json') as f:
    s = json.load(f)
for name in ('527_analcatdata_election2000', '663_rabe_266', '561_cpu'):
    r = s['ratios'].get(name, {}).get('ratio')
    print(f'  {name}: ratio={r:.3f}  → {\"PASS\" if r is not None and r <= 1.5 else \"FAIL\"}')
"
```

- [ ] **S-C (LGB story unchanged or tighter):** compute SB-vs-LGB median ratio. The Exp-15 number was 0.994 (basically tied). Read `summary.json`:

```bash
uv run python -c "
import json, statistics as st
with open('experiments/experiment17/summary.json') as f:
    s = json.load(f)
ratios_lgb = []
for name, agg in s['aggregate'].items():
    sb = agg.get('split_boost'); lg = agg.get('lightgbm')
    if sb and lg and lg['rmse_mean'] > 0:
        ratios_lgb.append(sb['rmse_mean'] / lg['rmse_mean'])
median = st.median(ratios_lgb)
print(f'SB-vs-LGB median ratio: {median:.3f} ({len(ratios_lgb)} datasets)')
print(f'  → {\"PASS (in [0.95, 1.05])\" if 0.95 <= median <= 1.05 else \"FAIL\"}')
"
```

- [ ] If S-A + S-B + S-C all pass: proceed to Task 3 (writeup).
- [ ] If any fails: STOP, report to user, do not proceed to writeup until investigation.

### Step 7: Commit the experiment artifacts

- [ ] Stage the generated files (logs are gitignored):

```bash
git add experiments/experiment17/
```

- [ ] Inspect `git status` to confirm what's staged. Expected: `summary.csv`, `summary.json`, `pmlb_rmse.png`, `failures.json`, `comparison_to_exp15.md`. NOT `run.log` (gitignored). NOT `report.md` yet (that's Task 3).

- [ ] Commit:

```bash
git commit -m "$(cat <<'EOF'
exp 17 done: full PMLB suite under matched hyperparameters

Headline (mean ratios SB / matched-XGB across 5 seeds, on the same 119
PMLB regression datasets as Exp 15):
- <fill in: within 10%, outright wins, median ratio>
- vs matched-LGB: <fill in: median ratio, win count>

Catastrophic regime: <fill in: how many of Exp-15's 3 catastrophic
datasets dropped to ratio ≤ 1.5>.

Per-dataset Δ vs Exp 15 (off-the-shelf): <fill in: mean Δ, # improved,
# regressed>. See experiments/experiment17/comparison_to_exp15.md for
the full per-dataset table.

Methodological caveats (LGB leaf-wise growth, SB EML mechanism — both
unmatchable as hyperparameters) documented in the report.md.

Wall time: <fill in> on RTX 3090. Logs gitignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `<fill in>` placeholders with actual values from the run BEFORE committing.)

---

## Task 3: Write Exp 17 report.md and update Exp 15 report.md with caveats note

**Goal:** Produce a narrative writeup of Exp 17's results following the same shape as `experiments/experiment15/report.md`, including the per-dataset shift analysis and the methodological-caveats discussion. Add a forward-pointing note from Exp 15's report to Exp 17.

**Files:**
- Create: `experiments/experiment17/report.md`
- Modify: `experiments/experiment15/report.md` (one new section near the top with a methodological note)

### Step 1: Read the Exp-17 outputs for the writeup

- [ ] Read in parallel for context:
  - `experiments/experiment17/summary.json` (full per-dataset aggregates + headline_stats + ratios)
  - `experiments/experiment17/comparison_to_exp15.md` (the auto-generated Δ table)
  - `experiments/experiment17/pmlb_rmse.png` (the headline plot)
  - `experiments/experiment15/report.md` (for style match and the existing narrative structure)

### Step 2: Draft `experiments/experiment17/report.md`

Mirror Exp 15's report.md structure. The required sections:

1. **Title + metadata** — `# Experiment 17: Full PMLB regression suite under matched hyperparameters`. Date 2026-04-25, commits, runtime, scripts.
2. **What the experiment was about** — paragraph explaining: Exp 15 used off-the-shelf defaults; this experiment matches three regularization axes (leaf-floor=1, L2=1.0, early-stopping=patience-15-with-15%-inner-val) across SB, XGB, LGB to produce a methodologically defensible baseline.
3. **Configuration** — table or code block showing the matched hyperparameters per algorithm. Reference the spec.
4. **Coverage** — datasets attempted, fittable (excluding the 3 PMLB stale-registry failures, same as Exp 15).
5. **Headline results (mean ratios SB / matched-XGB)** — table with: within-10%, within-5%, outright-wins, catastrophic, mean ratio, median ratio, P25, P75. Compare to Exp 15's same metrics in parens or a side-by-side row.
6. **Distribution of ratios** — same banding as Exp 15's report (deep wins, clear wins, narrow wins, narrow losses, losses, clear losses, catastrophic). Side-by-side count vs Exp 15.
7. **Top 10 wins / Top 10 losses** — tables showing how the bottom and top of the distribution shifted.
8. **vs LightGBM (matched)** — separate paragraph. Compare to Exp 15's "vs LGB" story.
9. **Catastrophic regime check** — explicit table of the 3 Exp-15 catastrophic datasets, their old ratios, their new ratios, and verdict (closed / partially-closed / unchanged).
10. **What Exp 17 actually shows** — interpretive paragraph(s). Possibilities to address:
    - If SB wins are roughly preserved: the architectural lead (EML) is real and survives matched comparison.
    - If SB wins dropped substantially: Exp 15's headline was largely an artifact of the unmatched defaults; SB is competitive with matched baselines but not dominant.
    - If catastrophic regime closed but win regime shrank: the diagnosis was correct AND the matching surfaced both effects.
11. **What's left as a loss** — datasets where SB still loses to XGB or LGB after matching.
12. **What Exp 17 does NOT show** — the standard caveats list, plus:
    - LGB's leaf-wise growth policy is unmatched (algorithmic).
    - SB's EML mechanism is unmatched (intentional differentiator).
    - No baseline tuning beyond the 3 axes.
    - Single 80/20 split per seed; no K-fold CV.
13. **Methodological caveats specific to Exp 17** — explain the inner-val-split same-seed-but-different-RNG note from the spec, the early-stopping behavior comparison, the LGB and EML confounds.
14. **Action taken** — based on outcomes, list any updates to the project's stated headline, library defaults (likely no library-default change in this experiment), or follow-up specs queued.
15. **Next experiments** — references the OpenML pivot (Exp 18+ per project memory) and any follow-up spec triggered by Exp 17's outcome (e.g., "if SB-vs-matched-XGB lands at 50% wins, consider tuning baselines via Optuna for an Exp 19 stronger-baselines comparison").

The writeup should be ~150-200 lines, similar in length to Exp 15's report.

### Step 3: Add the methodological-caveats note to `experiments/experiment15/report.md`

- [ ] Open `experiments/experiment15/report.md`. Find the "## What Experiment 15 actually shows" section (or wherever the narrative makes sense).

- [ ] Insert a new section at the top of that page (right after the metadata header), titled `## Methodological note (added 2026-04-25, post-Exp-17)`:

```markdown
## Methodological note (added 2026-04-25, post-Exp-17)

The headline numbers below (89% within 10%, 83% outright wins vs XGBoost,
median ratio 0.912) were obtained under an OFF-THE-SHELF DEFAULTS
comparison, not a matched-hyperparameter one. SB ran at
`min_samples_leaf=20` with `patience=15` early stopping; XGB ran at
`min_child_weight=1` (default), `reg_lambda=1.0` (default), and no
early stopping; LGB ran at `min_data_in_leaf=20` with `reg_lambda=0`
and no early stopping.

The mismatches biased the comparison in opposite directions on
different dataset regimes:
- On medium/large-n datasets, SB's `msl=20` acted as a regularizer
  XGB lacked, contributing to SB's 83% wins.
- On tiny-n datasets (n_train < 200), SB's `msl=20` blocked the algorithm
  from making the fine-grained leaf decisions XGB could (with `mcw=1`),
  contributing to all 13 of SB's losses including the 3 catastrophic
  cases.

Experiment 17 (`experiments/experiment17/report.md`,
`docs/superpowers/specs/2026-04-25-experiment17-matched-revalidation-design.md`)
re-ran the same 119-dataset suite under matched hyperparameters
(leaf-floor=1, L2 leaf-weight=1.0, early-stopping=patience-15 with
15% inner-val) across all three algorithms. Refer to that experiment
for the methodologically-defensible baseline.

The off-the-shelf comparison reported below is still meaningful
("which library is better with no per-dataset tuning?") but should
not be presented as architectural superiority of SB without the
Exp-17 caveat.
```

### Step 4: Commit both files

- [ ] Run:

```bash
git add experiments/experiment17/report.md experiments/experiment15/report.md
git commit -m "$(cat <<'EOF'
exp 17 report + Exp-15 methodological note

Writes experiments/experiment17/report.md following the same shape as
Exp 15's report (headline / distribution / wins / losses / vs-LGB /
catastrophic check / methodological caveats / next steps).

Adds a "Methodological note (added 2026-04-25, post-Exp-17)" section
at the top of experiments/experiment15/report.md acknowledging the
off-the-shelf-defaults framing of Exp 15 and pointing to Exp 17 for
the matched-hyperparameter baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 5: Report final summary to the user

After committing, report to the user:
- Final headline (Exp 17's matched-hyperparameter ratios vs Exp 15's off-the-shelf ratios)
- Catastrophic-regime verdict (did the 3 close to ≤ 1.5?)
- Whether Exp 15's narrative survives the matched comparison or shifts
- Recommendation for next steps based on outcome (typically: queue Exp 18 OpenML re-validation, OR queue a separate library-default-flip spec, OR investigate any S-A/B/C failures)

---

## Implementation order recap

1. **Task 1** — Build runner + smoke test on 3 datasets + commit.
2. **Task 2** — Pause for user confirmation; launch full ~1.5-2h run; verify success criteria; commit experiment artifacts.
3. **Task 3** — Read outputs, write report.md, add methodological note to Exp-15's report, commit.

After Task 3, regardless of outcome, report to the user. Do NOT autonomously kick off Exp 18 (OpenML) or any library-default-flip — those are separate decisions for the user.
