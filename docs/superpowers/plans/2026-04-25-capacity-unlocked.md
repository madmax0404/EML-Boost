# Capacity-Unlocked Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a single capacity-unlocked SplitBoost config (`max_depth=10, max_rounds=500, patience=30`) against matched-capacity XGBoost and LightGBM on the same 7 PMLB datasets × 3 seeds, then write a report comparing the result to Experiment 13's `D8_R200` reference.

**Architecture:** No library changes — both `max_depth` and `max_rounds` already accept any positive integer. A single new runner makes 63 fits total: 21 SplitBoost + 21 XGBoost + 21 LightGBM, all at depth 10 / 500 rounds. The report cross-references Exp-13's saved D8_R200 results to disentangle "is the architectural win still there" from "is the win still 7/7-within-10%."

**Tech Stack:** Python 3.11+, NumPy, PyTorch+CUDA, Triton 3.6+, PMLB, XGBoost, LightGBM, matplotlib. uv for environment management.

---

## Background an implementer needs

Experiment 13 (`experiments/experiment13/report.md`) flipped the `max_depth` default from 6 to 8 and showed a monotonic improvement: every dataset got strictly better or stayed the same vs. the Exp-12 baseline, with `557_analcatdata_apnea1` moving from 1.146 → 1.104 (just outside the strict 10% target). Experiment 14 tests whether the architectural lead holds at substantially higher capacity — `max_depth=10, max_rounds=500` — for both SplitBoost and XGBoost.

**Key wrinkle:** at 500 max rounds, the prior `patience=15` is too aggressive (3% of allowed rounds). Bump to `patience=30` (6%) to let high-capacity boosting actually use its rounds when the val set is improving slowly.

**Reference data (already on disk):** `experiments/experiment13/summary.json` has the D8_R200 results for the same 7 datasets × 3 seeds. The Exp-14 report compares against those values; no rerun needed.

**Reference runner:** `experiments/run_experiment13_apnea1_capacity.py` is the closest structural fork. The Exp-14 runner is simpler — one SplitBoost config and one matched baseline-depth, so no `BASELINE_DEPTHS` loop and no `matched_ratios` dict.

**Before starting Task 1, read:**
- `docs/superpowers/specs/2026-04-25-capacity-unlocked-design.md` — the spec.
- `experiments/run_experiment13_apnea1_capacity.py` — fork basis.
- `experiments/experiment13/report.md` — the prior result this builds on.
- `eml_boost/tree_split/ensemble.py` — confirm `max_rounds`, `patience`, and `max_depth` all accept the values we're about to use.

---

## Task 1: Create the Experiment 14 runner

**Goal:** Produce `experiments/run_experiment14_capacity_unlocked.py`. The runner loops over 7 PMLB datasets × 3 seeds × 3 models (SplitBoost D10_R500_C10, XGBoost at d=10/r=500, LightGBM at d=10/r=500). Writes CSV/JSON/PNG.

**Files:**
- Create: `experiments/run_experiment14_capacity_unlocked.py`
- Reference: `experiments/run_experiment13_apnea1_capacity.py` (fork basis — simplified)

- [ ] **Step 1: Create the runner file.**

Write `experiments/run_experiment14_capacity_unlocked.py` with the following complete contents:

```python
"""Experiment 14: capacity-unlocked mode.

A single SplitBoost config (max_depth=10, max_rounds=500, patience=30)
against matched-capacity XGBoost and LightGBM on the 7 PMLB datasets ×
3 seeds. Reference comparison against Experiment 13's D8_R200 numbers
happens in the report — those are already on disk in
experiments/experiment13/summary.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor

RESULTS_DIR = Path(__file__).resolve().parent / "experiment14"

DATASETS = [
    "192_vineyard",
    "210_cloud",
    "523_analcatdata_neavote",
    "557_analcatdata_apnea1",
    "529_pollen",
    "562_cpu_small",
    "564_fried",
]

# Capacity-unlocked configuration.
MAX_ROUNDS = 500
MAX_DEPTH = 10
PATIENCE = 30
LEARNING_RATE = 0.1
N_EML_CANDIDATES = 10
K_EML = 3
K_LEAF_EML = 1
MIN_SAMPLES_LEAF = 20
MIN_SAMPLES_LEAF_EML = 30
LEAF_EML_GAIN_THRESHOLD = 0.05
LEAF_EML_RIDGE = 0.0
LEAF_EML_CAP_K = 2.0
N_BINS = 256
TEST_SIZE = 0.20
SEEDS = [0, 1, 2]


@dataclass
class RunResult:
    dataset: str
    seed: int
    config: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
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
        patience=PATIENCE,
        val_fraction=0.15,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _fit_lgb(X_tr, y_tr, seed):
    start = time.time()
    m = lgb.train(
        dict(
            objective="regression_l2",
            max_depth=MAX_DEPTH,
            num_leaves=2**MAX_DEPTH,
            min_data_in_leaf=MIN_SAMPLES_LEAF,
            learning_rate=LEARNING_RATE,
            device="gpu",
            seed=seed,
            verbose=-1,
        ),
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=MAX_ROUNDS,
    )
    return m, time.time() - start


def _fit_xgb(X_tr, y_tr, seed):
    start = time.time()
    m = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=MAX_DEPTH,
        n_estimators=MAX_ROUNDS,
        learning_rate=LEARNING_RATE,
        device="cuda",
        verbosity=0,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[RunResult] = []

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed,
            )
            print(
                f"  [seed={seed}] n={len(X):>6}  k={X.shape[1]:>3}  "
                f"train={len(X_tr)}  test={len(X_te)}",
                flush=True,
            )

            m, t = _fit_split_boost(X_tr, y_tr, seed)
            rmse = _rmse(m.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="split_boost",
                rmse=rmse, fit_time=t, n_rounds=m.n_rounds,
            ))
            print(
                f"    split_boost ({t:6.1f}s, {m.n_rounds:>3} rounds) "
                f"RMSE={rmse:.4f}",
                flush=True,
            )

            m_lg, t_lg = _fit_lgb(X_tr, y_tr, seed)
            rmse_lg = _rmse(m_lg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="lightgbm",
                rmse=rmse_lg, fit_time=t_lg,
            ))
            print(
                f"    lightgbm    ({t_lg:6.1f}s) RMSE={rmse_lg:.4f}",
                flush=True,
            )

            m_xg, t_xg = _fit_xgb(X_tr, y_tr, seed)
            rmse_xg = _rmse(m_xg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="xgboost",
                rmse=rmse_xg, fit_time=t_xg,
            ))
            print(
                f"    xgboost     ({t_xg:6.1f}s) RMSE={rmse_xg:.4f}",
                flush=True,
            )

    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r.dataset, r.config)
        agg.setdefault(key, {"rmses": [], "times": [], "n_rounds": []})
        agg[key]["rmses"].append(r.rmse)
        agg[key]["times"].append(r.fit_time)
        agg[key]["n_rounds"].append(r.n_rounds)
    for key, d in agg.items():
        d["rmse_mean"] = float(mean(d["rmses"]))
        d["rmse_std"] = float(stdev(d["rmses"])) if len(d["rmses"]) > 1 else 0.0
        d["time_mean"] = float(mean(d["times"]))

    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write("dataset,seed,config,rmse,fit_time,n_rounds\n")
        for r in rows:
            fp.write(f"{r.dataset},{r.seed},{r.config},{r.rmse},{r.fit_time},{r.n_rounds}\n")
    print(f"\nwrote {csv_path}")

    json_path = RESULTS_DIR / "summary.json"
    out: dict = {"config": {
        "max_rounds": MAX_ROUNDS,
        "max_depth": MAX_DEPTH,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "n_eml_candidates": N_EML_CANDIDATES,
        "k_eml": K_EML,
        "k_leaf_eml": K_LEAF_EML,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "min_samples_leaf_eml": MIN_SAMPLES_LEAF_EML,
        "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
        "leaf_eml_ridge": LEAF_EML_RIDGE,
        "leaf_eml_cap_k": LEAF_EML_CAP_K,
        "n_bins": N_BINS,
        "test_size": TEST_SIZE,
        "seeds": SEEDS,
    }, "per_run": [r.__dict__ for r in rows], "aggregate": {}, "ratios": {}}
    for (ds, cfg), d in agg.items():
        out["aggregate"].setdefault(ds, {})[cfg] = {
            "rmse_mean": d["rmse_mean"], "rmse_std": d["rmse_std"],
            "time_mean": d["time_mean"],
            "rmses_per_seed": d["rmses"], "n_rounds_per_seed": d["n_rounds"],
        }
    # Convenience: ratios dict for the report.
    for ds in DATASETS:
        sb = agg[(ds, "split_boost")]["rmse_mean"]
        xg = agg[(ds, "xgboost")]["rmse_mean"]
        out["ratios"][ds] = {
            "split_boost_mean": sb,
            "xgboost_mean": xg,
            "ratio": sb / xg if xg > 0 else float("nan"),
        }
    with json_path.open("w") as fp:
        json.dump(out, fp, indent=2)
    print(f"wrote {json_path}")

    ordered = sorted(DATASETS, key=lambda n: agg[(n, "xgboost")]["rmse_mean"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), dpi=110)
    xs = np.arange(len(ordered))
    width = 0.27
    sb_means = [agg[(n, "split_boost")]["rmse_mean"] for n in ordered]
    sb_stds = [agg[(n, "split_boost")]["rmse_std"] for n in ordered]
    lg_means = [agg[(n, "lightgbm")]["rmse_mean"] for n in ordered]
    lg_stds = [agg[(n, "lightgbm")]["rmse_std"] for n in ordered]
    xg_means = [agg[(n, "xgboost")]["rmse_mean"] for n in ordered]
    xg_stds = [agg[(n, "xgboost")]["rmse_std"] for n in ordered]

    ax1.bar(xs - width, sb_means, width, yerr=sb_stds, label="SplitBoost", color="#2E86AB")
    ax1.bar(xs, lg_means, width, yerr=lg_stds, label="LightGBM", color="#588157")
    ax1.bar(xs + width, xg_means, width, yerr=xg_stds, label="XGBoost", color="#9B2226")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("test RMSE (mean ± std over 3 seeds)")
    ax1.set_yscale("log")
    ax1.set_title(
        f"Experiment 14: capacity-unlocked (max_depth={MAX_DEPTH}, "
        f"max_rounds={MAX_ROUNDS}, patience={PATIENCE})"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    ratios = [out["ratios"][n]["ratio"] for n in ordered]
    bar_colors = ["#2E86AB" if r < 1.10 else "#E63946" for r in ratios]
    ax2.bar(xs, ratios, color=bar_colors)
    ax2.axhline(1.0, color="black", linewidth=1, label="parity")
    ax2.axhline(1.1, color="gray", linestyle="--", linewidth=1, label="within 10% band")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("SplitBoost mean RMSE / XGBoost mean RMSE")
    ax2.set_title("Ratio vs. XGBoost — blue = within 10%, red = outside")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    print("\n=== Aggregate summary (mean over 3 seeds) ===")
    print(f"{'dataset':>28}  {'split':>10}  {'lgb':>10}  {'xgb':>10}  {'ratio':>7}")
    for n in ordered:
        sb = agg[(n, "split_boost")]["rmse_mean"]
        lg = agg[(n, "lightgbm")]["rmse_mean"]
        xg = agg[(n, "xgboost")]["rmse_mean"]
        r = sb / xg if xg > 0 else float("nan")
        print(f"{n:>28}  {sb:>10.4f}  {lg:>10.4f}  {xg:>10.4f}  {r:>7.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the runner on one small dataset.**

Temporarily edit `DATASETS` in place to only `["523_analcatdata_neavote"]` (one of the smaller datasets — quickest smoke) and `SEEDS` to `[0]`, then run:

```bash
uv run python experiments/run_experiment14_capacity_unlocked.py
```

Expected: completes in under 1 minute. Writes `summary.csv`, `summary.json`, `pmlb_rmse.png` to `experiments/experiment14/`. Console "Aggregate summary" prints one row.

Verify the JSON's `ratios` section has a single dataset with three keys (`split_boost_mean`, `xgboost_mean`, `ratio`):

```bash
python3 -c "import json; print(json.dumps(json.load(open('experiments/experiment14/summary.json'))['ratios'], indent=2))"
```

Then **revert** `DATASETS` and `SEEDS` to the full values before committing.

- [ ] **Step 3: Commit the runner.**

```bash
git add experiments/run_experiment14_capacity_unlocked.py
git commit -m "$(cat <<'EOF'
add: Experiment 14 runner for capacity-unlocked mode

Single SplitBoost config (max_depth=10, max_rounds=500, patience=30)
with matched-capacity XGBoost and LightGBM on the 7 PMLB datasets ×
3 seeds. Tests whether the architectural win established in
Experiment 13 holds at unlocked capacity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Self-review

- Did you create the file verbatim? No improvised additions?
- Did the smoke test produce all three output files (summary.csv, summary.json, pmlb_rmse.png)?
- Did you REVERT `DATASETS` and `SEEDS` to the full values before committing?

## Report back

- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- New commit SHA
- Brief note on the smoke-test ratio: what's the SplitBoost / XGBoost ratio for `523_analcatdata_neavote` seed 0?
- Any warnings or errors during the smoke run
- Anything surprising

---

## Task 2: Run Experiment 14 and write the report

**Goal:** Execute the full benchmark and write `experiments/experiment14/report.md` with a verdict against S-A/S-B/S-C. Compare against Experiment 13's D8_R200 numbers (already on disk) to disentangle "is the win still there?" from "is the win still 7/7-within-10%?". Optionally bump the `max_depth` default from 8 to 10 if D10_R500 is uniformly better.

**Files:**
- Execute: `experiments/run_experiment14_capacity_unlocked.py`
- Create: `experiments/experiment14/report.md`
- Commit: outputs + report + run.log (+ optional default flip)

- [ ] **Step 1: Run the full benchmark.**

```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment14_capacity_unlocked.py 2>&1 | tee experiments/experiment14/run.log
```

Expected runtime ~25-35 min on RTX 3090 (63 fits, dominated by `564_fried` at ~3-5 min per fit). All artifacts written to `experiments/experiment14/`. If a fit raises, investigate before writing the report.

- [ ] **Step 2: Read the outputs and compare against Exp 13.**

```bash
cat experiments/experiment14/summary.csv | head -30
python3 -c "import json; print(json.dumps(json.load(open('experiments/experiment14/summary.json'))['ratios'], indent=2))"
python3 -c "import json; d = json.load(open('experiments/experiment13/summary.json'))['matched_ratios']; print(json.dumps({k: v['D8_C10']['ratio'] for k, v in d.items()}, indent=2))"
```

The first command gives raw Exp 14 RMSEs. The second gives the D10_R500 ratio per dataset. The third gives the D8_R200 ratio per dataset (for comparison).

Identify per-dataset:
- D10_R500 mean ratio vs XGBoost.
- Δ vs D8_R200: does D10_R500 improve, tie, or regress vs D8_R200?
- Whether all 7 mean ratios stay ≤ 1.10 (S-A).
- Whether any dataset regresses by > 0.05 vs D8_R200 (S-B).
- Whether any RMSE on any dataset × seed exceeds 10× XGBoost RMSE (S-C).

- [ ] **Step 3: Write `experiments/experiment14/report.md`.**

Fill every `<…>` placeholder with concrete numbers. Use the Exp 13 report as a structural template.

Create the file with this structure:

```markdown
# Experiment 14: Capacity-Unlocked Mode

**Date:** 2026-04-25
**Commit:** <fill in from `git rev-parse HEAD` at this point>
**Runtime:** <fill in from run.log>
**Scripts:** `experiments/run_experiment14_capacity_unlocked.py`

## What the experiment was about

Experiment 13 established that bumping `max_depth` from 6 to 8 was a
monotonic Pareto improvement at matched capacity — every dataset got
better or tied. Experiment 14 tests whether the architectural lead
extends to substantially higher capacity: `max_depth=10, max_rounds=500`
for both SplitBoost and XGBoost. The question is "does SplitBoost
still win when both sides are unlocked?", not "is the win bigger?".

## Configuration

<copy from summary.json's config section>

## Results (mean ratios vs XGBoost, 3 seeds)

Side-by-side with the Experiment-13 D8_R200 reference (already on disk
in experiments/experiment13/summary.json).

| dataset | D8_R200 ratio (Exp 13) | **D10_R500 ratio (Exp 14)** | Δ | n_rounds (Exp 14, mean) | verdict |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

<one row per dataset; verdict says "improved", "tied", or "regressed">

### Per-seed picture on `557_analcatdata_apnea1`

<the Exp-13 marginal-miss dataset is worth seeing in detail>

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| SplitBoost RMSE | ... | ... | ... | ... |
| XGBoost RMSE | ... | ... | ... | ... |
| ratio | ... | ... | ... | ... |

## Success criteria verdict

- **S-A: all 7 datasets stay ≤ 1.10 mean ratio at unlocked capacity:**
  <MET / MARGINAL / NOT MET>
- **S-B: no dataset regresses > 0.05 vs D8_R200:** <MET / NOT MET>
- **S-C: no RMSE > 10× XGBoost on any dataset × seed:** <MET / NOT MET>

**Recommended action:** <one of: bump default max_depth from 8 to 10
because the unlocked-capacity result is uniformly better; keep d=8
default because the gain doesn't justify 3-5× runtime; flag a
specific dataset as problematic at d=10>.

## What Experiment 14 actually shows

- <headline: does SplitBoost still win at unlocked capacity?>
- <comparison to D8_R200: where does D10_R500 help, where does it
  hurt or break even?>
- <is XGBoost finally catching up at higher capacity, or does
  SplitBoost's regularization still keep the lead?>

## What's left as a loss

<any dataset still outside 10% of XGBoost under D10_R500>

## What Experiment 14 does NOT show

- Single 80/20 shuffle-split per seed; no CV.
- Full PMLB suite not tested (Experiment 15's job).
- No 2×2 sweep of (max_depth, max_rounds) — single capacity-unlocked
  config only.

## Action taken

<one of: bump default max_depth from 8 to 10; keep at 8>

## Consequence for the project

<update the headline claim from Experiment 13 if appropriate>

## Next possible experiments

<2-4 bullets>
```

Fill every `<…>` with concrete numbers. Do NOT leave placeholders.

- [ ] **Step 4: Optionally bump the `max_depth` default.**

If the report's verdict is that D10_R500 is uniformly better than D8_R200 by > 0.02 mean ratio on at least 3 datasets and not worse on any:

- `eml_boost/tree_split/tree.py`: change `max_depth: int = 8` to `max_depth: int = 10`.
- `eml_boost/tree_split/ensemble.py`: same change.

Run `uv run pytest tests/unit/ -v` to confirm all 28 in-scope tests still pass under the new default. If any test breaks under d=10 (analogous to the Experiment-12 `test_early_stopping_triggers` case), patch it by adding the relevant param to keep the test scoped to its intent, with a comment explaining why.

If the verdict is "keep d=8", skip this step. The report should explicitly state that d=10 is *available* but not the default — fine for users who want it, but the runtime cost (~3-5× per fit) doesn't justify auto-flipping.

- [ ] **Step 5: Run the unit test suite to confirm nothing drifted.**

```bash
uv run pytest tests/unit/ -v
```
Expected: all 28 in-scope tests pass (plus one pre-existing unrelated failure in `test_eml_weak_learner.py::test_fit_recovers_simple_formula`).

- [ ] **Step 6: Commit the run outputs, report, and any default change.**

```bash
git add experiments/experiment14/summary.csv experiments/experiment14/summary.json \
        experiments/experiment14/pmlb_rmse.png experiments/experiment14/report.md
git add -f experiments/experiment14/run.log
# If defaults changed in step 4:
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py
# If any unit test was patched in step 4:
git add tests/unit/
git commit -m "$(cat <<'EOF'
exp 14 done: capacity-unlocked mode on PMLB

D10_R500_C10 SplitBoost vs matched-capacity XGBoost and LightGBM on
7 PMLB datasets × 3 seeds. <fill in headline: e.g., "the architectural
win holds at unlocked capacity" or "X dataset regresses at d=10">.

<fill in default-flip note: "Default max_depth bumped from 8 to 10"
or "Kept d=8; d=10 is available but not the default given the 3-5×
runtime cost.">

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Before running the commit, edit the heredoc to fill in the verdict concretely.

---

## Self-review (plan author's checklist)

**Spec coverage:**
- Single config `D10_R500_C10` → Task 1 constants (`MAX_DEPTH=10, MAX_ROUNDS=500, PATIENCE=30`).
- Matched-capacity XGBoost and LightGBM → Task 1 `_fit_xgb` and `_fit_lgb` use the same `MAX_DEPTH` / `MAX_ROUNDS`.
- 7 datasets × 3 seeds → Task 1 `DATASETS` and `SEEDS` constants.
- All other defaults held at Exp-12/13 best (cap=2.0, ridge=0, blend=False, k_leaf_eml=1, min_samples_leaf_eml=30) → Task 1 constants.
- Outputs: summary CSV/JSON, PNG, report.md, run.log → Task 1 + Task 2.
- S-A/S-B/S-C verdict → Task 2 step 3.
- Optional default flip → Task 2 step 4.
- Reference comparison against Exp 13 D8_R200 → Task 2 step 2 reads Exp 13's summary.json.

No gaps.

**Placeholder scan:** No TBDs in task bodies. Task 2 step 3's report template has `<fill in…>` markers explicitly intended to be filled with run-time numbers.

**Type consistency:** All hyperparameter names match the existing `EmlSplitBoostRegressor.__init__` signature (`max_depth`, `max_rounds`, `patience`, `n_eml_candidates`, `leaf_eml_cap_k`, etc.). The runner's `_fit_split_boost` passes them as kwargs. The summary JSON's `ratios` dict uses `split_boost_mean` / `xgboost_mean` / `ratio` keys consistent with the report template's lookups.
