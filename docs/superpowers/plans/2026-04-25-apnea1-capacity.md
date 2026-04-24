# Close the apnea1 Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark whether a larger capacity sweep (max_depth × n_eml_candidates) can get `557_analcatdata_apnea1` into the 10% band without regressing the six datasets currently at ≤ 10%.

**Architecture:** No library changes — both `max_depth` and `n_eml_candidates` already exist as hyperparameters. A single new runner sweeps 5 `(max_depth, n_eml_candidates)` combinations across 3 seeds and 7 PMLB datasets, plus **matched-depth** XGBoost and LightGBM baselines (2 depths = 2 XGB runs + 2 LGB runs per (dataset, seed)). Writes summary + per-config ratios so the report can identify the apnea1-winning config and verify it doesn't regress the other 6.

**Tech Stack:** Python 3.11+, NumPy, PyTorch+CUDA, Triton 3.6+, PMLB, XGBoost, LightGBM, matplotlib. uv for environment management.

---

## Background an implementer needs

`EmlSplitBoostRegressor` (in `eml_boost/tree_split/ensemble.py`) already accepts `max_depth: int = 6` and `n_eml_candidates: int = 10` as hyperparameters. Both work at any reasonable positive value — this experiment is pure measurement, not code change.

**Why it matters:** Experiment 12 got to 6/7 outright wins against XGBoost at matched capacity, stable across 3 seeds. The lone remaining miss is `557_analcatdata_apnea1` (mean ratio 1.15). Experiment 12 showed that lowering `min_samples_leaf_eml` didn't help on apnea1 (1.152 → 1.147). This experiment tests the other obvious capacity levers: deeper trees (`max_depth=8`) and more internal-split candidates (`n_eml_candidates ∈ {30, 100}`).

**Matched-depth baselines**: when a SplitBoost config uses `max_depth=8`, the XGBoost and LightGBM baselines for that (dataset, seed) must *also* use `max_depth=8` so the "ratio vs XGBoost" number is fair. That means 2 depth values across the 5 SplitBoost configs → 2 XGBoost runs + 2 LightGBM runs per (dataset, seed), not 5 each. When computing the ratio for a SplitBoost config, look up the matching-depth XGBoost value.

**Reference runner:** `experiments/run_experiment12_min_leaf_sweep.py` is the closest structural fork basis — same 7 datasets, same 3 seeds. The sweep axis changes from `min_samples_leaf_eml` to `(max_depth, n_eml_candidates)`, and the baseline wiring gains matched-depth support.

**Before starting Task 1, read:**
- `docs/superpowers/specs/2026-04-25-apnea1-capacity-design.md` — the spec
- `experiments/run_experiment12_min_leaf_sweep.py` — structural template
- `experiments/experiment11/report.md` and `experiments/experiment12/report.md` — the prior results this builds on
- Current defaults on `EmlSplitBoostRegressor.__init__` in `eml_boost/tree_split/ensemble.py` (confirming `max_depth=6, n_eml_candidates=10, min_samples_leaf_eml=30, leaf_eml_cap_k=2.0`)

---

## Task 1: Create the Experiment 13 runner

**Goal:** Produce `experiments/run_experiment13_apnea1_capacity.py`. The runner loops over 7 PMLB datasets × 3 seeds × 5 (max_depth, n_eml_candidates) configs, plus 2 matched-depth XGBoost runs and 2 matched-depth LightGBM runs per (dataset, seed). Writes CSV/JSON/PNG/report. Ratio computation is per-config, using the matched-depth XGBoost as denominator.

**Files:**
- Create: `experiments/run_experiment13_apnea1_capacity.py`
- Reference: `experiments/run_experiment12_min_leaf_sweep.py` (fork basis)

- [ ] **Step 1: Create the runner file.**

Write `experiments/run_experiment13_apnea1_capacity.py` with the following complete contents:

```python
"""Experiment 13: close the 557_analcatdata_apnea1 gap.

Sweeps (max_depth, n_eml_candidates) over 5 combinations across 3
seeds and 7 PMLB datasets, with matched-depth XGBoost and LightGBM
baselines. Primary goal: get apnea1's 1.15 mean ratio inside the 10%
band without regressing the 6 current winners.

All other Exp-12-best defaults held fixed (leaf_eml_cap_k=2.0,
min_samples_leaf_eml=30).
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

RESULTS_DIR = Path(__file__).resolve().parent / "experiment13"

DATASETS = [
    "192_vineyard",
    "210_cloud",
    "523_analcatdata_neavote",
    "557_analcatdata_apnea1",
    "529_pollen",
    "562_cpu_small",
    "564_fried",
]

MAX_ROUNDS = 200
LEARNING_RATE = 0.1
MIN_SAMPLES_LEAF = 20
K_EML = 3
K_LEAF_EML = 1
MIN_SAMPLES_LEAF_EML = 30
LEAF_EML_GAIN_THRESHOLD = 0.05
LEAF_EML_RIDGE = 0.0
LEAF_EML_CAP_K = 2.0
N_BINS = 256
TEST_SIZE = 0.20
SEEDS = [0, 1, 2]

# (config_id, max_depth, n_eml_candidates). Gated path only.
SPLIT_CONFIGS = [
    ("D6_C10",  6, 10),   # Exp-12 baseline
    ("D6_C30",  6, 30),
    ("D6_C100", 6, 100),
    ("D8_C10",  8, 10),
    ("D8_C30",  8, 30),
]

# Unique depths for matched-depth baselines.
BASELINE_DEPTHS = sorted({d for _, d, _ in SPLIT_CONFIGS})  # -> [6, 8]


@dataclass
class RunResult:
    dataset: str
    seed: int
    config: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed, *, max_depth: int, n_eml_candidates: int):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=max_depth,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_eml_candidates=n_eml_candidates,
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
        patience=15,
        val_fraction=0.15,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _fit_lgb(X_tr, y_tr, seed, *, max_depth: int):
    start = time.time()
    m = lgb.train(
        dict(
            objective="regression_l2",
            max_depth=max_depth,
            num_leaves=2**max_depth,
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


def _fit_xgb(X_tr, y_tr, seed, *, max_depth: int):
    start = time.time()
    m = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=max_depth,
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

            for cfg_id, depth, n_eml in SPLIT_CONFIGS:
                m, t = _fit_split_boost(
                    X_tr, y_tr, seed,
                    max_depth=depth, n_eml_candidates=n_eml,
                )
                rmse = _rmse(m.predict(X_te), y_te)
                rows.append(RunResult(
                    dataset=name, seed=seed, config=cfg_id,
                    rmse=rmse, fit_time=t, n_rounds=m.n_rounds,
                ))
                print(
                    f"    {cfg_id:>8} ({t:6.1f}s, {m.n_rounds:>3} rounds) "
                    f"RMSE={rmse:.4f}",
                    flush=True,
                )

            for depth in BASELINE_DEPTHS:
                m_lg, t_lg = _fit_lgb(X_tr, y_tr, seed, max_depth=depth)
                rmse_lg = _rmse(m_lg.predict(X_te), y_te)
                rows.append(RunResult(
                    dataset=name, seed=seed, config=f"lightgbm_d{depth}",
                    rmse=rmse_lg, fit_time=t_lg,
                ))
                print(
                    f"    {'lgb_d' + str(depth):>8} ({t_lg:6.1f}s) RMSE={rmse_lg:.4f}",
                    flush=True,
                )

                m_xg, t_xg = _fit_xgb(X_tr, y_tr, seed, max_depth=depth)
                rmse_xg = _rmse(m_xg.predict(X_te), y_te)
                rows.append(RunResult(
                    dataset=name, seed=seed, config=f"xgboost_d{depth}",
                    rmse=rmse_xg, fit_time=t_xg,
                ))
                print(
                    f"    {'xgb_d' + str(depth):>8} ({t_xg:6.1f}s) RMSE={rmse_xg:.4f}",
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

    # CSV
    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write("dataset,seed,config,rmse,fit_time,n_rounds\n")
        for r in rows:
            fp.write(f"{r.dataset},{r.seed},{r.config},{r.rmse},{r.fit_time},{r.n_rounds}\n")
    print(f"\nwrote {csv_path}")

    # JSON with aggregates + matched-depth ratios
    json_path = RESULTS_DIR / "summary.json"
    out: dict = {"config": {
        "max_rounds": MAX_ROUNDS,
        "learning_rate": LEARNING_RATE,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "k_eml": K_EML,
        "k_leaf_eml": K_LEAF_EML,
        "min_samples_leaf_eml": MIN_SAMPLES_LEAF_EML,
        "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
        "leaf_eml_ridge": LEAF_EML_RIDGE,
        "leaf_eml_cap_k": LEAF_EML_CAP_K,
        "n_bins": N_BINS,
        "test_size": TEST_SIZE,
        "seeds": SEEDS,
        "baseline_depths": BASELINE_DEPTHS,
        "split_configs": [
            {"id": c, "max_depth": d, "n_eml_candidates": n}
            for c, d, n in SPLIT_CONFIGS
        ],
    }, "per_run": [r.__dict__ for r in rows], "aggregate": {}, "matched_ratios": {}}
    for (ds, cfg), d in agg.items():
        out["aggregate"].setdefault(ds, {})[cfg] = {
            "rmse_mean": d["rmse_mean"], "rmse_std": d["rmse_std"],
            "time_mean": d["time_mean"],
            "rmses_per_seed": d["rmses"], "n_rounds_per_seed": d["n_rounds"],
        }
    # Matched-depth ratios: for each SplitBoost config, divide by the
    # XGBoost run at the same max_depth.
    depth_map = {cfg_id: d for cfg_id, d, _ in SPLIT_CONFIGS}
    for ds in DATASETS:
        out["matched_ratios"][ds] = {}
        for cfg_id, d, _ in SPLIT_CONFIGS:
            sb_mean = agg[(ds, cfg_id)]["rmse_mean"]
            xgb_mean = agg[(ds, f"xgboost_d{d}")]["rmse_mean"]
            out["matched_ratios"][ds][cfg_id] = {
                "split_boost_mean": sb_mean,
                "xgboost_d{}_mean".format(d): xgb_mean,
                "ratio": sb_mean / xgb_mean if xgb_mean > 0 else float("nan"),
            }
    with json_path.open("w") as fp:
        json.dump(out, fp, indent=2)
    print(f"wrote {json_path}")

    # Plot: two-panel bar chart. Top: RMSE per dataset, grouped by all
    # configs (log scale). Bottom: matched-depth ratio vs XGBoost per
    # SplitBoost config.
    ordered = sorted(DATASETS, key=lambda n: agg[(n, "xgboost_d6")]["rmse_mean"])
    split_cfg_ids = [c for c, _, _ in SPLIT_CONFIGS]
    all_bars = split_cfg_ids + [f"xgboost_d{d}" for d in BASELINE_DEPTHS]
    colors = ["#2E86AB", "#4EA1B2", "#588157", "#E9C46A", "#F4A261", "#9B2226", "#B5374A"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=110)
    xs = np.arange(len(ordered))
    width = 0.8 / len(all_bars)

    for i, cfg_name in enumerate(all_bars):
        means_ = [agg[(n, cfg_name)]["rmse_mean"] for n in ordered]
        stds_ = [agg[(n, cfg_name)]["rmse_std"] for n in ordered]
        offset = (i - len(all_bars) / 2) * width + width / 2
        ax1.bar(xs + offset, means_, width, yerr=stds_, label=cfg_name, color=colors[i])

    ax1.set_xticks(xs)
    ax1.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("test RMSE (mean ± std over 3 seeds)")
    ax1.set_yscale("log")
    ax1.set_title("Experiment 13: capacity sweep (log scale)")
    ax1.legend(fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3, axis="y")

    # Ratio panel uses matched-depth XGBoost denominator per config.
    for i, cfg_id in enumerate(split_cfg_ids):
        d = depth_map[cfg_id]
        ratios = [
            agg[(n, cfg_id)]["rmse_mean"] / agg[(n, f"xgboost_d{d}")]["rmse_mean"]
            for n in ordered
        ]
        offset = (i - len(split_cfg_ids) / 2) * width + width / 2
        ax2.bar(xs + offset, ratios, width, label=cfg_id, color=colors[i])

    ax2.axhline(1.0, color="black", linewidth=1, label="parity")
    ax2.axhline(1.1, color="gray", linestyle="--", linewidth=1, label="within 10% band")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("mean SplitBoost RMSE / matched-depth XGBoost RMSE")
    ax2.set_title("Ratio vs. matched-depth XGBoost")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    # Console summary — matched-depth ratios.
    print("\n=== Aggregate summary (mean over 3 seeds, ratio vs matched-depth XGBoost) ===")
    header = f"{'dataset':>28}  " + "  ".join(f"{c:>8}" for c in split_cfg_ids)
    print(header)
    for n in ordered:
        cells = []
        for cfg_id in split_cfg_ids:
            d = depth_map[cfg_id]
            sb = agg[(n, cfg_id)]["rmse_mean"]
            xg = agg[(n, f"xgboost_d{d}")]["rmse_mean"]
            r = sb / xg if xg > 0 else float("nan")
            cells.append(f"{r:>8.3f}")
        print(f"{n:>28}  " + "  ".join(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the runner on one small dataset.**

Temporarily edit `DATASETS` in place to only `["557_analcatdata_apnea1"]` (the target of this experiment) and `SEEDS` to `[0]`, then run:

```bash
uv run python experiments/run_experiment13_apnea1_capacity.py
```

Expected: completes in under 2 minutes. Writes `summary.csv`, `summary.json`, `pmlb_rmse.png` to `experiments/experiment13/`. Console "Aggregate summary" prints a row for `557_analcatdata_apnea1` with ratios across all 5 configs.

Verify the matched-depth wiring by inspecting summary.json's `matched_ratios` section:

```bash
python3 -c "import json; d = json.load(open('experiments/experiment13/summary.json')); print(json.dumps(d['matched_ratios'], indent=2))"
```

Confirm each SplitBoost config's ratio uses the correct-depth XGBoost as denominator (D6_* configs should reference `xgboost_d6_mean`, D8_* configs should reference `xgboost_d8_mean`).

Then **revert** `DATASETS` and `SEEDS` to the full values before committing.

- [ ] **Step 3: Commit the runner.**

```bash
git add experiments/run_experiment13_apnea1_capacity.py
git commit -m "$(cat <<'EOF'
add: Experiment 13 runner for apnea1 capacity sweep

5 SplitBoost configs sweeping (max_depth, n_eml_candidates) over
(6,10), (6,30), (6,100), (8,10), (8,30). 3 seeds × 7 datasets with
matched-depth XGBoost and LightGBM baselines. Ratio computation is
per-config using the matching-depth XGBoost as denominator, so D8
configs aren't unfairly advantaged over a d=6 XGBoost baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Self-review

- Did you create the file verbatim? No improvised additions?
- Did the smoke test produce all three output files (summary.csv, summary.json, pmlb_rmse.png)?
- Did the matched-depth wiring produce the right XGBoost denominators? (D6_* should use xgboost_d6, D8_* should use xgboost_d8 — verify in `matched_ratios`.)
- Did you REVERT `DATASETS` and `SEEDS` before committing?

## Report back

- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- New commit SHA
- Brief note on the smoke-test `matched_ratios` for `557_analcatdata_apnea1` seed 0: which config ratio is lowest? This is informational only — the 3-seed mean is the real verdict.
- Any warnings or errors during smoke-run
- Anything surprising

---

## Task 2: Run Experiment 13 and write the report

**Goal:** Execute the full benchmark and write `experiments/experiment13/report.md` with a verdict against S-A/S-B/S-C. If a config closes apnea1 and doesn't regress the 6 winners, flip the default.

**Files:**
- Execute: `experiments/run_experiment13_apnea1_capacity.py`
- Create: `experiments/experiment13/report.md`
- Commit: outputs + report + run.log (+ optional default flip)

- [ ] **Step 1: Run the full benchmark.**

```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment13_apnea1_capacity.py 2>&1 | tee experiments/experiment13/run.log
```

Expected runtime ~30-40 min on RTX 3090 (189 fits). All artifacts written to `experiments/experiment13/`. If a fit raises, investigate before writing the report.

- [ ] **Step 2: Read the outputs.**

```bash
cat experiments/experiment13/summary.csv | head -40
python3 -c "import json; print(json.dumps(json.load(open('experiments/experiment13/summary.json'))['matched_ratios'], indent=2))" | head -80
```

Identify per-dataset:
- Which SplitBoost config has the lowest matched-depth ratio vs XGBoost on `557_analcatdata_apnea1` across 3 seeds.
- Whether that config's mean ratio on every other dataset is within 0.03 of the D6_C10 baseline (i.e., no regression on the 6 current winners).
- Any stability issues (single-seed RMSE > 10× matched-depth XGBoost).

- [ ] **Step 3: Write `experiments/experiment13/report.md`.**

Fill every `<…>` placeholder with concrete numbers. Use the Experiment 12 report as a structural template.

Create the file with this structure:

```markdown
# Experiment 13: Close the `557_analcatdata_apnea1` Gap

**Date:** 2026-04-25
**Commit:** <fill in from `git rev-parse HEAD` at this point>
**Runtime:** <fill in from run.log>
**Scripts:** `experiments/run_experiment13_apnea1_capacity.py`

## What the experiment was about

Experiment 12 got SplitBoost to 6/7 within 10% of XGBoost at matched
capacity, stable across 3 seeds. The lone remaining miss was
`557_analcatdata_apnea1` (mean ratio 1.15). Experiment 12 showed
lowering `min_samples_leaf_eml` didn't help. This experiment sweeps
the other two obvious capacity levers: `max_depth` (6 vs 8) and
`n_eml_candidates` (10 vs 30 vs 100). Baselines use matched depth so
the ratio is fair.

## Configuration

<fill in config + split-configs from summary.json>

## Results (mean ratios vs matched-depth XGBoost, 3 seeds)

| dataset | D6_C10 | D6_C30 | D6_C100 | D8_C10 | D8_C30 | verdict |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

<one row per dataset, mean ratio to 3 decimals, verdict column
summarizes which config is best for each dataset>

### Per-seed on `557_analcatdata_apnea1`

| config | seed 0 | seed 1 | seed 2 | mean | XGB @ matched depth |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

## Success criteria verdict

- **S-A (primary): some config gets apnea1 mean ratio below 1.10:**
  <MET / NOT MET> — best config is <id> at <ratio>
- **S-B (no regression on winners > 0.03 under apnea1-best config):**
  <MET / NOT MET>
- **S-C (no RMSE > 10× matched-depth XGBoost on any dataset × seed):**
  <MET / NOT MET>

**Recommended default:** <config or "keep Exp-12 defaults"> because <reason>.

## What Experiment 13 actually shows

- <headline: did any config close apnea1?>
- <axes comparison: deeper trees vs more candidates — which helped more?>
- <cross-dataset effects: did the apnea1-winning config regress anyone?>
- <stability: did depth=8 cause any seed explosions on smaller datasets?>

## What's left as a loss

<any dataset still outside the band under the recommended config>

## What Experiment 13 does NOT show

- Single 80/20 shuffle-split per seed; no CV.
- Full PMLB suite not tested.
- No `max_depth=10` or `n_eml_candidates=300` — diminishing returns.

## Action taken

<one of: flip default max_depth / n_eml_candidates to the winning
config; keep Exp-12 defaults; document apnea1 as structurally stuck>

## Consequence for the project

<update or reaffirm the headline claim from Exp 11/12>

## Next possible experiments

<2-4 bullets based on what Exp 13 revealed>
```

Fill every `<…>` placeholder. Do NOT leave placeholders in the committed file.

- [ ] **Step 4: Optionally flip the default.**

If the report recommends a different `(max_depth, n_eml_candidates)` pair (apnea1 closes without regressions), update both constructors:

- `eml_boost/tree_split/tree.py`: update `max_depth: int = 6` and/or `n_eml_candidates: int = 10` to the recommended values.
- `eml_boost/tree_split/ensemble.py`: same changes.

Run `uv run pytest tests/unit/ -v` to confirm all tests still pass under the new default. If any test's synthetic fixture now triggers EML leaves where before it didn't (analogous to the Experiment-12 `test_early_stopping_triggers` fix), patch the test by adding `k_leaf_eml=0` or similar to disable the new behavior, documenting why in a comment.

If the report recommends keeping the Exp-12 defaults, skip this step.

- [ ] **Step 5: Run the unit test suite to confirm nothing drifted.**

```bash
uv run pytest tests/unit/ -v
```
Expected: all 28 in-scope tests pass (plus one pre-existing unrelated failure in `test_eml_weak_learner.py::test_fit_recovers_simple_formula`).

- [ ] **Step 6: Commit the run outputs, report, and any default change.**

```bash
git add experiments/experiment13/summary.csv experiments/experiment13/summary.json \
        experiments/experiment13/pmlb_rmse.png experiments/experiment13/report.md
git add -f experiments/experiment13/run.log
# If defaults changed in step 4:
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py
# If any unit test was patched in step 4:
git add tests/unit/
git commit -m "$(cat <<'EOF'
exp 13 done: apnea1 capacity sweep on PMLB

5-config sweep (max_depth × n_eml_candidates) × 3 seeds × 7 datasets
with matched-depth XGBoost baselines. Report includes the ratio
curve across configs and a recommended default (<fill in "kept at
d=6, c=10" or the new values>). Targets Experiment 12's open loss on
557_analcatdata_apnea1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Before running the commit, edit the heredoc to fill in the verdict concretely.

---

## Self-review (plan author's checklist)

**Spec coverage:**
- 5 config grid → Task 1 `SPLIT_CONFIGS`.
- Matched-depth baselines (2 XGB + 2 LGB per (dataset, seed)) → Task 1 baseline loop (inner `for depth in BASELINE_DEPTHS`).
- Per-config ratio computed against matching-depth XGBoost → Task 1 `matched_ratios` JSON section + plot ratio panel.
- All 7 datasets × 3 seeds fixed → Task 1 top-level loops.
- All other defaults held at Exp-12 values → Task 1 `_fit_split_boost` body.
- Outputs: summary CSV/JSON, PNG, report.md, run.log → Task 1 + Task 2.
- S-A/S-B/S-C verdict → Task 2 step 3.
- Optional default flip → Task 2 step 4.

No gaps.

**Placeholder scan:** No TBDs in task bodies. Task 2 step 3's report template has `<fill in…>` markers explicitly intended to be filled with post-run numbers.

**Type consistency:** `SPLIT_CONFIGS` tuples are `(config_id, max_depth, n_eml_candidates)` — 3-tuples, referenced consistently via `for cfg_id, d, _ in SPLIT_CONFIGS` in the ratio-computation section. `BASELINE_DEPTHS` is derived from `SPLIT_CONFIGS` via `sorted({d for _, d, _ in ...})`, so it stays in sync if the grid changes. The per-run baseline config_id is `f"xgboost_d{depth}"` / `f"lightgbm_d{depth}"` — a consistent naming used in both the `rows.append(...)` and the `matched_ratios` lookup.
