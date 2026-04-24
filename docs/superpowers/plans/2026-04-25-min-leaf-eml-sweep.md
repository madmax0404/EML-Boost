# `min_samples_leaf_eml` Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark whether lowering `min_samples_leaf_eml` below the current default of 50 unlocks EML leaves on small-n datasets (especially `210_cloud`) without regressing the medium/large-n winners.

**Architecture:** No library changes — the `min_samples_leaf_eml` parameter already exists and works at any positive int. A single new runner sweeps the parameter over `{20, 30, 40, 50}` across 3 seeds and the 7 PMLB datasets, holding all Experiment-11-best defaults fixed (including `leaf_eml_cap_k=2.0`). Writes summary + a leaf-activation-rate side stat so we can actually see whether the lower threshold produces EML leaves where it didn't before.

**Tech Stack:** Python 3.11+, NumPy, PyTorch+CUDA, Triton 3.6+, PMLB, XGBoost, LightGBM, matplotlib. uv for environment management.

---

## Background an implementer needs

`EmlSplitBoostRegressor` (in `eml_boost/tree_split/ensemble.py`) accepts `min_samples_leaf_eml: int = 50` as a hyperparameter. The parameter gates whether a leaf attempts an EML-expression fit: if the leaf's sample count falls below the threshold, `_fit_leaf` returns `LeafNode(value=constant_value)` without calling the EML machinery. The parameter is already fully functional — any positive integer works — so this experiment is pure measurement, no code changes to the library.

**Why this matters:** Experiment 11 got to 5/7 outright wins against XGBoost with the new adaptive cap (`leaf_eml_cap_k=2.0`). The two losses are `210_cloud` (ratio 1.19, n_train=86) and `557_analcatdata_apnea1` (ratio 1.13, n_train=380). On `210_cloud`, EML leaves *never activate* under the current `min_samples_leaf_eml=50` because the tree's leaves end up 20-40 samples each. This experiment tests whether dropping the threshold gets EML working on that dataset without hurting the already-winning datasets (`192_vineyard`, `523_analcatdata_neavote`, `529_pollen`, `562_cpu_small`, `564_fried`).

**Reference runner:** `experiments/run_experiment11_leaf_cap.py` is the closest structural fork basis — same 7 datasets, same 3 seeds, same baseline models. Experiment 12 swaps the sweep axis from `leaf_eml_cap_k` to `min_samples_leaf_eml`.

**Before starting Task 1, read:**
- `docs/superpowers/specs/2026-04-25-min-leaf-eml-sweep-design.md` — the spec
- `experiments/run_experiment11_leaf_cap.py` — the structural template
- `experiments/experiment11/report.md` — the results this experiment builds on
- `eml_boost/tree_split/tree.py` around `_fit_leaf` (lines ~340-430) to confirm how `min_samples_leaf_eml` is consumed (the `too_small = n < self.min_samples_leaf_eml` gate and the val-split guard `if n - val_sz < self.min_samples_leaf_eml // 2`)
- `eml_boost/tree_split/nodes.py` to remember the `LeafNode` / `EmlLeafNode` / `InternalNode` dataclasses (the runner walks the fitted tree to count EML vs constant leaves)

---

## Task 1: Create the Experiment 12 runner

**Goal:** Produce `experiments/run_experiment12_min_leaf_sweep.py`. The runner loops over the 7 PMLB datasets × 3 seeds × 4 `min_samples_leaf_eml` values, plus XGBoost and LightGBM baselines. Writes CSV/JSON/PNG plus a leaf-activation stats JSON so the report can show whether lower thresholds actually produced more EML leaves on the small-n datasets.

**Files:**
- Create: `experiments/run_experiment12_min_leaf_sweep.py`
- Reference: `experiments/run_experiment11_leaf_cap.py` (fork basis)

- [ ] **Step 1: Create the runner file.**

Write `experiments/run_experiment12_min_leaf_sweep.py` with the following complete contents:

```python
"""Experiment 12: min_samples_leaf_eml sweep.

Tests whether lowering the EML-leaf-eligibility threshold below the
current default of 50 unlocks EML on small-n datasets (especially
210_cloud) without regressing the already-winning datasets.

Sweeps min_samples_leaf_eml ∈ {20, 30, 40, 50} on 7 PMLB datasets × 3
seeds, holding all other Experiment-11-best defaults fixed — most
notably leaf_eml_cap_k=2.0 for stability on tiny-leaf OLS fits.
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
from eml_boost.tree_split.nodes import EmlLeafNode, InternalNode, LeafNode

RESULTS_DIR = Path(__file__).resolve().parent / "experiment12"

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
DEPTH = 6
LEARNING_RATE = 0.1
N_EML_CANDIDATES = 10
K_EML = 3
K_LEAF_EML = 1
LEAF_EML_GAIN_THRESHOLD = 0.05
LEAF_EML_RIDGE = 0.0
LEAF_EML_CAP_K = 2.0  # Experiment 11 default — kept on
MIN_SAMPLES_LEAF = 20  # tree-structure threshold; fixed
N_BINS = 256
TEST_SIZE = 0.20
SEEDS = [0, 1, 2]

# (config_id, min_samples_leaf_eml) tuples. Gated path only.
SPLIT_CONFIGS = [
    ("M20", 20),
    ("M30", 30),
    ("M40", 40),
    ("M50", 50),
]


@dataclass
class RunResult:
    dataset: str
    seed: int
    config: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed, *, min_samples_leaf_eml: int):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_eml_candidates=N_EML_CANDIDATES,
        k_eml=K_EML,
        n_bins=N_BINS,
        histogram_min_n=500,
        use_gpu=True,
        k_leaf_eml=K_LEAF_EML,
        min_samples_leaf_eml=min_samples_leaf_eml,
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


def _fit_lgb(X_tr, y_tr, seed):
    start = time.time()
    m = lgb.train(
        dict(
            objective="regression_l2",
            max_depth=DEPTH,
            num_leaves=2**DEPTH,
            min_data_in_leaf=20,
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
        max_depth=DEPTH,
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


def _count_leaves(boost: EmlSplitBoostRegressor) -> dict:
    """Walk every tree in the boost; return activation stats per leaf type."""
    n_eml = 0
    n_const = 0
    def walk(node):
        nonlocal n_eml, n_const
        if isinstance(node, EmlLeafNode):
            n_eml += 1
        elif isinstance(node, LeafNode):
            n_const += 1
        elif isinstance(node, InternalNode):
            walk(node.left); walk(node.right)
    for tree in boost._trees:
        walk(tree._root)
    total = n_eml + n_const
    return {
        "n_eml_leaves": n_eml,
        "n_const_leaves": n_const,
        "n_total_leaves": total,
        "eml_fraction": (n_eml / total) if total else 0.0,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[RunResult] = []
    # dataset -> config -> seed -> activation_stats
    leaf_activation_stats: dict[str, dict[str, dict[int, dict]]] = {}

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        leaf_activation_stats[name] = {cfg_id: {} for cfg_id, _ in SPLIT_CONFIGS}

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed,
            )
            print(
                f"  [seed={seed}] n={len(X):>6}  k={X.shape[1]:>3}  "
                f"train={len(X_tr)}  test={len(X_te)}",
                flush=True,
            )

            for cfg_id, min_eml in SPLIT_CONFIGS:
                m, t = _fit_split_boost(X_tr, y_tr, seed, min_samples_leaf_eml=min_eml)
                rmse = _rmse(m.predict(X_te), y_te)
                rows.append(RunResult(
                    dataset=name, seed=seed, config=cfg_id,
                    rmse=rmse, fit_time=t, n_rounds=m.n_rounds,
                ))
                print(
                    f"    {cfg_id:>6} ({t:6.1f}s, {m.n_rounds:>3} rounds) "
                    f"RMSE={rmse:.4f}",
                    flush=True,
                )
                leaf_activation_stats[name][cfg_id][seed] = _count_leaves(m)

            m_lg, t_lg = _fit_lgb(X_tr, y_tr, seed)
            rmse_lg = _rmse(m_lg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="lightgbm",
                rmse=rmse_lg, fit_time=t_lg,
            ))
            print(f"    {'lightgbm':>6} ({t_lg:6.1f}s) RMSE={rmse_lg:.4f}", flush=True)

            m_xg, t_xg = _fit_xgb(X_tr, y_tr, seed)
            rmse_xg = _rmse(m_xg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="xgboost",
                rmse=rmse_xg, fit_time=t_xg,
            ))
            print(f"    {'xgboost':>6} ({t_xg:6.1f}s) RMSE={rmse_xg:.4f}", flush=True)

    # Per-(dataset, config) aggregates.
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
        "max_rounds": MAX_ROUNDS, "depth": DEPTH,
        "learning_rate": LEARNING_RATE,
        "n_eml_candidates": N_EML_CANDIDATES, "k_eml": K_EML,
        "k_leaf_eml": K_LEAF_EML,
        "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
        "leaf_eml_ridge": LEAF_EML_RIDGE,
        "leaf_eml_cap_k": LEAF_EML_CAP_K,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "n_bins": N_BINS, "test_size": TEST_SIZE, "seeds": SEEDS,
        "split_configs": [{"id": c, "min_samples_leaf_eml": m} for c, m in SPLIT_CONFIGS],
    }, "per_run": [r.__dict__ for r in rows], "aggregate": {}}
    for (ds, cfg), d in agg.items():
        out["aggregate"].setdefault(ds, {})[cfg] = {
            "rmse_mean": d["rmse_mean"], "rmse_std": d["rmse_std"],
            "time_mean": d["time_mean"],
            "rmses_per_seed": d["rmses"], "n_rounds_per_seed": d["n_rounds"],
        }
    with json_path.open("w") as fp:
        json.dump(out, fp, indent=2)
    print(f"wrote {json_path}")

    leaf_json_path = RESULTS_DIR / "leaf_activation_stats.json"
    with leaf_json_path.open("w") as fp:
        json.dump(leaf_activation_stats, fp, indent=2)
    print(f"wrote {leaf_json_path}")

    ordered = sorted(DATASETS, key=lambda n: agg[(n, "xgboost")]["rmse_mean"])
    split_cfg_ids = [c for c, _ in SPLIT_CONFIGS]
    all_bars = split_cfg_ids + ["xgboost"]
    colors = ["#2E86AB", "#588157", "#E9C46A", "#F4A261", "#9B2226"]

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
    ax1.set_title("Experiment 12: min_samples_leaf_eml sweep (log scale)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3, axis="y")

    for i, cfg_name in enumerate(split_cfg_ids):
        ratios = [agg[(n, cfg_name)]["rmse_mean"] / agg[(n, "xgboost")]["rmse_mean"]
                  for n in ordered]
        offset = (i - len(split_cfg_ids) / 2) * width + width / 2
        ax2.bar(xs + offset, ratios, width, label=cfg_name, color=colors[i])

    ax2.axhline(1.0, color="black", linewidth=1, label="parity")
    ax2.axhline(1.1, color="gray", linestyle="--", linewidth=1, label="within 10% band")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("mean SplitBoost RMSE / XGBoost RMSE")
    ax2.set_title("Ratio vs. XGBoost")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    print("\n=== Aggregate summary (mean over 3 seeds, ratio vs XGBoost) ===")
    header = f"{'dataset':>28}  " + "  ".join(f"{c:>8}" for c in split_cfg_ids)
    print(header)
    for n in ordered:
        xg_mean = agg[(n, "xgboost")]["rmse_mean"]
        cells = []
        for c in split_cfg_ids:
            r = agg[(n, c)]["rmse_mean"] / xg_mean
            cells.append(f"{r:>8.3f}")
        print(f"{n:>28}  " + "  ".join(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the runner on one small dataset.**

Temporarily edit `DATASETS` in place to only `["210_cloud"]` (the primary target of this experiment) and `SEEDS` to `[0]`, then run:

```bash
uv run python experiments/run_experiment12_min_leaf_sweep.py
```

Expected: completes in under 1 minute. Writes `summary.csv`, `summary.json`, `leaf_activation_stats.json`, `pmlb_rmse.png` to `experiments/experiment12/`. The console "Aggregate summary" prints a row for `210_cloud`. Inspect the `leaf_activation_stats.json` content to verify that lower thresholds (M20) produce more EML leaves than M50 on cloud — this confirms the runner's activation-stats walker is working before the full run.

Then **revert** `DATASETS` and `SEEDS` to the full values before committing.

- [ ] **Step 3: Commit the runner.**

```bash
git add experiments/run_experiment12_min_leaf_sweep.py
git commit -m "$(cat <<'EOF'
add: Experiment 12 runner for min_samples_leaf_eml sweep

4 SplitBoost configs (min_samples_leaf_eml ∈ {20, 30, 40, 50}) × 3
seeds × 7 datasets against XGBoost and LightGBM. All other Exp-11
defaults held fixed (leaf_eml_cap_k=2.0). Writes summary.csv/json,
per-config leaf-activation counts so we can see whether lower
thresholds actually produce more EML leaves on small-n datasets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Self-review

- Did you create the file with exactly the content above? No improvised additions?
- Did the smoke test produce all four output files?
- Did `leaf_activation_stats.json` show more EML leaves under M20 than M50 on `210_cloud`? (If no, flag immediately — that would mean the runner or the parameter isn't wired correctly.)
- Did you REVERT `DATASETS` and `SEEDS` to the full values before committing?

## Report back

- Status: DONE | DONE_WITH_CONCERNS | BLOCKED
- New commit SHA
- Did the smoke-test produce all four output files?
- Brief note on the smoke-test activation stats: for `210_cloud` seed 0, what were the n_eml_leaves counts across M20, M30, M40, M50? (Should decrease monotonically as threshold rises.)
- Any warnings or errors during smoke-run
- Anything surprising

---

## Task 2: Run Experiment 12 and write the report

**Goal:** Execute the full benchmark and write `experiments/experiment12/report.md` with a verdict against S-A/S-B/S-C. If a threshold < 50 cleanly improves `210_cloud` without regressing winners, optionally flip the default.

**Files:**
- Execute: `experiments/run_experiment12_min_leaf_sweep.py`
- Create: `experiments/experiment12/report.md`
- Commit: outputs + report + run.log (+ optional default flip)

- [ ] **Step 1: Run the full benchmark.**

```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment12_min_leaf_sweep.py 2>&1 | tee experiments/experiment12/run.log
```

Expected runtime ~12 min on RTX 3090 (126 fits). All artifacts written to `experiments/experiment12/`. If a fit raises, investigate before writing the report.

- [ ] **Step 2: Read the outputs.**

```bash
cat experiments/experiment12/summary.csv | head -40
python3 -c "import json; print(json.dumps(json.load(open('experiments/experiment12/summary.json'))['aggregate'], indent=2))" | head -80
python3 -c "import json; print(json.dumps(json.load(open('experiments/experiment12/leaf_activation_stats.json')), indent=2))" | head -80
```

Identify per-dataset:
- Which threshold yields the lowest mean ratio vs XGBoost on `210_cloud` across 3 seeds.
- Whether that threshold regresses any of the Experiment-11 winners (`192_vineyard`, `523_analcatdata_neavote`, `529_pollen`, `562_cpu_small`, `564_fried`) by > 0.03 on mean ratio.
- For `210_cloud` specifically: `n_eml_leaves` under M20 vs M50 — ideally M20 > M50, confirming that EML actually got activated.
- Any stability issues (single-seed RMSE > 10× XGBoost).

- [ ] **Step 3: Write `experiments/experiment12/report.md`.**

Fill in every `<…>` placeholder with concrete numbers from Step 2. Use the Experiment 11 report as a structural template.

Create the file with this structure:

```markdown
# Experiment 12: `min_samples_leaf_eml` Sweep

**Date:** 2026-04-25
**Commit:** <fill in from `git rev-parse HEAD` at this point>
**Runtime:** <fill in from run.log>
**Scripts:** `experiments/run_experiment12_min_leaf_sweep.py`

## What the experiment was about

Experiment 11's leaf-cap rescue got SplitBoost to 5/7 outright wins
against XGBoost at matched capacity, stable across 3 seeds. The two
holdouts were `210_cloud` (1.19) and `557_analcatdata_apnea1` (1.13).
On `210_cloud` the cause was structural: with `min_samples_leaf_eml=50`
and n_train=86, depth-6 tree leaves of size 20-40 never reached the
EML-attempt gate, so EML leaves never activated. This experiment
sweeps the threshold over `{20, 30, 40, 50}` to see if lowering it
unlocks EML on cloud without regressing the winners.

All other Exp-11-best hyperparameters held fixed, most importantly
`leaf_eml_cap_k=2.0` — the cap's tree-selection robustness matters
*more* on small leaves where OLS is noisier.

## Configuration

<fill in config + split-configs from summary.json>

## Results (mean ratios vs XGBoost, 3 seeds)

| dataset | M20 | M30 | M40 | M50 | XGB mean | verdict |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

<one row per dataset, mean ratio to 3 decimals, with verdict column
summarizing how threshold affects the dataset>

## Leaf activation (blend-off, gated path)

`n_eml_leaves` summed across 3 seeds × 200 trees per (dataset, config).
Shows how often the EML-attempt path actually fires under each
threshold. For `210_cloud` the Exp-11 baseline (M50) should be 0;
the sweep lower values should produce positive counts.

| dataset | M20 | M30 | M40 | M50 |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## Success criteria verdict

- **S-A (210_cloud improves under some M<50):** <MET / NOT MET> —
  <which threshold gives the best cloud ratio, and what is that ratio>
- **S-B (no regression on winners > 0.03 mean ratio under the best
  M<50):** <MET / NOT MET>
- **S-C (no RMSE > 10× XGBoost on any dataset × seed):** <MET / NOT MET>

**Recommended default:** `min_samples_leaf_eml = <value>` because <reason>.

## What Experiment 12 actually shows

- <headline: does lowering the threshold unlock cloud?>
- <EML activation pattern: which datasets gain EML leaves, which don't>
- <stability: does the cap still hold on tinier leaves?>

## What's left as a loss

<any dataset still outside 10% of XGBoost under the recommended config>

## What Experiment 12 does NOT show

- Single 80/20 shuffle-split per seed; no CV.
- Full PMLB suite not tested.
- No min_samples_leaf sweep — that would change tree structure, not EML eligibility.

## Action taken

<one of: set default to recommended value; keep default=50 with
per-dataset guidance; no change>

## Consequence for the project

<update or reaffirm the headline claim from Exp 11>

## Next possible experiments

<2-4 bullets based on what Exp 12 revealed>
```

Fill every `<…>` placeholder with the concrete numbers from Step 2. Do NOT leave placeholders in the committed file.

- [ ] **Step 4: Optionally flip the default.**

If the report recommends a threshold `< 50` (cloud improves cleanly without regressions), update both constructors:

- `eml_boost/tree_split/tree.py`: change `min_samples_leaf_eml: int = 50` to the recommended value.
- `eml_boost/tree_split/ensemble.py`: same change.

Run `uv run pytest tests/unit/ -v` to confirm all tests pass under the new default.

If the report recommends keeping `min_samples_leaf_eml = 50`, skip this step.

- [ ] **Step 5: Run the unit test suite to confirm nothing drifted.**

```bash
uv run pytest tests/unit/ -v
```
Expected: all 89 in-scope tests pass. One known pre-existing `test_eml_weak_learner.py::test_fit_recovers_simple_formula` failure remains.

- [ ] **Step 6: Commit the run outputs, report, and any default change.**

```bash
git add experiments/experiment12/summary.csv experiments/experiment12/summary.json \
        experiments/experiment12/leaf_activation_stats.json \
        experiments/experiment12/pmlb_rmse.png \
        experiments/experiment12/report.md
git add -f experiments/experiment12/run.log
# If defaults changed in step 4:
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py
git commit -m "$(cat <<'EOF'
exp 12 done: min_samples_leaf_eml sweep on PMLB

4-config sweep × 3 seeds × 7 datasets. Report includes the activation
curve across thresholds, per-dataset EML-leaf counts, and a
recommended default (<fill in "kept at 50" or the new value>).
Targets Experiment 11's open loss on 210_cloud.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Before running the commit, edit the heredoc to fill in the verdict concretely so the message is specific.

---

## Self-review (plan author's checklist)

**Spec coverage:**
- 4 config grid `{20, 30, 40, 50}` → Task 1 `SPLIT_CONFIGS`.
- Same 7 datasets, 3 seeds, `leaf_eml_cap_k=2.0` held fixed → Task 1 constants and `_fit_split_boost`.
- Outputs: summary CSV/JSON, leaf_activation_stats.json, PNG, report.md, run.log → Task 1 + Task 2.
- S-A/S-B/S-C success criteria → Task 2 step 3.
- Optional default flip → Task 2 step 4.
- Val-split guard considerations noted in the spec — confirmed these all pass at the sweep values, no code changes needed.

No gaps.

**Placeholder scan:** No TBDs in any task body. Task 2 step 3's report template has `<fill in…>` markers — those are explicit instructions to replace at run time.

**Type consistency:** `SPLIT_CONFIGS` tuples match the `(config_id, min_samples_leaf_eml)` shape. The runner's `_fit_split_boost` signature passes `min_samples_leaf_eml` as a keyword, matching the regressor's parameter name exactly. `_count_leaves` walks `LeafNode`/`EmlLeafNode`/`InternalNode` — all three imported from `eml_boost.tree_split.nodes`.
