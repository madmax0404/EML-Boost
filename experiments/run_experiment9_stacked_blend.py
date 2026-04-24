"""Experiment 9: PMLB regression benchmark with stacked-blend leaves.

Compares two configurations of EmlSplitBoostRegressor at the same commit:
  - blend-off: use_stacked_blend=False (legacy gate behavior).
  - blend-on:  use_stacked_blend=True  (val-fit convex blend per leaf).
Both evaluated against XGBoost and LightGBM at matched capacity, across
3 seeds, on the Experiment 8 dataset set. Reports mean±std of test RMSE
per dataset and per config, plus per-dataset leaf-stats aggregates for
the blend-on configuration.
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

RESULTS_DIR = Path(__file__).resolve().parent / "experiment9"

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
MIN_SAMPLES_LEAF_EML = 50
LEAF_EML_GAIN_THRESHOLD = 0.05
N_BINS = 256
TEST_SIZE = 0.20
SEEDS = [0, 1, 2]


@dataclass
class RunResult:
    dataset: str
    seed: int
    model: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed, *, use_stacked_blend: bool):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=20,
        n_eml_candidates=N_EML_CANDIDATES,
        k_eml=K_EML,
        n_bins=N_BINS,
        histogram_min_n=500,
        use_gpu=True,
        k_leaf_eml=K_LEAF_EML,
        min_samples_leaf_eml=MIN_SAMPLES_LEAF_EML,
        leaf_eml_gain_threshold=LEAF_EML_GAIN_THRESHOLD,
        use_stacked_blend=use_stacked_blend,
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


def _collect_leaf_stats(boost: EmlSplitBoostRegressor) -> dict:
    """Aggregate per-leaf (α, type) records across every tree in the boost."""
    all_alphas: list[float] = []
    n_eml_leaves = 0
    n_const_leaves = 0
    for tree in boost._trees:
        for s in getattr(tree, "_leaf_stats", []):
            all_alphas.append(s["alpha"])
            if s["leaf_type"] == "EmlLeafNode":
                n_eml_leaves += 1
            else:
                n_const_leaves += 1
    total = n_eml_leaves + n_const_leaves
    return {
        "n_leaf_records": total,
        "n_eml_leaves": n_eml_leaves,
        "n_const_leaves": n_const_leaves,
        "eml_leaf_fraction": n_eml_leaves / total if total else 0.0,
        "alpha_mean": float(mean(all_alphas)) if all_alphas else float("nan"),
        "alpha_stdev": float(stdev(all_alphas)) if len(all_alphas) > 1 else 0.0,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[RunResult] = []
    leaf_stats: dict[str, dict[int, dict]] = {}  # dataset -> seed -> stats

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        leaf_stats[name] = {}

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed,
            )
            print(
                f"  [seed={seed}] n={len(X):>6}  k={X.shape[1]:>3}  "
                f"train={len(X_tr)}  test={len(X_te)}",
                flush=True,
            )

            # blend-off
            m_off, t_off = _fit_split_boost(X_tr, y_tr, seed, use_stacked_blend=False)
            rmse_off = _rmse(m_off.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, model="split_boost_blend_off",
                rmse=rmse_off, fit_time=t_off, n_rounds=m_off.n_rounds,
            ))
            print(
                f"    SplitBoost/blend-off ({t_off:6.1f}s, "
                f"{m_off.n_rounds} rounds) RMSE={rmse_off:.4f}",
                flush=True,
            )

            # blend-on
            m_on, t_on = _fit_split_boost(X_tr, y_tr, seed, use_stacked_blend=True)
            rmse_on = _rmse(m_on.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, model="split_boost_blend_on",
                rmse=rmse_on, fit_time=t_on, n_rounds=m_on.n_rounds,
            ))
            print(
                f"    SplitBoost/blend-on  ({t_on:6.1f}s, "
                f"{m_on.n_rounds} rounds) RMSE={rmse_on:.4f}",
                flush=True,
            )
            leaf_stats[name][seed] = _collect_leaf_stats(m_on)

            # lightgbm
            m_lg, t_lg = _fit_lgb(X_tr, y_tr, seed)
            rmse_lg = _rmse(m_lg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, model="lightgbm",
                rmse=rmse_lg, fit_time=t_lg,
            ))
            print(f"    LightGBM             ({t_lg:6.1f}s) RMSE={rmse_lg:.4f}", flush=True)

            # xgboost
            m_xg, t_xg = _fit_xgb(X_tr, y_tr, seed)
            rmse_xg = _rmse(m_xg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, model="xgboost",
                rmse=rmse_xg, fit_time=t_xg,
            ))
            print(f"    XGBoost              ({t_xg:6.1f}s) RMSE={rmse_xg:.4f}", flush=True)

    # Per-(dataset, model) aggregates.
    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r.dataset, r.model)
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
        fp.write("dataset,seed,model,rmse,fit_time,n_rounds\n")
        for r in rows:
            fp.write(f"{r.dataset},{r.seed},{r.model},{r.rmse},{r.fit_time},{r.n_rounds}\n")
    print(f"\nwrote {csv_path}")

    # JSON with aggregates
    json_path = RESULTS_DIR / "summary.json"
    out: dict = {"config": {
        "max_rounds": MAX_ROUNDS, "depth": DEPTH,
        "learning_rate": LEARNING_RATE,
        "n_eml_candidates": N_EML_CANDIDATES, "k_eml": K_EML,
        "k_leaf_eml": K_LEAF_EML,
        "min_samples_leaf_eml": MIN_SAMPLES_LEAF_EML,
        "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
        "n_bins": N_BINS, "test_size": TEST_SIZE, "seeds": SEEDS,
    }, "per_run": [r.__dict__ for r in rows], "aggregate": {}}
    for (ds, model), d in agg.items():
        out["aggregate"].setdefault(ds, {})[model] = {
            "rmse_mean": d["rmse_mean"], "rmse_std": d["rmse_std"],
            "time_mean": d["time_mean"],
            "rmses_per_seed": d["rmses"], "n_rounds_per_seed": d["n_rounds"],
        }
    with json_path.open("w") as fp:
        json.dump(out, fp, indent=2)
    print(f"wrote {json_path}")

    # Leaf stats JSON
    leaf_json_path = RESULTS_DIR / "leaf_stats.json"
    with leaf_json_path.open("w") as fp:
        json.dump(leaf_stats, fp, indent=2)
    print(f"wrote {leaf_json_path}")

    # Plot: bars with error bars for blend-off, blend-on, xgboost.
    ordered = sorted(DATASETS, key=lambda n: agg[(n, "xgboost")]["rmse_mean"])
    means = {model: [agg[(n, model)]["rmse_mean"] for n in ordered]
             for model in ("split_boost_blend_off", "split_boost_blend_on", "xgboost")}
    stds = {model: [agg[(n, model)]["rmse_std"] for n in ordered]
            for model in ("split_boost_blend_off", "split_boost_blend_on", "xgboost")}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=110)
    xs = np.arange(len(ordered)); w = 0.25
    ax1.bar(xs - w, means["split_boost_blend_off"], w,
            yerr=stds["split_boost_blend_off"], color="#588157", label="blend-off")
    ax1.bar(xs,      means["split_boost_blend_on"], w,
            yerr=stds["split_boost_blend_on"], color="#2E86AB", label="blend-on")
    ax1.bar(xs + w,  means["xgboost"], w,
            yerr=stds["xgboost"], color="#9B2226", label="XGBoost")
    ax1.set_xticks(xs); ax1.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("test RMSE (mean ± std over 3 seeds)")
    ax1.set_title(f"Experiment 9: stacked-blend vs gate, {len(SEEDS)} seeds")
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="y")

    for n in ordered:
        if agg[(n, "xgboost")]["rmse_mean"] == 0.0:
            raise ValueError(f"XGBoost RMSE is 0 on {n}; cannot compute ratio")

    ratios_off = [agg[(n, "split_boost_blend_off")]["rmse_mean"]
                  / agg[(n, "xgboost")]["rmse_mean"] for n in ordered]
    ratios_on = [agg[(n, "split_boost_blend_on")]["rmse_mean"]
                 / agg[(n, "xgboost")]["rmse_mean"] for n in ordered]
    ax2.bar(xs - w/2, ratios_off, w, color="#588157", label="blend-off")
    ax2.bar(xs + w/2, ratios_on,  w, color="#2E86AB", label="blend-on")
    ax2.axhline(1.0, color="black", linewidth=1, label="parity")
    ax2.axhline(1.1, color="gray", linestyle="--", linewidth=1, label="within 10% band")
    ax2.set_xticks(xs); ax2.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("mean RMSE / XGBoost mean RMSE")
    ax2.set_title("Ratio vs. XGBoost")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    # Console summary
    print("\n=== Aggregate summary (mean±std over 3 seeds) ===")
    print(
        f"{'dataset':>28}  {'off_mean':>9}  {'off_std':>7}  "
        f"{'on_mean':>9}  {'on_std':>7}  {'off/xgb':>7}  {'on/xgb':>7}"
    )
    for n in ordered:
        off = agg[(n, "split_boost_blend_off")]
        on = agg[(n, "split_boost_blend_on")]
        xg = agg[(n, "xgboost")]
        print(
            f"{n:>28}  {off['rmse_mean']:>9.4f}  {off['rmse_std']:>7.4f}  "
            f"{on['rmse_mean']:>9.4f}  {on['rmse_std']:>7.4f}  "
            f"{off['rmse_mean']/xg['rmse_mean']:>7.3f}  "
            f"{on['rmse_mean']/xg['rmse_mean']:>7.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
