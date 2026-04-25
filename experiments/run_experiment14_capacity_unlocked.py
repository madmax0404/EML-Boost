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
