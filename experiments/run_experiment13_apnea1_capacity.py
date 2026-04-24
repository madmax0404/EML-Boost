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
