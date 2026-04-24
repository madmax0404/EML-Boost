"""Experiment 10: ridge-regularized EML-leaf OLS.

Sweeps 6 SplitBoost configurations across 3 seeds and the 7 PMLB
datasets used in Experiments 8/9, plus XGBoost and LightGBM baselines.
Writes aggregate stats and per-leaf η magnitude distributions.
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
from eml_boost.tree_split.nodes import EmlLeafNode, InternalNode

RESULTS_DIR = Path(__file__).resolve().parent / "experiment10"

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

# (config_id, use_stacked_blend, leaf_eml_ridge) tuples.
SPLIT_CONFIGS = [
    ("G0",        False, 0.0),
    ("G_weak",    False, 0.1),
    ("G_strong",  False, 1.0),
    ("G_vstrong", False, 10.0),
    ("B0",        True,  0.0),
    ("B_strong",  True,  1.0),
]


@dataclass
class RunResult:
    dataset: str
    seed: int
    config: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed, *, use_stacked_blend: bool, leaf_eml_ridge: float):
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
        leaf_eml_ridge=leaf_eml_ridge,
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


def _collect_eta_abs(boost: EmlSplitBoostRegressor) -> list[float]:
    """Walk every tree in the boost; return |η| for every EmlLeafNode."""
    out: list[float] = []
    def walk(node):
        if isinstance(node, EmlLeafNode):
            out.append(abs(float(node.eta)))
        elif isinstance(node, InternalNode):
            walk(node.left); walk(node.right)
    for tree in boost._trees:
        walk(tree._root)
    return out


def _summarize_etas(etas: list[float]) -> dict:
    if not etas:
        return {"count": 0, "mean_abs_eta": float("nan"),
                "max_abs_eta": float("nan"), "p99_abs_eta": float("nan")}
    arr = np.asarray(etas)
    return {
        "count": int(len(arr)),
        "mean_abs_eta": float(arr.mean()),
        "max_abs_eta": float(arr.max()),
        "p99_abs_eta": float(np.percentile(arr, 99)),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[RunResult] = []
    # dataset -> config -> seed -> eta_stats dict
    eta_stats: dict[str, dict[str, dict[int, dict]]] = {}

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        eta_stats[name] = {cfg_id: {} for cfg_id, _, _ in SPLIT_CONFIGS}

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed,
            )
            print(
                f"  [seed={seed}] n={len(X):>6}  k={X.shape[1]:>3}  "
                f"train={len(X_tr)}  test={len(X_te)}",
                flush=True,
            )

            for cfg_id, blend, ridge in SPLIT_CONFIGS:
                m, t = _fit_split_boost(
                    X_tr, y_tr, seed,
                    use_stacked_blend=blend, leaf_eml_ridge=ridge,
                )
                rmse = _rmse(m.predict(X_te), y_te)
                rows.append(RunResult(
                    dataset=name, seed=seed, config=cfg_id,
                    rmse=rmse, fit_time=t, n_rounds=m.n_rounds,
                ))
                print(
                    f"    {cfg_id:>10} ({t:6.1f}s, {m.n_rounds:>3} rounds) "
                    f"RMSE={rmse:.4f}",
                    flush=True,
                )
                eta_stats[name][cfg_id][seed] = _summarize_etas(_collect_eta_abs(m))

            m_lg, t_lg = _fit_lgb(X_tr, y_tr, seed)
            rmse_lg = _rmse(m_lg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="lightgbm",
                rmse=rmse_lg, fit_time=t_lg,
            ))
            print(f"    {'lightgbm':>10} ({t_lg:6.1f}s) RMSE={rmse_lg:.4f}", flush=True)

            m_xg, t_xg = _fit_xgb(X_tr, y_tr, seed)
            rmse_xg = _rmse(m_xg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, config="xgboost",
                rmse=rmse_xg, fit_time=t_xg,
            ))
            print(f"    {'xgboost':>10} ({t_xg:6.1f}s) RMSE={rmse_xg:.4f}", flush=True)

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

    # CSV
    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write("dataset,seed,config,rmse,fit_time,n_rounds\n")
        for r in rows:
            fp.write(f"{r.dataset},{r.seed},{r.config},{r.rmse},{r.fit_time},{r.n_rounds}\n")
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
        "split_configs": [{"id": c, "blend": b, "ridge": r} for c, b, r in SPLIT_CONFIGS],
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

    # Eta stats JSON
    eta_json_path = RESULTS_DIR / "eta_stats.json"
    with eta_json_path.open("w") as fp:
        json.dump(eta_stats, fp, indent=2)
    print(f"wrote {eta_json_path}")

    # Plot: grouped bars per dataset, one bar per SplitBoost config + XGBoost.
    ordered = sorted(DATASETS, key=lambda n: agg[(n, "xgboost")]["rmse_mean"])
    split_cfg_ids = [c for c, _, _ in SPLIT_CONFIGS]
    all_bars = split_cfg_ids + ["xgboost"]
    colors = ["#2E86AB", "#4EA1B2", "#588157", "#7FA65A", "#E9C46A", "#F4A261", "#9B2226"]

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
    ax1.set_title("Experiment 10: ridge-regularized EML leaves (log scale)")
    ax1.legend(fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3, axis="y")

    # Ratio panel: vs XGBoost for each SplitBoost config.
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
    ax2.set_yscale("log")
    ax2.set_title("Ratio vs. XGBoost (log scale)")
    ax2.legend(fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    # Console summary
    print("\n=== Aggregate summary (mean over 3 seeds, log ratio vs XGBoost) ===")
    header = f"{'dataset':>28}  " + "  ".join(f"{c:>11}" for c in split_cfg_ids)
    print(header)
    for n in ordered:
        xg_mean = agg[(n, "xgboost")]["rmse_mean"]
        cells = []
        for c in split_cfg_ids:
            r = agg[(n, c)]["rmse_mean"] / xg_mean
            cells.append(f"{r:>11.3f}")
        print(f"{n:>28}  " + "  ".join(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
