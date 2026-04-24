"""Experiment 8: PMLB regression benchmark with the NEW elementary-split
boosted regressor (`EmlSplitBoostRegressor`).

Everything else is identical to Experiment 7 — same datasets, same capacity
for all three contestants, same seed, same train/test split, same metric.
The algorithm under test is the pivot design: each tree's internal nodes
may split on a raw feature OR a sampled EML expression. The hybrid-selector
family arbitration of Experiments 3-7 is gone; this is single-family
boosting over "elementary decision trees."

Compared to Experiment 7's 2/7 within-10% of XGBoost result, we're hoping
the richer per-node expressiveness narrows the gap on medium-to-large
tabular datasets where the old hybrid's depth-2 EML grammar couldn't reach.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor

RESULTS_DIR = Path(__file__).resolve().parent / "experiment8"

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
N_BINS = 256
TEST_SIZE = 0.20
SEED = 0


@dataclass
class DatasetResult:
    name: str
    n_samples: int
    n_features: int
    hybrid_rmse: float = 0.0
    lightgbm_rmse: float = 0.0
    xgboost_rmse: float = 0.0
    hybrid_time: float = 0.0
    lightgbm_time: float = 0.0
    xgboost_time: float = 0.0
    n_rounds_hybrid: int = 0


def _fit_split_boost(X_tr, y_tr, seed):
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


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[DatasetResult] = []

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED,
        )
        print(f"  n={len(X):>6}  k={X.shape[1]:>3}  train={len(X_tr)}  test={len(X_te)}", flush=True)

        r = DatasetResult(name=name, n_samples=len(X), n_features=X.shape[1])

        try:
            m_hy, t_hy = _fit_split_boost(X_tr, y_tr, SEED)
            r.hybrid_rmse = _rmse(m_hy.predict(X_te), y_te)
            r.hybrid_time = t_hy
            r.n_rounds_hybrid = m_hy.n_rounds
            print(
                f"  SplitBoost({t_hy:6.1f}s, {r.n_rounds_hybrid} rounds)  "
                f"RMSE={r.hybrid_rmse:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"  SplitBoost FAILED: {type(e).__name__}: {e}", flush=True)
            r.hybrid_rmse = float("nan")

        try:
            m_lgb, t_lgb = _fit_lgb(X_tr, y_tr, SEED)
            r.lightgbm_rmse = _rmse(m_lgb.predict(X_te), y_te)
            r.lightgbm_time = t_lgb
            print(f"  LightGBM ({t_lgb:6.1f}s)  RMSE={r.lightgbm_rmse:.4f}", flush=True)
        except Exception as e:
            print(f"  LightGBM FAILED: {e}", flush=True)
            r.lightgbm_rmse = float("nan")

        try:
            m_xgb, t_xgb = _fit_xgb(X_tr, y_tr, SEED)
            r.xgboost_rmse = _rmse(m_xgb.predict(X_te), y_te)
            r.xgboost_time = t_xgb
            print(f"  XGBoost  ({t_xgb:6.1f}s)  RMSE={r.xgboost_rmse:.4f}", flush=True)
        except Exception as e:
            print(f"  XGBoost FAILED: {e}", flush=True)
            r.xgboost_rmse = float("nan")

        ratio = (
            r.hybrid_rmse / r.xgboost_rmse if r.xgboost_rmse > 0 else float("nan")
        )
        print(f"  SplitBoost/xgb ratio: {ratio:.3f}  (within 10% iff < 1.10)", flush=True)

        results.append(r)

    # CSV
    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write(
            "dataset,n_samples,n_features,"
            "split_boost_rmse,lightgbm_rmse,xgboost_rmse,"
            "split_boost_time,lightgbm_time,xgboost_time,"
            "split_boost_vs_xgb_ratio,n_rounds_hybrid\n"
        )
        for r in results:
            ratio = (
                r.hybrid_rmse / r.xgboost_rmse if r.xgboost_rmse > 0 else float("nan")
            )
            fp.write(
                f"{r.name},{r.n_samples},{r.n_features},"
                f"{r.hybrid_rmse},{r.lightgbm_rmse},{r.xgboost_rmse},"
                f"{r.hybrid_time},{r.lightgbm_time},{r.xgboost_time},"
                f"{ratio},{r.n_rounds_hybrid}\n"
            )
    print(f"\nwrote {csv_path}")

    # JSON
    json_path = RESULTS_DIR / "summary.json"
    with json_path.open("w") as fp:
        out = {}
        for r in results:
            out[r.name] = {
                "n_samples": r.n_samples,
                "n_features": r.n_features,
                "split_boost_rmse": r.hybrid_rmse,
                "lightgbm_rmse": r.lightgbm_rmse,
                "xgboost_rmse": r.xgboost_rmse,
                "split_boost_time": r.hybrid_time,
                "lightgbm_time": r.lightgbm_time,
                "xgboost_time": r.xgboost_time,
                "split_boost_vs_xgb_ratio": (
                    r.hybrid_rmse / r.xgboost_rmse
                    if r.xgboost_rmse > 0 else None
                ),
                "n_rounds_hybrid": r.n_rounds_hybrid,
            }
        json.dump(
            {
                "config": {
                    "max_rounds": MAX_ROUNDS,
                    "depth": DEPTH,
                    "learning_rate": LEARNING_RATE,
                    "n_eml_candidates": N_EML_CANDIDATES,
                    "k_eml": K_EML,
                    "n_bins": N_BINS,
                    "test_size": TEST_SIZE,
                    "seed": SEED,
                },
                "results": out,
            },
            fp, indent=2,
        )
    print(f"wrote {json_path}")

    # Plot
    ordered = sorted(results, key=lambda r: r.n_samples)
    names = [r.name for r in ordered]
    hy = [r.hybrid_rmse for r in ordered]
    lg = [r.lightgbm_rmse for r in ordered]
    xg = [r.xgboost_rmse for r in ordered]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=110)
    xs = np.arange(len(names))
    width = 0.25
    ax1.bar(xs - width, hy, width, color="#2E86AB", label="SplitBoost")
    ax1.bar(xs, lg, width, color="#588157", label="LightGBM")
    ax1.bar(xs + width, xg, width, color="#9B2226", label="XGBoost")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(
        [f"{n}\n(n={r.n_samples}, k={r.n_features})" for n, r in zip(names, ordered)],
        rotation=20, ha="right", fontsize=8,
    )
    ax1.set_ylabel("test RMSE")
    ax1.set_title(f"PMLB regression — matched capacity (max_rounds={MAX_ROUNDS}, depth={DEPTH})")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ratios = [
        r.hybrid_rmse / r.xgboost_rmse if r.xgboost_rmse > 0 else float("nan")
        for r in ordered
    ]
    bar_colors = ["#2E86AB" if v < 1.10 else "#E63946" for v in ratios]
    ax2.bar(xs, ratios, color=bar_colors)
    ax2.axhline(1.0, color="black", linewidth=1, label="parity")
    ax2.axhline(1.1, color="gray", linewidth=1, linestyle="--", label="within 10% band")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("SplitBoost RMSE / XGBoost RMSE")
    ax2.set_title("Ratio: blue = within 10% of XGBoost; red = worse")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    # Console summary
    print("\n=== Per-dataset summary ===")
    print(f"{'dataset':>28}  {'n':>7}  {'k':>3}  {'split':>8}  {'LGBM':>8}  {'XGB':>8}  {'ratio':>6}")
    for r in ordered:
        ratio = r.hybrid_rmse / r.xgboost_rmse if r.xgboost_rmse > 0 else float("nan")
        print(
            f"{r.name:>28}  {r.n_samples:>7}  {r.n_features:>3}  "
            f"{r.hybrid_rmse:>8.4f}  {r.lightgbm_rmse:>8.4f}  {r.xgboost_rmse:>8.4f}  "
            f"{ratio:>6.2f}"
        )

    within_10pct = sum(
        1 for r in results
        if r.xgboost_rmse > 0 and r.hybrid_rmse / r.xgboost_rmse < 1.10
    )
    print(f"\nDatasets within 10% of XGBoost: {within_10pct}/{len(results)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
