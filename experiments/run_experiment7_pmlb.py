"""Experiment 7: PMLB regression benchmark with fully matched capacity.

Every knob we can match between the hybrid and the tree baselines is matched:
  max_rounds / n_estimators = 200
  max_depth / depth_dt      = 6
  learning_rate             = 0.1
  seed                      = 0
  patience                  = disabled (full capacity used)
  device                    = CUDA/GPU where supported

The hybrid's EML branch runs exhaustive search at depth 2, k=3 (6,400 trees
per round); the `_EXHAUSTIVE_EVAL_CAP = 500` in the production code keeps the
per-round cost bounded on larger datasets.

Reported metric per dataset: test RMSE ratio relative to the strongest
baseline (XGBoost). `ratio < 1` means the hybrid beats XGBoost; the
spec's "within 10%" threshold corresponds to `ratio < 1.10`.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost import EmlBoostRegressor

RESULTS_DIR = Path(__file__).resolve().parent / "experiment7"

DATASETS = [
    "192_vineyard",
    "210_cloud",
    "523_analcatdata_neavote",
    "557_analcatdata_apnea1",
    "529_pollen",
    "562_cpu_small",
    "564_fried",
]

# Matched capacity across the three families.
MAX_ROUNDS = 200
DEPTH = 6
LEARNING_RATE = 0.1
DEPTH_EML = 2     # hybrid's closed-form branch (exhaustive-reach)
K = 3             # hybrid's top-k feature selection
N_RESTARTS = 6    # unused on exhaustive path; kept for softmax fallback

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
    hybrid_formula: str | None = None
    hybrid_coverage: float | None = None


def _fit_hybrid(X_tr, y_tr, seed):
    start = time.time()
    m = EmlBoostRegressor(
        max_rounds=MAX_ROUNDS,
        depth_eml=DEPTH_EML,
        depth_dt=DEPTH,
        n_restarts=N_RESTARTS,
        k=K,
        patience=MAX_ROUNDS,
        eta_dt=LEARNING_RATE,
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

    all_results: list[DatasetResult] = []

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===")
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Simple shuffle-split. Drop rows with NaN (PMLB data is clean but defensive).
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED,
        )
        print(f"  n={len(X):>6}  k={X.shape[1]:>3}  train={len(X_tr)}  test={len(X_te)}")

        r = DatasetResult(name=name, n_samples=len(X), n_features=X.shape[1])

        try:
            m_hy, t_hy = _fit_hybrid(X_tr, y_tr, SEED)
            r.hybrid_rmse = _rmse(m_hy.predict(X_te), y_te)
            r.hybrid_time = t_hy
            try:
                r.hybrid_coverage = float(m_hy.coverage(X_tr))
            except Exception:
                r.hybrid_coverage = None
            if m_hy.formula is not None:
                fs = str(m_hy.formula)
                r.hybrid_formula = fs[:200] + (" ...[truncated]" if len(fs) > 200 else "")
            print(
                f"  Hybrid   ({t_hy:6.1f}s)  RMSE={r.hybrid_rmse:.4f}"
                + (f"  coverage={r.hybrid_coverage:.3f}" if r.hybrid_coverage is not None else "")
            )
        except Exception as e:
            print(f"  Hybrid FAILED: {type(e).__name__}: {e}")
            r.hybrid_rmse = float("nan")

        try:
            m_lgb, t_lgb = _fit_lgb(X_tr, y_tr, SEED)
            r.lightgbm_rmse = _rmse(m_lgb.predict(X_te), y_te)
            r.lightgbm_time = t_lgb
            print(f"  LightGBM ({t_lgb:6.1f}s)  RMSE={r.lightgbm_rmse:.4f}")
        except Exception as e:
            print(f"  LightGBM FAILED: {type(e).__name__}: {e}")
            r.lightgbm_rmse = float("nan")

        try:
            m_xgb, t_xgb = _fit_xgb(X_tr, y_tr, SEED)
            r.xgboost_rmse = _rmse(m_xgb.predict(X_te), y_te)
            r.xgboost_time = t_xgb
            print(f"  XGBoost  ({t_xgb:6.1f}s)  RMSE={r.xgboost_rmse:.4f}")
        except Exception as e:
            print(f"  XGBoost FAILED: {type(e).__name__}: {e}")
            r.xgboost_rmse = float("nan")

        ratio = r.hybrid_rmse / r.xgboost_rmse if r.xgboost_rmse > 0 else float("nan")
        print(f"  hybrid/xgb ratio: {ratio:.3f} (within 10% iff < 1.10)")

        all_results.append(r)

    # CSV
    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write(
            "dataset,n_samples,n_features,"
            "hybrid_rmse,lightgbm_rmse,xgboost_rmse,"
            "hybrid_time,lightgbm_time,xgboost_time,"
            "hybrid_vs_xgb_ratio,hybrid_coverage\n"
        )
        for r in all_results:
            ratio = r.hybrid_rmse / r.xgboost_rmse if r.xgboost_rmse > 0 else float("nan")
            cov = "" if r.hybrid_coverage is None else f"{r.hybrid_coverage}"
            fp.write(
                f"{r.name},{r.n_samples},{r.n_features},"
                f"{r.hybrid_rmse},{r.lightgbm_rmse},{r.xgboost_rmse},"
                f"{r.hybrid_time},{r.lightgbm_time},{r.xgboost_time},"
                f"{ratio},{cov}\n"
            )
    print(f"\nwrote {csv_path}")

    # JSON
    json_path = RESULTS_DIR / "summary.json"
    with json_path.open("w") as fp:
        out = {}
        for r in all_results:
            out[r.name] = {
                "n_samples": r.n_samples,
                "n_features": r.n_features,
                "hybrid_rmse": r.hybrid_rmse,
                "lightgbm_rmse": r.lightgbm_rmse,
                "xgboost_rmse": r.xgboost_rmse,
                "hybrid_time": r.hybrid_time,
                "lightgbm_time": r.lightgbm_time,
                "xgboost_time": r.xgboost_time,
                "hybrid_vs_xgb_ratio": (
                    r.hybrid_rmse / r.xgboost_rmse if r.xgboost_rmse > 0 else None
                ),
                "hybrid_coverage": r.hybrid_coverage,
                "hybrid_formula": r.hybrid_formula,
            }
        json.dump(
            {
                "config": {
                    "max_rounds": MAX_ROUNDS,
                    "depth": DEPTH,
                    "depth_eml": DEPTH_EML,
                    "k": K,
                    "learning_rate": LEARNING_RATE,
                    "test_size": TEST_SIZE,
                    "seed": SEED,
                },
                "results": out,
            },
            fp, indent=2,
        )
    print(f"wrote {json_path}")

    # Plot: grouped bar chart of RMSE per dataset, ordered by n_samples
    ordered = sorted(all_results, key=lambda r: r.n_samples)
    names = [r.name for r in ordered]
    hy = [r.hybrid_rmse for r in ordered]
    lg = [r.lightgbm_rmse for r in ordered]
    xg = [r.xgboost_rmse for r in ordered]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=110)

    xs = np.arange(len(names))
    width = 0.25
    ax1.bar(xs - width, hy, width, color="#2E86AB", label="Hybrid")
    ax1.bar(xs,         lg, width, color="#588157", label="LightGBM")
    ax1.bar(xs + width, xg, width, color="#9B2226", label="XGBoost")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{n}\n(n={r.n_samples}, k={r.n_features})" for n, r in zip(names, ordered)], rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("test RMSE")
    ax1.set_title(f"PMLB regression — matched capacity (max_rounds={MAX_ROUNDS}, depth={DEPTH})")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Bottom: hybrid/XGBoost ratio per dataset
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
    ax2.set_ylabel("hybrid RMSE / XGBoost RMSE")
    ax2.set_title("Ratio: blue = within 10% of XGBoost; red = worse")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    # Console summary
    print("\n=== Per-dataset summary ===")
    print(f"{'dataset':>28}  {'n':>7}  {'k':>3}  {'hybrid':>8}  {'LGBM':>8}  {'XGB':>8}  {'ratio':>6}")
    for r in ordered:
        ratio = r.hybrid_rmse / r.xgboost_rmse if r.xgboost_rmse > 0 else float("nan")
        print(
            f"{r.name:>28}  {r.n_samples:>7}  {r.n_features:>3}  "
            f"{r.hybrid_rmse:>8.4f}  {r.lightgbm_rmse:>8.4f}  {r.xgboost_rmse:>8.4f}  "
            f"{ratio:>6.2f}"
        )

    within_10pct = sum(
        1 for r in all_results if r.hybrid_rmse / max(r.xgboost_rmse, 1e-12) < 1.10
    )
    print(f"\nDatasets within 10% of XGBoost: {within_10pct}/{len(all_results)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
