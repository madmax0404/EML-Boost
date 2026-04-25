"""Phase 1 root-cause investigation for the catastrophic small-n loss regime.

Hypothesis: XGBoost's small-n robustness comes from a SYSTEM (small leaves
+ aggressive shrinkage on small leaves + Hessian-weighted gain), not from
reg_lambda alone. SB's min_samples_leaf=20 makes leaf_l2 structurally
inactive (max 5% shrinkage). This script tests 8 conditions on each of the
3 catastrophic Exp-15 datasets to isolate which mechanism matters most.

EML is disabled in SB conditions for clean A/B vs XGB.
"""

from __future__ import annotations

import time

import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor


def _fit_sb(X_tr, y_tr, seed, *, min_samples_leaf, leaf_l2):
    t0 = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=min_samples_leaf,
        n_eml_candidates=0, k_eml=3,    # EML disabled for clean A/B
        n_bins=256, histogram_min_n=500, use_gpu=True,
        k_leaf_eml=0, min_samples_leaf_eml=30,
        leaf_eml_gain_threshold=0.05, leaf_eml_ridge=0.0,
        leaf_eml_cap_k=2.0, leaf_l2=leaf_l2,
        use_stacked_blend=False, patience=15, val_fraction=0.15,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - t0


def _fit_xgb(X_tr, y_tr, seed, *, min_child_weight, reg_lambda):
    t0 = time.time()
    m = xgb.XGBRegressor(
        objective="reg:squarederror", max_depth=8, n_estimators=200,
        learning_rate=0.1, device="cuda", verbosity=0,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - t0


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def main():
    SEEDS = [0, 1, 2, 3, 4]
    CONDITIONS = [
        ("C1  SB   msl=20  l2=1   ",   lambda X, y, s: _fit_sb(X, y, s, min_samples_leaf=20, leaf_l2=1.0)),
        ("C2  SB   msl=1   l2=1   ",   lambda X, y, s: _fit_sb(X, y, s, min_samples_leaf=1,  leaf_l2=1.0)),
        ("C3  SB   msl=20  l2=10  ",   lambda X, y, s: _fit_sb(X, y, s, min_samples_leaf=20, leaf_l2=10.0)),
        ("C4  SB   msl=20  l2=100 ",   lambda X, y, s: _fit_sb(X, y, s, min_samples_leaf=20, leaf_l2=100.0)),
        ("C5  SB   msl=1   l2=10  ",   lambda X, y, s: _fit_sb(X, y, s, min_samples_leaf=1,  leaf_l2=10.0)),
        ("C6  XGB  mcw=1   l=1    ",   lambda X, y, s: _fit_xgb(X, y, s, min_child_weight=1,  reg_lambda=1.0)),
        ("C7  XGB  mcw=20  l=1    ",   lambda X, y, s: _fit_xgb(X, y, s, min_child_weight=20, reg_lambda=1.0)),
        ("C8  XGB  mcw=20  l=0    ",   lambda X, y, s: _fit_xgb(X, y, s, min_child_weight=20, reg_lambda=0.0)),
    ]

    for name in ("561_cpu", "663_rabe_266", "527_analcatdata_election2000"):
        print(f"\n=== {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y); X, y = X[mask], y[mask]
        print(f"n={len(X):>4}  k={X.shape[1]:>3}", flush=True)

        for cond_name, cond_fn in CONDITIONS:
            rmses = []
            for seed in SEEDS:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
                m, _ = cond_fn(X_tr, y_tr, seed)
                rmses.append(_rmse(m.predict(X_te), y_te))
            mean_rmse = np.mean(rmses)
            std_rmse = np.std(rmses)
            print(f"  {cond_name}: rmse mean={mean_rmse:8.3f}  std={std_rmse:8.3f}", flush=True)


if __name__ == "__main__":
    main()
