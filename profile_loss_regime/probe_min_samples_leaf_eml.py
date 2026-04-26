"""Phase-1 hypothesis check: does lowering min_samples_leaf_eml fix SB's
small-n losses? On 659_sleuth_ex1714 (n=37) and 210_cloud (n=86), with
min_samples_leaf=1 (Exp-17 default) and leaf_l2=1.0, vary min_samples_leaf_eml
to see if EML actually firing more closes the gap to XGBoost."""

from __future__ import annotations

import time
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor


def _fit_sb(X_tr, y_tr, seed, *, msle):
    m = EmlSplitBoostRegressor(
        max_rounds=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=1, leaf_l2=1.0,
        n_eml_candidates=10, k_eml=3, n_bins=256, histogram_min_n=500,
        use_gpu=True, k_leaf_eml=1, min_samples_leaf_eml=msle,
        leaf_eml_gain_threshold=0.05, leaf_eml_ridge=0.0,
        leaf_eml_cap_k=2.0, use_stacked_blend=False,
        patience=15, val_fraction=0.15, random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m


def _fit_xgb(X_tr, y_tr, seed):
    X_tr_in, X_v, y_tr_in, y_v = train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed)
    m = xgb.XGBRegressor(
        objective="reg:squarederror", max_depth=8, n_estimators=200,
        learning_rate=0.1, device="cuda", verbosity=0,
        min_child_weight=1, reg_lambda=1.0,
        early_stopping_rounds=15, random_state=seed,
    )
    m.fit(X_tr_in, y_tr_in, eval_set=[(X_v, y_v)], verbose=False)
    return m


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def main():
    SEEDS = [0, 1, 2, 3, 4]
    for name in ("659_sleuth_ex1714", "210_cloud", "228_elusage", "657_fri_c2_250_10"):
        print(f"\n=== {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y); X, y = X[mask], y[mask]
        print(f"n={len(X):>4}  k={X.shape[1]:>3}", flush=True)

        # Baseline XGB
        rmses = []
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
            m = _fit_xgb(X_tr, y_tr, seed)
            rmses.append(_rmse(m.predict(X_te), y_te))
        xgb_mean = np.mean(rmses)
        print(f"  XGB                       : rmse mean={xgb_mean:8.3f}", flush=True)

        # SB with various msle
        for msle in (30, 15, 10, 5, 2):
            rmses = []
            for seed in SEEDS:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
                m = _fit_sb(X_tr, y_tr, seed, msle=msle)
                rmses.append(_rmse(m.predict(X_te), y_te))
            sb_mean = np.mean(rmses)
            ratio = sb_mean / xgb_mean
            verdict = "WIN" if ratio < 1.0 else f"loss ({ratio:.3f})"
            print(f"  SB  msle={msle:<2}                : rmse mean={sb_mean:8.3f}  ratio={ratio:.3f}  {verdict}", flush=True)


if __name__ == "__main__":
    main()
