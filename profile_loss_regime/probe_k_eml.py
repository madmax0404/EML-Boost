"""Phase-1 hypothesis check: does bumping k_eml on high-dim datasets close the gap?

k_eml controls how many top-correlation features the EML internal-split
candidates can use. Currently fixed at 3. On high-k datasets like 505_tecator
(k=124) and 651_fri_c0_100_25 (k=25), the top-3 selection may miss
information that XGB picks up via its all-features splits.

Test condition: Exp-17 matched config (msl=1, leaf_l2=1, EML enabled),
varying k_eml ∈ {3 (current), 5, 10, 25, k}. Run on:
- 505_tecator (n=192, k=124, Exp-17 ratio 1.038) — high-dim spectroscopy
- 651_fri_c0_100_25 (n=80, k=25, Exp-17 ratio 1.053) — small-n + mid-dim
- 344_mv (n=40k, k=10, Exp-17 winner ratio 0.583) — winner regression check
"""

from __future__ import annotations

import time
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor


def _fit_sb(X_tr, y_tr, seed, *, k_eml):
    m = EmlSplitBoostRegressor(
        max_rounds=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=1, leaf_l2=1.0,
        n_eml_candidates=10, k_eml=k_eml, n_bins=256, histogram_min_n=500,
        use_gpu=True, k_leaf_eml=1, min_samples_leaf_eml=30,
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
    cases = [
        ("651_fri_c0_100_25", [3, 5, 10, 25]),
        ("505_tecator",        [3, 10, 30, 124]),
        ("344_mv",              [3, 5, 10]),  # winner regression check
    ]
    for name, k_emls in cases:
        print(f"\n=== {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y); X, y = X[mask], y[mask]
        print(f"n={len(X):>6}  k={X.shape[1]:>4}", flush=True)

        # Baseline XGB
        rmses = []
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
            m = _fit_xgb(X_tr, y_tr, seed)
            rmses.append(_rmse(m.predict(X_te), y_te))
        xgb_mean = np.mean(rmses)
        print(f"  XGB                       : rmse mean={xgb_mean:10.4f}", flush=True)

        # SB with various k_eml
        for k_eml in k_emls:
            try:
                rmses = []
                for seed in SEEDS:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
                    m = _fit_sb(X_tr, y_tr, seed, k_eml=k_eml)
                    rmses.append(_rmse(m.predict(X_te), y_te))
                sb_mean = np.mean(rmses)
                ratio = sb_mean / xgb_mean
                verdict = "WIN " if ratio < 1.0 else "loss"
                print(f"  SB  k_eml={k_eml:<3}              : rmse mean={sb_mean:10.4f}  ratio={ratio:.3f}  {verdict}", flush=True)
            except Exception as e:
                print(f"  SB  k_eml={k_eml:<3}              : FAILED ({type(e).__name__}: {str(e)[:60]})", flush=True)


if __name__ == "__main__":
    main()
