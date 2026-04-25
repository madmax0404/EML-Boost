"""Isolate why SplitBoost loses 2.34x to XGBoost on 663_rabe_266 (n=96, k=2).

Hypothesis menu (testing each by toggling a single hyperparameter):
- A: EML internal splits overfit on tiny n.
    Toggle: n_eml_candidates=0 (raw-feature splits only).
- B: EML leaves overfit on tiny n.
    Toggle: k_leaf_eml=0 (constant leaves only).
- C: SplitBoost's tree depth/min_samples_leaf differs from XGBoost in a hurtful way.
    Already matched (max_depth=8, min_samples_leaf=20); no toggle.
- Both A and B disabled → SplitBoost reduces to a vanilla histogram GBDT.
    Should match XGBoost roughly if the architecture is the only differentiator.
"""

from __future__ import annotations

import time
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor


def _fit_sb(X_tr, y_tr, seed, *, n_eml_candidates, k_leaf_eml):
    t0 = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=20, n_eml_candidates=n_eml_candidates, k_eml=3,
        n_bins=256, histogram_min_n=500, use_gpu=True,
        k_leaf_eml=k_leaf_eml, min_samples_leaf_eml=30,
        leaf_eml_gain_threshold=0.05, leaf_eml_ridge=0.0,
        leaf_eml_cap_k=2.0, use_stacked_blend=False,
        patience=15, val_fraction=0.15, random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - t0


def _fit_xgb(X_tr, y_tr, seed):
    t0 = time.time()
    m = xgb.XGBRegressor(
        objective="reg:squarederror", max_depth=8, n_estimators=200,
        learning_rate=0.1, device="cuda", verbosity=0, random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - t0


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def main():
    SEEDS = [0, 1, 2, 3, 4]
    configs = [
        ("default        (cand=10, leaf_eml=1)",  dict(n_eml_candidates=10, k_leaf_eml=1)),
        ("no-EML-splits  (cand=0,  leaf_eml=1)",  dict(n_eml_candidates=0,  k_leaf_eml=1)),
        ("no-EML-leaves  (cand=10, leaf_eml=0)",  dict(n_eml_candidates=10, k_leaf_eml=0)),
        ("vanilla GBDT   (cand=0,  leaf_eml=0)",  dict(n_eml_candidates=0,  k_leaf_eml=0)),
    ]
    for name in ("663_rabe_266", "561_cpu"):
        print(f"\n=== {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y); X, y = X[mask], y[mask]
        print(f"n={len(X)}  k={X.shape[1]}", flush=True)

        for cfg_name, cfg in configs:
            rmses = []
            for seed in SEEDS:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
                m, t = _fit_sb(X_tr, y_tr, seed, **cfg)
                rmses.append(_rmse(m.predict(X_te), y_te))
            print(f"  SB {cfg_name}: rmse mean={np.mean(rmses):.3f} std={np.std(rmses):.3f}", flush=True)

        rmses = []
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)
            m, t = _fit_xgb(X_tr, y_tr, seed)
            rmses.append(_rmse(m.predict(X_te), y_te))
        print(f"  XGB                                   : rmse mean={np.mean(rmses):.3f} std={np.std(rmses):.3f}", flush=True)


if __name__ == "__main__":
    main()
