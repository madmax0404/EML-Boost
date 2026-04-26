"""20-seed paired re-validation of Exp-17's 9 SB-vs-XGB losses (ratio >= 1.05).

For each loser, fit SB and XGB with matched hyperparameters × 20 outer seeds.
Report mean ratio, paired 95% CI, win/loss counts, sign-test p-value.
Goal: distinguish real algorithmic losses from underpowered-comparison noise.
"""

from __future__ import annotations

from math import comb
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor

LOSERS = [
    ("210_cloud",                 1.147),
    ("659_sleuth_ex1714",         1.118),
    ("627_fri_c2_500_10",         1.096),
    ("banana",                    1.090),
    ("228_elusage",               1.073),
    ("615_fri_c4_250_10",         1.064),
    ("666_rmftsa_ladata",         1.054),
    ("651_fri_c0_100_25",         1.053),
    ("657_fri_c2_250_10",         1.051),
]


def fit_sb(X_tr, y_tr, seed):
    m = EmlSplitBoostRegressor(
        max_rounds=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=1, leaf_l2=1.0,
        n_eml_candidates=10, k_eml=3, n_bins=256, histogram_min_n=500,
        use_gpu=True, k_leaf_eml=1, min_samples_leaf_eml=30,
        leaf_eml_gain_threshold=0.05, leaf_eml_ridge=0.0,
        leaf_eml_cap_k=2.0, use_stacked_blend=False,
        patience=15, val_fraction=0.15, random_state=seed,
    )
    m.fit(X_tr, y_tr); return m


def fit_xgb(X_tr, y_tr, seed):
    Xtr, Xv, ytr, yv = train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed)
    m = xgb.XGBRegressor(
        objective="reg:squarederror", max_depth=8, n_estimators=200,
        learning_rate=0.1, device="cuda", verbosity=0,
        min_child_weight=1, reg_lambda=1.0,
        early_stopping_rounds=15, random_state=seed,
    )
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False); return m


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def sign_test_pval(k, n, p=0.5):
    pmf = lambda x: comb(n, x) * p**x * (1-p)**(n-x)
    obs = pmf(k)
    return sum(pmf(x) for x in range(n+1) if pmf(x) <= obs)


SEEDS = list(range(20))

print(f"{'dataset':<28}  {'n':>5} {'k':>3}  {'old':>5}  {'new mean':>9}  {'paired95CI':>17}  {'sb_wins':>8}  {'p':>7}  verdict")
print("-" * 130)

for name, exp17_ratio in LOSERS:
    try:
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y); X, y = X[mask], y[mask]
        n = len(X); k = X.shape[1]

        ratios = []
        for s in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=s)
            msb = fit_sb(X_tr, y_tr, s); mxgb = fit_xgb(X_tr, y_tr, s)
            sb_r = rmse(msb.predict(X_te), y_te)
            xgb_r = rmse(mxgb.predict(X_te), y_te)
            if xgb_r > 0:
                ratios.append(sb_r / xgb_r)

        ratios = np.array(ratios)
        mean = ratios.mean()
        sem = ratios.std() / np.sqrt(len(ratios))
        ci_lo = mean - 1.96 * sem
        ci_hi = mean + 1.96 * sem
        n_sb_wins = int((ratios < 1.0).sum())
        n_sb_loses = len(ratios) - n_sb_wins
        pval = sign_test_pval(n_sb_loses, len(ratios))

        if pval < 0.05 and mean > 1.0:
            verdict = "REAL LOSS"
        elif pval < 0.05 and mean < 1.0:
            verdict = "REAL WIN"
        else:
            verdict = "tied (noise)"

        print(f"{name:<28}  {int(n*0.8):>5} {k:>3}  {exp17_ratio:>5.3f}  {mean:>9.3f}  ({ci_lo:>5.3f}, {ci_hi:>5.3f})  {n_sb_wins:>2}/20  {pval:>7.3f}  {verdict}", flush=True)
    except Exception as e:
        print(f"{name:<28}  FETCH/FIT FAILED: {type(e).__name__}: {e}", flush=True)
