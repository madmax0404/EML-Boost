"""Phase-1 deep instrumentation: SB vs XGB on 659_sleuth_ex1714 (n=37).

Records per-fit:
- n_rounds_used (early stopping behavior)
- train RMSE, test RMSE, overfit gap (test - train)
- prediction magnitude stats (mean abs, max abs, std)

For SB only:
- total leaf count across all trees
- breakdown: constant leaves vs EML leaves
- mean / max leaf size

Goal: figure out WHY SB loses 1.118x to XGB on this tiny dataset, given
that simple parameter tweaks (msle, k_eml) didn't close the gap.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor
from eml_boost.tree_split.nodes import EmlLeafNode, InternalNode, LeafNode


def _walk_tree(node, leaves):
    """Append leaf info to `leaves`: list of (kind, n_leaf_unknown_in_inference)."""
    if isinstance(node, (LeafNode, EmlLeafNode)):
        leaves.append("constant" if isinstance(node, LeafNode) else "eml")
    elif isinstance(node, InternalNode):
        _walk_tree(node.left, leaves)
        _walk_tree(node.right, leaves)


def _count_sb_leaves(model: EmlSplitBoostRegressor) -> dict:
    total_const = 0
    total_eml = 0
    n_trees = len(model._trees)
    for tree in model._trees:
        leaves = []
        _walk_tree(tree._root, leaves)
        total_const += sum(1 for k in leaves if k == "constant")
        total_eml += sum(1 for k in leaves if k == "eml")
    return {
        "n_trees": n_trees,
        "total_leaves": total_const + total_eml,
        "constant_leaves": total_const,
        "eml_leaves": total_eml,
        "mean_leaves_per_tree": (total_const + total_eml) / max(n_trees, 1),
        "eml_leaf_fraction": total_eml / max(total_const + total_eml, 1),
    }


def _fit_sb(X_tr, y_tr, seed):
    m = EmlSplitBoostRegressor(
        max_rounds=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=1, leaf_l2=1.0,
        n_eml_candidates=10, k_eml=3, n_bins=256, histogram_min_n=500,
        use_gpu=True, k_leaf_eml=1, min_samples_leaf_eml=30,
        leaf_eml_gain_threshold=0.05, leaf_eml_ridge=0.0,
        leaf_eml_cap_k=2.0, use_stacked_blend=False,
        patience=15, val_fraction=0.15, random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m


def _fit_xgb(X_tr, y_tr, seed):
    X_inner_tr, X_inner_val, y_inner_tr, y_inner_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed,
    )
    m = xgb.XGBRegressor(
        objective="reg:squarederror", max_depth=8, n_estimators=200,
        learning_rate=0.1, device="cuda", verbosity=0,
        min_child_weight=1, reg_lambda=1.0,
        early_stopping_rounds=15, random_state=seed,
    )
    m.fit(X_inner_tr, y_inner_tr,
          eval_set=[(X_inner_val, y_inner_val)], verbose=False)
    return m


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def _instrument(name: str, m, X_tr, y_tr, X_te, y_te) -> dict:
    p_tr = np.asarray(m.predict(X_tr))
    p_te = np.asarray(m.predict(X_te))
    return {
        "algo": name,
        "train_rmse": _rmse(p_tr, y_tr),
        "test_rmse": _rmse(p_te, y_te),
        "overfit_gap": _rmse(p_te, y_te) - _rmse(p_tr, y_tr),
        "n_rounds": getattr(m, "n_rounds", 0) if name == "SB" else (m.best_iteration + 1 if m.best_iteration is not None else 200),
        "mean_abs_pred": float(np.mean(np.abs(p_te))),
        "max_abs_pred": float(np.max(np.abs(p_te))),
        "std_pred": float(np.std(p_te)),
        "y_te_mean_abs": float(np.mean(np.abs(y_te))),
        "y_te_std": float(np.std(y_te)),
    }


def main():
    name = "659_sleuth_ex1714"
    SEEDS = [0, 1, 2, 3, 4]
    X, y = fetch_data(name, return_X_y=True)
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y); X, y = X[mask], y[mask]
    print(f"=== {name} ===")
    print(f"n={len(X)}  k={X.shape[1]}  y_std={y.std():.1f}  y_mean={y.mean():.1f}\n")

    sb_results = []
    xgb_results = []
    sb_tree_stats = []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed)

        m_sb = _fit_sb(X_tr, y_tr, seed)
        sb_r = _instrument("SB", m_sb, X_tr, y_tr, X_te, y_te)
        sb_r["seed"] = seed
        sb_results.append(sb_r)
        sb_tree_stats.append(_count_sb_leaves(m_sb))

        m_xgb = _fit_xgb(X_tr, y_tr, seed)
        xgb_r = _instrument("XGB", m_xgb, X_tr, y_tr, X_te, y_te)
        xgb_r["seed"] = seed
        xgb_results.append(xgb_r)

    # Per-seed table
    print(f"{'seed':>4}  {'algo':>3}  {'train_rmse':>10}  {'test_rmse':>10}  {'gap':>8}  {'rounds':>6}  {'mean|p|':>9}  {'max|p|':>9}  {'std(p)':>8}")
    print("-" * 110)
    for sb_r, xgb_r in zip(sb_results, xgb_results):
        print(f"{sb_r['seed']:>4}  {'SB':>3}  {sb_r['train_rmse']:>10.2f}  {sb_r['test_rmse']:>10.2f}  {sb_r['overfit_gap']:>8.2f}  {sb_r['n_rounds']:>6}  {sb_r['mean_abs_pred']:>9.1f}  {sb_r['max_abs_pred']:>9.1f}  {sb_r['std_pred']:>8.1f}")
        print(f"      XGB  {xgb_r['train_rmse']:>10.2f}  {xgb_r['test_rmse']:>10.2f}  {xgb_r['overfit_gap']:>8.2f}  {xgb_r['n_rounds']:>6}  {xgb_r['mean_abs_pred']:>9.1f}  {xgb_r['max_abs_pred']:>9.1f}  {xgb_r['std_pred']:>8.1f}")
        print()

    # Aggregate
    def agg(rs, key):
        return np.mean([r[key] for r in rs])

    print(f"\n=== Aggregates (mean across {len(SEEDS)} seeds) ===")
    print(f"{'metric':<20}  {'SB':>10}  {'XGB':>10}  {'SB - XGB':>10}")
    print("-" * 60)
    for key in ("train_rmse", "test_rmse", "overfit_gap", "n_rounds", "mean_abs_pred", "max_abs_pred", "std_pred"):
        sb_v = agg(sb_results, key)
        xgb_v = agg(xgb_results, key)
        print(f"{key:<20}  {sb_v:>10.2f}  {xgb_v:>10.2f}  {sb_v - xgb_v:>+10.2f}")

    print(f"\ny_te std (target var):       {np.mean([r['y_te_std'] for r in sb_results]):>10.1f}")
    print(f"y_te mean abs:                {np.mean([r['y_te_mean_abs'] for r in sb_results]):>10.1f}")

    # SB tree structure
    print(f"\n=== SB tree structure (mean across {len(SEEDS)} seeds) ===")
    print(f"  trees per fit:              {np.mean([s['n_trees'] for s in sb_tree_stats]):>10.1f}")
    print(f"  total leaves per fit:       {np.mean([s['total_leaves'] for s in sb_tree_stats]):>10.1f}")
    print(f"  mean leaves per tree:       {np.mean([s['mean_leaves_per_tree'] for s in sb_tree_stats]):>10.1f}")
    print(f"  constant leaves:            {np.mean([s['constant_leaves'] for s in sb_tree_stats]):>10.0f}")
    print(f"  EML leaves:                 {np.mean([s['eml_leaves'] for s in sb_tree_stats]):>10.0f}")
    print(f"  EML leaf fraction:          {np.mean([s['eml_leaf_fraction'] for s in sb_tree_stats]):>10.3f}")


if __name__ == "__main__":
    main()
