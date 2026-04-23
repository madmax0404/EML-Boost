"""LightGBM single-tree wrapper used as DT weak learner."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np


@dataclass
class DtWeakLearner:
    booster: lgb.Booster
    n_features: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X).astype(np.float64)

    def params_count(self) -> int:
        """Number of leaves in the single tree."""
        dump = self.booster.dump_model()
        tree = dump["tree_info"][0]["tree_structure"]
        return _count_leaves(tree)


def _count_leaves(node: dict) -> int:
    if "leaf_value" in node:
        return 1
    return _count_leaves(node["left_child"]) + _count_leaves(node["right_child"])


def fit_dt_stump(
    X: np.ndarray,
    y: np.ndarray,
    depth: int,
    categorical_feature: list[int] | str = "auto",
) -> DtWeakLearner:
    """Fit a single regression tree via LightGBM.

    learning_rate is forced to 1.0: the outer boosting loop applies shrinkage.
    """
    dataset = lgb.Dataset(X, label=y, categorical_feature=categorical_feature)
    params = dict(
        objective="regression_l2",
        max_depth=depth,
        num_leaves=2**depth,
        min_data_in_leaf=20,
        learning_rate=1.0,
        verbose=-1,
    )
    booster = lgb.train(params, dataset, num_boost_round=1)
    return DtWeakLearner(booster=booster, n_features=X.shape[1])
