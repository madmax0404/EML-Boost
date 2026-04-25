"""Boosting wrapper around `EmlSplitTreeRegressor`.

Single-family gradient boosting: each round fits one elementary-split tree
to the current pseudo-residual (for L2 loss, `y − F_{m-1}`) and adds
``learning_rate · tree`` to the ensemble. Optional early stopping on a
held-out inner validation split.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from eml_boost.tree_split.tree import EmlSplitTreeRegressor


class EmlSplitBoostRegressor(BaseEstimator, RegressorMixin):
    """Boosted elementary-split regression trees.

    Each weak learner is an `EmlSplitTreeRegressor` grown on the residual.
    Every internal tree node may split on a raw feature or on an EML
    expression sampled from the depth-2 grammar.

    Parameters
    ----------
    max_rounds : int
        Number of boosting iterations.
    max_depth : int
        Tree depth within each weak learner.
    learning_rate : float
        Shrinkage applied to every tree's contribution.
    min_samples_leaf : int
        Prevents splits that would produce a leaf smaller than this.
    n_eml_candidates : int
        EML expressions sampled and considered per tree node.
    k_eml : int
        Features passed into each sampled EML expression.
    eml_depth : int
        Grammar depth for the EML sampling pool (must be 2 in Phase 2).
    n_bins : int
        Histogram bin count for split-finding on large nodes.
    histogram_min_n : int
        Minimum node size to activate histogram split-finding.
    use_gpu : bool
        Route EML candidate evaluation through the Triton kernel if CUDA is
        available; falls back to torch CPU otherwise.
    use_stacked_blend : bool
        Threads through to each round's ``EmlSplitTreeRegressor``. If True,
        EML leaves use a val-fit convex blend instead of the binary gate.
        Default False per Experiment 9's negative-outcome verdict on
        heavy-tailed datasets (see ``experiments/experiment9/report.md``).
    patience : int or None
        Early-stopping patience on an inner validation set. Set to None
        (or 0) to run the full `max_rounds`.
    val_fraction : float
        Fraction of training rows reserved for early-stopping validation.
    random_state : int or None
        Master seed. Per-tree seeds are derived deterministically from this.
    """

    def __init__(
        self,
        *,
        max_rounds: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        n_eml_candidates: int = 10,
        k_eml: int = 3,
        eml_depth: int = 2,
        n_bins: int = 256,
        histogram_min_n: int = 500,
        use_gpu: bool = True,
        k_leaf_eml: int = 1,
        min_samples_leaf_eml: int = 30,
        leaf_eml_gain_threshold: float = 0.05,
        leaf_eml_ridge: float = 0.0,
        leaf_eml_cap_k: float = 2.0,
        leaf_l2: float = 1.0,                   # was 0.0; mirrors EmlSplitTreeRegressor
        use_stacked_blend: bool = False,
        patience: int | None = 15,
        val_fraction: float = 0.15,
        random_state: int | None = None,
    ):
        self.max_rounds = max_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.n_eml_candidates = n_eml_candidates
        self.k_eml = k_eml
        self.eml_depth = eml_depth
        self.n_bins = n_bins
        self.histogram_min_n = histogram_min_n
        self.use_gpu = use_gpu
        self.k_leaf_eml = k_leaf_eml
        self.min_samples_leaf_eml = min_samples_leaf_eml
        self.leaf_eml_gain_threshold = leaf_eml_gain_threshold
        self.leaf_eml_ridge = leaf_eml_ridge
        self.leaf_eml_cap_k = leaf_eml_cap_k
        if leaf_l2 < 0.0:
            raise ValueError(f"leaf_l2 must be >= 0, got {leaf_l2}")
        self.leaf_l2 = float(leaf_l2)
        self.use_stacked_blend = use_stacked_blend
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "EmlSplitBoostRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self._F_0 = float(y.mean())
        self._trees: list[EmlSplitTreeRegressor] = []
        self._history: list[dict] = []

        patience = self.patience if self.patience is not None else 0
        if patience > 0 and self.val_fraction > 0:
            perm = rng.permutation(len(X))
            n_val = max(int(self.val_fraction * len(X)), 10)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[val_idx], y[val_idx]
        else:
            X_tr, y_tr = X, y
            X_va, y_va = None, None

        tree_seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=self.max_rounds)]

        if self.use_gpu and torch.cuda.is_available():
            return self._fit_gpu_loop(X_tr, y_tr, X_va, y_va, tree_seeds, patience)
        return self._fit_cpu_loop(X_tr, y_tr, X_va, y_va, tree_seeds, patience)

    def _fit_cpu_loop(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: "np.ndarray | None",
        y_va: "np.ndarray | None",
        tree_seeds: list,
        patience: int,
    ) -> "EmlSplitBoostRegressor":
        """CPU boost loop (numpy throughout). Existing behavior preserved."""
        F_tr = np.full(len(X_tr), self._F_0)
        F_va = np.full(len(X_va), self._F_0) if X_va is not None else None

        best_val_mse = float("inf")
        since_improve = 0

        for m in range(self.max_rounds):
            r = y_tr - F_tr
            tree = EmlSplitTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_eml_candidates=self.n_eml_candidates,
                k_eml=self.k_eml,
                eml_depth=self.eml_depth,
                n_bins=self.n_bins,
                histogram_min_n=self.histogram_min_n,
                use_gpu=self.use_gpu,
                k_leaf_eml=self.k_leaf_eml,
                min_samples_leaf_eml=self.min_samples_leaf_eml,
                leaf_eml_gain_threshold=self.leaf_eml_gain_threshold,
                leaf_eml_ridge=self.leaf_eml_ridge,
                leaf_eml_cap_k=self.leaf_eml_cap_k,
                leaf_l2=self.leaf_l2,
                use_stacked_blend=self.use_stacked_blend,
                random_state=tree_seeds[m],
            ).fit(X_tr, r)
            self._trees.append(tree)

            tree_pred_tr = tree.predict(X_tr)
            F_tr = F_tr + self.learning_rate * tree_pred_tr
            train_mse = float(np.mean((y_tr - F_tr) ** 2))

            record = {"round": m, "train_mse": train_mse}
            if F_va is not None and X_va is not None and y_va is not None:
                F_va = F_va + self.learning_rate * tree.predict(X_va)
                val_mse = float(np.mean((y_va - F_va) ** 2))
                record["val_mse"] = val_mse
                if val_mse < best_val_mse - 1e-10:
                    best_val_mse = val_mse
                    since_improve = 0
                else:
                    since_improve += 1
                    if patience > 0 and since_improve >= patience:
                        self._history.append(record)
                        break
            self._history.append(record)

        return self

    def _fit_gpu_loop(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: "np.ndarray | None",
        y_va: "np.ndarray | None",
        tree_seeds: list,
        patience: int,
    ) -> "EmlSplitBoostRegressor":
        """GPU-resident boost loop. X_tr, y_tr, and F_tr are allocated once
        on GPU and reused across all rounds, eliminating per-round H2D copies."""
        device = torch.device("cuda")
        X_tr_gpu = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_tr_gpu = torch.tensor(y_tr, dtype=torch.float32, device=device)
        F_tr_gpu = torch.full_like(y_tr_gpu, self._F_0)

        X_va_gpu = (
            torch.tensor(X_va, dtype=torch.float32, device=device)
            if X_va is not None else None
        )
        y_va_gpu = (
            torch.tensor(y_va, dtype=torch.float32, device=device)
            if y_va is not None else None
        )
        F_va_gpu = (
            torch.full_like(y_va_gpu, self._F_0) if y_va_gpu is not None else None
        )

        best_val_mse = float("inf")
        since_improve = 0

        for m in range(self.max_rounds):
            r_gpu = y_tr_gpu - F_tr_gpu
            tree = EmlSplitTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_eml_candidates=self.n_eml_candidates,
                k_eml=self.k_eml,
                eml_depth=self.eml_depth,
                n_bins=self.n_bins,
                histogram_min_n=self.histogram_min_n,
                use_gpu=self.use_gpu,
                k_leaf_eml=self.k_leaf_eml,
                min_samples_leaf_eml=self.min_samples_leaf_eml,
                leaf_eml_gain_threshold=self.leaf_eml_gain_threshold,
                leaf_eml_ridge=self.leaf_eml_ridge,
                leaf_eml_cap_k=self.leaf_eml_cap_k,
                leaf_l2=self.leaf_l2,
                use_stacked_blend=self.use_stacked_blend,
                random_state=tree_seeds[m],
            )._fit_xy_gpu(X_tr_gpu, r_gpu)
            self._trees.append(tree)

            tree_pred_tr_gpu = tree._predict_x_gpu(X_tr_gpu)
            F_tr_gpu = F_tr_gpu + self.learning_rate * tree_pred_tr_gpu
            train_mse = float(((y_tr_gpu - F_tr_gpu) ** 2).mean().item())
            record = {"round": m, "train_mse": train_mse}

            if F_va_gpu is not None and X_va_gpu is not None and y_va_gpu is not None:
                F_va_gpu = F_va_gpu + self.learning_rate * tree._predict_x_gpu(X_va_gpu)
                val_mse = float(((y_va_gpu - F_va_gpu) ** 2).mean().item())
                record["val_mse"] = val_mse
                if val_mse < best_val_mse - 1e-10:
                    best_val_mse = val_mse
                    since_improve = 0
                else:
                    since_improve += 1
                    if patience > 0 and since_improve >= patience:
                        self._history.append(record)
                        break
            self._history.append(record)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.full(len(X), self._F_0, dtype=np.float64)
        for tree in self._trees:
            out = out + self.learning_rate * tree.predict(X)
        return out

    @property
    def history(self) -> list[dict]:
        return self._history

    @property
    def n_rounds(self) -> int:
        return len(self._trees)
