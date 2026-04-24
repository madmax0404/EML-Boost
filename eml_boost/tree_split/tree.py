"""Single-tree MVP for elementary-split regression trees.

Phase 1: exact greedy split-finding (no histogram yet) over the union of
raw features and a sampled pool of EML expressions evaluated at each node.

The split-finding algorithm:
  1. Pick the top-k_eml raw features by absolute correlation with residual.
  2. Sample n_eml_candidates descriptors uniformly from the non-constant
     depth-2 tree space at k = k_eml.
  3. Evaluate each descriptor on X[:, top_features] using the torch
     evaluator. (Triton becomes relevant in Phase 2.)
  4. Concatenate raw-feature columns and EML-candidate columns. For each
     column, compute the best (threshold, gain) via exact sorted scan.
  5. Keep the highest-gain split; partition and recurse until the tree
     hits max_depth or the leaf-size threshold.
"""

from __future__ import annotations

import numpy as np
import torch

from eml_boost._triton_exhaustive import (
    descriptor_feature_mask_numpy,
    enumerate_depth2_descriptor,
    evaluate_trees_torch,
    evaluate_trees_triton,
)
from eml_boost.symbolic.snap import SnappedTree
from eml_boost.tree_split.nodes import (
    EmlSplit,
    InternalNode,
    LeafNode,
    Node,
    RawSplit,
)


class EmlSplitTreeRegressor:
    """Regression tree whose internal nodes split on raw features OR
    randomly-sampled elementary expressions.

    Parameters
    ----------
    max_depth : int
        Stop growing a branch at this depth.
    min_samples_leaf : int
        Don't split a node whose sample count would drop below this on either side.
    n_eml_candidates : int
        Number of EML expressions to sample and consider at each internal node.
        Set to 0 to reduce to a plain axis-aligned regression tree.
    k_eml : int
        Number of raw features each EML expression receives. The node picks
        the top-k_eml by residual correlation and samples expressions over them.
    eml_depth : int
        Depth of the EML grammar for the sampling pool. Currently only 2.
    random_state : int or None
        Seed for the sampler.
    """

    def __init__(
        self,
        *,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        n_eml_candidates: int = 10,
        k_eml: int = 3,
        eml_depth: int = 2,
        n_bins: int = 256,
        histogram_min_n: int = 500,
        use_gpu: bool = True,
        random_state: int | None = None,
    ):
        if eml_depth != 2:
            raise ValueError("Phase 1/2 only supports eml_depth=2")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_eml_candidates = n_eml_candidates
        self.k_eml = k_eml
        self.eml_depth = eml_depth
        self.n_bins = n_bins
        self.histogram_min_n = histogram_min_n
        self.use_gpu = use_gpu
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmlSplitTreeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self._root: Node = self._grow(X, y, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        self._predict_vec(self._root, X, np.arange(len(X)), out)
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator) -> Node:
        if depth >= self.max_depth or len(y) <= 2 * self.min_samples_leaf:
            return LeafNode(value=float(y.mean()) if len(y) > 0 else 0.0)

        best = self._find_best_split(X, y, rng)
        if best is None:
            return LeafNode(value=float(y.mean()))

        split, gain = best
        mask = self._evaluate_split(split, X)
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return LeafNode(value=float(y.mean()))

        return InternalNode(
            split=split,
            left=self._grow(X[mask], y[mask], depth + 1, rng),
            right=self._grow(X[~mask], y[~mask], depth + 1, rng),
        )

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[RawSplit | EmlSplit, float] | None:
        n_features = X.shape[1]
        best_gain = 0.0
        best_split: RawSplit | EmlSplit | None = None

        use_histogram = len(y) >= self.histogram_min_n
        threshold_fn = self._best_threshold_histogram if use_histogram else self._best_threshold

        # 1. Evaluate every raw-feature split.
        for j in range(n_features):
            t, gain = threshold_fn(X[:, j], y)
            if gain > best_gain:
                best_gain = gain
                best_split = RawSplit(feature_idx=j, threshold=float(t))

        # 2. Evaluate sampled EML candidates.
        if self.n_eml_candidates > 0 and n_features > 0:
            k = min(self.k_eml, n_features)
            top_features = self._top_features_by_corr(X, y, k)
            candidates = self._sample_descriptors(k, self.n_eml_candidates, rng)
            eml_values = self._eval_eml_candidates(X, top_features, candidates, k)
            finite = np.isfinite(eml_values).all(axis=1)  # (n_candidates,)

            for c_idx in range(candidates.shape[0]):
                if not finite[c_idx]:
                    continue
                t, gain = threshold_fn(eml_values[c_idx], y)
                if gain > best_gain:
                    best_gain = gain
                    best_split = EmlSplit(
                        snapped=SnappedTree(
                            depth=2,
                            k=k,
                            internal_input_count=2,
                            leaf_input_count=4,
                            terminal_choices=tuple(int(v) for v in candidates[c_idx]),
                        ),
                        feature_subset=tuple(int(v) for v in top_features),
                        threshold=float(t),
                    )

        if best_split is None:
            return None
        return best_split, best_gain

    @staticmethod
    def _best_threshold(values: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Exact sorted-scan best threshold for squared-error gain.

        Returns (threshold, gain). Gain is the reduction in total SSE from
        splitting at this threshold; threshold is midway between the two
        data points that define the best boundary.
        """
        n = len(y)
        if n < 2:
            return 0.0, 0.0
        order = np.argsort(values, kind="stable")
        v = values[order]
        yr = y[order]

        cumsum = np.cumsum(yr)
        cumsum_sq = np.cumsum(yr ** 2)
        total_sum = cumsum[-1]
        total_sq = cumsum_sq[-1]
        total_sse = total_sq - total_sum ** 2 / n

        # For split after position i-1 (i ∈ {1..n-1}): left has i samples,
        # right has n-i samples. Vectorize and mask off zero-gain positions
        # where v[i] == v[i-1] (can't legally split there).
        i = np.arange(1, n)
        left_sum = cumsum[i - 1]
        left_sq = cumsum_sq[i - 1]
        left_sse = left_sq - left_sum ** 2 / i
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq
        right_sse = right_sq - right_sum ** 2 / (n - i)

        gain = total_sse - left_sse - right_sse
        # Only legal splits are those where the two adjacent values differ.
        legal = v[1:] > v[:-1]
        gain = np.where(legal, gain, -np.inf)

        best_i = int(np.argmax(gain))
        if not np.isfinite(gain[best_i]) or gain[best_i] <= 0:
            return 0.0, 0.0
        threshold = 0.5 * (v[best_i] + v[best_i + 1])
        return float(threshold), float(gain[best_i])

    def _best_threshold_histogram(
        self, values: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Histogram-based split-finding. Approximate, O(n + B) per feature.

        Bins ``values`` into ``self.n_bins`` uniform-width bins, scatters
        y and y^2 into bins, then evaluates every possible bin-boundary
        threshold via precomputed cumulative sums. Loses precision in
        threshold placement (picks a bin edge, not a true data value
        midpoint) but that error is O(1/B) and matches LightGBM/XGBoost.
        """
        n = len(y)
        if n < 2:
            return 0.0, 0.0
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            return 0.0, 0.0

        # Uniform bin edges over [vmin, vmax].
        B = self.n_bins
        edges = np.linspace(vmin, vmax, B + 1)
        # np.digitize(..., right=False) maps values in [edges[b], edges[b+1])
        # to bin b+1; shift to 0-indexed then clamp the last edge into the last bin.
        bin_idx = np.clip(np.digitize(values, edges) - 1, 0, B - 1)

        # Per-bin sums (bincount handles integer indices efficiently).
        hist_count = np.bincount(bin_idx, minlength=B).astype(np.float64)
        hist_sum = np.bincount(bin_idx, weights=y, minlength=B)
        hist_sq = np.bincount(bin_idx, weights=y * y, minlength=B)

        # Cumulative sums for fast left/right split evaluation.
        c_count = np.cumsum(hist_count)
        c_sum = np.cumsum(hist_sum)
        c_sq = np.cumsum(hist_sq)
        total_count = c_count[-1]
        total_sum = c_sum[-1]
        total_sq = c_sq[-1]
        total_sse = total_sq - total_sum ** 2 / total_count

        # For split at bin boundary b (samples with bin<=b go left):
        # left_count = c_count[b], right_count = total_count - c_count[b].
        # Only b in 0..B-2 are legal (need non-empty right side).
        bs = np.arange(B - 1)
        left_count = c_count[bs]
        left_sum = c_sum[bs]
        left_sq = c_sq[bs]
        right_count = total_count - left_count
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq
        # Guard against empty side
        legal = (left_count >= 1) & (right_count >= 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            left_sse = left_sq - left_sum ** 2 / left_count
            right_sse = right_sq - right_sum ** 2 / right_count
        gain = total_sse - left_sse - right_sse
        gain = np.where(legal, gain, -np.inf)

        best_b = int(np.argmax(gain))
        if not np.isfinite(gain[best_b]) or gain[best_b] <= 0:
            return 0.0, 0.0
        # Threshold is the right edge of bin `best_b`.
        threshold = float(edges[best_b + 1])
        return threshold, float(gain[best_b])

    def _eval_eml_candidates(
        self,
        X: np.ndarray,
        top_features: np.ndarray,
        candidates: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Evaluate sampled descriptors on X[:, top_features].

        Returns a (n_candidates, n_samples) float64 numpy array. Uses
        the Triton kernel when CUDA is available and `use_gpu` is set,
        else falls back to the torch CPU evaluator.
        """
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            X_sub = torch.tensor(X[:, top_features], dtype=torch.float32, device=device)
            desc_t = torch.tensor(candidates, dtype=torch.int32, device=device)
            eml_values_t = evaluate_trees_triton(desc_t, X_sub, k)
            return eml_values_t.detach().cpu().numpy().astype(np.float64)
        # CPU fallback
        X_sub = torch.tensor(X[:, top_features], dtype=torch.float64)
        desc_t = torch.tensor(candidates, dtype=torch.int32)
        eml_values_t = evaluate_trees_torch(desc_t, X_sub, k)
        return eml_values_t.detach().numpy()

    @staticmethod
    def _top_features_by_corr(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Indices of the top-k features by |Pearson correlation| with y."""
        col_std = X.std(axis=0) + 1e-12
        y_centered = y - y.mean()
        denom = col_std * (y.std() + 1e-12) * len(y) + 1e-12
        corrs = (X - X.mean(axis=0)).T @ y_centered / denom
        return np.argsort(-np.abs(corrs))[:k]

    def _sample_descriptors(
        self, k: int, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Uniform random draw from the non-constant depth-2 tree space."""
        all_desc = enumerate_depth2_descriptor(k)
        mask = descriptor_feature_mask_numpy(all_desc, k)
        valid_desc = all_desc[mask]
        if len(valid_desc) == 0:
            return np.empty((0, 6), dtype=np.int32)
        idx = rng.integers(0, len(valid_desc), size=n_samples)
        return valid_desc[idx]

    @staticmethod
    def _evaluate_split(split: RawSplit | EmlSplit, X: np.ndarray) -> np.ndarray:
        """Boolean mask: True iff the sample goes LEFT under this split."""
        if isinstance(split, RawSplit):
            return X[:, split.feature_idx] <= split.threshold
        # EmlSplit
        feat_idx = np.asarray(split.feature_subset, dtype=np.int64)
        X_sub = torch.tensor(X[:, feat_idx], dtype=torch.float64)
        desc_t = torch.tensor([split.snapped.terminal_choices], dtype=torch.int32)
        values_t = evaluate_trees_torch(desc_t, X_sub, split.snapped.k)
        values = values_t.detach().numpy().flatten()
        # Non-finite values go right (arbitrary tie-break).
        mask = np.where(np.isfinite(values), values <= split.threshold, False)
        return mask

    @classmethod
    def _predict_vec(
        cls, node: Node, X: np.ndarray, idx: np.ndarray, out: np.ndarray
    ) -> None:
        """Recursively fill ``out[idx]`` by walking the tree in batched fashion.

        At each internal node we evaluate the split ONCE on the current
        `idx` slice and recurse into both children. This is O(depth · splits)
        torch calls rather than O(n_samples · depth) — a big win during
        boosting predict.
        """
        if isinstance(node, LeafNode):
            out[idx] = node.value
            return
        if len(idx) == 0:
            return
        mask = cls._evaluate_split(node.split, X[idx])
        left_idx = idx[mask]
        right_idx = idx[~mask]
        cls._predict_vec(node.left, X, left_idx, out)
        cls._predict_vec(node.right, X, right_idx, out)
