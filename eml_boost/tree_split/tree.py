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
    get_descriptor_gpu,
    get_descriptor_np,
    get_feature_mask_gpu,
)
from eml_boost.symbolic.snap import SnappedTree
from eml_boost.tree_split._gpu_split import gpu_histogram_split
from eml_boost.tree_split.nodes import (
    EmlLeafNode,
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
    use_stacked_blend : bool
        If True, EML leaves are fit as a val-fit convex blend
        ``α·ȳ + (1−α)·(η·eml + β)`` with no minimum-gain gate. If False
        (default), the legacy binary accept/reject gate using
        ``leaf_eml_gain_threshold`` is used. See
        ``experiments/experiment9/report.md`` for why the default is False.
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
        k_leaf_eml: int = 1,
        min_samples_leaf_eml: int = 50,
        leaf_eml_gain_threshold: float = 0.05,
        leaf_eml_ridge: float = 0.0,
        use_stacked_blend: bool = False,
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
        # EML-at-leaves hyperparameters. `k_leaf_eml=0` disables EML leaves
        # entirely; default k_leaf_eml=1 keeps the search space small (144
        # trees) to bound overfitting. `leaf_eml_gain_threshold` is the
        # minimum fractional SSE improvement over a constant leaf required
        # to accept the EML form — XGBoost's γ concept applied to leaves.
        self.k_leaf_eml = k_leaf_eml
        self.min_samples_leaf_eml = min_samples_leaf_eml
        self.leaf_eml_gain_threshold = leaf_eml_gain_threshold
        self.leaf_eml_ridge = leaf_eml_ridge
        self.use_stacked_blend = use_stacked_blend
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmlSplitTreeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        self._X_cpu = X
        self._leaf_stats: list[dict] = []
        # Global mean/std across the full training set, used to standardize
        # features inside EML leaves. Local (per-leaf) stats produce narrow
        # ranges that blow up at predict time when same-leaf test samples
        # lie slightly outside that local window — see cpu_small diagnosis.
        self._global_mean = X.mean(axis=0)
        self._global_std = np.maximum(X.std(axis=0), 1e-6)
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._X_gpu = torch.tensor(X, dtype=torch.float32, device=self._device)
            self._global_mean_gpu = torch.tensor(
                self._global_mean, dtype=torch.float32, device=self._device,
            )
            self._global_std_gpu = torch.tensor(
                self._global_std, dtype=torch.float32, device=self._device,
            )
        else:
            self._device = None
            self._X_gpu = None
            self._global_mean_gpu = None
            self._global_std_gpu = None

        indices = np.arange(len(X))
        self._root: Node = self._grow(indices, y, depth=0, rng=rng)

        # Release GPU handles after fit; tree stores only CPU Node objects.
        self._X_gpu = None
        self._global_mean_gpu = None
        self._global_std_gpu = None
        self._device = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)
        self._predict_vec(self._root, X, np.arange(len(X)), out)
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _grow(
        self, indices: np.ndarray, y_sub: np.ndarray, depth: int, rng: np.random.Generator
    ) -> Node:
        if depth >= self.max_depth or len(y_sub) <= 2 * self.min_samples_leaf:
            return self._fit_leaf(indices, y_sub)

        if self._X_gpu is not None:
            best = self._find_best_split_gpu(indices, y_sub, rng)
        else:
            X_node = self._X_cpu[indices]
            best = self._find_best_split_cpu(X_node, y_sub, rng)
        if best is None:
            return self._fit_leaf(indices, y_sub)

        split, _gain, left_mask = best
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return self._fit_leaf(indices, y_sub)

        return InternalNode(
            split=split,
            left=self._grow(indices[left_mask], y_sub[left_mask], depth + 1, rng),
            right=self._grow(indices[~left_mask], y_sub[~left_mask], depth + 1, rng),
        )

    def _find_best_split_cpu(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[RawSplit | EmlSplit, float, np.ndarray] | None:
        n_features = X.shape[1]
        best_gain = 0.0
        best_split: RawSplit | EmlSplit | None = None
        best_mask: np.ndarray | None = None

        use_histogram = len(y) >= self.histogram_min_n
        threshold_fn = self._best_threshold_histogram if use_histogram else self._best_threshold

        for j in range(n_features):
            t, gain = threshold_fn(X[:, j], y)
            if gain > best_gain:
                best_gain = gain
                best_split = RawSplit(feature_idx=j, threshold=float(t))
                best_mask = X[:, j] <= t

        if self.n_eml_candidates > 0 and n_features > 0:
            k = min(self.k_eml, n_features)
            top_features = self._top_features_by_corr(X, y, k)
            candidates = self._sample_descriptors(k, self.n_eml_candidates, rng)
            eml_values = self._eval_eml_candidates(X, top_features, candidates, k)
            finite = np.isfinite(eml_values).all(axis=1)

            for c_idx in range(candidates.shape[0]):
                if not finite[c_idx]:
                    continue
                t, gain = threshold_fn(eml_values[c_idx], y)
                if gain > best_gain:
                    best_gain = gain
                    best_split = EmlSplit(
                        snapped=SnappedTree(
                            depth=2, k=k,
                            internal_input_count=2, leaf_input_count=4,
                            terminal_choices=tuple(int(v) for v in candidates[c_idx]),
                        ),
                        feature_subset=tuple(int(v) for v in top_features),
                        threshold=float(t),
                    )
                    best_mask = eml_values[c_idx] <= t

        if best_split is None or best_mask is None:
            return None
        return best_split, best_gain, best_mask

    def _find_best_split_gpu(
        self, indices: np.ndarray, y_sub: np.ndarray, rng: np.random.Generator
    ) -> tuple[RawSplit | EmlSplit, float, np.ndarray] | None:
        """GPU-batched histogram split-finding.

        Stacks raw features and sampled EML-transformed features into one
        (n_node, d_total) tensor, then calls `gpu_histogram_split` which
        argmaxes over all features × bin boundaries in a single torch pass.
        """
        device = self._device
        assert device is not None and self._X_gpu is not None
        idx_gpu = torch.from_numpy(indices).to(device=device, dtype=torch.long)
        X_node = self._X_gpu[idx_gpu]                                # (n, n_raw)
        y_node = torch.tensor(y_sub, dtype=torch.float32, device=device)

        n_raw = X_node.shape[1]
        feat_cols: list[torch.Tensor] = [X_node]
        valid_candidates: np.ndarray | None = None
        top_features: np.ndarray | None = None
        k_used = 0

        if self.n_eml_candidates > 0 and n_raw > 0:
            k_used = min(self.k_eml, n_raw)
            top_features = self._top_features_by_corr(self._X_cpu[indices], y_sub, k_used)
            candidates = self._sample_descriptors(k_used, self.n_eml_candidates, rng)
            if len(candidates) > 0:
                X_sub = X_node[:, top_features]
                desc_gpu = torch.tensor(candidates, dtype=torch.int32, device=device)
                eml_values = evaluate_trees_triton(desc_gpu, X_sub, k_used)  # (n_cand, n)
                finite = torch.isfinite(eml_values).all(dim=1)
                if finite.any():
                    valid_candidates = candidates[finite.cpu().numpy()]
                    feat_cols.append(eml_values[finite].T.contiguous())
                else:
                    valid_candidates = candidates[:0]
            else:
                valid_candidates = candidates[:0]

        all_feats = torch.cat(feat_cols, dim=1) if len(feat_cols) > 1 else feat_cols[0]
        best_idx, best_t, best_gain = gpu_histogram_split(
            all_feats, y_node, self.n_bins, min_leaf_count=self.min_samples_leaf,
        )

        if best_gain <= 0:
            return None

        if best_idx < n_raw:
            split: RawSplit | EmlSplit = RawSplit(feature_idx=int(best_idx), threshold=float(best_t))
            left_mask_gpu = all_feats[:, best_idx] <= best_t
        else:
            c_idx = int(best_idx) - n_raw
            assert valid_candidates is not None
            assert top_features is not None
            desc = valid_candidates[c_idx]
            split = EmlSplit(
                snapped=SnappedTree(
                    depth=2, k=k_used,
                    internal_input_count=2, leaf_input_count=4,
                    terminal_choices=tuple(int(v) for v in desc),
                ),
                feature_subset=tuple(int(v) for v in top_features),
                threshold=float(best_t),
            )
            left_mask_gpu = all_feats[:, best_idx] <= best_t

        left_mask = left_mask_gpu.cpu().numpy()
        return split, best_gain, left_mask

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

    def _fit_leaf(self, indices: np.ndarray, y_sub: np.ndarray) -> Node:
        """Build a leaf node. Tries an EML expression leaf if enabled and the
        sample count is large enough; falls back to a constant leaf otherwise.

        Two tree-selection policies are available behind the
        ``use_stacked_blend`` flag:
          - False: binary accept/reject gate on val-SSE improvement over a
            constant leaf (legacy).
          - True: val-fit convex blend between constant and EML predictions;
            α selected in closed form per candidate tree; tree chosen by
            α-optimized val-SSE.
        """
        n = len(y_sub)
        constant_value = float(y_sub.mean()) if n > 0 else 0.0

        # Early-out gates.
        eml_disabled = self.k_leaf_eml <= 0
        too_small = n < self.min_samples_leaf_eml
        no_gpu = self._X_gpu is None or self._device is None
        n_raw = self._X_cpu.shape[1] if self._X_cpu is not None else 0
        if eml_disabled or too_small or no_gpu or n_raw == 0:
            return LeafNode(value=constant_value)

        device = self._device
        assert device is not None and self._X_gpu is not None

        k = min(self.k_leaf_eml, n_raw)
        top_features = self._top_features_by_corr(self._X_cpu[indices], y_sub, k)
        idx_gpu = torch.from_numpy(indices).to(device=device, dtype=torch.long)
        X_sub_raw = self._X_gpu[idx_gpu][:, top_features]
        y_full = torch.tensor(y_sub, dtype=torch.float32, device=device)

        # Standardize using GLOBAL (fit-time) mean/std — local leaf stats
        # produce narrow ranges that explode at predict time on same-leaf
        # test samples lying slightly outside the local window. Global
        # stats give a consistent transform across all leaves.
        # Then CLAMP to [-3, 3] so that outliers (heavy-tailed PMLB
        # features like cpu_small's 10+σ samples) can't push exp(exp(·))
        # into overflow territory; the snapped grammar allows nested
        # exponentials and those are catastrophic at |arg| >> 3.
        assert self._global_mean_gpu is not None and self._global_std_gpu is not None
        top_features_t = torch.from_numpy(top_features).to(device=device, dtype=torch.long)
        mean_x = self._global_mean_gpu[top_features_t]
        std_x = self._global_std_gpu[top_features_t]
        X_sub = torch.clamp((X_sub_raw - mean_x) / std_x, -3.0, 3.0)

        # Deterministic leaf train/val split. The val portion (25%) is held
        # out from the per-tree OLS fit so the tree-selection policy
        # (either the legacy gate or the Task 2 blend) can evaluate
        # generalization rather than training fit.
        seed = int(indices[0]) if len(indices) else 0
        rng_leaf = np.random.default_rng(seed)
        perm = rng_leaf.permutation(n)
        val_sz = max(n // 4, 5)
        if n - val_sz < self.min_samples_leaf_eml // 2:
            return LeafNode(value=constant_value)
        val_local = perm[:val_sz]
        fit_local = perm[val_sz:]
        fit_idx_gpu = torch.from_numpy(fit_local).to(device=device, dtype=torch.long)
        val_idx_gpu = torch.from_numpy(val_local).to(device=device, dtype=torch.long)
        X_fit = X_sub[fit_idx_gpu]
        X_val = X_sub[val_idx_gpu]
        y_fit = y_full[fit_idx_gpu]
        y_val = y_full[val_idx_gpu]

        # Batched evaluation of all 144 depth-2 candidate trees.
        descriptor_gpu = get_descriptor_gpu(depth=2, k=k, device=device)
        feature_mask = get_feature_mask_gpu(depth=2, k=k, device=device)
        preds_fit = evaluate_trees_triton(descriptor_gpu, X_fit, k)  # (n_trees, n_fit)
        preds_val = evaluate_trees_triton(descriptor_gpu, X_val, k)  # (n_trees, n_val)

        # Closed-form OLS per tree on the fit portion. When
        # self.leaf_eml_ridge > 0 we regularize the slope (η) with a ridge
        # penalty λ·η². The centered-ridge closed form adds n_fit·λ to the
        # normal-equation diagonal; on the original sufficient statistics
        # that means replacing det with det + n_fit·λ. The bias term is
        # then the conditional-OLS intercept given the shrunk slope:
        # β = (Σy − η·Σp) / n_fit.
        n_fit = float(X_fit.shape[0])
        sum_p = preds_fit.sum(dim=1)
        sum_p2 = (preds_fit * preds_fit).sum(dim=1)
        sum_y_f = y_fit.sum()
        sum_py_f = (preds_fit * y_fit.unsqueeze(0)).sum(dim=1)
        det = sum_p2 * n_fit - sum_p * sum_p
        lam = float(self.leaf_eml_ridge)
        det_ridged = det + n_fit * lam
        # Guard against the remaining zero case (λ = 0 and det = 0 — a
        # genuinely degenerate tree with p_val constant and λ off).
        det_safe = torch.where(
            det_ridged.abs() > 1e-6, det_ridged, torch.ones_like(det_ridged)
        )
        eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_safe
        if lam == 0.0:
            bias = (sum_p2 * sum_y_f - sum_p * sum_py_f) / det_safe
        else:
            bias = (sum_y_f - eta * sum_p) / n_fit

        # Validity mask.
        finite_preds = (
            torch.isfinite(preds_fit).all(dim=1)
            & torch.isfinite(preds_val).all(dim=1)
        )
        finite_coefs = torch.isfinite(eta) & torch.isfinite(bias)
        valid = feature_mask & finite_preds & finite_coefs & (det.abs() > 1e-6)

        ctx = dict(
            y_full=y_full, y_val=y_val, eta=eta, bias=bias,
            preds_val=preds_val, valid=valid, k=k, top_features=top_features,
            mean_x=mean_x, std_x=std_x, constant_value=constant_value,
        )
        if self.use_stacked_blend:
            return self._select_leaf_blended(**ctx)
        return self._select_leaf_gated(**ctx)

    def _select_leaf_gated(
        self,
        *,
        y_full: "torch.Tensor",
        y_val: "torch.Tensor",
        eta: "torch.Tensor",
        bias: "torch.Tensor",
        preds_val: "torch.Tensor",
        valid: "torch.Tensor",
        k: int,
        top_features: np.ndarray,
        mean_x: "torch.Tensor",
        std_x: "torch.Tensor",
        constant_value: float,
    ) -> Node:
        """Legacy binary-gate tree selection. Picks the tree with smallest
        val-SSE on the pure-EML prediction; accepts it only if val-SSE beats
        the constant-leaf val-SSE by ``leaf_eml_gain_threshold``."""
        val_pred = eta.unsqueeze(1) * preds_val + bias.unsqueeze(1)
        val_res = y_val.unsqueeze(0) - val_pred
        val_sse = (val_res * val_res).sum(dim=1)
        val_sse = torch.where(valid, val_sse, torch.full_like(val_sse, float("inf")))

        best_idx = int(val_sse.argmin().item())
        if not bool(valid[best_idx].item()):
            return LeafNode(value=constant_value)

        best_val_sse = float(val_sse[best_idx].item())
        constant_val_sse = float(((y_val - y_full.mean()) ** 2).sum().item())
        if best_val_sse >= constant_val_sse * (1.0 - self.leaf_eml_gain_threshold):
            return LeafNode(value=constant_value)

        desc_np = get_descriptor_np(2, k)
        desc_row = desc_np[best_idx]
        return EmlLeafNode(
            snapped=SnappedTree(
                depth=2, k=k,
                internal_input_count=2, leaf_input_count=4,
                terminal_choices=tuple(int(v) for v in desc_row),
            ),
            feature_subset=tuple(int(v) for v in top_features),
            feature_mean=tuple(float(v) for v in mean_x.cpu().numpy()),
            feature_std=tuple(float(v) for v in std_x.cpu().numpy()),
            eta=float(eta[best_idx].item()),
            bias=float(bias[best_idx].item()),
        )

    def _select_leaf_blended(
        self,
        *,
        y_full: "torch.Tensor",
        y_val: "torch.Tensor",
        eta: "torch.Tensor",
        bias: "torch.Tensor",
        preds_val: "torch.Tensor",
        valid: "torch.Tensor",
        k: int,
        top_features: np.ndarray,
        mean_x: "torch.Tensor",
        std_x: "torch.Tensor",
        constant_value: float,
    ) -> Node:
        """Stacked-blend tree selection. Per candidate tree, fits the optimal
        α ∈ [0, 1] on the val portion in closed form; picks the tree with
        smallest α-optimized val-SSE; folds α into (η, β) for storage. No
        gate — α=1 collapse to LeafNode replaces the accept/reject decision.

        Fold:  η' = (1−α)·η,  β' = α·ȳ + (1−α)·β.
        Collapse: emit LeafNode(β') when |η'| < 1e-6 (float32-calibrated).
        Degenerate guard: α forced to 1 when ‖ȳ − val_pred‖² < 1e-12.
        """
        ybar = y_full.mean()
        # Pure-EML val predictions per candidate.
        val_pred = eta.unsqueeze(1) * preds_val + bias.unsqueeze(1)  # (n_trees, n_val)

        # Prediction under the blend: blend = α·ȳ + (1−α)·val_pred.
        # Loss = ||y_val − blend||² = ||(y_val − val_pred) − α·(ȳ − val_pred)||².
        # Let s = ȳ − val_pred (per tree, per val sample). Closed form:
        #   α* = sum(s · (y_val − val_pred)) / sum(s · s)
        s = ybar - val_pred                                     # (n_trees, n_val)
        y_minus_p = y_val.unsqueeze(0) - val_pred                # (n_trees, n_val)
        s_dot_diff = (s * y_minus_p).sum(dim=1)                  # (n_trees,)
        s_sq_sum = (s * s).sum(dim=1)                            # (n_trees,)

        # When s_sq_sum ≈ 0 the EML prediction equals ȳ on val — the blend
        # degenerates. Force α=1 in that case (constant beats nothing).
        degenerate = s_sq_sum.abs() < 1e-12
        s_sq_safe = torch.where(degenerate, torch.ones_like(s_sq_sum), s_sq_sum)
        alpha = s_dot_diff / s_sq_safe
        alpha = torch.clamp(alpha, 0.0, 1.0)
        alpha = torch.where(degenerate, torch.ones_like(alpha), alpha)

        # Blended val-SSE per tree.
        blend_pred = alpha.unsqueeze(1) * ybar + (1.0 - alpha).unsqueeze(1) * val_pred
        blend_res = y_val.unsqueeze(0) - blend_pred
        blend_sse = (blend_res * blend_res).sum(dim=1)           # (n_trees,)

        # Extend validity with finite-α.
        finite_alpha = torch.isfinite(alpha)
        valid_blend = valid & finite_alpha
        blend_sse = torch.where(
            valid_blend, blend_sse, torch.full_like(blend_sse, float("inf"))
        )

        best_idx = int(blend_sse.argmin().item())
        if not bool(valid_blend[best_idx].item()):
            self._leaf_stats.append({
                "n_leaf": int(y_full.shape[0]),
                "alpha": 1.0,  # no valid tree — treat as full collapse to constant
                "leaf_type": "LeafNode",
            })
            return LeafNode(value=constant_value)

        alpha_star = float(alpha[best_idx].item())
        eta_raw = float(eta[best_idx].item())
        bias_raw = float(bias[best_idx].item())
        ybar_py = float(ybar.item())

        # Fold α into (η, β).
        eta_folded = (1.0 - alpha_star) * eta_raw
        bias_folded = alpha_star * ybar_py + (1.0 - alpha_star) * bias_raw

        # If the blend collapsed the EML contribution, emit a LeafNode so
        # leaf-type counts remain interpretable.
        # Threshold calibrated to float32 one-ULP precision: when alpha is
        # one ULP below 1.0 in float32 (~0.99999988), eta_folded lands at
        # ~1.2e-7 * eta_raw — we want to catch that as "collapsed."
        if abs(eta_folded) < 1e-6:
            self._leaf_stats.append({
                "n_leaf": int(y_full.shape[0]),
                "alpha": alpha_star,
                "leaf_type": "LeafNode",
            })
            return LeafNode(value=bias_folded)

        desc_np = get_descriptor_np(2, k)
        desc_row = desc_np[best_idx]
        self._leaf_stats.append({
            "n_leaf": int(y_full.shape[0]),
            "alpha": alpha_star,
            "leaf_type": "EmlLeafNode",
        })
        return EmlLeafNode(
            snapped=SnappedTree(
                depth=2, k=k,
                internal_input_count=2, leaf_input_count=4,
                terminal_choices=tuple(int(v) for v in desc_row),
            ),
            feature_subset=tuple(int(v) for v in top_features),
            feature_mean=tuple(float(v) for v in mean_x.cpu().numpy()),
            feature_std=tuple(float(v) for v in std_x.cpu().numpy()),
            eta=eta_folded,
            bias=bias_folded,
        )

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
        `idx` slice and recurse into both children. At leaves, we either
        write the stored constant (``LeafNode``) or evaluate the stored
        elementary expression on the leaf's samples (``EmlLeafNode``).
        """
        if isinstance(node, LeafNode):
            out[idx] = node.value
            return
        if isinstance(node, EmlLeafNode):
            if len(idx) == 0:
                return
            X_leaf = X[idx][:, list(node.feature_subset)]
            # Re-apply the leaf's fit-time standardization AND clamp to the
            # same [-3, 3] range used during fit. Without this clamp, test
            # outliers (same-leaf but wider feature range than fit) blow up
            # via nested-exp in the grammar.
            mean = np.asarray(node.feature_mean, dtype=np.float64)
            std = np.asarray(node.feature_std, dtype=np.float64)
            X_leaf_std = np.clip((X_leaf - mean) / std, -3.0, 3.0)
            X_t = torch.tensor(X_leaf_std, dtype=torch.float64)
            desc_t = torch.tensor(
                [node.snapped.terminal_choices], dtype=torch.int32,
            )
            preds = evaluate_trees_torch(desc_t, X_t, node.snapped.k)  # (1, n)
            vals = preds.squeeze(0).cpu().numpy().astype(np.float64)
            out[idx] = node.eta * vals + node.bias
            return
        if len(idx) == 0:
            return
        mask = cls._evaluate_split(node.split, X[idx])
        left_idx = idx[mask]
        right_idx = idx[~mask]
        cls._predict_vec(node.left, X, left_idx, out)
        cls._predict_vec(node.right, X, right_idx, out)
