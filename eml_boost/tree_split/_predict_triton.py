"""Triton kernel for whole-tree GPU prediction.

Replaces the torch-loop implementation in tree.py's _predict_x_gpu with
a single kernel launch. Each thread block handles a chunk of samples;
each thread walks the tree from root to leaf using the tensorized tree
representation built by `_tensorize_tree`. Both internal-node EML splits
and EML leaves are evaluated inline within the kernel.

If the Triton kernel fails to compile or run for any reason, the
caller (tree.py's `_predict_x_gpu`) catches the exception and falls
back to the torch implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# These mirror the constants in eml_boost/_triton_exhaustive.py.
_EXP_CLAMP = 50.0
_LOG_EPS = 1e-6


@triton.jit
def _predict_tree_kernel(
    X_ptr,                    # (N_SAMPLES, N_FEATURES) float32
    out_ptr,                  # (N_SAMPLES,) float32
    node_kind_ptr,            # (n_nodes,) int8
    left_child_ptr,           # (n_nodes,) int32
    right_child_ptr,          # (n_nodes,) int32
    feature_idx_ptr,          # (n_nodes,) int32
    threshold_ptr,            # (n_nodes,) float32
    leaf_value_ptr,           # (n_nodes,) float32
    split_eml_descriptor_ptr, # (n_nodes, 6) int32, row-major
    leaf_eml_descriptor_ptr,  # (n_nodes, 6) int32, row-major
    split_feat_subset_ptr,    # (n_nodes, K_SPLIT) int32, row-major
    leaf_feat_subset_ptr,     # (n_nodes, K_LEAF) int32, row-major
    leaf_feat_mean_ptr,       # (n_nodes, K_LEAF) float32, row-major
    leaf_feat_std_ptr,        # (n_nodes, K_LEAF) float32, row-major
    leaf_eta_ptr,             # (n_nodes,) float32
    leaf_bias_ptr,            # (n_nodes,) float32
    leaf_cap_ptr,             # (n_nodes,) float32
    N_SAMPLES,
    N_FEATURES,
    MAX_DEPTH: tl.constexpr,
    K_SPLIT: tl.constexpr,
    K_LEAF: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
    EXP_CLAMP: tl.constexpr,
    LOG_EPS: tl.constexpr,
):
    """Walk the fitted tree from root to leaf for each sample, and write
    the per-sample prediction to ``out_ptr``.

    Algorithm: a (BLOCK_SAMPLES,)-vector of `current` node ids starts at 0
    (the root). For MAX_DEPTH+1 iterations we evaluate the split at the
    current node for every sample, and step the still-internal samples
    into their child. After the loop, every sample lands at a leaf
    (kind ∈ {2, 3}); we evaluate the leaf inline.

    Both raw splits (kind=0) and EML splits (kind=1) are evaluated for
    every sample on every iteration — the inactive one's result is masked
    away via tl.where. This is wasteful per-sample work but eliminates
    branching and keeps the kernel uniform across the warp.
    """
    pid = tl.program_id(0)
    sample_idx = pid * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    sample_mask = sample_idx < N_SAMPLES
    # Use int64 so node indices are safe even on big trees.
    current = tl.zeros((BLOCK_SAMPLES,), dtype=tl.int64)

    # Per-iteration "any internal still" check would require a tl.reduce,
    # but unrolling MAX_DEPTH+1 times unconditionally is correct: once a
    # sample reaches a leaf, the `is_internal` mask keeps it pinned in
    # place via `current = tl.where(is_internal, next_node, current)`.
    for _ in tl.static_range(MAX_DEPTH + 1):
        kind = tl.load(node_kind_ptr + current, mask=sample_mask, other=2)
        is_internal = kind < 2
        is_eml_split = kind == 1
        feat = tl.load(feature_idx_ptr + current, mask=sample_mask, other=0).to(tl.int64)
        thr = tl.load(threshold_ptr + current, mask=sample_mask, other=0.0)
        # Raw value: gather X[sample_idx, feat] in row-major layout.
        raw_x_offset = sample_idx * N_FEATURES + feat
        raw_val = tl.load(
            X_ptr + raw_x_offset, mask=sample_mask, other=0.0,
        )

        # ====== INLINE INTERNAL-EML EVALUATION ======
        # For each sample at the current node, build the depth-2 EML
        # output using:
        #   - split_eml_descriptor[current, 0..5]  (per-sample descriptor)
        #   - split_feat_subset[current, 0..K_SPLIT-1]  (per-sample feature subset)
        # The grammar (matching evaluate_trees_torch_per_sample):
        #   v_c2..v_c5 = leaf-terminal selection over {0, 1..K_SPLIT}
        #   node_0 = exp(clamp(v_c2)) - log(clamp(v_c3, LOG_EPS))
        #   node_1 = exp(clamp(v_c4)) - log(clamp(v_c5, LOG_EPS))
        #   left, right = root-internal selection over {0, 1..K_SPLIT, K_SPLIT+1}
        #   eml_val = exp(clamp(left)) - log(clamp(right, LOG_EPS))
        c0_s = tl.load(
            split_eml_descriptor_ptr + current * 6 + 0,
            mask=sample_mask, other=0,
        )
        c1_s = tl.load(
            split_eml_descriptor_ptr + current * 6 + 1,
            mask=sample_mask, other=0,
        )
        c2_s = tl.load(
            split_eml_descriptor_ptr + current * 6 + 2,
            mask=sample_mask, other=0,
        )
        c3_s = tl.load(
            split_eml_descriptor_ptr + current * 6 + 3,
            mask=sample_mask, other=0,
        )
        c4_s = tl.load(
            split_eml_descriptor_ptr + current * 6 + 4,
            mask=sample_mask, other=0,
        )
        c5_s = tl.load(
            split_eml_descriptor_ptr + current * 6 + 5,
            mask=sample_mask, other=0,
        )

        # Leaf-position selectors (range {0, 1..K_SPLIT}).
        v_c2_s = (c2_s == 0).to(tl.float32)
        v_c3_s = (c3_s == 0).to(tl.float32)
        v_c4_s = (c4_s == 0).to(tl.float32)
        v_c5_s = (c5_s == 0).to(tl.float32)
        for j in tl.static_range(K_SPLIT):
            feat_j = tl.load(
                split_feat_subset_ptr + current * K_SPLIT + j,
                mask=sample_mask, other=0,
            ).to(tl.int64)
            x_j_off = sample_idx * N_FEATURES + feat_j
            x_j = tl.load(X_ptr + x_j_off, mask=sample_mask, other=0.0)
            v_c2_s = v_c2_s + (c2_s == (j + 1)).to(tl.float32) * x_j
            v_c3_s = v_c3_s + (c3_s == (j + 1)).to(tl.float32) * x_j
            v_c4_s = v_c4_s + (c4_s == (j + 1)).to(tl.float32) * x_j
            v_c5_s = v_c5_s + (c5_s == (j + 1)).to(tl.float32) * x_j

        node_0_s = (
            tl.exp(tl.minimum(tl.maximum(v_c2_s, -EXP_CLAMP), EXP_CLAMP))
            - tl.log(tl.maximum(v_c3_s, LOG_EPS))
        )
        node_1_s = (
            tl.exp(tl.minimum(tl.maximum(v_c4_s, -EXP_CLAMP), EXP_CLAMP))
            - tl.log(tl.maximum(v_c5_s, LOG_EPS))
        )

        # Root-internal selectors (range {0, 1..K_SPLIT, K_SPLIT+1}).
        left_s = (c0_s == 0).to(tl.float32)
        right_s = (c1_s == 0).to(tl.float32)
        for j in tl.static_range(K_SPLIT):
            feat_j = tl.load(
                split_feat_subset_ptr + current * K_SPLIT + j,
                mask=sample_mask, other=0,
            ).to(tl.int64)
            x_j_off = sample_idx * N_FEATURES + feat_j
            x_j = tl.load(X_ptr + x_j_off, mask=sample_mask, other=0.0)
            left_s = left_s + (c0_s == (j + 1)).to(tl.float32) * x_j
            right_s = right_s + (c1_s == (j + 1)).to(tl.float32) * x_j
        left_s = left_s + (c0_s == (K_SPLIT + 1)).to(tl.float32) * node_0_s
        right_s = right_s + (c1_s == (K_SPLIT + 1)).to(tl.float32) * node_1_s

        eml_val = (
            tl.exp(tl.minimum(tl.maximum(left_s, -EXP_CLAMP), EXP_CLAMP))
            - tl.log(tl.maximum(right_s, LOG_EPS))
        )
        # ============================================

        split_val = tl.where(is_eml_split, eml_val, raw_val)
        go_left = split_val <= thr
        next_lc = tl.load(
            left_child_ptr + current, mask=sample_mask, other=0,
        ).to(tl.int64)
        next_rc = tl.load(
            right_child_ptr + current, mask=sample_mask, other=0,
        ).to(tl.int64)
        next_node = tl.where(go_left, next_lc, next_rc)
        current = tl.where(is_internal, next_node, current)

    # At leaves: kind in {2, 3}.
    final_kind = tl.load(node_kind_ptr + current, mask=sample_mask, other=2)
    leaf_const_val = tl.load(
        leaf_value_ptr + current, mask=sample_mask, other=0.0,
    )

    # ====== INLINE EML-LEAF EVALUATION ======
    # For samples landing on kind==3 leaves:
    #   1. x_raw[k] = X[sample, leaf_feat_subset[node, k]]
    #   2. x_std[k] = clamp((x_raw[k] - mean[k]) / std[k], -3, 3)
    #   3. Evaluate depth-2 grammar on x_std using leaf_eml_descriptor.
    #   4. pred = leaf_eta * eml_pred + leaf_bias.
    #   5. If leaf_cap < inf: clamp(pred, -cap, cap).
    c0_l = tl.load(
        leaf_eml_descriptor_ptr + current * 6 + 0,
        mask=sample_mask, other=0,
    )
    c1_l = tl.load(
        leaf_eml_descriptor_ptr + current * 6 + 1,
        mask=sample_mask, other=0,
    )
    c2_l = tl.load(
        leaf_eml_descriptor_ptr + current * 6 + 2,
        mask=sample_mask, other=0,
    )
    c3_l = tl.load(
        leaf_eml_descriptor_ptr + current * 6 + 3,
        mask=sample_mask, other=0,
    )
    c4_l = tl.load(
        leaf_eml_descriptor_ptr + current * 6 + 4,
        mask=sample_mask, other=0,
    )
    c5_l = tl.load(
        leaf_eml_descriptor_ptr + current * 6 + 5,
        mask=sample_mask, other=0,
    )

    v_c2_l = (c2_l == 0).to(tl.float32)
    v_c3_l = (c3_l == 0).to(tl.float32)
    v_c4_l = (c4_l == 0).to(tl.float32)
    v_c5_l = (c5_l == 0).to(tl.float32)
    for j in tl.static_range(K_LEAF):
        feat_j = tl.load(
            leaf_feat_subset_ptr + current * K_LEAF + j,
            mask=sample_mask, other=0,
        ).to(tl.int64)
        mean_j = tl.load(
            leaf_feat_mean_ptr + current * K_LEAF + j,
            mask=sample_mask, other=0.0,
        )
        std_j = tl.load(
            leaf_feat_std_ptr + current * K_LEAF + j,
            mask=sample_mask, other=1.0,
        )
        x_j_off = sample_idx * N_FEATURES + feat_j
        x_raw = tl.load(X_ptr + x_j_off, mask=sample_mask, other=0.0)
        x_std = tl.minimum(
            tl.maximum((x_raw - mean_j) / std_j, -3.0), 3.0,
        )
        v_c2_l = v_c2_l + (c2_l == (j + 1)).to(tl.float32) * x_std
        v_c3_l = v_c3_l + (c3_l == (j + 1)).to(tl.float32) * x_std
        v_c4_l = v_c4_l + (c4_l == (j + 1)).to(tl.float32) * x_std
        v_c5_l = v_c5_l + (c5_l == (j + 1)).to(tl.float32) * x_std

    node_0_l = (
        tl.exp(tl.minimum(tl.maximum(v_c2_l, -EXP_CLAMP), EXP_CLAMP))
        - tl.log(tl.maximum(v_c3_l, LOG_EPS))
    )
    node_1_l = (
        tl.exp(tl.minimum(tl.maximum(v_c4_l, -EXP_CLAMP), EXP_CLAMP))
        - tl.log(tl.maximum(v_c5_l, LOG_EPS))
    )

    left_l = (c0_l == 0).to(tl.float32)
    right_l = (c1_l == 0).to(tl.float32)
    for j in tl.static_range(K_LEAF):
        feat_j = tl.load(
            leaf_feat_subset_ptr + current * K_LEAF + j,
            mask=sample_mask, other=0,
        ).to(tl.int64)
        mean_j = tl.load(
            leaf_feat_mean_ptr + current * K_LEAF + j,
            mask=sample_mask, other=0.0,
        )
        std_j = tl.load(
            leaf_feat_std_ptr + current * K_LEAF + j,
            mask=sample_mask, other=1.0,
        )
        x_j_off = sample_idx * N_FEATURES + feat_j
        x_raw = tl.load(X_ptr + x_j_off, mask=sample_mask, other=0.0)
        x_std = tl.minimum(
            tl.maximum((x_raw - mean_j) / std_j, -3.0), 3.0,
        )
        left_l = left_l + (c0_l == (j + 1)).to(tl.float32) * x_std
        right_l = right_l + (c1_l == (j + 1)).to(tl.float32) * x_std
    left_l = left_l + (c0_l == (K_LEAF + 1)).to(tl.float32) * node_0_l
    right_l = right_l + (c1_l == (K_LEAF + 1)).to(tl.float32) * node_1_l

    eml_pred = (
        tl.exp(tl.minimum(tl.maximum(left_l, -EXP_CLAMP), EXP_CLAMP))
        - tl.log(tl.maximum(right_l, LOG_EPS))
    )

    eta = tl.load(leaf_eta_ptr + current, mask=sample_mask, other=0.0)
    bias = tl.load(leaf_bias_ptr + current, mask=sample_mask, other=0.0)
    cap = tl.load(
        leaf_cap_ptr + current, mask=sample_mask, other=float("inf"),
    )
    leaf_eml_val = eta * eml_pred + bias
    cap_finite = cap < float("inf")
    leaf_eml_val = tl.where(
        cap_finite,
        tl.minimum(tl.maximum(leaf_eml_val, -cap), cap),
        leaf_eml_val,
    )
    # =========================================

    out = tl.where(final_kind == 2, leaf_const_val, leaf_eml_val)
    tl.store(out_ptr + sample_idx, out, mask=sample_mask)


def predict_tree_triton(
    X_gpu: torch.Tensor, gpu_tree: dict, max_depth: int,
) -> torch.Tensor:
    """Python wrapper. Validates inputs, allocates output, launches kernel.

    Caller is expected to have already moved the gpu_tree dict's tensors
    onto the same device as X_gpu — see tree.py's _predict_x_gpu
    dispatcher for the migration path.
    """
    device = X_gpu.device
    n_samples, n_features = X_gpu.shape
    out_gpu = torch.zeros(n_samples, dtype=torch.float32, device=device)

    if n_samples == 0:
        return out_gpu

    desc_split_flat = gpu_tree["split_eml_descriptor"].contiguous()
    desc_leaf_flat = gpu_tree["leaf_eml_descriptor"].contiguous()
    split_fs_flat = gpu_tree["split_feat_subset"].contiguous()
    leaf_fs_flat = gpu_tree["leaf_feat_subset"].contiguous()
    leaf_mean_flat = gpu_tree["leaf_feat_mean"].contiguous()
    leaf_std_flat = gpu_tree["leaf_feat_std"].contiguous()

    K_split = int(gpu_tree["k_split_eml"])
    K_leaf = int(gpu_tree["k_leaf_eml"])

    BLOCK_SAMPLES = 256
    grid = (triton.cdiv(n_samples, BLOCK_SAMPLES),)

    _predict_tree_kernel[grid](
        X_gpu.contiguous(), out_gpu,
        gpu_tree["node_kind"],
        gpu_tree["left_child"],
        gpu_tree["right_child"],
        gpu_tree["feature_idx"],
        gpu_tree["threshold"],
        gpu_tree["leaf_value"],
        desc_split_flat,
        desc_leaf_flat,
        split_fs_flat,
        leaf_fs_flat,
        leaf_mean_flat,
        leaf_std_flat,
        gpu_tree["leaf_eta"],
        gpu_tree["leaf_bias"],
        gpu_tree["leaf_cap"],
        N_SAMPLES=n_samples,
        N_FEATURES=n_features,
        MAX_DEPTH=max_depth,
        K_SPLIT=K_split,
        K_LEAF=K_leaf,
        BLOCK_SAMPLES=BLOCK_SAMPLES,
        EXP_CLAMP=_EXP_CLAMP,
        LOG_EPS=_LOG_EPS,
    )
    return out_gpu
