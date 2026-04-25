"""GPU-accelerated exhaustive snapped-tree evaluation for depth=2 EML.

Two implementations with identical numerics:

  evaluate_trees_torch  — pure PyTorch, runs on CPU or CUDA. Oracle: numerics
                          match the sympy lambdify path bit-for-bit within
                          floating-point tolerance. Used when Triton is
                          unavailable or for kernel validation in tests.

  evaluate_trees_triton — Triton kernel fused for depth-2 evaluation. One
                          program per (tree-block × sample-block); terminal
                          selection inside the kernel via constexpr-unrolled
                          arithmetic. ~100-300x faster than the per-tree
                          sympy path, eliminates Python dispatch overhead.

Both functions take a descriptor tensor encoding the discrete tree structure
(one int per terminal position) and an X tensor of feature values, and
return per-tree predictions. No sympy involvement in the hot path.

Descriptor layout (depth=2, single program-id interpretation):
    positions 0..1 are the root EML's two internal inputs
    positions 2..5 are the deepest level's leaf inputs (node 0 left/right,
                   node 1 left/right)
Internal choices ∈ {0, 1, ..., k-1, k+1}:  (k+1 == f_prev from matching child)
    0            → constant 1
    1..k         → feature x_{choice-1}
    k+1          → f_prev (child node output)  [only valid at positions 0..1]
Leaf choices  ∈ {0, 1, ..., k}:
    0            → constant 1
    1..k         → feature x_{choice-1}
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from eml_boost.symbolic.snap import SnappedTree

_EXP_CLAMP = 50.0
# Clamp the argument of log to at least this small positive value. Prevents
# log of zero / negative (which would be NaN) from erasing every tree whose
# snap happens to invoke log on standardized features; a tree using log of
# an effectively-zero argument gets a very-negative value that MSE-ranks
# poorly without taking out the entire search.
_LOG_EPS = 1e-6
_MAX_K = 7  # per-kernel upper bound on features; we fall back to torch above

# Module-level caches keyed on (depth, k) so the ~6k-entry descriptor and
# the feature-mask vector are built once and reused across every boosting
# round in the fit. Further, per-(depth, k, device) cache avoids repeated
# host→device transfers.
_descriptor_cache: dict[tuple[int, int], np.ndarray] = {}
_descriptor_gpu_cache: dict[tuple[int, int, str], torch.Tensor] = {}
_feature_mask_cache: dict[tuple[int, int], np.ndarray] = {}
_feature_mask_gpu_cache: dict[tuple[int, int, str], torch.Tensor] = {}
_valid_desc_cache: dict[tuple[int, int], np.ndarray] = {}


# ---------------------------------------------------------------------------
# Descriptor construction
# ---------------------------------------------------------------------------


def build_descriptor_depth2(snapped_trees: list[SnappedTree]) -> np.ndarray:
    """Pack a list of depth-2 SnappedTrees into an (n_trees, 6) int32 array.

    The array is laid out in the order (c0, c1, c2, c3, c4, c5) matching the
    `terminal_choices` tuple of SnappedTree: internals first (root inputs),
    then leaves (child 0 left, child 0 right, child 1 left, child 1 right).
    """
    rows: list[list[int]] = []
    for tree in snapped_trees:
        if tree.depth != 2:
            raise ValueError(
                f"build_descriptor_depth2 only supports depth=2, got {tree.depth}"
            )
        if len(tree.terminal_choices) != 6:
            raise ValueError(
                f"expected 6 terminal choices at depth 2, got {len(tree.terminal_choices)}"
            )
        rows.append(list(tree.terminal_choices))
    return np.asarray(rows, dtype=np.int32)


def enumerate_depth2_descriptor(k: int) -> np.ndarray:
    """Enumerate every SnappedTree at depth=2, k features and return its descriptor.

    The enumeration order is: outer product of internal choices (two
    positions, each with `k+2` options) × leaf choices (four positions, each
    with `k+1` options). Shape (n_trees, 6).
    """
    internal_dim = k + 2
    leaf_dim = k + 1
    n_internal_combos = internal_dim**2
    n_leaf_combos = leaf_dim**4
    n_trees = n_internal_combos * n_leaf_combos

    desc = np.empty((n_trees, 6), dtype=np.int32)
    row = 0
    for c0 in range(internal_dim):
        for c1 in range(internal_dim):
            for c2 in range(leaf_dim):
                for c3 in range(leaf_dim):
                    for c4 in range(leaf_dim):
                        for c5 in range(leaf_dim):
                            desc[row, 0] = c0
                            desc[row, 1] = c1
                            desc[row, 2] = c2
                            desc[row, 3] = c3
                            desc[row, 4] = c4
                            desc[row, 5] = c5
                            row += 1
    return desc


# ---------------------------------------------------------------------------
# Torch evaluator (CPU/CUDA oracle)
# ---------------------------------------------------------------------------


def evaluate_trees_torch(
    descriptor: torch.Tensor,
    X: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Evaluate all depth-2 trees on all samples using pure PyTorch ops.

    Args:
        descriptor: int tensor (n_trees, 6).
        X: float tensor (n_samples, k).
        k: number of features (must match X.shape[1]).

    Returns:
        predictions: float tensor (n_trees, n_samples). NaN where the
            snapped tree evaluates log of a non-positive value on that sample.
    """
    if X.shape[1] != k:
        raise ValueError(f"X.shape[1]={X.shape[1]} != k={k}")
    if descriptor.shape[1] != 6:
        raise ValueError(f"expected descriptor.shape[1]=6, got {descriptor.shape[1]}")

    n_trees = descriptor.shape[0]
    n_samples = X.shape[0]
    device = X.device
    dtype = X.dtype

    # Gather leaf-level inputs: leaf_terminals shape (n_samples, k+1)
    # column 0 = 1, columns 1..k = features
    ones_col = torch.ones(n_samples, 1, device=device, dtype=dtype)
    leaf_terminals = torch.cat([ones_col, X], dim=1)  # (n_samples, k+1)

    # For each leaf position (c2, c3, c4, c5), gather per-tree per-sample values.
    # leaf_terminals.T is (k+1, n_samples); fancy-indexing by descriptor[:, pos]
    # (int tensor of shape (n_trees,)) yields (n_trees, n_samples).
    lt_T = leaf_terminals.transpose(0, 1).contiguous()  # (k+1, n_samples)
    v_c2 = lt_T[descriptor[:, 2].long()]  # (n_trees, n_samples)
    v_c3 = lt_T[descriptor[:, 3].long()]
    v_c4 = lt_T[descriptor[:, 4].long()]
    v_c5 = lt_T[descriptor[:, 5].long()]

    # Leaf-level EML nodes: eml(a, b) = exp(a) - log(b)
    node_0 = (
        torch.exp(v_c2.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(v_c3.clamp(min=_LOG_EPS))
    )
    node_1 = (
        torch.exp(v_c4.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(v_c5.clamp(min=_LOG_EPS))
    )

    # Root's internal inputs. Choice c ∈ {0, 1..k, k+1}:
    #   c == 0: value = 1
    #   c == j (1..k): value = x_{j-1}   (broadcast over trees)
    #   c == k+1: value = node_? (child output; left uses node_0, right uses node_1)
    c0 = descriptor[:, 0].long()  # (n_trees,)
    c1 = descriptor[:, 1].long()

    # Build left input via vectorized conditional sums.
    # Masks shape: (n_trees, 1) so they broadcast over samples.
    left = torch.zeros(n_trees, n_samples, device=device, dtype=dtype)
    right = torch.zeros_like(left)

    # Constant 1 branch
    left += ((c0 == 0).to(dtype).unsqueeze(1))
    right += ((c1 == 0).to(dtype).unsqueeze(1))

    # Feature branches
    for j in range(k):
        feat = X[:, j].unsqueeze(0)  # (1, n_samples)
        left += ((c0 == j + 1).to(dtype).unsqueeze(1)) * feat
        right += ((c1 == j + 1).to(dtype).unsqueeze(1)) * feat

    # f_prev branches (only valid at internal positions)
    left += ((c0 == k + 1).to(dtype).unsqueeze(1)) * node_0
    right += ((c1 == k + 1).to(dtype).unsqueeze(1)) * node_1

    # Root EML
    output = (
        torch.exp(left.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(right.clamp(min=_LOG_EPS))
    )
    return output


def evaluate_trees_torch_per_sample(
    descriptor: torch.Tensor, X: torch.Tensor, k: int,
) -> torch.Tensor:
    """Evaluate one tree per sample.

    descriptor: (m, 6) int32, one descriptor per sample
    X: (m, k) float, one feature subset per sample
    k: feature count per sample

    Returns: (m,) float — sample i's descriptor evaluated on X[i].
    """
    m = descriptor.shape[0]
    if m == 0:
        return torch.empty(0, dtype=X.dtype, device=X.device)
    dtype = X.dtype
    device = X.device

    leaf_terminals = torch.cat(
        [torch.ones(m, 1, device=device, dtype=dtype), X], dim=1
    )  # (m, k+1)

    def gather_leaf(pos):  # pos in {2, 3, 4, 5}
        idx = descriptor[:, pos].long().unsqueeze(1)  # (m, 1)
        return leaf_terminals.gather(1, idx).squeeze(1)  # (m,)

    v_c2 = gather_leaf(2)
    v_c3 = gather_leaf(3)
    v_c4 = gather_leaf(4)
    v_c5 = gather_leaf(5)

    node_0 = (
        torch.exp(v_c2.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(v_c3.clamp(min=_LOG_EPS))
    )
    node_1 = (
        torch.exp(v_c4.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(v_c5.clamp(min=_LOG_EPS))
    )

    c0 = descriptor[:, 0].long()
    c1 = descriptor[:, 1].long()

    left = torch.zeros(m, device=device, dtype=dtype)
    right = torch.zeros(m, device=device, dtype=dtype)

    left = left + (c0 == 0).to(dtype)
    right = right + (c1 == 0).to(dtype)
    for j in range(k):
        feat_j = X[:, j]
        left = left + ((c0 == j + 1).to(dtype)) * feat_j
        right = right + ((c1 == j + 1).to(dtype)) * feat_j
    left = left + ((c0 == k + 1).to(dtype)) * node_0
    right = right + ((c1 == k + 1).to(dtype)) * node_1

    output = (
        torch.exp(left.clamp(-_EXP_CLAMP, _EXP_CLAMP))
        - torch.log(right.clamp(min=_LOG_EPS))
    )
    return output


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _eval_depth2_kernel(
        X_ptr,
        desc_ptr,
        out_ptr,
        n_trees,
        n_samples,
        X_stride_sample,
        K: tl.constexpr,
        EXP_CLAMP: tl.constexpr,
        LOG_EPS: tl.constexpr,
        BLOCK_TREES: tl.constexpr,
        BLOCK_SAMPLES: tl.constexpr,
    ):
        """Evaluate all depth-2 trees on a block of (trees × samples).

        Terminal selection is done via constexpr-unrolled arithmetic mask
        summation — no branch per choice, no indirect load. Triton's JIT
        does not support inner function definitions, so the terminal
        selection is inlined four times for leaves and twice for internals.
        """
        pid_tree = tl.program_id(0)
        pid_sample = tl.program_id(1)
        tree_offs = pid_tree * BLOCK_TREES + tl.arange(0, BLOCK_TREES)
        sample_offs = pid_sample * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
        tree_mask = tree_offs < n_trees
        sample_mask = sample_offs < n_samples

        # Load descriptor row per tree: 6 int32s per tree.
        c0 = tl.load(desc_ptr + tree_offs * 6 + 0, mask=tree_mask, other=0)
        c1 = tl.load(desc_ptr + tree_offs * 6 + 1, mask=tree_mask, other=0)
        c2 = tl.load(desc_ptr + tree_offs * 6 + 2, mask=tree_mask, other=0)
        c3 = tl.load(desc_ptr + tree_offs * 6 + 3, mask=tree_mask, other=0)
        c4 = tl.load(desc_ptr + tree_offs * 6 + 4, mask=tree_mask, other=0)
        c5 = tl.load(desc_ptr + tree_offs * 6 + 5, mask=tree_mask, other=0)

        # Broadcast choices to (BLOCK_TREES, BLOCK_SAMPLES).
        c0b = c0[:, None]
        c1b = c1[:, None]
        c2b = c2[:, None]
        c3b = c3[:, None]
        c4b = c4[:, None]
        c5b = c5[:, None]

        one = tl.full((BLOCK_TREES, BLOCK_SAMPLES), 1.0, dtype=tl.float32)

        # Load all K feature columns for the sample block: feat[j] has shape (1, BLOCK_SAMPLES)
        # Use the constexpr K to statically unroll the loads into a small list we can
        # reference from every terminal-selection expression.
        feat_base = X_ptr + sample_offs[None, :] * X_stride_sample

        # v_c2: leaf-terminal selection on c2 ∈ {0..K}
        v_c2 = one * (c2b == 0).to(tl.float32)
        for j in tl.static_range(K):
            x_j = tl.load(feat_base + j, mask=sample_mask[None, :], other=0.0)
            v_c2 = v_c2 + (c2b == (j + 1)).to(tl.float32) * x_j

        v_c3 = one * (c3b == 0).to(tl.float32)
        for j in tl.static_range(K):
            x_j = tl.load(feat_base + j, mask=sample_mask[None, :], other=0.0)
            v_c3 = v_c3 + (c3b == (j + 1)).to(tl.float32) * x_j

        v_c4 = one * (c4b == 0).to(tl.float32)
        for j in tl.static_range(K):
            x_j = tl.load(feat_base + j, mask=sample_mask[None, :], other=0.0)
            v_c4 = v_c4 + (c4b == (j + 1)).to(tl.float32) * x_j

        v_c5 = one * (c5b == 0).to(tl.float32)
        for j in tl.static_range(K):
            x_j = tl.load(feat_base + j, mask=sample_mask[None, :], other=0.0)
            v_c5 = v_c5 + (c5b == (j + 1)).to(tl.float32) * x_j

        # Leaf-level EML nodes: eml(a, b) = exp(clamp(a)) - log(b)
        node_0 = tl.exp(tl.minimum(tl.maximum(v_c2, -EXP_CLAMP), EXP_CLAMP)) - tl.log(tl.maximum(v_c3, LOG_EPS))
        node_1 = tl.exp(tl.minimum(tl.maximum(v_c4, -EXP_CLAMP), EXP_CLAMP)) - tl.log(tl.maximum(v_c5, LOG_EPS))

        # Root's internal terminals (choice ∈ {0..K+1}): extra branch for f_prev.
        left = one * (c0b == 0).to(tl.float32)
        for j in tl.static_range(K):
            x_j = tl.load(feat_base + j, mask=sample_mask[None, :], other=0.0)
            left = left + (c0b == (j + 1)).to(tl.float32) * x_j
        left = left + (c0b == (K + 1)).to(tl.float32) * node_0

        right = one * (c1b == 0).to(tl.float32)
        for j in tl.static_range(K):
            x_j = tl.load(feat_base + j, mask=sample_mask[None, :], other=0.0)
            right = right + (c1b == (j + 1)).to(tl.float32) * x_j
        right = right + (c1b == (K + 1)).to(tl.float32) * node_1

        out = tl.exp(tl.minimum(tl.maximum(left, -EXP_CLAMP), EXP_CLAMP)) - tl.log(tl.maximum(right, LOG_EPS))

        out_ptrs = out_ptr + tree_offs[:, None] * n_samples + sample_offs[None, :]
        mask = tree_mask[:, None] & sample_mask[None, :]
        tl.store(out_ptrs, out, mask=mask)


def descriptor_feature_mask_numpy(descriptor: np.ndarray, k: int) -> np.ndarray:
    """Boolean (n_trees,) mask: True where the depth-2 tree genuinely references
    at least one input feature after the argmax snap (not a dead-branch constant).
    """
    c0, c1, c2, c3 = descriptor[:, 0], descriptor[:, 1], descriptor[:, 2], descriptor[:, 3]
    c4, c5 = descriptor[:, 4], descriptor[:, 5]

    def is_feat(c: np.ndarray) -> np.ndarray:
        return (c >= 1) & (c <= k)

    def is_fprev(c: np.ndarray) -> np.ndarray:
        return c == k + 1

    node_0_uses = is_feat(c2) | is_feat(c3)
    node_1_uses = is_feat(c4) | is_feat(c5)
    root_left_uses = is_feat(c0) | (is_fprev(c0) & node_0_uses)
    root_right_uses = is_feat(c1) | (is_fprev(c1) & node_1_uses)
    return root_left_uses | root_right_uses


def get_descriptor_np(depth: int, k: int) -> np.ndarray:
    """Module-level cached enumerated descriptor (n_trees, 6) int32."""
    if depth != 2:
        raise ValueError("GPU path only supports depth=2")
    key = (depth, k)
    cached = _descriptor_cache.get(key)
    if cached is None:
        cached = enumerate_depth2_descriptor(k)
        _descriptor_cache[key] = cached
    return cached


def get_descriptor_gpu(depth: int, k: int, device: torch.device) -> torch.Tensor:
    key = (depth, k, str(device))
    cached = _descriptor_gpu_cache.get(key)
    if cached is None:
        desc_np = get_descriptor_np(depth, k)
        cached = torch.tensor(desc_np, dtype=torch.int32, device=device)
        _descriptor_gpu_cache[key] = cached
    return cached


def get_feature_mask_np(depth: int, k: int) -> np.ndarray:
    """Cached per-tree constant-detection mask."""
    if depth != 2:
        raise ValueError("GPU path only supports depth=2")
    key = (depth, k)
    cached = _feature_mask_cache.get(key)
    if cached is None:
        desc = get_descriptor_np(depth, k)
        cached = descriptor_feature_mask_numpy(desc, k)
        _feature_mask_cache[key] = cached
    return cached


def get_valid_descriptors_np(depth: int, k: int) -> np.ndarray:
    """Cached enumeration of non-constant depth-`depth` descriptors at k inputs.

    Returns a contiguous (n_valid, 6) int32 array, where n_valid is the count
    of descriptors that pass `get_feature_mask_np`. Process-global cache:
    same array is returned on every call with the same (depth, k) — callers
    must not mutate it.
    """
    if depth != 2:
        raise ValueError("GPU path only supports depth=2")
    key = (depth, k)
    cached = _valid_desc_cache.get(key)
    if cached is None:
        all_desc = get_descriptor_np(depth, k)
        mask = get_feature_mask_np(depth, k)
        cached = np.ascontiguousarray(all_desc[mask])
        _valid_desc_cache[key] = cached
    return cached


def get_feature_mask_gpu(depth: int, k: int, device: torch.device) -> torch.Tensor:
    key = (depth, k, str(device))
    cached = _feature_mask_gpu_cache.get(key)
    if cached is None:
        mask_np = get_feature_mask_np(depth, k)
        cached = torch.tensor(mask_np, dtype=torch.bool, device=device)
        _feature_mask_gpu_cache[key] = cached
    return cached


def evaluate_trees_triton(
    descriptor: torch.Tensor,
    X: torch.Tensor,
    k: int,
    block_trees: int = 32,
    block_samples: int = 64,
) -> torch.Tensor:
    """GPU-accelerated Triton kernel evaluator.

    Falls back to evaluate_trees_torch if Triton is unavailable or k > _MAX_K.
    """
    if not _TRITON_AVAILABLE or k > _MAX_K:
        return evaluate_trees_torch(descriptor, X, k)
    if not X.is_cuda:
        return evaluate_trees_torch(descriptor, X, k)

    descriptor = descriptor.to(torch.int32).contiguous()
    X = X.contiguous().to(torch.float32)
    n_trees = descriptor.shape[0]
    n_samples = X.shape[0]
    out = torch.empty(n_trees, n_samples, device=X.device, dtype=torch.float32)

    grid = (triton.cdiv(n_trees, block_trees), triton.cdiv(n_samples, block_samples))
    _eval_depth2_kernel[grid](
        X,
        descriptor,
        out,
        n_trees,
        n_samples,
        X.stride(0),
        K=k,
        EXP_CLAMP=_EXP_CLAMP,
        LOG_EPS=_LOG_EPS,
        BLOCK_TREES=block_trees,
        BLOCK_SAMPLES=block_samples,
    )
    return out
