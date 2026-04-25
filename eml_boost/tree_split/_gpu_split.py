"""GPU-accelerated histogram split-finding for a batch of features.

Given a value matrix `(n_samples, n_features)` and residuals `(n_samples,)`,
compute the best `(feature, threshold)` split across ALL features in one
pass of torch ops. No per-feature Python loop.

The math mirrors the CPU histogram implementation in `tree.py`:

  bin_idx[i, j] = bucket of values[i, j] into n_bins uniform bins
  hist_sum[j, b] = Σ_i y_i · [bin_idx[i, j] == b]
  cumulative sums give O(1) left/right SSE per candidate threshold
  argmax over (feature, bin) yields the best split

Running this fused on CUDA turns ~20 numpy histogram calls per node into
a single GPU kernel launch. At n=40k, d=20, the per-node cost drops from
several milliseconds of Python dispatch to a few microseconds of GPU time.

This module provides two implementations:

* ``gpu_histogram_split_torch`` — the original pure-torch implementation
  using ``scatter_add`` + ``cumsum``. Acts as the numerical oracle and
  the fallback path.
* ``gpu_histogram_split`` — top-level dispatcher that tries the Triton
  kernel from ``_gpu_split_triton.py`` first and falls back to the
  torch path on any error, warning once per process on the first
  fallback.
"""

from __future__ import annotations

import torch


# Module-level state for the dispatcher's warn-once-on-fallback contract.
# Set to True after the first time we fall back from the Triton path so
# subsequent fallbacks don't spam the user's stderr.
_TRITON_HIST_FALLBACK_WARNED = False


def gpu_histogram_split_torch(
    values: torch.Tensor,
    y: torch.Tensor,
    n_bins: int,
    min_leaf_count: int = 1,
    leaf_l2: float = 0.0,                       # NEW; defaults to 0.0 for backwards-compat
) -> tuple[int, float, float]:
    """Batched histogram split-finding on GPU.

    Parameters
    ----------
    values : torch.Tensor, shape (n_samples, n_features), float32
        Feature values (raw or EML-transformed), must be on CUDA.
    y : torch.Tensor, shape (n_samples,), float32
        Residuals (targets), must be on the same device as ``values``.
    n_bins : int
        Uniform bin count per feature.
    min_leaf_count : int
        Minimum samples on either side of the split.

    Returns
    -------
    (feature_idx, threshold, gain) : tuple[int, float, float]
        ``gain`` will be non-positive if no legal split was found; callers
        should treat that as "no split" rather than applying the result.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {values.shape}")
    n, d = values.shape
    if n < 2 * min_leaf_count:
        return 0, 0.0, 0.0

    device = values.device
    dtype = values.dtype

    # Per-feature range. If a feature is constant (vmax == vmin), its bins
    # collapse and no legal split exists → gain will be -inf for that feat.
    vmin = values.min(dim=0).values          # (d,)
    vmax = values.max(dim=0).values          # (d,)
    rng_ = (vmax - vmin).clamp(min=1e-12)    # (d,)
    bin_width = rng_ / n_bins                # (d,)

    # Bin indices in [0, n_bins-1]
    bin_idx = torch.clamp(
        ((values - vmin) / bin_width).long(),
        min=0,
        max=n_bins - 1,
    )                                        # (n, d)

    # Flat index per (sample, feature): feat_j * n_bins + bin_b
    feat_offsets = torch.arange(d, device=device) * n_bins
    flat_idx = (bin_idx + feat_offsets).reshape(-1)  # (n*d,) long

    y_rep = y.unsqueeze(1).expand(-1, d).reshape(-1).to(dtype)
    y2_rep = (y * y).unsqueeze(1).expand(-1, d).reshape(-1).to(dtype)
    ones = torch.ones_like(y_rep)

    total_slots = d * n_bins
    hist_sum = torch.zeros(total_slots, device=device, dtype=dtype).scatter_add_(0, flat_idx, y_rep)
    hist_sq = torch.zeros(total_slots, device=device, dtype=dtype).scatter_add_(0, flat_idx, y2_rep)
    hist_cnt = torch.zeros(total_slots, device=device, dtype=dtype).scatter_add_(0, flat_idx, ones)

    hist_sum = hist_sum.view(d, n_bins)
    hist_sq = hist_sq.view(d, n_bins)
    hist_cnt = hist_cnt.view(d, n_bins)

    # Cumulative along bin dim.
    c_sum = torch.cumsum(hist_sum, dim=1)      # (d, n_bins)
    c_sq = torch.cumsum(hist_sq, dim=1)
    c_cnt = torch.cumsum(hist_cnt, dim=1)

    total_sum = c_sum[:, -1:]                  # (d, 1)
    total_sq = c_sq[:, -1:]
    total_cnt = c_cnt[:, -1:]
    total_sse = total_sq - total_sum ** 2 / (total_cnt.clamp(min=1.0) + leaf_l2)

    # For split at bin boundary b (left = bins 0..b inclusive, right = b+1..),
    # left has c_cnt[:, b] samples, right has total - c_cnt[:, b].
    # Valid b ∈ [0, n_bins-2] (need non-empty right side).
    left_cnt = c_cnt[:, :-1]                   # (d, n_bins-1)
    left_sum = c_sum[:, :-1]
    left_sq = c_sq[:, :-1]
    right_cnt = total_cnt - left_cnt
    right_sum = total_sum - left_sum
    right_sq = total_sq - left_sq

    legal = (left_cnt >= min_leaf_count) & (right_cnt >= min_leaf_count)
    left_sse = left_sq - left_sum ** 2 / (left_cnt.clamp(min=1.0) + leaf_l2)
    right_sse = right_sq - right_sum ** 2 / (right_cnt.clamp(min=1.0) + leaf_l2)
    gain = total_sse - left_sse - right_sse
    gain = torch.where(legal, gain, torch.full_like(gain, float("-inf")))

    # argmax over (d, n_bins-1) → best (feature, bin-boundary) pair.
    flat_best = torch.argmax(gain.view(-1))
    n_legal_per_feat = n_bins - 1
    best_feat = int((flat_best // n_legal_per_feat).item())
    best_bin = int((flat_best % n_legal_per_feat).item())
    best_gain = float(gain.view(-1)[flat_best].item())
    if not (best_gain > 0):
        return 0, 0.0, 0.0

    # Threshold is the right edge of the "left" side's last included bin.
    threshold = float(
        (vmin[best_feat] + bin_width[best_feat] * (best_bin + 1)).item()
    )
    return best_feat, threshold, best_gain


def gpu_histogram_split(
    feats: torch.Tensor,
    y: torch.Tensor,
    n_bins: int,
    min_leaf_count: int = 1,
    leaf_l2: float = 0.0,                       # NEW
) -> tuple[int, float, float]:
    """Best-split-finding via histogram. Tries Triton kernel first;
    falls back to the torch implementation on any error.

    Warns once per process the first time the fallback fires so silent
    Triton failures don't go unnoticed in production runs. Mirrors the
    same dispatcher pattern used for the predict kernel (commit a4df96d).
    """
    try:
        from eml_boost.tree_split._gpu_split_triton import (
            gpu_histogram_split_triton,
        )
        return gpu_histogram_split_triton(feats, y, n_bins, min_leaf_count, leaf_l2)
    except Exception as exc:  # broad catch: any kernel-level failure (compile, OOM, illegal mem) falls back to torch; warn-once below surfaces the issue.
        global _TRITON_HIST_FALLBACK_WARNED
        if not _TRITON_HIST_FALLBACK_WARNED:
            import warnings
            warnings.warn(
                f"Triton histogram-split kernel failed; falling back to torch. "
                f"This warning fires once per process. "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _TRITON_HIST_FALLBACK_WARNED = True
        return gpu_histogram_split_torch(feats, y, n_bins, min_leaf_count, leaf_l2)
