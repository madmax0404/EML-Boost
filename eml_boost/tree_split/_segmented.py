"""Per-segment (per-node / per-leaf) GPU statistics for batched tree growth.

All helpers take a `seg_id` long tensor mapping each row to its segment and
compute one statistic per segment in O(1) kernel launches via `index_add_`
/ `scatter_reduce_`, replacing per-node Python loops.

Determinism note: `index_add_` on float32 CUDA uses atomic adds whose
accumulation order is nondeterministic; resulting corr values can wobble at
float32-ulp scale between runs. This only affects top-k selection on EXACT
corr ties (measure-zero on continuous data). Histogram statistics — the
load-bearing split decision — do NOT use this module's float path; they use
the fixed-point deterministic path in `_multinode_hist.py`.
"""

from __future__ import annotations

import torch


def segment_counts(seg_id: torch.Tensor, n_segments: int) -> torch.Tensor:
    """(S,) float32 row count per segment."""
    return torch.bincount(seg_id, minlength=n_segments).float()


def segment_corr(
    X: torch.Tensor,
    y: torch.Tensor,
    seg_id: torch.Tensor,
    n_segments: int,
) -> torch.Tensor:
    """|Pearson corr| per (segment, feature): (S, D) float32.

    Mirrors `EmlSplitTreeRegressor._top_features_by_corr_gpu` per segment:
    two-pass centered computation, denominator guarded with +1e-12.
    Segments with no rows return 0 for every feature.
    """
    n, d = X.shape
    device = X.device
    cnt = segment_counts(seg_id, n_segments).clamp(min=1.0)  # (S,)

    sum_x = torch.zeros(n_segments, d, device=device).index_add_(0, seg_id, X)
    mean_x = sum_x / cnt.unsqueeze(1)
    sum_y = torch.zeros(n_segments, device=device).index_add_(0, seg_id, y)
    mean_y = sum_y / cnt

    xc = X - mean_x[seg_id]
    yc = y - mean_y[seg_id]

    num = torch.zeros(n_segments, d, device=device).index_add_(
        0, seg_id, xc * yc.unsqueeze(1)
    )
    sq_x = torch.zeros(n_segments, d, device=device).index_add_(0, seg_id, xc * xc)
    sq_y = torch.zeros(n_segments, device=device).index_add_(0, seg_id, yc * yc)

    denom = sq_x.sqrt() * sq_y.sqrt().unsqueeze(1) + 1e-12
    return (num / denom).abs()


def segment_topk_corr(
    X: torch.Tensor,
    y: torch.Tensor,
    seg_id: torch.Tensor,
    n_segments: int,
    k: int,
) -> torch.Tensor:
    """Top-k feature indices per segment by |corr|, corr-descending: (S, k) long.

    NOTE: returns `sorted=True` order (corr descending). The per-node
    reference uses `sorted=False`; order only matters for k > 1 where it
    changes descriptor semantics — an accepted statistically-neutral
    deviation (see plan Global Constraints).
    """
    corr = segment_corr(X, y, seg_id, n_segments)
    k_clipped = min(k, X.shape[1])
    return torch.topk(corr, k_clipped, dim=1, sorted=True).indices.to(torch.long)


def segment_minmax(
    values: torch.Tensor, seg_id: torch.Tensor, n_segments: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(segment, column) min and max: ((S, C), (S, C)) float32.

    min/max are associative+commutative, so scatter_reduce_ is bitwise
    deterministic regardless of atomic ordering. Empty segments return
    (+inf, -inf) — callers must mask them (their bin width degenerates).
    """
    n, c = values.shape
    idx = seg_id.unsqueeze(1).expand(n, c)
    mn = torch.full(
        (n_segments, c), float("inf"), device=values.device
    ).scatter_reduce_(0, idx, values, reduce="amin", include_self=True)
    mx = torch.full(
        (n_segments, c), -float("inf"), device=values.device
    ).scatter_reduce_(0, idx, values, reduce="amax", include_self=True)
    return mn, mx
