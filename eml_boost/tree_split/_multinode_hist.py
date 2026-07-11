# eml_boost/tree_split/_multinode_hist.py
"""Deterministic multi-node histogram split-finding.

Generalizes `gpu_histogram_split_torch` over a batch of frontier nodes:
one scatter pass builds (n_segments, n_cols, n_bins) histograms for every
node at a tree level, one reduction finds each node's best (column, bin).
The node-wise engine routes through this same core with n_segments=1, so
both growth engines share ONE histogram backend.

Determinism: float atomic adds are accumulation-order nondeterministic,
which breaks same-seed reproducibility of fitted trees (1-ULP gain wobble
flips near-tied argmax decisions and cascades). y and y**2 are therefore
quantized to int64 fixed-point before scatter_add_ — integer addition is
associative, so histogram totals are bit-identical regardless of row
order. The quantization scale is 2**20 relative to each SEGMENT's own
max|y| (amax is order-independent), so a single-segment call (node-wise
dispatcher) and a multi-segment call (level-wise engine) yield
bit-identical per-node sums, giving ~1e-6 relative error on bin sums — far below
the O(1/n_bins) threshold-placement error inherent to histogram splitting.
Implied bound: max|y| * n < ~2**43 per segment to keep int64 exact
(residuals in this project are far below that).

Gain math (incl. leaf_l2 and min-leaf legality) mirrors
`gpu_histogram_split_torch` line for line; see _gpu_split.py.
"""

from __future__ import annotations

import torch

from eml_boost.tree_split._segmented import segment_minmax

_FP_BITS = 20  # fixed-point fractional bits for y-sum quantization


def multinode_histogram_split(
    values: torch.Tensor,
    y: torch.Tensor,
    seg_id: torch.Tensor,
    n_segments: int,
    n_bins: int,
    min_leaf_count: int,
    leaf_l2: float,
    col_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c = values.shape
    device = values.device
    S, B = n_segments, n_bins

    if n == 0:
        return (
            torch.zeros(S, dtype=torch.long, device=device),
            torch.zeros(S, device=device),
            torch.zeros(S, device=device),
        )

    # Per-(segment, column) uniform bins — same rule as the single-node path.
    vmin, vmax = segment_minmax(values, seg_id, S)  # (S, C) each
    empty = torch.isinf(vmin)  # empty-segment marker
    vmin = torch.where(empty, torch.zeros_like(vmin), vmin)
    vmax = torch.where(empty, torch.zeros_like(vmax), vmax)
    rng_ = (vmax - vmin).clamp(min=1e-12)
    bin_width = rng_ / B  # (S, C)

    bin_idx = torch.clamp(
        ((values - vmin[seg_id]) / bin_width[seg_id]).long(), min=0, max=B - 1
    )  # (N, C)

    # Fixed-point quantization with a PER-SEGMENT scale. A single-segment
    # call (the node-wise dispatcher) and a multi-segment call (the level-
    # wise engine) must produce bit-identical per-node histogram sums, so
    # the scale must depend only on the segment's own rows — never on which
    # other segments share the batch. amax is order-independent, so the
    # scale itself is deterministic.
    seg_absmax = torch.zeros(S, device=device).scatter_reduce_(
        0, seg_id, y.abs(), reduce="amax", include_self=True
    ).clamp(min=1e-12)                              # (S,)
    scale = float(1 << _FP_BITS) / seg_absmax       # (S,)
    row_scale = scale[seg_id]                       # (N,)
    y_q = torch.round(y * row_scale).to(torch.int64)
    y2_q = torch.round((y * y) * row_scale).to(torch.int64)

    col_offs = torch.arange(c, device=device) * B  # (C,)
    flat = (seg_id.unsqueeze(1) * (c * B) + col_offs + bin_idx).reshape(-1)

    slots = S * c * B
    # Fuse the three per-stat scatters into ONE scatter_add_ over a
    # (3 * slots,) buffer: stat k accumulates into the disjoint block
    # [k*slots : (k+1)*slots], sharing the same `flat` bin locations shifted
    # by k*slots. Integer adds are exact and order-independent, so the
    # sum/sq/cnt totals are bit-identical to three separate calls — one
    # kernel launch instead of three on a launch-bound workload.
    y_q_e = y_q.unsqueeze(1).expand(n, c).reshape(-1)
    y2_q_e = y2_q.unsqueeze(1).expand(n, c).reshape(-1)
    ones_e = torch.ones(n * c, dtype=torch.int64, device=device)
    fused_idx = torch.cat([flat, flat + slots, flat + 2 * slots])
    fused_val = torch.cat([y_q_e, y2_q_e, ones_e])
    hist = torch.zeros(3 * slots, dtype=torch.int64, device=device).scatter_add_(
        0, fused_idx, fused_val
    )
    hist_sum = hist[:slots]
    hist_sq = hist[slots : 2 * slots]
    hist_cnt = hist[2 * slots :]

    # Prefix-sum the histograms in int64 BEFORE dequantizing. Integer prefix
    # sums are exact and order-independent, so a segment's cumulative bins
    # are identical whether it is scanned alone (S=1 node-wise call) or
    # inside a batch (S=A level-wise call). Float32 CUDA cumsum is NOT
    # batch-invariant — it switches kernels by tensor size and perturbs the
    # last ULP — which would desync the two engines' gains at quantization
    # level and reintroduce the flaky-oracle failure this module exists to
    # prevent. Dequantizing the already-summed integers keeps the entire
    # gain computation batch-invariant (every op below is elementwise).
    inv = (1.0 / scale).view(S, 1, 1)
    c_sum = torch.cumsum(hist_sum.view(S, c, B), dim=2).float() * inv
    c_sq = torch.cumsum(hist_sq.view(S, c, B), dim=2).float() * inv
    c_cnt = torch.cumsum(hist_cnt.view(S, c, B), dim=2).float()

    total_sum = c_sum[:, :, -1:]
    total_sq = c_sq[:, :, -1:]
    total_cnt = c_cnt[:, :, -1:]
    total_sse = total_sq - total_sum**2 / (total_cnt.clamp(min=1.0) + leaf_l2)

    left_cnt = c_cnt[:, :, :-1]
    left_sum = c_sum[:, :, :-1]
    left_sq = c_sq[:, :, :-1]
    right_cnt = total_cnt - left_cnt
    right_sum = total_sum - left_sum
    right_sq = total_sq - left_sq

    legal = (left_cnt >= min_leaf_count) & (right_cnt >= min_leaf_count)
    if col_valid is not None:
        legal = legal & col_valid.unsqueeze(2)
    left_sse = left_sq - left_sum**2 / (left_cnt.clamp(min=1.0) + leaf_l2)
    right_sse = right_sq - right_sum**2 / (right_cnt.clamp(min=1.0) + leaf_l2)
    gain = total_sse - left_sse - right_sse
    gain = torch.where(legal, gain, torch.full_like(gain, float("-inf")))

    flat_gain = gain.reshape(S, c * (B - 1))
    flat_best = torch.argmax(flat_gain, dim=1)  # (S,)
    best_gain = flat_gain.gather(1, flat_best.unsqueeze(1)).squeeze(1)
    best_col = flat_best // (B - 1)
    best_bin = flat_best % (B - 1)

    ar = torch.arange(S, device=device)
    thr = vmin[ar, best_col] + bin_width[ar, best_col] * (best_bin + 1).float()

    found = torch.isfinite(best_gain) & (best_gain > 0)
    best_gain = torch.where(found, best_gain, torch.zeros_like(best_gain))
    thr = torch.where(found, thr, torch.zeros_like(thr))
    best_col = torch.where(found, best_col, torch.zeros_like(best_col))
    return best_col, thr, best_gain
