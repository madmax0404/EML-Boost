"""Triton kernels for histogram-based best-split-finding.

This module fuses the histogram-build + best-split-scan path that
``_gpu_split.py``'s torch implementation otherwise expresses as a chain
of ~6-8 torch ops (3 ``scatter_add_``, 3 ``cumsum``, 1 ``argmax``,
plus boolean masking). We compress that chain into two Triton kernel
launches:

* ``_hist_build_kernel`` — one program per (sample-block × feature-block).
  Each program reads the per-feature ``(vmin, bin_width)`` precomputed by
  the host wrapper, computes the bin index for every (sample, feature)
  in its tile, and atomic-adds ``(1, y, y²)`` into a global histogram of
  shape ``(n_features, n_bins, 3)``.
* ``_hist_scan_kernel`` — one program per feature. Reads its histogram
  row, runs an in-register cumulative sum over bins, computes
  ``gain = total_sse - left_sse - right_sse`` for every legal bin
  boundary (subject to ``min_leaf_count``), and writes the best
  ``(gain, bin_idx)`` for the feature.

The host wrapper precomputes ``vmin``/``vmax`` (cheap reductions, two
launches), launches the two kernels, and selects the global best feature
via ``out_gain.max(dim=0)``. Total: ~4 launches replacing ~8 torch ops,
plus the per-call Python dispatch round trip is shorter.

If the Triton kernels fail for any reason, the dispatcher in
``_gpu_split.py`` catches the exception and falls back to the
``gpu_histogram_split_torch`` oracle, emitting a one-time warning.

Numerical contract: every formula matches the torch oracle exactly.
* Bin index: ``clamp(((x - vmin) / bin_width).to(int), 0, n_bins-1)``
  with ``bin_width = max(vmax - vmin, 1e-12) / n_bins``.
* Gain at boundary ``b`` (left = bins ``0..b``, right = ``b+1..``):
  ``gain = total_sse - left_sse - right_sse`` where
  ``sse(sum, sq, cnt) = sq - sum²/max(cnt, 1)``.
* Threshold: ``vmin[best_feat] + bin_width[best_feat] * (best_bin + 1)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hist_build_kernel(
    feats_ptr,      # (N_SAMPLES, N_FEATURES) float32
    y_ptr,          # (N_SAMPLES,) float32
    vmin_ptr,       # (N_FEATURES,) float32
    bin_width_ptr,  # (N_FEATURES,) float32
    hist_ptr,       # (N_FEATURES, N_BINS, 3) float32 — pre-zeroed
    n_samples,
    n_features,
    feats_stride_sample,
    N_BINS: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
    BLOCK_FEATS: tl.constexpr,
):
    """Build the per-feature histograms via global atomic-add.

    Each program covers a tile of (BLOCK_SAMPLES × BLOCK_FEATS). Inside
    the tile each lane computes its bin index and atomically adds
    ``(1, y, y²)`` into ``hist_ptr[feat, bin, {0,1,2}]``.

    Writing to global atomics is the canonical Triton idiom for
    scatter-add-into-array; ``tl.atomic_add`` is supported on float32 on
    all current CUDA targets and avoids the local-array scatter-add
    issues that bedevil the in-register histogram approach.

    The grid is ``(ceil(n_samples / BLOCK_SAMPLES), ceil(n_features / BLOCK_FEATS))``.
    """
    pid_s = tl.program_id(0)
    pid_f = tl.program_id(1)

    sample_offs = pid_s * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    feat_offs = pid_f * BLOCK_FEATS + tl.arange(0, BLOCK_FEATS)
    sample_mask = sample_offs < n_samples
    feat_mask = feat_offs < n_features

    # Per-feature edge parameters: shape (BLOCK_FEATS,) broadcast over samples.
    vmin = tl.load(vmin_ptr + feat_offs, mask=feat_mask, other=0.0)
    bin_width = tl.load(
        bin_width_ptr + feat_offs, mask=feat_mask, other=1.0,
    )

    # Load y for the sample slab: (BLOCK_SAMPLES,)
    y_vals = tl.load(y_ptr + sample_offs, mask=sample_mask, other=0.0)
    y_sq_vals = y_vals * y_vals

    # Load the (BLOCK_SAMPLES, BLOCK_FEATS) tile of feature values.
    # feats is row-major (sample-major): feats[s, f] = feats_ptr[s * feats_stride_sample + f].
    x_offsets = (
        sample_offs[:, None] * feats_stride_sample + feat_offs[None, :]
    )
    tile_mask = sample_mask[:, None] & feat_mask[None, :]
    x = tl.load(feats_ptr + x_offsets, mask=tile_mask, other=0.0)

    # Compute bin index per (sample, feat) → shape (BLOCK_SAMPLES, BLOCK_FEATS).
    # Match torch oracle: ((x - vmin) / bin_width).long().clamp(0, N_BINS-1).
    rel = (x - vmin[None, :]) / bin_width[None, :]
    # Cast via int32; tl.minimum/maximum clamp.
    b = rel.to(tl.int32)
    b = tl.minimum(tl.maximum(b, 0), N_BINS - 1)

    # Per-tile histogram offsets in the flat (n_features, n_bins, 3) layout.
    # base[s, f] = feat_offs[f] * (N_BINS * 3) + b[s, f] * 3
    base = feat_offs[None, :] * (N_BINS * 3) + b * 3

    # Broadcast y / y² across the feature dim.
    y_b = tl.broadcast_to(y_vals[:, None], (BLOCK_SAMPLES, BLOCK_FEATS))
    y2_b = tl.broadcast_to(y_sq_vals[:, None], (BLOCK_SAMPLES, BLOCK_FEATS))
    ones = tl.full((BLOCK_SAMPLES, BLOCK_FEATS), 1.0, dtype=tl.float32)

    # Three atomic adds per (sample, feat). Mask off out-of-range lanes.
    tl.atomic_add(hist_ptr + base + 0, ones, mask=tile_mask)
    tl.atomic_add(hist_ptr + base + 1, y_b, mask=tile_mask)
    tl.atomic_add(hist_ptr + base + 2, y2_b, mask=tile_mask)


@triton.jit
def _hist_scan_kernel(
    hist_ptr,       # (N_FEATURES, N_BINS, 3) float32 — populated histogram
    out_gain_ptr,   # (N_FEATURES,) float32 — best gain per feature
    out_bin_ptr,    # (N_FEATURES,) int32   — best bin boundary per feature
    n_features,
    MIN_LEAF: tl.constexpr,
    N_BINS: tl.constexpr,
):
    """Per-feature cumulative-sum + gain scan in registers.

    Loads the entire histogram row for one feature into a register array
    of shape (N_BINS,) (× 3 for cnt/sum/sq), runs an in-place
    ``tl.cumsum``, computes the per-bin-boundary gain matching the torch
    oracle exactly, masks out illegal boundaries (left or right side
    fewer than MIN_LEAF samples), and writes the best (gain, bin) for
    this feature.
    """
    feat_pid = tl.program_id(0)
    if feat_pid >= n_features:
        return

    # Load the full per-feature histogram into registers: (N_BINS,) × 3.
    bin_offs = tl.arange(0, N_BINS)
    base = feat_pid * (N_BINS * 3) + bin_offs * 3
    cnt = tl.load(hist_ptr + base + 0)
    s = tl.load(hist_ptr + base + 1)
    sq = tl.load(hist_ptr + base + 2)

    # Cumulative sums along the bin axis.
    c_cnt = tl.cumsum(cnt, axis=0)
    c_sum = tl.cumsum(s, axis=0)
    c_sq = tl.cumsum(sq, axis=0)

    # Totals = last element of cumsum.
    # tl.sum reduces; we use it on the original arrays to avoid indexing
    # into the last cumsum slot (more portable across Triton versions).
    total_cnt = tl.sum(cnt, axis=0)
    total_sum = tl.sum(s, axis=0)
    total_sq = tl.sum(sq, axis=0)

    # Parent SSE = total_sq - total_sum² / max(total_cnt, 1).
    total_cnt_safe = tl.maximum(total_cnt, 1.0)
    total_sse = total_sq - total_sum * total_sum / total_cnt_safe

    # For boundary b ∈ [0, N_BINS-1], left side = bins 0..b inclusive.
    # Boundary b == N_BINS-1 is illegal (right side empty). The torch
    # oracle slices c_cnt[:, :-1] to drop it, then masks with `legal`;
    # we instead compute gain at every bin and force the last-bin gain
    # to -inf via the legality mask below.
    left_cnt = c_cnt
    left_sum = c_sum
    left_sq = c_sq
    right_cnt = total_cnt - left_cnt
    right_sum = total_sum - left_sum
    right_sq = total_sq - left_sq

    left_cnt_safe = tl.maximum(left_cnt, 1.0)
    right_cnt_safe = tl.maximum(right_cnt, 1.0)
    left_sse = left_sq - left_sum * left_sum / left_cnt_safe
    right_sse = right_sq - right_sum * right_sum / right_cnt_safe
    gain = total_sse - left_sse - right_sse

    # Legality: left and right sides each ≥ MIN_LEAF, AND bin index is
    # not the rightmost (right side empty).
    is_last_bin = bin_offs == (N_BINS - 1)
    legal = (
        (left_cnt >= MIN_LEAF)
        & (right_cnt >= MIN_LEAF)
        & (~is_last_bin)
    )
    NEG_INF = -float("inf")
    gain_masked = tl.where(legal, gain, NEG_INF)

    # Argmax: best gain, best bin. tl.argmax returns the index of the max.
    best_gain = tl.max(gain_masked, axis=0)
    best_bin = tl.argmax(gain_masked, axis=0)

    tl.store(out_gain_ptr + feat_pid, best_gain)
    tl.store(out_bin_ptr + feat_pid, best_bin.to(tl.int32))


def gpu_histogram_split_triton(
    feats: torch.Tensor,
    y: torch.Tensor,
    n_bins: int,
    min_leaf_count: int = 1,
) -> tuple[int, float, float]:
    """Triton-accelerated best-split-finding.

    Same return contract as ``gpu_histogram_split_torch``: a triple
    ``(best_feat_idx, threshold, gain)`` where a non-positive ``gain``
    signals "no legal split". The kernel matches the torch oracle's
    bin-edge math, gain formula, and threshold reconstruction exactly
    (modulo float32 reduction ordering, which is why the equivalence
    test allows ``rtol=5e-3`` on gain and ~1 bin width on threshold).

    Edge cases preserved from the torch oracle:
    * ``feats.ndim != 2`` → ``ValueError``.
    * ``n_samples < 2 * min_leaf_count`` → ``(0, 0.0, 0.0)``.
    * ``best_gain <= 0`` → ``(0, 0.0, 0.0)``.
    * Constant features (vmax == vmin) are clamped via ``rng_.clamp(min=1e-12)``
      so their bin width is non-zero; their gain ends up non-positive
      (every sample lands in bin 0 → right side empty for every legal
      boundary), so they're filtered out by the global argmax.
    """
    if feats.ndim != 2:
        raise ValueError(f"feats must be 2D, got shape {feats.shape}")
    n, d = feats.shape
    if n < 2 * min_leaf_count:
        return 0, 0.0, 0.0

    if n_bins <= 0 or (n_bins & (n_bins - 1)) != 0:
        raise ValueError(
            f"n_bins must be a positive power of 2 for the Triton kernel "
            f"(got {n_bins}). Caller can use the torch fallback "
            f"(gpu_histogram_split_torch) for arbitrary n_bins."
        )

    device = feats.device
    dtype = torch.float32

    # Cast to float32 if needed; the kernel is float32-only.
    if feats.dtype != dtype:
        feats_f = feats.to(dtype).contiguous()
    else:
        feats_f = feats.contiguous()
    if y.dtype != dtype:
        y_f = y.to(dtype).contiguous()
    else:
        y_f = y.contiguous()

    # Per-feature edge parameters via cheap torch reductions on GPU.
    vmin = feats_f.min(dim=0).values                       # (d,)
    vmax = feats_f.max(dim=0).values                       # (d,)
    rng_ = (vmax - vmin).clamp(min=1e-12)                  # (d,) — matches torch oracle
    bin_width = (rng_ / n_bins).contiguous()               # (d,)

    # Pre-zeroed global histogram of shape (d, n_bins, 3).
    hist = torch.zeros((d, n_bins, 3), dtype=dtype, device=device)

    # Histogram-build kernel.
    BLOCK_SAMPLES = 128
    BLOCK_FEATS = 16
    grid_build = (
        triton.cdiv(n, BLOCK_SAMPLES),
        triton.cdiv(d, BLOCK_FEATS),
    )
    _hist_build_kernel[grid_build](
        feats_f,
        y_f,
        vmin.contiguous(),
        bin_width,
        hist,
        n_samples=n,
        n_features=d,
        feats_stride_sample=feats_f.stride(0),
        N_BINS=n_bins,
        BLOCK_SAMPLES=BLOCK_SAMPLES,
        BLOCK_FEATS=BLOCK_FEATS,
    )

    # Scan kernel: one program per feature.
    out_gain = torch.full(
        (d,), float("-inf"), dtype=dtype, device=device,
    )
    out_bin = torch.zeros((d,), dtype=torch.int32, device=device)
    grid_scan = (d,)
    _hist_scan_kernel[grid_scan](
        hist,
        out_gain,
        out_bin,
        n_features=d,
        MIN_LEAF=min_leaf_count,
        N_BINS=n_bins,
    )

    # Host-side argmax over features. Returning `(0, 0.0, 0.0)` for
    # the no-legal-split case mirrors the torch oracle.
    best_gain_t, best_feat_t = out_gain.max(dim=0)
    best_gain_f = float(best_gain_t.item())
    if not (best_gain_f > 0):
        return 0, 0.0, 0.0
    best_feat = int(best_feat_t.item())
    best_bin = int(out_bin[best_feat].item())

    # Threshold reconstruction matches the torch oracle exactly:
    #   threshold = vmin[best_feat] + bin_width[best_feat] * (best_bin + 1)
    threshold = float(
        (vmin[best_feat] + bin_width[best_feat] * (best_bin + 1)).item()
    )
    return best_feat, threshold, best_gain_f
