# eml_boost/tree_split/_leaf_batch.py
"""Batched leaf finalization for Stage 1 of the level-wise growth plan.

Replaces ~250 sequential `_fit_leaf` calls per tree (each ~30 kernel
launches + 2-3 device syncs) with one batched pass over all pending
leaves: segmented constant values, segmented top-k correlation, one
Triton evaluation of the shared descriptor set over all leaf rows,
segmented OLS + vectorized gate, and a handful of batched D2H readbacks
(a fixed small count independent of leaf count, vs. ~3 per leaf before).

Semantics mirror `EmlSplitTreeRegressor._fit_leaf` (gated policy) exactly;
see that method for the reference formulas. The stacked-blend policy is
NOT handled here — `_finalize_leaves` routes it to the per-leaf path.

Determinism note: the depth-2 / k=1 descriptor grammar is structurally
redundant — e.g. any tree whose root doesn't select `f_prev` on a given
side has FOUR functionally-identical descriptors (that side's unused
leaf-terminal choices are multiplied by zero), so `evaluate_trees_triton`
produces bit-identical prediction rows for several distinct tree indices.
The reference `_fit_leaf` breaks such ties deterministically (lowest tree
index wins `argmin`) because its per-leaf reductions are plain dense
`.sum(dim=1)` calls, which are value-only-dependent. This module instead
reduces ACROSS leaves sharing one kernel launch via `index_add_`/
`scatter_reduce_`, and `index_add_`'s CUDA atomic accumulation order is a
per-call race — confirmed empirically to make bit-identical duplicate rows
sum to different float32 results depending on scheduling, which flips
`argmin` on exact ties and diverges from the reference's tree choice (same
bug class Task 2A fixed for split-finding's histogram atomics). We enable
`torch.use_deterministic_algorithms` for the duration of this function so
`index_add_` uses its deterministic (value-only-dependent) CUDA path
instead; `scatter_reduce_` here only ever uses amin/amax (already
order-independent per `_segmented.segment_minmax`'s docstring) so it is
unaffected either way. Scoped via try/finally so the global flag doesn't
leak into caller code; accepted cost is `index_add_`'s slower deterministic
kernel, mirrored from Task 2A's own precedent of trading Triton histogram
throughput for determinism.
"""

from __future__ import annotations

import numpy as np
import torch

from eml_boost._triton_exhaustive import (
    evaluate_trees_triton,
    get_descriptor_gpu,
    get_descriptor_np,
    get_feature_mask_gpu,
)
from eml_boost.symbolic.snap import SnappedTree
from eml_boost.tree_split._segmented import segment_topk_corr
from eml_boost.tree_split.nodes import EmlLeafNode, LeafNode, Node

# Chunk the (n_trees × total_rows) prediction matrix past this many rows
# to bound peak memory (~144 trees × 500k rows × 4B ≈ 288 MB per matrix).
_CHUNK_ROWS = 500_000


def fit_leaves_batched(tree, pending) -> list[Node]:
    if len(pending) == 0:
        return []
    device = tree._device
    assert device is not None and tree._X_gpu is not None
    y_gpu = tree._y_gpu
    leaf_l2 = float(tree.leaf_l2)
    L = len(pending)

    sizes = [int(p.indices.shape[0]) for p in pending]  # shape metadata, no sync
    # float32 count sums (index_add_ of fit/val 0/1 masks) require exact
    # integer representation; float32 mantissa is 24 bits, so total n < 2**24.
    n = sum(sizes)
    assert n < (1 << 24), (
        f"levelwise float32 count sums require n < 2**24, got n={n}"
    )
    n_raw = tree._X_cpu.shape[1] if tree._X_cpu is not None else 0

    # See module docstring: index_add_'s atomic accumulation order can break
    # exact ties between functionally-duplicate descriptors differently than
    # the reference's dense per-leaf sums. Force the deterministic (value-
    # only-dependent) CUDA kernels for every index_add_/scatter_reduce_ call
    # below, restoring the caller's setting on every exit path.
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        # ---- constant values for ALL leaves (one segmented pass) ----
        nonempty = [i for i, sz in enumerate(sizes) if sz > 0]
        const_vals = np.zeros(L, dtype=np.float64)
        if nonempty:
            rows_all = torch.cat([pending[i].indices for i in nonempty])
            rep = torch.tensor(
                [sizes[i] for i in nonempty], dtype=torch.long, device=device
            )
            seg_all = torch.repeat_interleave(
                torch.arange(len(nonempty), dtype=torch.long, device=device), rep
            )
            sum_y = torch.zeros(len(nonempty), device=device).index_add_(
                0, seg_all, y_gpu[rows_all]
            )
            n_t = rep.float()
            cv = (sum_y / (n_t + leaf_l2)).cpu().numpy()
            for slot, i in enumerate(nonempty):
                const_vals[i] = float(cv[slot])

        # ---- eligibility (pure CPU metadata; mirrors _fit_leaf's early-outs) ----
        def _eligible(sz: int) -> bool:
            if tree.k_leaf_eml <= 0 or sz < tree.min_samples_leaf_eml or n_raw == 0:
                return False
            val_sz = max(sz // 4, 5)
            return sz - val_sz >= tree.min_samples_leaf_eml // 2

        e_ids = [i for i, sz in enumerate(sizes) if sz > 0 and _eligible(sz)]
        e_id_set = set(e_ids)
        out: list[Node | None] = [None] * L
        for i in range(L):
            if i not in e_id_set:
                out[i] = LeafNode(value=float(const_vals[i]))
        if not e_ids:
            return [n for n in out]  # type: ignore[list-item]

        # ---- eligible sub-batch ----
        E = len(e_ids)
        k = min(tree.k_leaf_eml, n_raw)
        rows_e = torch.cat([pending[i].indices for i in e_ids])
        sizes_e = [sizes[i] for i in e_ids]
        rep_e = torch.tensor(sizes_e, dtype=torch.long, device=device)
        seg_e = torch.repeat_interleave(
            torch.arange(E, dtype=torch.long, device=device), rep_e
        )
        Ne = int(rows_e.shape[0])
        X_e = tree._X_gpu[rows_e]  # (Ne, d)
        y_e = y_gpu[rows_e]  # (Ne,)

        # Per-leaf top-k features (corr-descending; identical to reference at k=1).
        top_feats = segment_topk_corr(X_e, y_e, seg_e, E, k)  # (E, k)
        mean_x = tree._global_mean_gpu[top_feats]  # (E, k)
        std_x = tree._global_std_gpu[top_feats]  # (E, k)

        Xsub = torch.clamp(
            (X_e.gather(1, top_feats[seg_e]) - mean_x[seg_e]) / std_x[seg_e], -3.0, 3.0
        )  # (Ne, k)

        # Per-leaf deterministic train/val masks (seed = first row index — the
        # exact rule _fit_leaf uses; independent of the shared tree RNG).
        firsts = (
            torch.stack([pending[i].indices[0] for i in e_ids]).cpu().numpy()
        )  # one small D2H
        fit_mask_np = np.empty(Ne, dtype=bool)
        pos = 0
        for slot, sz in enumerate(sizes_e):
            rng_leaf = np.random.default_rng(int(firsts[slot]))
            perm = rng_leaf.permutation(sz)
            val_sz = max(sz // 4, 5)
            m = np.ones(sz, dtype=bool)
            m[perm[:val_sz]] = False  # False = val row
            fit_mask_np[pos : pos + sz] = m
            pos += sz
        fit_f = torch.tensor(fit_mask_np, device=device).float()  # (Ne,)
        val_f = 1.0 - fit_f

        # cap per leaf: leaf_eml_cap_k * max|y_leaf| (order-independent amax).
        cap_k = float(tree.leaf_eml_cap_k)
        if cap_k > 0.0:
            cap = torch.full((E,), -float("inf"), device=device).scatter_reduce_(
                0, seg_e, y_e.abs(), reduce="amax", include_self=True
            ) * cap_k
        else:
            cap = torch.full((E,), float("inf"), device=device)

        # ---- evaluate the shared descriptor set over all leaf rows ----
        desc_gpu = get_descriptor_gpu(depth=2, k=k, device=device)  # (T, 6)
        feature_mask = get_feature_mask_gpu(depth=2, k=k, device=device)  # (T,)
        T = int(desc_gpu.shape[0])

        if Ne <= _CHUNK_ROWS:
            preds = evaluate_trees_triton(desc_gpu, Xsub, k)  # (T, Ne)
            preds_list = [(preds, seg_e, y_e, fit_f, val_f, 0)]
        else:
            preds_list = []
            for s in range(0, Ne, _CHUNK_ROWS):
                e = min(s + _CHUNK_ROWS, Ne)
                preds_list.append(
                    (
                        evaluate_trees_triton(desc_gpu, Xsub[s:e], k),
                        seg_e[s:e],
                        y_e[s:e],
                        fit_f[s:e],
                        val_f[s:e],
                        s,
                    )
                )

        def _acc(fn):
            acc = torch.zeros(T, E, device=device)
            for preds, seg, yy, ff, vf, _s in preds_list:
                acc.index_add_(1, seg, fn(preds, yy, ff, vf))
            return acc

        n_fit = torch.zeros(E, device=device).index_add_(0, seg_e, fit_f)  # (E,)
        sum_p = _acc(lambda p, yy, ff, vf: p * ff)
        sum_p2 = _acc(lambda p, yy, ff, vf: p * p * ff)
        sum_py = _acc(lambda p, yy, ff, vf: p * (yy * ff).unsqueeze(0))
        sum_y_f = torch.zeros(E, device=device).index_add_(0, seg_e, y_e * fit_f)
        bad = _acc(lambda p, yy, ff, vf: (~torch.isfinite(p)).float())  # any row (fit+val)

        # ---- closed-form OLS per (tree, leaf) — formulas verbatim from _fit_leaf ----
        n_fit_reg = n_fit + leaf_l2
        det = sum_p2 * n_fit - sum_p * sum_p
        lam = float(tree.leaf_eml_ridge)
        det_ridged = det + n_fit * lam
        det_safe = torch.where(
            det_ridged.abs() > 1e-6, det_ridged, torch.ones_like(det_ridged)
        )
        eta = (n_fit * sum_py - sum_p * sum_y_f) / det_safe
        if lam == 0.0:
            bias = (sum_p2 * sum_y_f - sum_p * sum_py) / det_safe
            if leaf_l2 > 0.0:
                bias = bias * n_fit / n_fit_reg
        else:
            bias = (sum_y_f - eta * sum_p) / n_fit_reg

        finite_coefs = torch.isfinite(eta) & torch.isfinite(bias)
        valid = (
            feature_mask.unsqueeze(1)
            & (bad == 0)
            & finite_coefs
            & (det.abs() > 1e-6)
        )  # (T, E)

        # ---- val SSE per (tree, leaf) with cap; constant val SSE per leaf ----
        val_sse = torch.zeros(T, E, device=device)
        const_sse = torch.zeros(E, device=device)
        mean_full = (
            torch.zeros(E, device=device).index_add_(0, seg_e, y_e)
            / torch.zeros(E, device=device).index_add_(0, seg_e, torch.ones_like(y_e))
        )
        for preds, seg, yy, _ff, vf, _s in preds_list:
            vp = eta[:, seg] * preds + bias[:, seg]
            vp = torch.clamp(vp, min=-cap[seg], max=cap[seg])
            res = (yy.unsqueeze(0) - vp) * vf
            val_sse.index_add_(1, seg, res * res)
            cres = (yy - mean_full[seg]) * vf
            const_sse.index_add_(0, seg, cres * cres)
        val_sse = torch.where(valid, val_sse, torch.full_like(val_sse, float("inf")))

        # ---- gate + select + readback ----
        best_idx = val_sse.argmin(dim=0)  # (E,)
        ar = torch.arange(E, device=device)
        best_sse = val_sse[best_idx, ar]
        valid_b = valid[best_idx, ar]
        eta_b = eta[best_idx, ar]
        bias_b = bias[best_idx, ar]
        thr = float(tree.leaf_eml_gain_threshold)
        accept = valid_b & (best_sse < const_sse * (1.0 - thr))

        # Consolidate the eight adjacent D2H readbacks into three transfers,
        # grouped by shape/dtype. .cpu() is a pure copy (no compute), so every
        # extracted value is bit-identical to the per-tensor version; this only
        # collapses eight device->host launches into three.
        flt_e = torch.stack([eta_b, bias_b, cap]).cpu().numpy()  # (3, E)
        flt_ek = torch.stack([mean_x, std_x]).cpu().numpy()  # (2, E, k)
        lng = (
            torch.cat(
                [best_idx.unsqueeze(1), accept.long().unsqueeze(1), top_feats], dim=1
            )
            .cpu()
            .numpy()
        )  # (E, k+2)
        eta_np, bias_np, cap_np = flt_e[0], flt_e[1], flt_e[2]
        mean_np, std_np = flt_ek[0], flt_ek[1]
        best_idx_np = lng[:, 0]
        accept_np = lng[:, 1].astype(bool)
        feats_np = lng[:, 2:]

        desc_np = get_descriptor_np(2, k)
        for slot, i in enumerate(e_ids):
            if not bool(accept_np[slot]):
                out[i] = LeafNode(value=float(const_vals[i]))
                continue
            drow = desc_np[int(best_idx_np[slot])]
            out[i] = EmlLeafNode(
                snapped=SnappedTree(
                    depth=2,
                    k=k,
                    internal_input_count=2,
                    leaf_input_count=4,
                    terminal_choices=tuple(int(v) for v in drow),
                ),
                feature_subset=tuple(int(v) for v in feats_np[slot]),
                feature_mean=tuple(float(v) for v in mean_np[slot]),
                feature_std=tuple(float(v) for v in std_np[slot]),
                eta=float(eta_np[slot]),
                bias=float(bias_np[slot]),
                cap=float(cap_np[slot]),
            )
        return [n for n in out]  # type: ignore[list-item]
    finally:
        torch.use_deterministic_algorithms(prev_deterministic)
