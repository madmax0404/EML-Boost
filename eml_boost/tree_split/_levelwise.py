# eml_boost/tree_split/_levelwise.py
"""Level-wise (breadth-first) batched tree growth — Stage 2 of the plan.

Grows all nodes of a depth level in one batched pass (~20 kernel launches
+ 2 small syncs per LEVEL) instead of visiting ~2^depth nodes sequentially
(~25 launches + 3 syncs per NODE). Split semantics mirror
`_find_best_split_gpu` + `_grow_gpu`: per-node uniform-bin histograms,
identical gain formula, identical partition rule (val <= thr), identical
child-count re-check. Leaves are emitted as _PendingLeaf via
tree._make_pending_leaf and fitted by the Stage-1 batched finalize.

Semantic deviations from the node-wise engine (accepted by the spec):
  * descriptor RNG draws happen in BFS order, not DFS;
  * per-node top-k features are corr-descending (topk sorted=True);
  * histogram sums accumulate in fixed-point (deterministic);
  * leaf row order follows the level's stable argsort, not the node-wise
    DFS partition order. Leaf STRUCTURE and constant-leaf values are
    unaffected (both order-independent), so the no-EML structural oracle
    stays exact; only which row seeds a leaf's val/fit RNG can differ
    (see _leaf_batch.fit_leaves_batched). Do NOT try to replicate the
    node-wise row order.

In no-EML mode (n_eml_candidates=0) the RNG is never consumed and the
per-node histogram core (`multinode_histogram_split`) produces bit-identical
per-node decisions to the node-wise S=1 dispatcher, so structure, split
features, and thresholds match the node-wise engine exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from eml_boost._triton_exhaustive import (
    evaluate_trees_triton_nodewise,
    get_valid_descriptors_np,
)
from eml_boost.symbolic.snap import SnappedTree
from eml_boost.tree_split._multinode_hist import multinode_histogram_split
from eml_boost.tree_split._segmented import segment_topk_corr
from eml_boost.tree_split.nodes import EmlSplit, InternalNode, Node, RawSplit


class _Slot:
    """Python-side record for one frontier node."""

    __slots__ = ("attach",)

    def __init__(self, attach):
        # attach(node) grafts the finished Node into the parent (or root).
        self.attach = attach


def grow_levelwise(tree, indices: torch.Tensor, rng: np.random.Generator) -> Node:
    device = tree._device
    X = tree._X_gpu
    y = tree._y_gpu
    assert device is not None and X is not None and y is not None
    d = X.shape[1]
    msl = int(tree.min_samples_leaf)
    C = int(tree.n_eml_candidates)
    B = int(tree.n_bins)

    root_box: list[Node | None] = [None]

    def _attach_root(node):
        root_box[0] = node

    rows = indices  # active rows, any order
    seg = torch.zeros(rows.shape[0], dtype=torch.long, device=device)
    slots = [_Slot(_attach_root)]

    for depth in range(tree.max_depth + 1):
        L = len(slots)
        if L == 0:
            break
        # NOTE: do NOT break on rows.shape[0] == 0 — with min_samples_leaf=0
        # a slot can legitimately own zero rows and still needs its leaf
        # attached (bincount/argsort/slicing below all handle empty tensors).

        # ---- sync 1: per-slot row counts ----
        counts = torch.bincount(seg, minlength=L)
        counts_np = counts.cpu().numpy()

        # Sort rows by slot so per-slot extraction is slicing (stable →
        # row order within a slot is preserved deterministically, which
        # keeps each leaf's indices[0] seed rule reproducible).
        order = torch.argsort(seg, stable=True)
        rows = rows[order]
        seg = seg[order]
        offsets = np.zeros(L + 1, dtype=np.int64)
        np.cumsum(counts_np, out=offsets[1:])

        # Leaf-vs-attempt decision, vectorized in numpy (was a per-slot
        # Python int()+if). A slot leafs at max depth or when it holds too
        # few rows to yield two min-legal children; the rest attempt a split.
        # np.nonzero returns ascending slot ids, so leaf creation order — and
        # therefore each leaf's finalize segment — is identical to the loop.
        if depth >= tree.max_depth:
            is_leaf_np = np.ones(L, dtype=bool)
        else:
            is_leaf_np = counts_np <= 2 * msl
        for s in np.nonzero(is_leaf_np)[0]:
            s = int(s)
            slots[s].attach(tree._make_pending_leaf(rows[offsets[s] : offsets[s + 1]]))
        attempt = np.nonzero(~is_leaf_np)[0]  # ascending slot ids (int64)
        A = int(attempt.shape[0])
        if A == 0:
            break

        # Compact the attempting slots to 0..A-1. attempt_t is built once and
        # reused for the right-count gather below (was two torch.tensor calls).
        attempt_t = torch.as_tensor(attempt, device=device)  # long
        remap = torch.full((L,), -1, dtype=torch.long, device=device)
        remap[attempt_t] = torch.arange(A, dtype=torch.long, device=device)
        keep = remap[seg] >= 0
        rows_a = rows[keep]
        seg_a = remap[seg[keep]]
        y_a = y[rows_a]
        X_a = X[rows_a]

        # ---- EML candidate columns ----
        eml_vals = None
        col_valid = None
        desc_a = None
        topk_np = None
        k_used = 0
        if C > 0 and d > 0:
            k_used = min(tree.k_eml, d)
            topk = segment_topk_corr(X_a, y_a, seg_a, A, k_used)  # (A, k)
            valid_desc = get_valid_descriptors_np(2, k_used)
            if len(valid_desc) > 0:
                # BFS-order RNG consumption: one block draw per level.
                draw = rng.integers(0, len(valid_desc), size=(A, C))
                desc_a = valid_desc[draw]  # (A, C, 6) int32 numpy
                desc_gpu = torch.tensor(desc_a, dtype=torch.int32, device=device)
                Xk = X_a.gather(1, topk[seg_a])  # (Na, k)
                eml = evaluate_trees_triton_nodewise(desc_gpu, seg_a, Xk, k_used)
                # per-(slot, candidate) finiteness -> col_valid for EML cols
                nonfinite = torch.zeros(A, C, device=device).index_add_(
                    0, seg_a, (~torch.isfinite(eml)).float().T
                )
                eml_ok = nonfinite == 0
                eml_vals = torch.nan_to_num(
                    eml.T, nan=0.0, posinf=0.0, neginf=0.0
                )  # neutralized; invalid cols are masked out of the gain argmax
                col_valid = torch.cat(
                    [torch.ones(A, d, dtype=torch.bool, device=device), eml_ok],
                    dim=1,
                )
                topk_np = topk.cpu().numpy()

        values = X_a if eml_vals is None else torch.cat([X_a, eml_vals], dim=1)

        # ---- batched split decision ----
        best_col, best_thr, best_gain = multinode_histogram_split(
            values, y_a, seg_a, A, B,
            min_leaf_count=msl, leaf_l2=tree.leaf_l2, col_valid=col_valid,
        )

        # ---- partition + child-count re-check (mirrors _grow_gpu) ----
        row_col = best_col[seg_a]
        row_thr = best_thr[seg_a]
        go_left = values.gather(1, row_col.unsqueeze(1)).squeeze(1) <= row_thr
        left_cnt = torch.zeros(A, device=device).index_add_(
            0, seg_a, go_left.float()
        )
        right_cnt = counts[attempt_t].float() - left_cnt

        # ---- sync 2: level decisions ----
        dec = torch.cat(
            [
                best_col.float(), best_thr, best_gain, left_cnt, right_cnt,
            ]
        ).cpu().numpy()
        col_np = dec[0 * A : 1 * A].astype(np.int64)
        thr_np = dec[1 * A : 2 * A]
        gain_np = dec[2 * A : 3 * A]
        lcnt_np = dec[3 * A : 4 * A].astype(np.int64)
        rcnt_np = dec[4 * A : 5 * A].astype(np.int64)

        # Build next frontier.
        new_slots: list[_Slot] = []
        child_of = torch.full((A,), -1, dtype=torch.long, device=device)
        for a, s in enumerate(attempt):
            s = int(s)
            sl = rows[offsets[s] : offsets[s + 1]]
            if gain_np[a] <= 0 or lcnt_np[a] < msl or rcnt_np[a] < msl:
                slots[s].attach(tree._make_pending_leaf(sl))
                continue
            if col_np[a] < d:
                split: RawSplit | EmlSplit = RawSplit(
                    feature_idx=int(col_np[a]), threshold=float(thr_np[a])
                )
            else:
                c_idx = int(col_np[a]) - d
                split = EmlSplit(
                    snapped=SnappedTree(
                        depth=2, k=k_used,
                        internal_input_count=2, leaf_input_count=4,
                        terminal_choices=tuple(int(v) for v in desc_a[a, c_idx]),
                    ),
                    feature_subset=tuple(int(v) for v in topk_np[a]),
                    threshold=float(thr_np[a]),
                )
            node = InternalNode(split=split, left=None, right=None)  # type: ignore[arg-type]
            slots[s].attach(node)
            base = len(new_slots)
            new_slots.append(_Slot(lambda n_, nd=node: setattr(nd, "left", n_)))
            new_slots.append(_Slot(lambda n_, nd=node: setattr(nd, "right", n_)))
            child_of[a] = base

        if not new_slots:
            break

        # Advance active rows to child slots; drop rows of nodes that leafed.
        row_child = child_of[seg_a]
        alive = row_child >= 0
        seg = row_child[alive] + (~go_left[alive]).long()
        rows = rows_a[alive]
        slots = new_slots

    root = root_box[0]
    assert root is not None
    return root
