"""Oracle tests for per-segment GPU statistics against per-node reference ops."""
import numpy as np
import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_segments(n=3000, d=8, n_seg=7, seed=0):
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float32, device="cuda")
    # y correlates with different features per segment so top-k is nontrivial
    seg = torch.tensor(rng.integers(0, n_seg, size=n), dtype=torch.long, device="cuda")
    y = torch.zeros(n, device="cuda")
    for s in range(n_seg):
        m = seg == s
        j = s % d
        y[m] = 3.0 * X[m, j] + 0.5 * X[m, (j + 1) % d]
    y = y + 0.01 * torch.tensor(rng.standard_normal(n), dtype=torch.float32, device="cuda")
    return X, y, seg, n_seg


@requires_cuda
def test_segment_counts_matches_bincount():
    from eml_boost.tree_split._segmented import segment_counts
    _X, _y, seg, n_seg = _make_segments()
    got = segment_counts(seg, n_seg)
    want = torch.bincount(seg, minlength=n_seg).float()
    assert torch.equal(got, want)


@requires_cuda
def test_segment_corr_matches_per_node_reference():
    from eml_boost.tree_split._segmented import segment_corr
    X, y, seg, n_seg = _make_segments()
    got = segment_corr(X, y, seg, n_seg)  # (S, D)
    for s in range(n_seg):
        m = seg == s
        Xs, ys = X[m], y[m]
        Xc = Xs - Xs.mean(dim=0, keepdim=True)
        yc = ys - ys.mean()
        num = (Xc * yc.unsqueeze(1)).sum(dim=0)
        denom = Xc.norm(dim=0) * yc.norm() + 1e-12
        want = (num / denom).abs()
        np.testing.assert_allclose(
            got[s].cpu().numpy(), want.cpu().numpy(), rtol=1e-4, atol=1e-6
        )


@requires_cuda
def test_segment_topk_corr_picks_reference_features():
    from eml_boost.tree_split._segmented import segment_topk_corr
    X, y, seg, n_seg = _make_segments()
    idx = segment_topk_corr(X, y, seg, n_seg, k=2)  # (S, 2)
    for s in range(n_seg):
        j = s % 8
        # segment s was built as 3*x_j + 0.5*x_{j+1}: top-1 must be j
        assert int(idx[s, 0]) == j
        assert int(idx[s, 1]) == (j + 1) % 8


@requires_cuda
def test_segment_corr_empty_segment_is_zero():
    from eml_boost.tree_split._segmented import segment_corr
    X, y, seg, n_seg = _make_segments()
    got = segment_corr(X, y, seg, n_seg + 3)  # 3 segments with no rows
    assert torch.isfinite(got).all()
    assert torch.equal(got[n_seg:], torch.zeros_like(got[n_seg:]))


@requires_cuda
def test_segment_minmax_matches_loop():
    from eml_boost.tree_split._segmented import segment_minmax
    rng = np.random.default_rng(1)
    vals = torch.tensor(rng.standard_normal((5000, 4)), dtype=torch.float32, device="cuda")
    seg = torch.tensor(rng.integers(0, 6, size=5000), dtype=torch.long, device="cuda")
    mn, mx = segment_minmax(vals, seg, 8)  # segments 6, 7 empty
    for s in range(6):
        m = seg == s
        assert torch.equal(mn[s], vals[m].min(dim=0).values)
        assert torch.equal(mx[s], vals[m].max(dim=0).values)
    assert bool(torch.isinf(mn[6:]).all()) and bool(torch.isinf(mx[6:]).all())
