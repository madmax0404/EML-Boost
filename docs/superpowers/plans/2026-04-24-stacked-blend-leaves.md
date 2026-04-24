# Stacked-Blend Leaves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binary EML-leaf gate with a val-fit convex blend between constant and EML predictions, evaluate across 3 seeds on the Experiment 8 datasets.

**Architecture:** Per leaf, fit `(η, β)` on 75% train / evaluate each of 144 candidate depth-2 EML trees on the 25% val portion. For each candidate compute the closed-form optimal α ∈ [0, 1] that minimizes `||y_val − α·ȳ − (1−α)·(η·eml + β)||²`. Pick the tree with the smallest α-optimized val-SSE. Fold α into `(η, β)` so the post-fit representation (and predict-time path) is unchanged. Blend-off vs. blend-on are same-commit via a `use_stacked_blend` flag for a clean ablation.

**Tech Stack:** Python 3.11+, NumPy, PyTorch+CUDA, Triton 3.6+, pytest, PMLB, XGBoost, LightGBM, matplotlib. uv for environment management.

---

## Background an implementer needs

**The codebase under change** is `EmlSplitTreeRegressor` (`eml_boost/tree_split/tree.py`), a regression tree whose *internal* nodes may split on either a raw feature or a sampled depth-2 EML expression, and whose *leaves* may store a constant (mean of residuals in the leaf) or an EML expression `η · eml((x − μ)/σ clamped) + β` (Phase 4 — the code currently under modification). `EmlSplitBoostRegressor` (`eml_boost/tree_split/ensemble.py`) is a gradient-boosting wrapper over those trees.

The EML Sheffer operator is `eml(a, b) = exp(a) − ln(b)`. The Option-A grammar `S → 1 | x_j | eml(S, S)` at depth 2 with `k=1` features produces **144 enumerable trees**. They are enumerated and evaluated on GPU via a Triton kernel in `eml_boost/_triton_exhaustive.py`.

**Existing leaf-fit code** lives in `EmlSplitTreeRegressor._fit_leaf` at `eml_boost/tree_split/tree.py:330-444`. It:

1. Standardizes the top-`k_leaf_eml` residual-correlated features using global (fit-time) mean/std, clamps to `[−3, 3]`.
2. Splits the leaf samples 75%/25% fit/val using a deterministic per-leaf seed.
3. On the fit portion, closed-form OLS fits `(η, β)` for every one of 144 candidate trees in one batched GPU operation.
4. On the val portion, computes per-tree val-SSE and picks the minimum.
5. Rejects the EML leaf if val-SSE improvement over the constant baseline is less than `leaf_eml_gain_threshold` (default 0.05).

This plan changes steps 4-5 into a val-fit α blend, keeping steps 1-3 intact.

**Test infrastructure:** existing unit tests are in `tests/unit/test_eml_split_tree.py` (11 tests) and `tests/unit/test_eml_split_boost.py` (5 tests). Run with `uv run pytest tests/unit/ -v`. Some tests require CUDA (they use `torch.cuda.is_available()` and `pytest.skip()` if not present). The RTX 3090 in the dev environment has CUDA; these tests run.

**Before starting Task 1, read:**
- `eml_boost/tree_split/tree.py:1-100` (class header and constructor)
- `eml_boost/tree_split/tree.py:330-444` (current `_fit_leaf`)
- `eml_boost/tree_split/nodes.py` (LeafNode, EmlLeafNode, RawSplit, EmlSplit, InternalNode dataclasses)
- `eml_boost/tree_split/ensemble.py` (wrapper; ~180 lines)
- `tests/unit/test_eml_split_tree.py` (to see the test style)
- The spec: `docs/superpowers/specs/2026-04-24-stacked-blend-leaves-design.md`

---

## Task 1: Refactor — extract shared setup, add `use_stacked_blend` param

**Goal:** Add the constructor parameter and refactor `_fit_leaf` so the shared GPU setup is in one place and the gated tree-selection logic is in a clearly named helper. Behavior unchanged: `use_stacked_blend=False` is the default and is equivalent to today's code.

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (the `EmlSplitTreeRegressor` class — constructor + `_fit_leaf`)
- Modify: `eml_boost/tree_split/ensemble.py` (the `EmlSplitBoostRegressor` class — pass-through in constructor and `fit`)
- Modify: `tests/unit/test_eml_split_tree.py` (one new smoke test)

- [ ] **Step 1: Write a smoke test for the new parameter.**

Add this test at the bottom of `tests/unit/test_eml_split_tree.py`:

```python
def test_use_stacked_blend_false_matches_current_behavior():
    """With `use_stacked_blend=False` (and the rest of the config identical),
    the regressor should behave exactly like the current gated implementation:
    pure-noise training data should leave most leaves as constants."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = rng.normal(size=800)
    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        leaf_eml_gain_threshold=0.05,
        use_stacked_blend=False,
        random_state=0,
    ).fit(X, y)
    n_eml = _count_eml_leaves(m._root)
    n_total = _count_leaves(m._root)
    assert n_eml < 0.4 * n_total
```

- [ ] **Step 2: Run the test; confirm it fails with "unexpected keyword argument `use_stacked_blend`".**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_use_stacked_blend_false_matches_current_behavior -v
```
Expected: FAIL with `TypeError: EmlSplitTreeRegressor.__init__() got an unexpected keyword argument 'use_stacked_blend'`.

- [ ] **Step 3: Add `use_stacked_blend` parameter to `EmlSplitTreeRegressor.__init__`.**

In `eml_boost/tree_split/tree.py`, locate the `__init__` signature and the hyperparameter storage block around lines 60-100. Add `use_stacked_blend: bool = False` at the end of the keyword-only args (just before `random_state`), and store it as an attribute.

Constructor signature addition (default False in this task; Task 3 flips to True):
```python
use_stacked_blend: bool = False,
random_state: int | None = None,
```

Attribute storage (add alongside the other hyperparams):
```python
self.use_stacked_blend = use_stacked_blend
```

- [ ] **Step 4: Add `use_stacked_blend` parameter to `EmlSplitBoostRegressor.__init__` and thread it to each round's tree.**

In `eml_boost/tree_split/ensemble.py`, locate the `__init__` signature at around line 56 and the per-round `EmlSplitTreeRegressor(...)` construction at around line 131.

Add parameter to constructor (default False; Task 3 flips to True) just after `leaf_eml_gain_threshold`:
```python
leaf_eml_gain_threshold: float = 0.05,
use_stacked_blend: bool = False,
patience: int | None = 15,
```

Add attribute storage alongside the others:
```python
self.use_stacked_blend = use_stacked_blend
```

In the per-round tree construction inside `fit`, add the kwarg:
```python
tree = EmlSplitTreeRegressor(
    max_depth=self.max_depth,
    ...
    leaf_eml_gain_threshold=self.leaf_eml_gain_threshold,
    use_stacked_blend=self.use_stacked_blend,
    random_state=tree_seeds[m],
).fit(X_tr, r)
```

- [ ] **Step 5: Run the smoke test; confirm it passes.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_use_stacked_blend_false_matches_current_behavior -v
```
Expected: PASS.

- [ ] **Step 6: Refactor `_fit_leaf` — extract shared GPU setup, move gated logic to `_select_leaf_gated`.**

In `eml_boost/tree_split/tree.py`, replace the `_fit_leaf` method (current lines 330-444) with this structure. The point of the refactor is that steps 1-6 (the early-out gates through the per-tree OLS) are shared across both blend-off and blend-on; the branch is only steps 7-9 (tree selection + gate / blend + leaf emission).

Replace the entire `_fit_leaf` method body with:

```python
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

    # Global-stat standardization + clamp to [-3, 3].
    assert self._global_mean_gpu is not None and self._global_std_gpu is not None
    top_features_t = torch.from_numpy(top_features).to(device=device, dtype=torch.long)
    mean_x = self._global_mean_gpu[top_features_t]
    std_x = self._global_std_gpu[top_features_t]
    X_sub = torch.clamp((X_sub_raw - mean_x) / std_x, -3.0, 3.0)

    # Deterministic 75/25 leaf-local split.
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

    # Closed-form OLS per tree on the fit portion.
    n_fit = float(X_fit.shape[0])
    sum_p = preds_fit.sum(dim=1)
    sum_p2 = (preds_fit * preds_fit).sum(dim=1)
    sum_y_f = y_fit.sum()
    sum_py_f = (preds_fit * y_fit.unsqueeze(0)).sum(dim=1)
    det = sum_p2 * n_fit - sum_p * sum_p
    det_safe = torch.where(det.abs() > 1e-6, det, torch.ones_like(det))
    eta = (n_fit * sum_py_f - sum_p * sum_y_f) / det_safe
    bias = (sum_p2 * sum_y_f - sum_p * sum_py_f) / det_safe

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
```

Then, immediately below `_fit_leaf`, add the extracted `_select_leaf_gated` method. This is the exact tree-selection logic from the old `_fit_leaf` (lines 408-444 in the current file):

```python
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
```

And add a placeholder `_select_leaf_blended` that just delegates to the gated path for now — we implement it in Task 2:

```python
def _select_leaf_blended(self, **ctx) -> Node:
    """Stacked-blend tree selection (Task 2). Temporarily routes to the
    gated path so the refactor in Task 1 is behavior-preserving."""
    return self._select_leaf_gated(**ctx)
```

- [ ] **Step 7: Run the full unit test suite; confirm all tests pass.**

Run:
```bash
uv run pytest tests/unit/ -v
```
Expected: all tests pass (17 tests total: 16 existing + 1 new smoke test from Step 1).

- [ ] **Step 8: Commit.**

```bash
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
refactor: extract shared leaf setup + add use_stacked_blend flag

Split the EML-leaf fit into a shared GPU-setup preamble and a tree-
selection helper. Adds the use_stacked_blend flag (default False, no
behavior change yet); True currently routes to the gated path. Next
task implements the blended path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement the stacked-blend tree selection

**Goal:** Implement `_select_leaf_blended` with the val-fit convex blend. When `use_stacked_blend=True`, per-candidate-tree it picks the optimal α ∈ [0, 1] in closed form, selects the tree with smallest blended val-SSE, folds α into `(η, β)`, and emits either `LeafNode` (α ≈ 1 collapse) or `EmlLeafNode`.

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (replace the placeholder `_select_leaf_blended`)
- Modify: `tests/unit/test_eml_split_tree.py` (add 3 tests)

- [ ] **Step 1: Write the failing test for α ≈ 1 on pure noise.**

Add this test at the bottom of `tests/unit/test_eml_split_tree.py`:

```python
def test_stacked_blend_collapses_to_constant_on_pure_noise():
    """With `use_stacked_blend=True`, a regressor trained on pure Gaussian
    noise should produce mostly constant leaves because α ≈ 1 for every
    candidate — there's no EML signal to latch onto."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = rng.normal(size=800)
    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        use_stacked_blend=True, random_state=0,
    ).fit(X, y)
    n_eml = _count_eml_leaves(m._root)
    n_total = _count_leaves(m._root)
    # Blend-on should be at LEAST as regularizing as the 5% gate on noise;
    # 40% was the gate's tolerance and is also what we require here.
    assert n_eml < 0.4 * n_total, f"{n_eml}/{n_total} EML leaves on pure noise"
```

- [ ] **Step 2: Run the test; confirm it fails.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_stacked_blend_collapses_to_constant_on_pure_noise -v
```

Expected: this may actually *accidentally pass* because `_select_leaf_blended` currently delegates to `_select_leaf_gated`. That's fine — note it and proceed; the real test that will fail is the clean-signal one in Step 4.

- [ ] **Step 3: Write the failing test for α ≈ 0 on a clean elementary signal.**

Add this test just below the previous one:

```python
def test_stacked_blend_activates_on_clean_elementary_signal():
    """On `y = exp(x_0) + tiny_noise`, the blend should latch onto the EML
    tree (α ≈ 0) and produce EML leaves that outperform constant leaves."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0]) + 0.01 * rng.normal(size=800)

    m_blend = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_stacked_blend=True, random_state=0,
    ).fit(X, y)
    m_const = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=0, use_stacked_blend=True, random_state=0,
    ).fit(X, y)

    mse_blend = _mse(m_blend.predict(X), y)
    mse_const = _mse(m_const.predict(X), y)
    assert mse_blend < mse_const, f"blend={mse_blend:.4f} vs const={mse_const:.4f}"
    assert _count_eml_leaves(m_blend._root) >= 1
```

- [ ] **Step 4: Run this test; confirm it fails.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_stacked_blend_activates_on_clean_elementary_signal -v
```

Expected: FAIL. Because `_select_leaf_blended` delegates to the gated path, the gate's 5% threshold may reject some leaves that the blend would've kept, or the folded-α collapse logic isn't there, so the EML-leaf count/behavior may not match. The specific failure depends on the 0.05 gate's decisions on this signal; proceed regardless.

(If the test accidentally passes, that means the gated path already satisfies the assertions on this data. Still proceed to implement the blended path — the next test will require it.)

- [ ] **Step 5: Write the failing test for numerical stability on heavy-tailed features.**

Add this test just below the previous one:

```python
def test_stacked_blend_no_numerical_blowup_on_heavy_tails():
    """A leaf-local feature with magnitudes into the millions (like
    PMLB 562_cpu_small) must not produce NaN/inf predictions under the
    blended path."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 2)) * 1e6
    # Targets that are a small, well-behaved transformation of the big feature.
    y = 0.001 * (X[:, 0] / 1e6)

    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=50, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=50,
        use_stacked_blend=True, random_state=0,
    ).fit(X, y)
    pred = m.predict(X)
    assert np.all(np.isfinite(pred)), (
        "prediction contains NaN or inf — numerical stability failure"
    )
```

- [ ] **Step 6: Run this test; confirm it passes or fails.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_stacked_blend_no_numerical_blowup_on_heavy_tails -v
```

Expected: PASS (the existing `[-3, 3]` clamp already handles this in the shared setup preamble). Record the result; if it fails, the blended path's fold-into-(η, β) logic must not re-introduce overflow. The fold multiplies η by `(1−α)`, which can only *reduce* magnitude, so failure here would signal a deeper issue — investigate before proceeding.

- [ ] **Step 7: Implement `_select_leaf_blended`.**

In `eml_boost/tree_split/tree.py`, replace the placeholder `_select_leaf_blended` with the actual implementation:

```python
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
    if abs(eta_folded) < 1e-10:
        return LeafNode(value=bias_folded)

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
        eta=eta_folded,
        bias=bias_folded,
    )
```

- [ ] **Step 8: Run the three new blend tests; confirm all pass.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py -v -k "stacked_blend"
```

Expected: all four tests pass (the Task 1 smoke test plus the three new ones):
- `test_use_stacked_blend_false_matches_current_behavior` PASS
- `test_stacked_blend_collapses_to_constant_on_pure_noise` PASS
- `test_stacked_blend_activates_on_clean_elementary_signal` PASS
- `test_stacked_blend_no_numerical_blowup_on_heavy_tails` PASS

- [ ] **Step 9: Run the FULL unit test suite; confirm every test still passes.**

Run:
```bash
uv run pytest tests/unit/ -v
```

Expected: all 20 tests pass (16 original + 1 from Task 1 + 3 new).

- [ ] **Step 10: Commit.**

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: implement stacked-blend tree selection for EML leaves

Per candidate tree, fit the optimal α ∈ [0, 1] in closed form on the
val portion to minimize ||y_val − α·ȳ − (1−α)·(η·eml + β)||². Pick the
tree with smallest α-optimized val-SSE; fold α into (η, β); emit a
LeafNode when the blend collapses (α≈1) or an EmlLeafNode otherwise.
No gate — α subsumes the accept/reject decision.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Flip default to `use_stacked_blend=True`

**Goal:** Make the blended path the default. Verify all existing tests still pass under the new default. No test file changes are expected — the blended path should behave equivalently or better on the existing test fixtures (noise → α≈1; elementary signal → α≈0).

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (change default in `__init__`)
- Modify: `eml_boost/tree_split/ensemble.py` (change default in `__init__`)

- [ ] **Step 1: Flip the default in `EmlSplitTreeRegressor`.**

In `eml_boost/tree_split/tree.py`, locate the `use_stacked_blend` parameter in the `__init__` signature and change its default:

```python
use_stacked_blend: bool = True,
```

- [ ] **Step 2: Flip the default in `EmlSplitBoostRegressor`.**

In `eml_boost/tree_split/ensemble.py`, locate the `use_stacked_blend` parameter and change its default to True:

```python
use_stacked_blend: bool = True,
```

- [ ] **Step 3: Run the full unit test suite.**

Run:
```bash
uv run pytest tests/unit/ -v
```

Expected: all 20 tests pass. If `test_eml_leaf_activates_on_elementary_target` or `test_eml_leaf_gate_rejects_weak_fits` fail under the new default, the blended algorithm's behavior on those fixtures does not match the gated path's. Investigate — do NOT "fix" by forcing `use_stacked_blend=False` in the test; diagnose whether the blended behavior is genuinely wrong, or whether the test's tolerance needs a small relaxation.

- [ ] **Step 4: Commit.**

```bash
git add eml_boost/tree_split/tree.py eml_boost/tree_split/ensemble.py
git commit -m "$(cat <<'EOF'
feat: flip use_stacked_blend default to True

Make the blend the default leaf-fitting behavior. The gate path remains
reachable via use_stacked_blend=False for ablation and backward compat.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `_leaf_stats` instrumentation

**Goal:** Record per-leaf `(n_leaf, alpha, leaf_type)` triples on each `EmlSplitTreeRegressor` fit, so Experiment 9 can report "mean α across all EML leaves" and "fraction of attempted EML leaves that collapsed to constant."

**Files:**
- Modify: `eml_boost/tree_split/tree.py` (add attribute + populate in `_select_leaf_blended`)
- Modify: `tests/unit/test_eml_split_tree.py` (add one test)

- [ ] **Step 1: Write the failing test.**

Add this test at the bottom of `tests/unit/test_eml_split_tree.py`:

```python
def test_leaf_stats_populated_when_blend_enabled():
    """With `use_stacked_blend=True`, each leaf that reaches the EML
    decision point should append a record to `_leaf_stats` with the
    chosen α and the emitted leaf type."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("EML leaf fit requires CUDA")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(800, 2))
    y = np.exp(X[:, 0]) + 0.01 * rng.normal(size=800)
    m = EmlSplitTreeRegressor(
        max_depth=3, min_samples_leaf=20, n_eml_candidates=0,
        k_leaf_eml=1, min_samples_leaf_eml=30,
        use_stacked_blend=True, random_state=0,
    ).fit(X, y)
    stats = m._leaf_stats
    assert len(stats) >= 1
    for s in stats:
        assert s["n_leaf"] >= 30
        assert 0.0 <= s["alpha"] <= 1.0
        assert s["leaf_type"] in ("LeafNode", "EmlLeafNode")
    # On a clean exp(x_0) signal we expect at least one non-collapsed EML leaf.
    assert any(s["leaf_type"] == "EmlLeafNode" for s in stats)
```

- [ ] **Step 2: Run the test; confirm it fails with an AttributeError on `_leaf_stats`.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_stats_populated_when_blend_enabled -v
```

Expected: FAIL with `AttributeError: 'EmlSplitTreeRegressor' object has no attribute '_leaf_stats'`.

- [ ] **Step 3: Add `_leaf_stats` attribute initialization in `fit`.**

In `eml_boost/tree_split/tree.py`, locate the `fit` method (search for `def fit`). At the top of `fit`, immediately after any input-validation / coercion but before tree growth starts, add:

```python
self._leaf_stats: list[dict] = []
```

Place it alongside other per-fit state initialization (you'll see assignments like `self._X_cpu = ...`). It must be fresh on each `fit` call.

- [ ] **Step 4: Populate `_leaf_stats` in `_select_leaf_blended`.**

In `eml_boost/tree_split/tree.py`, in `_select_leaf_blended`, add record-keeping just before each of the two return statements.

Before the `return LeafNode(value=bias_folded)` (the collapse case):
```python
self._leaf_stats.append({
    "n_leaf": int(y_full.shape[0]),
    "alpha": alpha_star,
    "leaf_type": "LeafNode",
})
return LeafNode(value=bias_folded)
```

Before the `return EmlLeafNode(...)` (the normal case):
```python
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
```

Also add a record when the early validity check fails and we return a `LeafNode(value=constant_value)`:
```python
if not bool(valid_blend[best_idx].item()):
    self._leaf_stats.append({
        "n_leaf": int(y_full.shape[0]),
        "alpha": 1.0,
        "leaf_type": "LeafNode",
    })
    return LeafNode(value=constant_value)
```

- [ ] **Step 5: Run the test; confirm it passes.**

Run:
```bash
uv run pytest tests/unit/test_eml_split_tree.py::test_leaf_stats_populated_when_blend_enabled -v
```

Expected: PASS.

- [ ] **Step 6: Run the full unit test suite; confirm no regressions.**

Run:
```bash
uv run pytest tests/unit/ -v
```

Expected: all 21 tests pass.

- [ ] **Step 7: Commit.**

```bash
git add eml_boost/tree_split/tree.py tests/unit/test_eml_split_tree.py
git commit -m "$(cat <<'EOF'
feat: record per-leaf (n, alpha, type) stats during blended fit

Enables interpretability analysis in Experiment 9: mean α across all
EML leaves and fraction of EML-eligible leaves that collapsed to
constants.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Create the Experiment 9 runner

**Goal:** Produce `experiments/run_experiment9_stacked_blend.py` that loops over the 7 PMLB datasets × 3 seeds × 4 models (SplitBoost blend-off, SplitBoost blend-on, LightGBM, XGBoost), aggregates mean/std across seeds, and writes CSV/JSON/PNG/leaf-stats artifacts to `experiments/experiment9/`.

**Files:**
- Create: `experiments/run_experiment9_stacked_blend.py`
- Reference: `experiments/run_experiment8_pmlb_split.py` (fork basis)

- [ ] **Step 1: Create the runner file.**

Write `experiments/run_experiment9_stacked_blend.py` with the following complete contents:

```python
"""Experiment 9: PMLB regression benchmark with stacked-blend leaves.

Compares two configurations of EmlSplitBoostRegressor at the same commit:
  - blend-off: use_stacked_blend=False (legacy gate behavior).
  - blend-on:  use_stacked_blend=True  (val-fit convex blend per leaf).
Both evaluated against XGBoost and LightGBM at matched capacity, across
3 seeds, on the Experiment 8 dataset set. Reports mean±std of test RMSE
per dataset and per config, plus per-dataset leaf-stats aggregates for
the blend-on configuration.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

from eml_boost.tree_split import EmlSplitBoostRegressor

RESULTS_DIR = Path(__file__).resolve().parent / "experiment9"

DATASETS = [
    "192_vineyard",
    "210_cloud",
    "523_analcatdata_neavote",
    "557_analcatdata_apnea1",
    "529_pollen",
    "562_cpu_small",
    "564_fried",
]

MAX_ROUNDS = 200
DEPTH = 6
LEARNING_RATE = 0.1
N_EML_CANDIDATES = 10
K_EML = 3
K_LEAF_EML = 1
MIN_SAMPLES_LEAF_EML = 50
LEAF_EML_GAIN_THRESHOLD = 0.05
N_BINS = 256
TEST_SIZE = 0.20
SEEDS = [0, 1, 2]


@dataclass
class RunResult:
    dataset: str
    seed: int
    model: str
    rmse: float
    fit_time: float
    n_rounds: int = 0


def _fit_split_boost(X_tr, y_tr, seed, *, use_stacked_blend: bool):
    start = time.time()
    m = EmlSplitBoostRegressor(
        max_rounds=MAX_ROUNDS,
        max_depth=DEPTH,
        learning_rate=LEARNING_RATE,
        min_samples_leaf=20,
        n_eml_candidates=N_EML_CANDIDATES,
        k_eml=K_EML,
        n_bins=N_BINS,
        histogram_min_n=500,
        use_gpu=True,
        k_leaf_eml=K_LEAF_EML,
        min_samples_leaf_eml=MIN_SAMPLES_LEAF_EML,
        leaf_eml_gain_threshold=LEAF_EML_GAIN_THRESHOLD,
        use_stacked_blend=use_stacked_blend,
        patience=15,
        val_fraction=0.15,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _fit_lgb(X_tr, y_tr, seed):
    start = time.time()
    m = lgb.train(
        dict(
            objective="regression_l2",
            max_depth=DEPTH,
            num_leaves=2**DEPTH,
            min_data_in_leaf=20,
            learning_rate=LEARNING_RATE,
            device="gpu",
            seed=seed,
            verbose=-1,
        ),
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=MAX_ROUNDS,
    )
    return m, time.time() - start


def _fit_xgb(X_tr, y_tr, seed):
    start = time.time()
    m = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=DEPTH,
        n_estimators=MAX_ROUNDS,
        learning_rate=LEARNING_RATE,
        device="cuda",
        verbosity=0,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m, time.time() - start


def _rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))


def _collect_leaf_stats(boost: EmlSplitBoostRegressor) -> dict:
    """Aggregate per-leaf (α, type) records across every tree in the boost."""
    all_alphas: list[float] = []
    n_eml_leaves = 0
    n_const_leaves = 0
    for tree in boost._trees:
        for s in getattr(tree, "_leaf_stats", []):
            all_alphas.append(s["alpha"])
            if s["leaf_type"] == "EmlLeafNode":
                n_eml_leaves += 1
            else:
                n_const_leaves += 1
    total = n_eml_leaves + n_const_leaves
    return {
        "n_leaf_records": total,
        "n_eml_leaves": n_eml_leaves,
        "n_const_leaves": n_const_leaves,
        "eml_leaf_fraction": n_eml_leaves / total if total else 0.0,
        "alpha_mean": float(mean(all_alphas)) if all_alphas else float("nan"),
        "alpha_stdev": float(stdev(all_alphas)) if len(all_alphas) > 1 else 0.0,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[RunResult] = []
    leaf_stats: dict[str, dict[int, dict]] = {}  # dataset -> seed -> stats

    for name in DATASETS:
        print(f"\n=== dataset: {name} ===", flush=True)
        X, y = fetch_data(name, return_X_y=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        leaf_stats[name] = {}

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed,
            )
            print(
                f"  [seed={seed}] n={len(X):>6}  k={X.shape[1]:>3}  "
                f"train={len(X_tr)}  test={len(X_te)}",
                flush=True,
            )

            # blend-off
            m_off, t_off = _fit_split_boost(X_tr, y_tr, seed, use_stacked_blend=False)
            rmse_off = _rmse(m_off.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, model="split_boost_blend_off",
                rmse=rmse_off, fit_time=t_off, n_rounds=m_off.n_rounds,
            ))
            print(
                f"    SplitBoost/blend-off ({t_off:6.1f}s, "
                f"{m_off.n_rounds} rounds) RMSE={rmse_off:.4f}",
                flush=True,
            )

            # blend-on
            m_on, t_on = _fit_split_boost(X_tr, y_tr, seed, use_stacked_blend=True)
            rmse_on = _rmse(m_on.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, model="split_boost_blend_on",
                rmse=rmse_on, fit_time=t_on, n_rounds=m_on.n_rounds,
            ))
            print(
                f"    SplitBoost/blend-on  ({t_on:6.1f}s, "
                f"{m_on.n_rounds} rounds) RMSE={rmse_on:.4f}",
                flush=True,
            )
            leaf_stats[name][seed] = _collect_leaf_stats(m_on)

            # lightgbm
            m_lg, t_lg = _fit_lgb(X_tr, y_tr, seed)
            rmse_lg = _rmse(m_lg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, model="lightgbm",
                rmse=rmse_lg, fit_time=t_lg,
            ))
            print(f"    LightGBM             ({t_lg:6.1f}s) RMSE={rmse_lg:.4f}", flush=True)

            # xgboost
            m_xg, t_xg = _fit_xgb(X_tr, y_tr, seed)
            rmse_xg = _rmse(m_xg.predict(X_te), y_te)
            rows.append(RunResult(
                dataset=name, seed=seed, model="xgboost",
                rmse=rmse_xg, fit_time=t_xg,
            ))
            print(f"    XGBoost              ({t_xg:6.1f}s) RMSE={rmse_xg:.4f}", flush=True)

    # Per-(dataset, model) aggregates.
    agg: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r.dataset, r.model)
        agg.setdefault(key, {"rmses": [], "times": [], "n_rounds": []})
        agg[key]["rmses"].append(r.rmse)
        agg[key]["times"].append(r.fit_time)
        agg[key]["n_rounds"].append(r.n_rounds)
    for key, d in agg.items():
        d["rmse_mean"] = float(mean(d["rmses"]))
        d["rmse_std"] = float(stdev(d["rmses"])) if len(d["rmses"]) > 1 else 0.0
        d["time_mean"] = float(mean(d["times"]))

    # CSV
    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w") as fp:
        fp.write("dataset,seed,model,rmse,fit_time,n_rounds\n")
        for r in rows:
            fp.write(f"{r.dataset},{r.seed},{r.model},{r.rmse},{r.fit_time},{r.n_rounds}\n")
    print(f"\nwrote {csv_path}")

    # JSON with aggregates
    json_path = RESULTS_DIR / "summary.json"
    out: dict = {"config": {
        "max_rounds": MAX_ROUNDS, "depth": DEPTH,
        "learning_rate": LEARNING_RATE,
        "n_eml_candidates": N_EML_CANDIDATES, "k_eml": K_EML,
        "k_leaf_eml": K_LEAF_EML,
        "min_samples_leaf_eml": MIN_SAMPLES_LEAF_EML,
        "leaf_eml_gain_threshold": LEAF_EML_GAIN_THRESHOLD,
        "n_bins": N_BINS, "test_size": TEST_SIZE, "seeds": SEEDS,
    }, "per_run": [r.__dict__ for r in rows], "aggregate": {}}
    for (ds, model), d in agg.items():
        out["aggregate"].setdefault(ds, {})[model] = {
            "rmse_mean": d["rmse_mean"], "rmse_std": d["rmse_std"],
            "time_mean": d["time_mean"],
            "rmses_per_seed": d["rmses"], "n_rounds_per_seed": d["n_rounds"],
        }
    with json_path.open("w") as fp:
        json.dump(out, fp, indent=2)
    print(f"wrote {json_path}")

    # Leaf stats JSON
    leaf_json_path = RESULTS_DIR / "leaf_stats.json"
    with leaf_json_path.open("w") as fp:
        json.dump(leaf_stats, fp, indent=2)
    print(f"wrote {leaf_json_path}")

    # Plot: bars with error bars for blend-off, blend-on, xgboost.
    ordered = sorted(DATASETS, key=lambda n: agg[(n, "xgboost")]["rmse_mean"])
    means = {model: [agg[(n, model)]["rmse_mean"] for n in ordered]
             for model in ("split_boost_blend_off", "split_boost_blend_on", "xgboost")}
    stds = {model: [agg[(n, model)]["rmse_std"] for n in ordered]
            for model in ("split_boost_blend_off", "split_boost_blend_on", "xgboost")}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=110)
    xs = np.arange(len(ordered)); w = 0.25
    ax1.bar(xs - w, means["split_boost_blend_off"], w,
            yerr=stds["split_boost_blend_off"], color="#588157", label="blend-off")
    ax1.bar(xs,      means["split_boost_blend_on"], w,
            yerr=stds["split_boost_blend_on"], color="#2E86AB", label="blend-on")
    ax1.bar(xs + w,  means["xgboost"], w,
            yerr=stds["xgboost"], color="#9B2226", label="XGBoost")
    ax1.set_xticks(xs); ax1.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("test RMSE (mean ± std over 3 seeds)")
    ax1.set_title(f"Experiment 9: stacked-blend vs gate, {len(SEEDS)} seeds")
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="y")

    ratios_off = [agg[(n, "split_boost_blend_off")]["rmse_mean"]
                  / agg[(n, "xgboost")]["rmse_mean"] for n in ordered]
    ratios_on = [agg[(n, "split_boost_blend_on")]["rmse_mean"]
                 / agg[(n, "xgboost")]["rmse_mean"] for n in ordered]
    ax2.bar(xs - w/2, ratios_off, w, color="#588157", label="blend-off")
    ax2.bar(xs + w/2, ratios_on,  w, color="#2E86AB", label="blend-on")
    ax2.axhline(1.0, color="black", linewidth=1, label="parity")
    ax2.axhline(1.1, color="gray", linestyle="--", linewidth=1, label="within 10% band")
    ax2.set_xticks(xs); ax2.set_xticklabels(ordered, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("mean RMSE / XGBoost mean RMSE")
    ax2.set_title("Ratio vs. XGBoost")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "pmlb_rmse.png"
    plt.savefig(plot_path)
    print(f"wrote {plot_path}")

    # Console summary
    print("\n=== Aggregate summary (mean±std over 3 seeds) ===")
    print(
        f"{'dataset':>28}  {'off_mean':>9}  {'off_std':>7}  "
        f"{'on_mean':>9}  {'on_std':>7}  {'off/xgb':>7}  {'on/xgb':>7}"
    )
    for n in ordered:
        off = agg[(n, "split_boost_blend_off")]
        on = agg[(n, "split_boost_blend_on")]
        xg = agg[(n, "xgboost")]
        print(
            f"{n:>28}  {off['rmse_mean']:>9.4f}  {off['rmse_std']:>7.4f}  "
            f"{on['rmse_mean']:>9.4f}  {on['rmse_std']:>7.4f}  "
            f"{off['rmse_mean']/xg['rmse_mean']:>7.3f}  "
            f"{on['rmse_mean']/xg['rmse_mean']:>7.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the runner on a single small dataset.**

A dry-run over all 7 datasets × 3 seeds takes ~5 minutes. To validate the runner without that cost, temporarily edit `DATASETS` in place to only `["523_analcatdata_neavote"]` and `SEEDS` to `[0]`, run:

```bash
uv run python experiments/run_experiment9_stacked_blend.py
```

Expected: completes in under 1 minute; writes `experiments/experiment9/{summary.csv, summary.json, leaf_stats.json, pmlb_rmse.png}`; console prints the "Aggregate summary" table with one row.

Then **revert** `DATASETS` and `SEEDS` to the full values (all 7 datasets; `[0, 1, 2]`) before committing.

- [ ] **Step 3: Commit the runner.**

```bash
git add experiments/run_experiment9_stacked_blend.py
git commit -m "$(cat <<'EOF'
add: Experiment 9 runner for stacked-blend leaves

3 seeds × 4 models × 7 datasets. Compares blend-off vs blend-on at the
same commit against XGBoost and LightGBM. Writes summary CSV/JSON,
per-seed leaf-stats, and a PNG plot.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Run Experiment 9 and write the report

**Goal:** Execute the full runner and write `experiments/experiment9/report.md`. The report evaluates the success criteria (S-A/S-B/S-C) from the spec and decides whether the blend is a keeper.

**Files:**
- Execute: `experiments/run_experiment9_stacked_blend.py`
- Create: `experiments/experiment9/report.md`
- Commit: the outputs (`summary.csv`, `summary.json`, `leaf_stats.json`, `pmlb_rmse.png`) plus the report.

- [ ] **Step 1: Run the full benchmark.**

Run:
```bash
PYTHONUNBUFFERED=1 uv run python -u experiments/run_experiment9_stacked_blend.py 2>&1 | tee experiments/experiment9/run.log
```

Expected: runs in ~5-7 minutes on RTX 3090. Confirms all 84 fits complete (7 datasets × 3 seeds × 4 models). Writes `experiments/experiment9/{summary.csv, summary.json, leaf_stats.json, pmlb_rmse.png, run.log}`.

If any dataset's SplitBoost fit raises an exception, the runner will stop. Investigate before writing the report — this indicates a real bug, not a flaky benchmark. Re-run after fixing.

- [ ] **Step 2: Read the outputs.**

Run:
```bash
cat experiments/experiment9/summary.csv
cat experiments/experiment9/summary.json | python -m json.tool | head -80
cat experiments/experiment9/leaf_stats.json | python -m json.tool | head -40
```

Identify:
- Per-dataset mean ratio blend-off vs blend-on vs XGBoost.
- Per-dataset seed-std on the ratio.
- For blend-on: eml_leaf_fraction and alpha_mean per dataset.
- Number of datasets within 10% of XGBoost under each config.
- Which of S-A, S-B, S-C from the spec are satisfied.

- [ ] **Step 3: Write `experiments/experiment9/report.md`.**

Create the report file with the following structure; fill in the concrete numbers from Step 2. Use the Experiment 8 report (`experiments/experiment8/report.md`) as a template for tone and formatting.

```markdown
# Experiment 9: Stacked-Blend Leaves

**Date:** 2026-04-24
**Commit:** <fill in: output of `git rev-parse HEAD` at this point>
**Runtime:** <fill in: from run.log>
**Scripts:** `experiments/run_experiment9_stacked_blend.py`

## What the experiment was about

Experiment 8 added EML leaves (Phase 4) and closed the PMLB 7-dataset
gap to 5/7 outright wins against XGBoost, but regressed `562_cpu_small`
(ratio 0.81 → 0.90) and left the wins on `564_fried` (0.99) and
`529_pollen` (0.97) narrow enough that a single-seed result was noisy.
Experiment 9 replaces the binary accept/reject EML-leaf gate with a
val-fit convex blend `α·constant + (1−α)·EML`, and runs the full
benchmark across 3 seeds to simultaneously measure the blend's impact
and the seed-to-seed variance.

## Configuration

<copy-paste from summary.json config section>

## Results (mean ± std over 3 seeds)

| dataset | n | k | blend-off RMSE | blend-on RMSE | XGBoost RMSE | off/xgb | on/xgb | Δ ratio | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

<fill in one row per dataset with: mean ± std, ratios, and whether the
blend helped (Δ ratio < 0), hurt (Δ > 0), or was neutral (|Δ| < 0.01)>

**Within 10% of XGBoost (mean ratio < 1.10):** blend-off = <N>/7, blend-on = <N>/7.
**Outright beats XGBoost (mean ratio < 1.00):** blend-off = <N>/7, blend-on = <N>/7.

## Success criteria

- **S-A (primary):** 5/7 within-10% under blend-on across all 3 seeds,
  AND ≥ 1 dataset improves by ≥ 0.03 mean ratio: <MET / NOT MET>
- **S-B:** cpu_small mean ratio < 0.85 under blend-on with no win→loss
  switches: <MET / NOT MET>
- **S-C:** mean σ of ratios is lower under blend-on than blend-off: <MET / NOT MET>

Verdict: <BLEND IS A KEEPER / BLEND IS NEUTRAL / BLEND HURTS>

## Leaf-level behavior (blend-on)

| dataset | n_eml_leaves (summed over 3 seeds) | eml_leaf_fraction | α mean |
|---|---|---|---|
| ... | ... | ... | ... |

Observations:
- <e.g. "on `562_cpu_small` α mean is 0.42: the blend significantly
  shrank EML contributions, explaining the recovery from 0.90 → 0.85">
- <e.g. "on `529_pollen` α mean is 0.05: the blend barely shrinks,
  confirming the Phase-4 EML leaves were already well-calibrated there">

## What Experiment 9 actually shows

- <bullet pointing to the headline result, e.g. "the blend recovers
  cpu_small without regressing any other winner">
- <bullet on seed stability: is the variance actually lower under
  blend-on? how much?>
- <bullet on the remaining losses (210_cloud, 557_analcatdata_apnea1)
  — did the blend move them at all?>

## What's left as a loss

<per-dataset analysis of anything still > 1.10 ratio>

## What Experiment 9 does NOT show

- Does not test the full PMLB suite (55 datasets).
- Does not test CV (single 80/20 shuffle-split per seed).
- Does not test the capacity-unlocked regime.

## Consequence for the project

<one paragraph updating the headline claim from Experiment 8>

## Next possible experiments

<2-4 bullets — e.g. full PMLB, CV, capacity unlock, interpretability metric>
```

Fill in every `<…>` placeholder with the concrete numbers from Step 2. Do NOT leave placeholders in the committed file.

- [ ] **Step 4: Run the unit test suite one more time to confirm nothing drifted.**

Run:
```bash
uv run pytest tests/unit/ -v
```

Expected: all 21 tests pass.

- [ ] **Step 5: Commit the run outputs and report.**

```bash
git add experiments/experiment9/summary.csv experiments/experiment9/summary.json \
        experiments/experiment9/leaf_stats.json experiments/experiment9/pmlb_rmse.png \
        experiments/experiment9/report.md experiments/experiment9/run.log
git commit -m "$(cat <<'EOF'
exp 9 done: stacked-blend leaves on PMLB

3 seeds × 7 datasets. Report includes aggregate mean±std, seed variance,
leaf-level α distributions, and a keep/revert verdict on the blend
against the S-A/S-B/S-C criteria from the spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Act on the verdict.**

If the report's verdict is **KEEPER**: nothing more to do — the blend-on default from Task 3 stays, and the gate path remains reachable via `use_stacked_blend=False`.

If the report's verdict is **NEUTRAL** or **HURTS**: open a follow-up commit that flips the default back to `use_stacked_blend=False` in both `EmlSplitTreeRegressor.__init__` and `EmlSplitBoostRegressor.__init__`, and amend the report's "Consequence for the project" section to state why the blend was reverted. Do NOT delete the blended code path or its tests — they remain for future experiments.

---

## Self-review (plan author's checklist)

**Spec coverage:**
- Core algorithm change → Task 1 (refactor) + Task 2 (implement).
- Fold α into (η, β) → Task 2, step 7.
- No new node type → confirmed in Task 2 (emits existing `EmlLeafNode`).
- Tree selection under the blend (policy Y) → Task 2, step 7.
- `use_stacked_blend` config API with `leaf_eml_gain_threshold` retained but unused → Task 1 (param added; gate code path preserved) + Task 3 (default flip).
- `_leaf_stats` instrumentation → Task 4.
- Tests for blend behavior (α≈0 clean, α≈1 noise, heavy-tails stable, blend-off preserves gate) → Task 1 step 1 + Task 2 steps 1-7 + Task 4 step 1.
- Experiment 9 runner with 3 seeds × 4 models × 7 datasets → Task 5.
- Outputs: summary.csv/json, leaf_stats.json, pmlb_rmse.png, report.md → Task 5 + Task 6.
- Success criteria S-A/S-B/S-C evaluation → Task 6 step 3.
- Negative-outcome revert → Task 6 step 6.

No gaps.

**Placeholder scan:** No TBDs or "implement later" in any task body. Task 6 step 3's report template has concrete placeholders (`<fill in…>`) with explicit instructions to replace them — these are for run-time data, not plan gaps.

**Type consistency:** `_fit_leaf` dispatches to `_select_leaf_gated` or `_select_leaf_blended`, both with the same kwargs-only `**ctx` signature. `_leaf_stats` is a `list[dict]` with keys `{n_leaf: int, alpha: float, leaf_type: str}` consistently in Task 4. `EmlLeafNode` signature matches existing usage (checked against `eml_boost/tree_split/nodes.py`). Runner imports `EmlSplitBoostRegressor` from the package's existing public surface.
