import numpy as np
import pytest

from eml_boost.weak_learners.eml import EmlWeakLearner, fit_eml_tree


def test_fit_returns_weak_learner():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X[:, 0] ** 2 - np.log(np.abs(X[:, 1]) + 1)
    h = fit_eml_tree(X, y, depth=2, n_restarts=3, k=2, random_state=0)
    assert isinstance(h, EmlWeakLearner)
    # The fit should at least produce predictions of the right shape
    preds = h.predict(X)
    assert preds.shape == y.shape


def test_fit_raises_on_k_too_large():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = rng.normal(size=40)
    with pytest.raises(ValueError, match="k=5"):
        fit_eml_tree(X, y, depth=2, n_restarts=1, k=5, random_state=0)


def test_random_state_none_succeeds_twice():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(60, 2))
    y = X[:, 0]
    h1 = fit_eml_tree(X, y, depth=2, n_restarts=2, k=2, random_state=None)
    h2 = fit_eml_tree(X, y, depth=2, n_restarts=2, k=2, random_state=None)
    assert h1 is not None and h2 is not None
    # Predictions shape check on both
    assert h1.predict(X).shape == (60,)
    assert h2.predict(X).shape == (60,)


@pytest.mark.slow
def test_fit_recovers_simple_formula():
    """Depth-2 fit on y = exp(x) should recover the formula on most seeds."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1.5, 1.5, size=(200, 1))
    y = np.exp(X[:, 0])
    h = fit_eml_tree(X, y, depth=2, n_restarts=20, k=1, random_state=42)
    residual_rms = np.sqrt(np.mean((h.predict(X) - y) ** 2))
    assert residual_rms < 0.1, f"fit should approximate exp well; got {residual_rms}"
