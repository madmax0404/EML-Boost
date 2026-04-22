import numpy as np

from eml_boost.weak_learners.dt import DtWeakLearner, fit_dt_stump


def test_fit_returns_dt_weak_learner():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(float) * 2 - 1
    h = fit_dt_stump(X, y, depth=3)
    assert isinstance(h, DtWeakLearner)
    assert h.predict(X).shape == y.shape


def test_fit_handles_missing_values():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    X[::5, 0] = np.nan
    y = np.where(np.isnan(X[:, 0]), 1.0, X[:, 1])
    h = fit_dt_stump(X, y, depth=3)
    pred = h.predict(X)
    assert np.isfinite(pred).all()


def test_params_count_is_leaves():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = X[:, 0] + X[:, 1]
    h = fit_dt_stump(X, y, depth=3)
    assert h.params_count() >= 1
    assert h.params_count() <= 2**3
