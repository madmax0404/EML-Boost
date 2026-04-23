import numpy as np

from eml_boost.training import boost
from eml_boost.weak_learners.base import WeakLearnerKind


def test_boost_reduces_residual():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 2))
    y = X[:, 0] ** 2 + 0.1 * rng.normal(size=300)

    model = boost(
        X,
        y,
        M=15,
        depth_eml=2,
        depth_dt=2,
        n_restarts=3,
        k=2,
        patience=5,
        random_state=0,
    )
    pred = model.predict(X)
    initial_var = np.var(y)
    residual_var = np.var(y - pred)
    assert residual_var < initial_var


def test_boost_returns_history():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = X[:, 0]
    model = boost(
        X, y, M=5, depth_eml=2, depth_dt=2, n_restarts=2, k=2, random_state=0
    )
    assert len(model.history) <= 5
    for rec in model.history:
        assert rec.kind in (WeakLearnerKind.EML, WeakLearnerKind.DT)
