import numpy as np

from eml_boost.selection import bic_score, select_winner
from eml_boost.weak_learners.base import WeakLearnerKind
from eml_boost.weak_learners.dt import fit_dt_stump
from eml_boost.weak_learners.eml import fit_eml_tree


def test_bic_score_penalizes_more_params():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    y = rng.normal(size=100)
    constant_prediction = np.zeros_like(y)
    score_one_param = bic_score(y, constant_prediction, params=1)
    score_many_params = bic_score(y, constant_prediction, params=20)
    assert score_many_params > score_one_param


def test_select_winner_returns_tuple():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] ** 2
    X_iv = X[150:]
    y_iv = y[150:]

    eml = fit_eml_tree(X[:150], y[:150], depth=2, n_restarts=2, k=2, random_state=0)
    dt = fit_dt_stump(X[:150], y[:150], depth=2)
    winner, kind, eta, score = select_winner(eml, dt, X_iv, y_iv)
    assert kind in (WeakLearnerKind.EML, WeakLearnerKind.DT)
    assert isinstance(eta, float)
    assert isinstance(score, float)
