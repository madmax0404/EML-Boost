"""Weak learner protocol and round record used by the boosting loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class WeakLearnerKind(str, Enum):
    EML = "EML"
    DT = "DT"


@runtime_checkable
class WeakLearner(Protocol):
    """Common interface for EML and DT weak learners."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def params_count(self) -> int:
        ...


@dataclass
class RoundRecord:
    """Per-round bookkeeping emitted by the boosting loop."""

    round_index: int
    kind: WeakLearnerKind
    eta: float
    score: float
    mse_inner_val: float
