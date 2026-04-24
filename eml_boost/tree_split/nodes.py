"""Node types for elementary-split regression trees.

Internal nodes split on either a raw feature (`RawSplit`) or an elementary
expression over a feature subset (`EmlSplit`). Leaves store a constant
value (for squared-error regression, the mean of residuals in the leaf).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from eml_boost.symbolic.snap import SnappedTree


@dataclass(frozen=True)
class RawSplit:
    """Axis-aligned split: go left iff X[:, feature_idx] <= threshold."""

    feature_idx: int
    threshold: float


@dataclass(frozen=True)
class EmlSplit:
    """Elementary-expression split.

    go left iff eval(snapped_tree on X[:, feature_subset]) <= threshold.
    """

    snapped: SnappedTree
    feature_subset: tuple[int, ...]  # indices into the full feature matrix
    threshold: float


@dataclass
class LeafNode:
    """Terminal: predict this constant for every sample reaching here."""

    value: float


@dataclass
class InternalNode:
    """Non-terminal: route samples left/right via `split`."""

    split: Union[RawSplit, EmlSplit]
    left: "Node"
    right: "Node"


Node = Union[LeafNode, InternalNode]
