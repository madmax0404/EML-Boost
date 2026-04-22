"""Argmax-snap of parameterized EML logits to a fixed structural tree.

After snap, each input position resolves to an integer terminal index:
  0       -> constant 1
  1..k    -> input feature x_{index-1}
  k+1     -> previous EML output (f_prev, only valid for internal inputs)
"""

from __future__ import annotations

from dataclasses import dataclass

from eml_boost.symbolic.master_formula import MasterFormula

CONSTANT_ONE_IDX = 0


@dataclass(frozen=True)
class SnappedTree:
    """A SnappedTree records the argmax choice for each input position.

    terminal_choices[i] is the chosen terminal index for input position i,
    where i ranges over the same ordering as MasterFormula.logits_list
    (internals first, then leaves).
    """

    depth: int
    k: int
    internal_input_count: int
    leaf_input_count: int
    terminal_choices: tuple[int, ...]


def snap_master_formula(mf: MasterFormula) -> SnappedTree:
    """Argmax each logit vector, return the resulting structural tree."""
    choices = tuple(int(vec.argmax().item()) for vec in mf.logits_list)
    return SnappedTree(
        depth=mf.depth,
        k=mf.k,
        internal_input_count=mf.internal_input_count,
        leaf_input_count=mf.leaf_input_count,
        terminal_choices=choices,
    )


def count_active_positions(snapped: SnappedTree) -> int:
    """Count input positions that are NOT snapped to the constant 1.

    Proxy complexity measure used as a BIC params() fallback when the
    RPN length of the simplified expression is unavailable.
    """
    return sum(1 for c in snapped.terminal_choices if c != CONSTANT_ONE_IDX)
