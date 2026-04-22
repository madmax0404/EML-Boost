import torch

from eml_boost.symbolic.master_formula import MasterFormula
from eml_boost.symbolic.snap import (
    SnappedTree,
    count_active_positions,
    snap_master_formula,
)


class TestSnap:
    def test_snap_produces_one_hot_vectors(self):
        mf = MasterFormula(depth=2, k=2)
        snapped = snap_master_formula(mf)
        for vec in snapped.terminal_choices:
            # Each choice is an integer index in range [0, dim)
            assert 0 <= vec < (mf.k + 2)  # at most, for internal inputs

    def test_snap_returns_snapped_tree(self):
        mf = MasterFormula(depth=2, k=2)
        snapped = snap_master_formula(mf)
        assert isinstance(snapped, SnappedTree)
        assert snapped.depth == 2
        assert snapped.k == 2

    def test_count_active_positions_lte_total(self):
        mf = MasterFormula(depth=2, k=2)
        snapped = snap_master_formula(mf)
        total_positions = mf.internal_input_count + mf.leaf_input_count
        assert 0 <= count_active_positions(snapped) <= total_positions
