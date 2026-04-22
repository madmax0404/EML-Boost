"""Parameterized EML tree (paper's 'master formula'), PyTorch nn.Module.

Implements a depth-n full binary tree of EML nodes. Each input position
is a (k+2)-way softmax over [1, x_1, ..., x_k, f_prev] for internal
inputs, or (k+1)-way over [1, x_1, ..., x_k] for leaf inputs (no f_prev).

After argmax-snap, each input resolves to a single terminal; the tree
becomes a finite elementary expression.
"""

from __future__ import annotations

import torch
from torch import nn

from eml_boost._numerics import eml


class MasterFormula(nn.Module):
    """Depth-n, k-feature parameterized EML tree.

    Parameters
    ----------
    depth : int
        Tree depth n (number of EML levels). depth >= 1.
    k : int
        Number of input features the tree sees.
    """

    def __init__(self, depth: int, k: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.depth = depth
        self.k = k
        # Counts:
        # - leaf inputs (no f_prev option): 2^depth of them
        # - internal inputs (have f_prev option): 2^depth - 2 when depth >= 2, else 0
        internal_input_count = (2**depth) - 2 if depth >= 2 else 0
        leaf_input_count = 2**depth

        self.internal_input_count = internal_input_count
        self.leaf_input_count = leaf_input_count

        internal_dim = k + 2  # [1, x_1, ..., x_k, f_prev]
        leaf_dim = k + 1      # [1, x_1, ..., x_k]

        # Layout convention — internal logits are stored DEEPEST FIRST:
        #   indices [0 .. 2^(depth-1) - 1): inputs of EML nodes at level depth-2
        #   ...
        #   last two indices: inputs of the root EML node
        # Leaf logits follow, in node order 0..2^(depth-1)-1.
        # Both MasterFormula.forward and snapped_to_sympy assume this layout.
        self.logits_list = nn.ParameterList(
            [nn.Parameter(torch.randn(internal_dim)) for _ in range(internal_input_count)]
            + [nn.Parameter(torch.randn(leaf_dim)) for _ in range(leaf_input_count)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the EML tree on a batch of inputs.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, k), dtype complex128
            Feature vectors.

        Returns
        -------
        torch.Tensor, shape (batch,), dtype complex128
            Tree output for each sample.
        """
        batch = x.shape[0]
        device = x.device
        ones = torch.ones(batch, dtype=torch.complex128, device=device)

        # Leaf-input softmax terminals: [1, x_1, ..., x_k]
        leaf_terminals = torch.cat([ones.unsqueeze(1), x], dim=1)  # (batch, k+1)

        # Compute outputs of EML nodes at deepest level (level depth-1).
        # These nodes have 2 LEAF inputs each.
        leaf_logits = self.logits_list[self.internal_input_count:]
        node_outputs = self._apply_eml_level(leaf_terminals, leaf_logits, n_nodes=2 ** (self.depth - 1))

        # Walk up the tree, from the deepest internal level to the root.
        # Per the layout convention (deepest first), each iteration consumes
        # logits from the front of the remaining internals list.
        internal_logits = list(self.logits_list[:self.internal_input_count])

        for level in range(self.depth - 2, -1, -1):
            n_nodes_at_level = 2 ** level
            level_logit_count = 2 * n_nodes_at_level
            level_logits = internal_logits[:level_logit_count]
            internal_logits = internal_logits[level_logit_count:]

            node_outputs = self._apply_eml_level_internal(
                x=x,
                ones=ones,
                child_outputs=node_outputs,
                logits_list=level_logits,
                n_nodes=n_nodes_at_level,
            )

        # node_outputs now has shape (batch, 1): the root EML output.
        return node_outputs.squeeze(1)

    @staticmethod
    def _apply_eml_level(
        terminals: torch.Tensor,  # (batch, n_terminals)
        logits_list,  # flat list of 2*n_nodes tensors
        n_nodes: int,
    ) -> torch.Tensor:
        """Apply EML to pairs of softmax-weighted terminals.

        Returns (batch, n_nodes) complex128.
        """
        outs = []
        for i in range(n_nodes):
            left_logits = logits_list[2 * i]
            right_logits = logits_list[2 * i + 1]
            left = _softmax_weighted(terminals, left_logits)
            right = _softmax_weighted(terminals, right_logits)
            outs.append(eml(left, right))
        return torch.stack(outs, dim=1)  # (batch, n_nodes)

    @staticmethod
    def _apply_eml_level_internal(
        x: torch.Tensor,
        ones: torch.Tensor,
        child_outputs: torch.Tensor,  # (batch, 2*n_nodes)
        logits_list,  # flat list of 2*n_nodes tensors
        n_nodes: int,
    ) -> torch.Tensor:
        outs = []
        for i in range(n_nodes):
            left_logits = logits_list[2 * i]
            right_logits = logits_list[2 * i + 1]
            left_terminals = torch.cat(
                [ones.unsqueeze(1), x, child_outputs[:, 2 * i].unsqueeze(1)], dim=1
            )
            right_terminals = torch.cat(
                [ones.unsqueeze(1), x, child_outputs[:, 2 * i + 1].unsqueeze(1)], dim=1
            )
            left = _softmax_weighted(left_terminals, left_logits)
            right = _softmax_weighted(right_terminals, right_logits)
            outs.append(eml(left, right))
        return torch.stack(outs, dim=1)  # (batch, n_nodes)


def _softmax_weighted(terminals: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Return softmax(logits) dot terminals, broadcasting along batch.

    terminals: (batch, dim), complex.
    logits: (dim,), real.
    Returns: (batch,), complex.
    """
    weights = torch.softmax(logits, dim=0).to(terminals.dtype)  # (dim,)
    return terminals @ weights  # (batch,)
