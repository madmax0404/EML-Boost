import torch

from eml_boost.symbolic.master_formula import MasterFormula


class TestMasterFormulaShape:
    def test_depth_2_k_3_has_expected_logits(self):
        # 62 pre-snap logits for depth 3, k 3 per spec section 5.1.
        # Depth 2, k 3 has fewer — let's just assert they exist and shape is right.
        mf = MasterFormula(depth=2, k=3)
        assert mf.logits_list  # non-empty ParameterList
        # Forward pass shape: batch in, batch out
        x = torch.randn(5, 3, dtype=torch.complex128)
        out = mf(x)
        assert out.shape == (5,)

    def test_depth_3_batch_forward(self):
        mf = MasterFormula(depth=3, k=3)
        x = torch.randn(11, 3, dtype=torch.complex128)
        out = mf(x)
        assert out.shape == (11,)
        assert out.dtype == torch.complex128


class TestMasterFormulaLogits:
    def test_all_logits_trainable(self):
        mf = MasterFormula(depth=2, k=2)
        trainable = [p for p in mf.parameters() if p.requires_grad]
        assert len(trainable) == len(mf.logits_list)


class TestMasterFormulaGradient:
    def test_backward_pass_produces_gradients(self):
        mf = MasterFormula(depth=2, k=2)
        x = torch.randn(4, 2, dtype=torch.complex128)
        out = mf(x)
        loss = out.real.pow(2).mean()
        loss.backward()
        for p in mf.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
