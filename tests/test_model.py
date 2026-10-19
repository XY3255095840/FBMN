import torch
from torch import nn

from fbmn import (
    FBMN,
    FactorizedBasisBranch,
    FourierBranch,
    MultiBasisFBMN,
    RBFBranch,
)


class DummyBasis(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 2

    def forward(self, x):
        return torch.cat((x, torch.ones_like(x)), dim=1)


def test_factorized_branch_matches_manual_recurrence():
    branch = FactorizedBasisBranch(
        2, [DummyBasis(), DummyBasis()], dims=[0, 1]
    )
    branch.mappings[0].weight.data = torch.tensor([[1.0], [2.0]])
    branch.mappings[0].bias.data = torch.zeros(2)
    branch.mappings[1].weight.data = torch.eye(2)
    branch.mappings[1].bias.data = torch.tensor([1.0, -1.0])
    x = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(branch(x), torch.tensor([[9.0, 1.0]]))


def test_all_affine_layers_use_bias():
    branch = FourierBranch(2, [1.0], dims=[0, 1])
    assert all(mapping.bias is not None for mapping in branch.mappings)

    model = FBMN(2, branch, hidden_dims=[4])
    assert model.output_layer.bias is not None

    multi = MultiBasisFBMN(
        2,
        [branch, RBFBranch(2, 0.0, 1.0, 4, dims=[0, 1])],
        hidden_dims=[4],
        fusion_dim=8,
    )
    assert multi.encoder_projection.bias is not None
    assert all(
        projection.bias is not None
        for projection in multi.basis_projections
    )
    assert multi.output_layer.bias is not None


def test_single_and_multi_basis_forward_shapes():
    x = torch.rand(5, 2)
    branch = FourierBranch(2, [1.0], dims=[0, 1])
    model = FBMN(2, branch, hidden_dims=[4])
    assert model(x).shape == (5, 1)

    multi = MultiBasisFBMN(
        2,
        [branch, RBFBranch(2, 0.0, 1.0, 4, dims=[0, 1])],
        hidden_dims=[4],
        fusion_dim=8,
    )
    assert multi(x).shape == (5, 1)

    loss = model(x).square().mean() + multi(x).square().mean()
    loss.backward()
    assert model.output_layer.weight.grad is not None
    assert multi.output_layer.weight.grad is not None
