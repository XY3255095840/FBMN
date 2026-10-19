import math

import torch

from fbmn import FourierBasis, GaussianRBFBasis


def test_fourier_basis_interleaves_channels_and_adds_constant():
    frequencies = [1.0, 2.0]
    basis = FourierBasis(frequencies)
    x = torch.tensor([[0.25], [0.5]])
    result = basis(x)
    angles = math.pi * x * torch.tensor(frequencies)
    expected = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    expected = expected.reshape(x.shape[0], -1)
    torch.testing.assert_close(result[:, :-1], expected)
    torch.testing.assert_close(result[:, -1], torch.ones(x.shape[0]))
    assert result.shape == (2, 5)
    assert basis.frequencies.requires_grad


def test_fixed_fourier_frequencies_are_buffers():
    basis = FourierBasis([1.0], trainable=False)
    assert not any(parameter.requires_grad for parameter in basis.parameters())
    assert "frequencies" in dict(basis.named_buffers())


def test_rbf_basis_has_constant_channel_and_trainable_widths():
    basis = GaussianRBFBasis.from_interval(0.0, 1.0, 5)
    x = torch.linspace(0.0, 1.0, 7).reshape(-1, 1)
    result = basis(x)
    assert result.shape == (7, 6)
    torch.testing.assert_close(result[:, -1], torch.ones(7))
    assert basis.log_widths.requires_grad
