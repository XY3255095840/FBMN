"""Coordinate-wise basis units with a constant bypass channel."""

import math

import torch
from torch import nn


def _as_vector(values, name):
    vector = torch.as_tensor(values, dtype=torch.get_default_dtype()).detach().clone()
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    return vector


class FourierBasis(nn.Module):
    """Interleaved sine/cosine features followed by a constant channel."""

    def __init__(self, frequencies, trainable=True):
        super().__init__()
        frequencies = _as_vector(frequencies, "frequencies")
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.out_dim = 2 * frequencies.numel() + 1

    def forward(self, x):
        angles = math.pi * x * self.frequencies.reshape(1, -1)
        values = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        values = values.reshape(x.shape[0], -1)
        return torch.cat((values, x.new_ones((x.shape[0], 1))), dim=1)


class GaussianRBFBasis(nn.Module):
    """Gaussian kernels with fixed centers and positive log-widths."""

    def __init__(self, centers, widths, trainable_widths=True):
        super().__init__()
        centers = _as_vector(centers, "centers")
        widths = _as_vector(widths, "widths")
        if widths.numel() == 1 and centers.numel() > 1:
            widths = widths.expand_as(centers).clone()
        if widths.numel() != centers.numel():
            raise ValueError("widths must be scalar or match centers")
        if torch.any(widths <= 0):
            raise ValueError("widths must be positive")

        self.register_buffer("centers", centers)
        log_widths = torch.log(widths)
        if trainable_widths:
            self.log_widths = nn.Parameter(log_widths)
        else:
            self.register_buffer("log_widths", log_widths)
        self.out_dim = centers.numel() + 1

    @classmethod
    def from_interval(cls, low, high, num_centers, width_scale=1.2, **kwargs):
        if num_centers < 2:
            raise ValueError("num_centers must be at least 2")
        low, high = float(low), float(high)
        if high <= low:
            raise ValueError("high must exceed low")
        width_scale = float(width_scale)
        if width_scale <= 0:
            raise ValueError("width_scale must be positive")
        centers = torch.linspace(low, high, num_centers)
        spacing = (high - low) / (num_centers - 1)
        widths = torch.full_like(centers, spacing * width_scale)
        return cls(centers, widths, **kwargs)

    def forward(self, x):
        widths = torch.exp(self.log_widths).reshape(1, -1)
        centers = self.centers.reshape(1, -1)
        values = torch.exp(-0.5 * ((x - centers) / widths).square())
        return torch.cat((values, x.new_ones((x.shape[0], 1))), dim=1)
