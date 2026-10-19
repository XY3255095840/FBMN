"""Single- and multi-basis factorized basis modulation networks."""

from torch import nn

from .bases import FourierBasis, GaussianRBFBasis


def _init_linears(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)


class MLPEncoder(nn.Module):
    """A tanh MLP mapping coordinates to a neural feature."""

    def __init__(self, input_dim, widths):
        super().__init__()
        layers = []
        previous = input_dim
        for width in widths:
            layers.extend((nn.Linear(previous, width, bias=True), nn.Tanh()))
            previous = width
        self.network = nn.Sequential(*layers)
        self.out_dim = previous
        _init_linears(self.network)

    def forward(self, x):
        return self.network(x)


class FactorizedBasisBranch(nn.Module):
    """Chain coordinate basis units with affine mixing and modulation."""

    def __init__(self, input_dim, bases, dims=None):
        super().__init__()
        bases = list(bases)
        dims = tuple(range(input_dim)) if dims is None else tuple(dims)
        if not dims or len(set(dims)) != len(dims):
            raise ValueError("dims must be nonempty and unique")
        if any(
            not isinstance(dim, int) or dim < 0 or dim >= input_dim
            for dim in dims
        ):
            raise ValueError("dims contain an invalid coordinate")
        if len(bases) != len(dims):
            raise ValueError("len(bases) must match len(dims)")

        self.input_dim = input_dim
        self.dims = dims
        self.bases = nn.ModuleList(bases)
        self.mappings = nn.ModuleList()
        previous = 1
        for basis in self.bases:
            self.mappings.append(nn.Linear(previous, basis.out_dim, bias=True))
            previous = basis.out_dim
        self.out_dim = previous
        _init_linears(self.mappings)

    def forward(self, x):
        hidden = x.new_ones((x.shape[0], 1))
        for dim, basis, mapping in zip(self.dims, self.bases, self.mappings):
            hidden = mapping(hidden) * basis(x[:, dim : dim + 1])
        return hidden


class FourierBranch(FactorizedBasisBranch):
    """Convenience branch with one Fourier unit per selected coordinate."""

    def __init__(self, input_dim, frequencies, dims=None, trainable=True):
        dims = tuple(range(input_dim)) if dims is None else tuple(dims)
        bases = [FourierBasis(frequencies, trainable=trainable) for _ in dims]
        super().__init__(input_dim, bases, dims)


class RBFBranch(FactorizedBasisBranch):
    """Convenience branch with one Gaussian RBF unit per coordinate."""

    def __init__(
        self,
        input_dim,
        low,
        high,
        num_centers,
        dims=None,
        width_scale=1.2,
        trainable_widths=True,
    ):
        dims = tuple(range(input_dim)) if dims is None else tuple(dims)
        bases = [
            GaussianRBFBasis.from_interval(
                low,
                high,
                num_centers,
                width_scale=width_scale,
                trainable_widths=trainable_widths,
            )
            for _ in dims
        ]
        super().__init__(input_dim, bases, dims)


class FBMN(nn.Module):
    """Single-basis factorized basis modulation network."""

    def __init__(
        self,
        input_dim,
        branch,
        hidden_dims=(32, 64, 32),
        output_dim=1,
    ):
        super().__init__()
        if branch.input_dim != input_dim:
            raise ValueError("branch input_dim must match model input_dim")
        self.encoder = MLPEncoder(
            input_dim, tuple(hidden_dims) + (branch.out_dim,)
        )
        self.branch = branch
        self.output_layer = nn.Linear(branch.out_dim, output_dim, bias=True)
        _init_linears(self.output_layer)

    def forward(self, x):
        return self.output_layer(self.encoder(x) * self.branch(x))


class MultiBasisFBMN(nn.Module):
    """Fuse several basis branches and the MLP feature in a shared space."""

    def __init__(
        self,
        input_dim,
        branches,
        hidden_dims=(32, 64, 64),
        fusion_dim=64,
        output_dim=1,
    ):
        super().__init__()
        branches = list(branches)
        if not branches:
            raise ValueError("branches must be nonempty")
        if any(branch.input_dim != input_dim for branch in branches):
            raise ValueError("every branch input_dim must match model input_dim")

        self.encoder = MLPEncoder(input_dim, hidden_dims)
        self.branches = nn.ModuleList(branches)
        self.encoder_projection = nn.Linear(
            self.encoder.out_dim, fusion_dim, bias=True
        )
        self.basis_projections = nn.ModuleList(
            nn.Linear(branch.out_dim, fusion_dim, bias=True)
            for branch in branches
        )
        self.output_layer = nn.Linear(fusion_dim, output_dim, bias=True)
        _init_linears(self.encoder_projection)
        _init_linears(self.basis_projections)
        _init_linears(self.output_layer)

    def forward(self, x):
        features = self.encoder_projection(self.encoder(x))
        for branch, projection in zip(self.branches, self.basis_projections):
            features = features * projection(branch(x))
        return self.output_layer(features)
