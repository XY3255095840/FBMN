"""Physics-informed training utilities."""

from dataclasses import dataclass

import torch
from torch import nn


def partial_derivative(values, inputs, dim, order=1):
    """Differentiate a [batch, 1] field with respect to one input coordinate."""
    if inputs.ndim != 2 or not inputs.requires_grad:
        raise ValueError(
            "inputs must be [batch, dimensions] with requires_grad=True"
        )
    if values.shape != (inputs.shape[0], 1):
        raise ValueError("values must have shape [batch, 1]")
    if not isinstance(dim, int) or not 0 <= dim < inputs.shape[1]:
        raise ValueError("dim must select an input coordinate")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")

    result = values
    for _ in range(order):
        if result.requires_grad:
            gradient = torch.autograd.grad(
                result,
                inputs,
                grad_outputs=torch.ones_like(result),
                create_graph=True,
                allow_unused=True,
            )[0]
        else:
            gradient = None
        result = (
            inputs[:, dim : dim + 1] * 0
            if gradient is None
            else gradient[:, dim : dim + 1]
        )
    return result


def laplacian(values, inputs, dims=None):
    """Sum second partial derivatives along the selected coordinates."""
    dims = tuple(range(inputs.shape[1])) if dims is None else tuple(dims)
    if not dims or len(set(dims)) != len(dims):
        raise ValueError("dims must be a nonempty set of distinct coordinates")
    return sum(
        partial_derivative(values, inputs, dim, order=2) for dim in dims
    )


def weighted_residual_loss(residuals, weights):
    """Return weighted mean squares and the per-term components."""
    if not residuals or set(residuals) != set(weights):
        raise ValueError("residuals and weights must have the same keys")
    components = {
        name: value.square().mean() for name, value in residuals.items()
    }
    total = sum(components[name] * weights[name] for name in components)
    return total, components


@dataclass(frozen=True)
class TrainingConfig:
    rounds: int = 30
    adam_steps: int = 2000
    adam_lr: float = 1e-3
    lbfgs_rounds: int = 5
    lbfgs_steps: int = 300
    lbfgs_lr: float = 0.1
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 50

    def __post_init__(self):
        if self.rounds < 1 or self.adam_steps < 1:
            raise ValueError("rounds and adam_steps must be positive")
        if not 0 <= self.lbfgs_rounds <= self.rounds:
            raise ValueError("lbfgs_rounds must be between 0 and rounds")
        if self.lbfgs_steps < 1 or self.lbfgs_max_iter < 1:
            raise ValueError("lbfgs steps and max_iter must be positive")


class PINNTrainer:
    """Resample, run Adam, and refine with double-precision L-BFGS."""

    def __init__(
        self,
        model,
        config=None,
        loss_weights=None,
        extra_parameters=(),
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.loss_weights = dict(loss_weights or {"pde": 1.0})
        self.extra_parameters = tuple(extra_parameters)
        self.history = []

    def fit(self, sample_fn, residual_fn, on_round_end=None):
        model_parameters = tuple(self.model.parameters())
        parameters = [
            parameter
            for parameter in (
                *model_parameters,
                *self.extra_parameters,
            )
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("model has no trainable parameters")

        reference = model_parameters[0] if model_parameters else parameters[0]
        device = reference.device
        entry_dtype = reference.dtype
        extra_dtypes = tuple(p.dtype for p in self.extra_parameters)
        if any(parameter.device != device for parameter in parameters):
            raise ValueError("all trainable parameters must share one device")

        def set_precision(dtype):
            self.model.to(dtype=dtype)
            for parameter in self.extra_parameters:
                parameter.data = parameter.data.to(dtype=dtype)
                parameter.grad = None

        def move(batch, dtype):
            return {
                name: value.detach().to(
                    device=device,
                    dtype=dtype if value.is_floating_point() else value.dtype,
                )
                for name, value in batch.items()
            }

        def loss(batch):
            total, components = weighted_residual_loss(
                residual_fn(self.model, batch), self.loss_weights
            )
            if not torch.isfinite(total):
                raise FloatingPointError("training loss became nonfinite")
            return total, components

        self.history = []
        try:
            set_precision(torch.float32)
            adam = torch.optim.Adam(parameters, lr=self.config.adam_lr)
            for round_index in range(self.config.rounds):
                raw_batch = sample_fn(round_index)
                set_precision(torch.float32)
                batch = move(raw_batch, torch.float32)
                self.model.train()

                for _ in range(self.config.adam_steps):
                    adam.zero_grad(set_to_none=True)
                    total, _ = loss(batch)
                    total.backward()
                    adam.step()

                total, components = loss(batch)
                adam_loss = float(total.detach())
                lbfgs_loss = None

                if round_index >= self.config.rounds - self.config.lbfgs_rounds:
                    set_precision(torch.float64)
                    batch = move(raw_batch, torch.float64)
                    lbfgs = torch.optim.LBFGS(
                        parameters,
                        lr=self.config.lbfgs_lr,
                        max_iter=self.config.lbfgs_max_iter,
                        history_size=self.config.lbfgs_history_size,
                        line_search_fn="strong_wolfe",
                    )

                    def closure():
                        lbfgs.zero_grad(set_to_none=True)
                        value, _ = loss(batch)
                        value.backward()
                        return value

                    for _ in range(self.config.lbfgs_steps):
                        lbfgs.step(closure)
                    total, components = loss(batch)
                    lbfgs_loss = float(total.detach())

                record = {
                    "round": round_index + 1,
                    "loss": float(total.detach()),
                    "adam_loss": adam_loss,
                    "lbfgs_loss": lbfgs_loss,
                    **{
                        name: float(value.detach())
                        for name, value in components.items()
                    },
                }
                self.history.append(record)
                if on_round_end is not None:
                    on_round_end(dict(record))
        finally:
            set_precision(entry_dtype)
            for parameter, dtype in zip(self.extra_parameters, extra_dtypes):
                parameter.data = parameter.data.to(dtype=dtype)
                parameter.grad = None
        return self.history
