import math

import torch

from fbmn import (
    FBMN,
    FourierBranch,
    PINNTrainer,
    TrainingConfig,
    laplacian,
    partial_derivative,
    weighted_residual_loss,
)


def test_partial_derivative_and_laplacian():
    x = torch.rand(6, 2, dtype=torch.float64, requires_grad=True)
    u = x[:, 0:1].square() + x[:, 1:2].square()
    torch.testing.assert_close(
        partial_derivative(u, x, 0), 2.0 * x[:, 0:1]
    )
    torch.testing.assert_close(
        laplacian(u, x), torch.full_like(u, 4.0)
    )


def test_higher_derivative_of_linear_field_is_zero():
    x = torch.rand(4, 2, dtype=torch.float64, requires_grad=True)
    u = x[:, 0:1]
    torch.testing.assert_close(
        partial_derivative(u, x, 0, order=2),
        torch.zeros_like(u),
    )


def test_weighted_residual_loss():
    residuals = {
        "a": torch.ones(4, 1),
        "b": torch.full((4, 1), 2.0),
    }
    total, components = weighted_residual_loss(
        residuals, {"a": 1.0, "b": 2.0}
    )
    torch.testing.assert_close(total, torch.tensor(9.0))
    torch.testing.assert_close(components["a"], torch.tensor(1.0))
    torch.testing.assert_close(components["b"], torch.tensor(4.0))


def test_trainer_smoke():
    torch.manual_seed(0)
    model = FBMN(
        1,
        FourierBranch(1, [1.0], dims=[0]),
        hidden_dims=[8],
    )

    def sample_fn(_round_index):
        return {"x": torch.rand(16, 1)}

    def residual_fn(network, batch):
        x = batch["x"].detach().clone().requires_grad_(True)
        return {"pde": network(x) - torch.sin(math.pi * x)}

    trainer = PINNTrainer(
        model,
        TrainingConfig(
            rounds=1,
            adam_steps=2,
            lbfgs_rounds=1,
            lbfgs_steps=1,
            lbfgs_max_iter=2,
        ),
        loss_weights={"pde": 1.0},
    )
    history = trainer.fit(sample_fn, residual_fn)
    assert len(history) == 1
    assert math.isfinite(history[0]["loss"])
    assert model.output_layer.weight.dtype == torch.float32


def test_trainer_optimizes_extra_parameter():
    torch.manual_seed(0)
    model = FBMN(
        1,
        FourierBranch(1, [1.0], dims=[0]),
        hidden_dims=[8],
    )
    parameter = torch.nn.Parameter(torch.tensor(0.0))

    def sample_fn(_round_index):
        return {"x": torch.rand(16, 1)}

    def residual_fn(network, batch):
        return {
            "pde": network(batch["x"])
            - parameter
            - torch.sin(math.pi * batch["x"])
        }

    trainer = PINNTrainer(
        model,
        TrainingConfig(
            rounds=1,
            adam_steps=5,
            lbfgs_rounds=0,
        ),
        loss_weights={"pde": 1.0},
        extra_parameters=[parameter],
    )
    trainer.fit(sample_fn, residual_fn)
    assert parameter.item() != 0.0
    assert parameter.dtype == torch.float32
