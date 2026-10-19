"""High-frequency heat equation example from the paper."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from fbmn import (
    FBMN,
    FourierBranch,
    PINNTrainer,
    TrainingConfig,
    laplacian,
    partial_derivative,
)


def exact_solution(x, m=18, n=19):
    """Return exp(-t) sin(m*pi*x) sin(n*pi*y) for coordinates [x, y, t]."""
    return (
        torch.exp(-x[:, 2:3])
        * torch.sin(m * math.pi * x[:, 0:1])
        * torch.sin(n * math.pi * x[:, 1:2])
    )


def heat_residual(model, points, m=18, n=19):
    """Evaluate u_t - c(u_xx + u_yy) without interior labels."""
    points = points.detach().clone().requires_grad_(True)
    u = model(points)
    c = 1.0 / (math.pi**2 * (m**2 + n**2))
    return partial_derivative(u, points, 2) - c * laplacian(
        u, points, dims=(0, 1)
    )


class HeatSampler:
    """Draw interior, spatial-boundary, and initial-face points each round."""

    def __init__(
        self,
        n_interior=20000,
        n_boundary=3000,
        n_initial=2000,
        seed=42,
    ):
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, _round_index):
        interior = torch.rand(
            self.n_interior, 3, generator=self.generator
        )
        boundary = torch.rand(self.n_boundary, 3, generator=self.generator)
        sides = torch.arange(self.n_boundary) % 4
        boundary[torch.arange(self.n_boundary), sides // 2] = (
            sides % 2
        ).float()
        initial = torch.rand(self.n_initial, 3, generator=self.generator)
        initial[:, 2] = 0.0
        return {
            "interior": interior,
            "boundary": boundary,
            "initial": initial,
        }


def residuals(model, batch, m=18, n=19):
    return {
        "pde": heat_residual(model, batch["interior"], m, n),
        "bc": model(batch["boundary"]),
        "ic": model(batch["initial"])
        - exact_solution(batch["initial"], m, n),
    }


def evaluate(model, m=18, n=19, nx=128, ny=128, nt=20):
    axes = [torch.linspace(0.0, 1.0, count) for count in (nx, ny, nt)]
    points = torch.stack(
        torch.meshgrid(*axes, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    parameter = next(model.parameters())
    points = points.to(device=parameter.device, dtype=parameter.dtype)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        prediction = model(points)
        reference = exact_solution(points, m, n)
        relative_l2 = (prediction - reference).norm() / reference.norm()
        maximum = (prediction - reference).abs().max()
    model.train(was_training)
    return float(relative_l2), float(maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interior", type=int, default=20000)
    parser.add_argument("--boundary", type=int, default=3000)
    parser.add_argument("--initial", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-rounds", type=int, default=5)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("runs/heat"))
    args = parser.parse_args()

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if device == "auto":
        device = "cpu"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    branch = FourierBranch(
        input_dim=3,
        dims=[0, 1, 2],
        frequencies=[1.5 + index for index in range(20)],
    )
    model = FBMN(
        input_dim=3,
        branch=branch,
        hidden_dims=[32, 64, 32],
    ).to(device)
    sampler = HeatSampler(
        n_interior=args.interior,
        n_boundary=args.boundary,
        n_initial=args.initial,
        seed=args.seed,
    )
    trainer = PINNTrainer(
        model,
        TrainingConfig(
            rounds=args.rounds,
            adam_steps=args.adam_steps,
            lbfgs_rounds=args.lbfgs_rounds,
            lbfgs_steps=args.lbfgs_steps,
        ),
        loss_weights={"pde": 1.0, "bc": 20.0, "ic": 20.0},
    )

    start = time.perf_counter()
    history = trainer.fit(
        sampler,
        residuals,
        on_round_end=lambda record: print(
            f"round {record['round']}: loss={record['loss']:.3e}"
        ),
    )
    elapsed = time.perf_counter() - start
    relative_l2, maximum = evaluate(model)
    result = {
        "relative_l2": relative_l2,
        "max_absolute_error": maximum,
        "training_seconds": elapsed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    print(json.dumps(result, indent=2))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(model.state_dict(), args.output / "model.pt")


if __name__ == "__main__":
    main()
