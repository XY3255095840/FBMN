# FBMN: Factorized Basis Modulation Network

A compact PyTorch implementation of the FBMN architecture and its
physics-informed training loop. The repository contains the model, the trainer,
and one runnable example.

## Installation

Python 3.10+ and PyTorch 2.0+ are required.

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Quick Start

```python
import torch

from fbmn import FBMN, FourierBranch

frequencies = [1.5 + index for index in range(20)]
branch = FourierBranch(
    input_dim=3,
    frequencies=frequencies,
    dims=[0, 1, 2],
)
model = FBMN(
    input_dim=3,
    branch=branch,
    hidden_dims=[32, 64, 32],
)
print(model(torch.rand(8, 3)).shape)  # [8, 1]
```

Each basis unit returns its basis channels followed by a constant channel. The
branch applies

```text
h0 = 1
h_j = (W_j h_{j-1} + b_j) * Phi_j(x_j)
```

and the model computes

```text
u = W_out (E(x) * h_d) + b_out
```

All affine mappings and output layers use bias, following Eqs. (7)-(15).

## Multiple Bases

```python
from fbmn import FourierBranch, MultiBasisFBMN, RBFBranch

fourier = FourierBranch(
    input_dim=2,
    frequencies=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    dims=[0, 1],
)
rbf = RBFBranch(
    input_dim=2,
    low=0.0,
    high=1.0,
    num_centers=95,
    dims=[0, 1],
)
model = MultiBasisFBMN(
    input_dim=2,
    branches=[fourier, rbf],
    hidden_dims=[32, 64, 64],
    fusion_dim=64,
)
```

`FourierBranch` and `RBFBranch` apply one shared configuration to every selected
coordinate. For per-coordinate frequencies, domains, or basis sizes, construct
a `FactorizedBasisBranch` with one basis unit per coordinate.

## Training

```python
from fbmn import PINNTrainer, TrainingConfig

trainer = PINNTrainer(
    model,
    TrainingConfig(),
    loss_weights={"pde": 1.0, "bc": 20.0, "ic": 20.0},
)
history = trainer.fit(sample_fn, residual_fn)
```

`sample_fn` and `residual_fn` are user-defined; `examples/heat.py` provides a
complete implementation.

`sample_fn(round_index)` returns a dictionary of tensors. The
`residual_fn(model, batch)` callback returns named pointwise residuals. The
trainer resamples once per round, runs Adam in float32, and applies
double-precision L-BFGS in the final rounds.

## Example

The included example solves the high-frequency heat equation from Section 4.3.

```bash
python examples/heat.py --device auto
```

For a quick smoke run:

```bash
python examples/heat.py --device cpu --interior 256 --boundary 64 \
  --initial 64 --rounds 1 --adam-steps 2 --lbfgs-rounds 1 --lbfgs-steps 1
```

Each run writes `history.json`, `result.json`, and `model.pt` to `runs/heat`
unless another output directory is supplied.

## Code Map

| Code | Paper |
|---|---|
| `FourierBasis`, `GaussianRBFBasis` | Eqs. (10), (11) |
| `FactorizedBasisBranch` | Eqs. (6)-(8) |
| `FBMN` | Eq. (9) |
| `MultiBasisFBMN` | Eqs. (12)-(15) |
| `PINNTrainer` | Eq. (5), Section 4.2 |

## Notes

- The default schedule uses 30 rounds of 2,000 Adam steps at `1e-3`, followed
  by five rounds of 300 L-BFGS outer steps at `0.1` in float64. Each outer step
  uses an internal L-BFGS cap of 20 iterations.
- The heat example uses trainable Fourier frequencies initialized at
  `1.5, 2.5, ..., 20.5`. Within the paper's 5,000 boundary and initial points,
  it uses 3,000 boundary points and 2,000 initial points.

## Citation

Yang Xu, Zihao Li, Wenbo Shen. *Factorized basis modulation network: A flexible
and efficient basis-utilization framework for solving multiscale PDEs*.
Neurocomputing 709 (2027), 135308. DOI: `10.1016/j.neucom.2026.135308`.

MIT license; see [LICENSE](LICENSE).
