#!/usr/bin/env python3
"""Regression tests for block-equivariant many-well NHMC."""

from __future__ import annotations

import torch

from neural_nhmc_dw import (
    BlockEquivariantMomentumNet,
    ContinuousNHMCKernel,
    ManyWellTarget,
    rollout,
)
from train_block_dw_nhmc import training_loss


def permute_blocks(x: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1, 2)[:, permutation].reshape_as(x)


def test_block_permutation_equivariance() -> None:
    torch.manual_seed(3)
    network = BlockEquivariantMomentumNet(8, hidden=32).to(torch.float64)
    for parameter in network.parameters():
        if parameter.ndim:
            parameter.data.normal_(0.0, 0.1)
    x = torch.randn(7, 8, dtype=torch.float64)
    t = torch.rand(7, dtype=torch.float64)
    permutation = torch.tensor([2, 0, 3, 1])
    expected = permute_blocks(network(x, t), permutation)
    actual = network(permute_blocks(x, permutation), t)
    error = float((actual - expected).abs().max().detach())
    assert error < 1.0e-10, error


def test_stagewise_epsilon_and_rollout_are_finite() -> None:
    torch.manual_seed(5)
    target = ManyWellTarget(8)
    kernel = ContinuousNHMCKernel(
        target,
        n_leapfrog=2,
        hidden=32,
        architecture="block_equivariant",
        num_steps=4,
        stagewise_epsilon=True,
    )
    kernel.log_epsilon_offsets.data.copy_(torch.tensor([0.0, -0.2, 0.1, -0.1, 0.2]))
    _, work = rollout(kernel, target, 32, 4, torch.device("cpu"))
    assert work.shape == (32,)
    assert torch.isfinite(work).all()
    assert float(kernel.epsilon_at(1).detach()) != float(kernel.epsilon_at(2).detach())


def test_heavy_tail_loss_has_gradients() -> None:
    work = torch.tensor([-8.0, -1.0, 0.0, 2.0], requires_grad=True)
    loss, log_inverse_ess = training_loss(work, var_weight=0.02, ess_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert float(log_inverse_ess.detach()) > 0.0
    assert torch.isfinite(work.grad).all()


def test_legacy_state_dict_compatibility() -> None:
    target = ManyWellTarget(4)
    original = ContinuousNHMCKernel(target, n_leapfrog=2, hidden=16)
    restored = ContinuousNHMCKernel(target, n_leapfrog=2, hidden=16)
    restored.load_state_dict(original.state_dict())


if __name__ == "__main__":
    test_block_permutation_equivariance()
    test_stagewise_epsilon_and_rollout_are_finite()
    test_heavy_tail_loss_has_gradients()
    test_legacy_state_dict_compatibility()
    print("BLOCK_DW_REGRESSION_OK")
